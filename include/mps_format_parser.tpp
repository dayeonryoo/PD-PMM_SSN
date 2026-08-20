#pragma once
#include "mps_format_parser.hpp"

// Default choices of the parser:
//  - Column bounds: default is [0, +inf) unless overridden by a BOUNDS entry. A negative UP
//    value with no other explicit lower-bound entry (LO/FX/MI/BV/FR) for that column relaxes
//    the lower bound to -inf.
//  - Row bounds: E/L/G rows follow [rhs-|range|, rhs+|range|];
//    a RANGES entry's sign is ignored (only its magnitude matters).
//    A second N-type row (i.e. not the objective) is always free (-inf, +inf),
//    and any RHS or RANGES entry against it is accepted but silently ignored.
//  - Objective: the first N row encountered becomes the objective row;
//    a file with no N row at all gets an implicit "OBJ" row.
//    An RHS entry naming the objective row is added to obj_const,
//    and to_kspqp() negates obj_const on the way out since it was
//    read off the RHS side of the equation.
//  - RHS/RANGES/BOUNDS set names (e.g. "RHS", "BND") are read only to
//    determine token layout on a given line; they are never validated for
//    consistency across lines, so a second block under a different set
//    name still applies on top of the first.
//  - Duplicate entries for the same (row, col) in COLUMNS, or the same
//    (row, col) pair in QUADOBJ, are summed rather than overwritten or
//    rejected. QUADOBJ is stored lower-triangular only: (i, j) with i < j
//    is swapped to (j, i), and the upper triangle is never populated.
//  - Format detection (fixed vs. free columns) happens once per file, from
//    the first value-bearing line: fixed-column parsing is attempted first,
//    and the file is only concluded to be free-format if that fixed-column
//    reading disagrees with a whitespace-delimited reading of the same
//    line. Once decided, the format is used for the rest of the file.
//  - Bounds/RHS/RANGES magnitudes at or beyond to_kspqp()'s inf_cap are
//    treated as infinite, and lower/upper pairs within eq_tol of each other
//    are snapped to their exact midpoint (inclusive at diff == eq_tol).

template <typename T>
ParsedModel<T> MpsFormatParser<T>::parse(const std::string& filename) {
    model_ = ParsedModel<T>();
    section_ = Section::NONE;
    format_ = Format::UNKNOWN;
    row_map_.clear(); col_map_.clear();
    A_triplets_.clear(); Q_triplets_.clear();
    obj_name_.clear();
    rhs_values_.clear(); range_values_.clear();
    col_lower_explicit_.clear();
    ws_tokens_.clear(); tokens_.clear();

    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open MPS file: " + filename);

    std::string line;
    while (std::getline(f, line)) {
        if (is_comment_or_blank(line)) continue;

        // Section header parsing.
        split_ws(line, ws_tokens_);
        bool is_header = set_section(ws_tokens_);
        if (section_ == Section::ENDATA) break;
        if (is_header) continue; // Skip section header lines.

        decide_format_from(line, section_);

        // Section content parsing depending on format. Reuses ws_tokens_
        // (already split above) instead of re-scanning the line.
        tokenize_line(line, section_);
        switch (section_) {
            case Section::OBJSENSE: parse_objsense(tokens_); break;
            case Section::ROWS: parse_rows(tokens_); break;
            case Section::COLUMNS: parse_columns(tokens_); break;
            case Section::RHS: parse_rhs(tokens_); break;
            case Section::RANGES: parse_ranges(tokens_); break;
            case Section::BOUNDS: parse_bounds(tokens_); break;
            case Section::QUADOBJ: parse_quadobj(tokens_); break;
            default: break; // Ignore lines outside of known sections.
        }
    }

    finalize_defaults();
    finalize_row_bounds();
    build_sparse_matrices();

    return model_;
}

template <typename T>
KSPQPdata<T> MpsFormatParser<T>::to_kspqp(const ParsedModel<T>& model, T eq_tol, T inf_cap) {
    using Vec = typename MpsFormatParser<T>::Vec;
    using SpMat = typename MpsFormatParser<T>::SpMat;

    auto is_inf = [inf_cap](T val) {
        return std::isinf((double)val) || std::abs((double)val) >= (double)inf_cap;
    };

    auto cap_inf = [inf_cap, is_inf](T val) {
        if (is_inf(val)) return (val > 0 ? inf_cap : -inf_cap);
        else return val;
    };

    KSPQPdata<T> pd;
    pd.n = model.num_cols;

    pd.obj_const = -model.obj_const; // Negate since it's from RHS.
    pd.c = model.c;
    pd.lx = model.col_lower;
    pd.ux = model.col_upper;
    for (int i = 0; i < pd.n; ++i) {
        pd.lx(i) = cap_inf(pd.lx(i));
        pd.ux(i) = cap_inf(pd.ux(i));
        // Snap near-constant bounds to their exact midpoint.
        if (!is_inf(pd.lx(i)) && !is_inf(pd.ux(i)) && std::abs(pd.lx(i) - pd.ux(i)) <= eq_tol) {
            T mid = T(0.5) * (pd.lx(i) + pd.ux(i));
            pd.lx(i) = mid;
            pd.ux(i) = mid;
        }
    }

    if (model.is_qp) pd.Q = model.Q;
    else { pd.Q = SpMat(pd.n, pd.n); pd.Q.setZero(); }

    // If MAX, convert to MIN by negating objective.
    if (!model.is_min) {
        pd.c = -pd.c;
        if (model.is_qp) pd.Q.coeffs() *= -1;
    }

    // Process constraints.
    std::vector<int> eq_rows, ineq_rows;
    eq_rows.reserve(model.num_rows);
    ineq_rows.reserve(model.num_rows);

    for (int i = 0; i < model.num_rows; ++i) {
        T lb = cap_inf(model.row_lower(i));
        T ub = cap_inf(model.row_upper(i));
        if (is_inf(lb) && is_inf(ub)) {
            continue; // Skip free constraints.
        } else if (!is_inf(lb) && !is_inf(ub) && std::abs(lb - ub) <= eq_tol) {
            eq_rows.push_back(i);
        } else {
            ineq_rows.push_back(i);
        }
    }

    pd.m = (int)eq_rows.size();
    pd.l = (int)ineq_rows.size();

    // Map each original row to its compacted index within its partition (-1 if in
    // neither, i.e. a free row). Mutually exclusive, so between them every row of
    // model.A is claimed by at most one of pd.A/pd.B below.
    std::vector<int> eq_row_map(model.num_rows, -1), ineq_row_map(model.num_rows, -1);

    pd.b = Vec(pd.m);
    for (int i = 0; i < (int)eq_rows.size(); ++i) {
        eq_row_map[eq_rows[i]] = i;
        pd.b(i) = model.row_lower(eq_rows[i]);
    }
    pd.lw = Vec(pd.l);
    pd.uw = Vec(pd.l);
    for (int i = 0; i < (int)ineq_rows.size(); ++i) {
        int r = ineq_rows[i];
        ineq_row_map[r] = i;
        pd.lw(i) = model.row_lower(r);
        pd.uw(i) = model.row_upper(r);
    }

    // Construct A (equality rows) and B (inequality rows) together in a single pass
    // over model.A. eq_row_map/ineq_row_map partition its nonzeros between the two,
    // so together they hold at most nnz(model.A) entries -- reserving nnz(model.A)
    // for each separately, as two independent passes would, double-counts peak
    // memory. A cheap counting pass gets the exact split; and since model.A's
    // InnerIterator is already column-major with strictly increasing row indices --
    // a property eq_row_map/ineq_row_map preserve within each partition -- the fill
    // pass can append straight into pd.A/pd.B's compressed storage via
    // reserve()+startVec()+insertBack(), skipping the Triplet stage (and
    // setFromTriplets's own working copy) entirely.
    Eigen::Index eq_nnz = 0, ineq_nnz = 0;
    for (int col = 0; col < model.A.outerSize(); ++col)
        for (typename SpMat::InnerIterator it(model.A, col); it; ++it) {
            if (eq_row_map[it.row()] != -1) ++eq_nnz;
            else if (ineq_row_map[it.row()] != -1) ++ineq_nnz;
        }

    pd.A = SpMat(pd.m, pd.n);
    pd.B = SpMat(pd.l, pd.n);
    pd.A.reserve(eq_nnz);
    pd.B.reserve(ineq_nnz);
    for (int col = 0; col < model.A.outerSize(); ++col) {
        pd.A.startVec(col);
        pd.B.startVec(col);
        for (typename SpMat::InnerIterator it(model.A, col); it; ++it) {
            const int r = it.row();
            const int eq_loc = eq_row_map[r];
            if (eq_loc != -1) {
                pd.A.insertBack(eq_loc, col) = it.value();
                continue;
            }
            const int ineq_loc = ineq_row_map[r];
            if (ineq_loc != -1) pd.B.insertBack(ineq_loc, col) = it.value();
        }
    }
    pd.A.finalize();
    pd.B.finalize();

    return pd;
}

template <typename T>
bool MpsFormatParser<T>::set_section(const std::vector<std::string_view>& tokens) {
    if (tokens.empty()) return false;
    if (tokens.size() != 1) return false;

    std::string_view sec = tokens[0];
    if      (sec == "NAME")     { section_ = Section::NAME;     return true;}
    else if (sec == "OBJSENSE") { section_ = Section::OBJSENSE; return true;}
    else if (sec == "ROWS")     { section_ = Section::ROWS;     return true;}
    else if (sec == "COLUMNS")  { section_ = Section::COLUMNS;  return true;}
    else if (sec == "RHS")      { section_ = Section::RHS;      return true;}
    else if (sec == "RANGES")   { section_ = Section::RANGES;   return true;}
    else if (sec == "BOUNDS")   { section_ = Section::BOUNDS;   return true;}
    else if (sec == "QUADOBJ")  { section_ = Section::QUADOBJ;  model_.is_qp = true; return true;}
    else if (sec == "ENDATA")   { section_ = Section::ENDATA;   return true;}
    
    return false; // Not a section header.
}

template <typename T>
void MpsFormatParser<T>::decide_format_from(const std::string& line, Section sec) {
    if (format_ != Format::UNKNOWN) return; // already decided for this file

    // Sections that are value-bearing (i.e. have a fixed-column layout):
    switch (sec) {
        case Section::ROWS: case Section::COLUMNS: case Section::RHS:
        case Section::RANGES: case Section::BOUNDS: case Section::QUADOBJ:
            break;
        default:
            return;
    }

    std::vector<std::string_view> fixed;
    split_fixed_by_section(line, sec, fixed);
    std::vector<std::string_view> free;
    split_free_by_section(ws_tokens_, sec, free); // reuse this line's whitespace split
    // Disagreement means the fixed reading was a false positive.
    // Conclude free-format for the rest of this file.
    format_ = (!fixed.empty() && fixed == free) ? Format::FIXED : Format::FREE;
}

template <typename T>
void MpsFormatParser<T>::parse_objsense(const std::vector<std::string_view>& tokens) {
    if (tokens.size() == 1) sense_ = tokens[0];
    else if (tokens.size() >= 2 && (tokens[0] == "OBJSENSE" || tokens[0] == "'OBJSENSE'")) sense_ = tokens[1];
    else sense_ = tokens.back(); // Take last token as sense.

    if (sense_ == "MIN") {
        model_.is_min = true;
    } else if (sense_ == "MAX") {
        model_.is_min = false;
    } else {
        throw std::runtime_error("Unknown optimization sense in OBJSENSE section: " + sense_);
    }
}

template <typename T>
void MpsFormatParser<T>::parse_rows(const std::vector<std::string_view>& tokens) {
    // Free-format ROWS line: <type> <row_name>.
    if (tokens.size() < 2)
        throw std::runtime_error("Malformed ROWS line: expected at least 2 tokens, got " +
                                  std::to_string(tokens.size()) + ".");
    char type = tokens[0].empty() ? 'N' : tokens[0][0]; // Default to 'N' if type is missing.
    std::string rname(tokens[1]);

    RowInfo info;
    info.type = type;

    if (type == 'N') {
        if (obj_name_.empty()) {
            obj_name_ = rname;
            info.idx = -1;
        } else {
            // Non-objective row with type 'N' is treated as a constraint row with no bounds (free) in finalize_row_bounds().
            info.idx = model_.num_rows++;
        }
        row_map_[std::move(rname)] = info;
    } else {
        // try_emplace does the find-or-insert in a single hash lookup, instead
        // of a separate find() followed by an unconditional operator[] insert.
        auto [it, inserted] = row_map_.try_emplace(std::move(rname), info);
        if (!inserted) {
            throw std::runtime_error("Duplicate row name in ROWS section: " + it->first);
        }
        it->second.idx = model_.num_rows++; // Assign new index for constraint row.
    }
}

template <typename T>
void MpsFormatParser<T>::parse_columns(const std::vector<std::string_view>& tokens) {
    // Free-format COLUMNS line: <col_name> {<row_name> <value>} ...
    if (tokens.size() < 3)
        throw std::runtime_error("Malformed COLUMNS line: expected at least 3 tokens, got " +
                                  std::to_string(tokens.size()) + ".");
    if (tokens.size() >= 2 && (tokens[1] == "MARKER" || tokens[1] == "'MARKER'")) return; // Ignore marker lines.

    // try_emplace does the find-or-insert in a single hash lookup.
    auto [cit, inserted] = col_map_.try_emplace(std::string(tokens[0]), model_.num_cols);
    int col_idx = cit->second;
    if (inserted) ++model_.num_cols; // Assign new index for variable.

    // Ensure c is large enough to hold the coefficient for this variable.
    int c_size = model_.c.size();
    if (c_size < model_.num_cols) {
        model_.c.conservativeResize(model_.num_cols);
        model_.c.segment(c_size, model_.num_cols - c_size).setZero(); // Initialize new entries to 0.
    }

    for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
        std::string_view rname = tokens[i];
        T value = static_cast<T>(std::stod(std::string(tokens[i + 1])));

        auto it = row_map_.find(std::string(rname));
        if (it == row_map_.end()) {
            throw std::runtime_error("Row name in COLUMNS section not defined in ROWS section: " + std::string(rname));
        }
        const RowInfo& info = it->second;
        if (rname == obj_name_) {
            // Objective coefficient
            model_.c(col_idx) += value;
        } else {
            // Constraint matrix entry
            if (info.idx < 0) {
                throw std::runtime_error("Invalid row index for constraint in COLUMNS section: " + std::string(rname));
            }
            A_triplets_.emplace_back(info.idx, col_idx, value);
        }
    }
}

template <typename T>
void MpsFormatParser<T>::parse_rhs(const std::vector<std::string_view>& tokens) {
    // Free-format RHS line: <rhs_name> {<row_name> <value>} ...
    if (tokens.size() < 2)
        throw std::runtime_error("Malformed RHS line: expected at least 2 tokens, got " +
                                  std::to_string(tokens.size()) + ".");

    // The RHS set name (if present) is read only to detect which token
    // layout this line uses; it is not tracked or validated across lines.
    const size_t start_idx = (tokens.size() == 3 || tokens.size() == 5) ? 1 : 0;

    // Ensure rhs_values is large enough to hold the RHS for all constraints.
    if ((int)rhs_values_.size() < model_.num_rows) {
        rhs_values_.resize(model_.num_rows, static_cast<T>(0)); // Default RHS is 0.
    }

    for (size_t i = start_idx; i + 1 < tokens.size(); i += 2) {
        std::string_view rname = tokens[i];
        T value = static_cast<T>(std::stod(std::string(tokens[i + 1])));

        auto it = row_map_.find(std::string(rname));
        if (it == row_map_.end()) {
            throw std::runtime_error("Row name in RHS section not defined in ROWS section: " + std::string(rname));
        }
        int idx = it->second.idx;
        if (idx < 0) {
            if (it->second.type == 'N') model_.obj_const += value; // Add to objective constant if it's the objective row.
            continue;
        }

        // Store the RHS value by row index.
        rhs_values_[idx] = value;
    }
}

template <typename T>
void MpsFormatParser<T>::parse_ranges(const std::vector<std::string_view>& tokens) {
    // Free-format RANGES line: <range_name> {<row_name> <value>} ...
    if (tokens.size() < 3)
        throw std::runtime_error("Malformed RANGES line: expected at least 3 tokens, got " +
                                  std::to_string(tokens.size()) + ".");

    // Ensure range_values is large enough to hold the range for all constraints.
    if ((int)range_values_.size() < model_.num_rows) {
        range_values_.resize(model_.num_rows, static_cast<T>(0)); // Default range is 0 (no range).
    }

    for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
        std::string_view rname = tokens[i];
        T value = static_cast<T>(std::stod(std::string(tokens[i + 1])));

        auto it = row_map_.find(std::string(rname));
        if (it == row_map_.end()) {
            throw std::runtime_error("Row name in RANGES section not defined in ROWS section: " + std::string(rname));
        }
        int idx = it->second.idx;
        if (idx < 0) continue; // Ignore range for objective row or invalid row.

        // Store the range value by row index.
        range_values_[idx] = value;
    }
}

template <typename T>
void MpsFormatParser<T>::parse_bounds(const std::vector<std::string_view>& tokens) {
    // Free-format BOUNDS line:
    // <bound_type> <bound_name> <col_name> {<value>} ...
    // OR <bound_type> <col_name> {<value>} ... (if bound name is omitted, use default)

    if (tokens.size() < 2)
        throw std::runtime_error("Malformed BOUNDS line: expected at least 2 tokens, got " +
                                  std::to_string(tokens.size()) + ".");

    const T inf = std::numeric_limits<T>::infinity();
    std::string_view btype = tokens[0];

    const bool needs_value = (btype == "LO" || btype == "UP" || btype == "FX" || btype == "LI" || btype == "UI");

    std::string_view cname;
    std::string_view value_str;

    if (tokens.size() >= 4) {
        // <bound_type> <bound_name> <col_name> <value>
        cname = tokens[2];
        value_str = tokens[3];
    } else if (tokens.size() == 3) {
        // tokens[1] could be either bound name or column name.
        // If tokens[1] is an existing column name or tokens[2] parses as a number,
        // treat it as column name and use default bound name.
        auto is_number = [](std::string_view s) {
            if (s.empty()) return false;
            try {
                size_t pos;
                std::stod(std::string(s), &pos);
                return pos == s.size();
            } catch (...) {
                return false;
            }
        };

        if (col_map_.find(std::string(tokens[1])) != col_map_.end() || is_number(tokens[2])) {
            cname = tokens[1];
            value_str = tokens[2];
        } else {
            cname = tokens[2];
        }
    } else {
        // tokens.size == 2: we only have bound type and column name, no bound name or value.
        cname = tokens[1];
    }

    // Check if value is provided when required.
    if (needs_value && value_str.empty()) {
        throw std::runtime_error("Bound type " + std::string(btype) + " requires a value in BOUNDS section.");
    }

    T value = T(0);
    if (!value_str.empty()) value = static_cast<T>(std::stod(std::string(value_str)));

    // Get or create column index (try_emplace: one hash lookup, not two).
    auto [it, inserted] = col_map_.try_emplace(std::string(cname), model_.num_cols);
    int col_idx = it->second;
    if (inserted) ++model_.num_cols; // Assign new index for variable.

    // Ensure col_lower and col_upper are large enough to hold bounds for this variable.
    int size = model_.col_lower.size();
    if (size < model_.num_cols) {
        model_.col_lower.conservativeResize(model_.num_cols);
        model_.col_upper.conservativeResize(model_.num_cols);
        model_.col_lower.segment(size, model_.num_cols - size).setZero();         // Default lower bound is 0.
        model_.col_upper.segment(size, model_.num_cols - size).setConstant(inf);  // Default upper bound is inf.
    }
    if (static_cast<int>(col_lower_explicit_.size()) < model_.num_cols)
        col_lower_explicit_.resize(model_.num_cols, 0);

    // Set bounds based on bound type.
    if      (btype == "LO") { model_.col_lower(col_idx) = value; col_lower_explicit_[col_idx] = 1; } // Lower bound
    else if (btype == "UP") { model_.col_upper(col_idx) = value; } // Upper bound
    else if (btype == "FX") { model_.col_lower(col_idx) = value; model_.col_upper(col_idx) = value; col_lower_explicit_[col_idx] = 1; } // Fixed variable
    else if (btype == "FR") { model_.col_lower(col_idx) = -inf;  model_.col_upper(col_idx) = inf; col_lower_explicit_[col_idx] = 1; }   // Free variable
    else if (btype == "MI") { model_.col_lower(col_idx) = -inf; col_lower_explicit_[col_idx] = 1; } // No lower bound
    else if (btype == "PL") { model_.col_upper(col_idx) = inf; }  // No upper bound
    else if (btype == "BV") { model_.col_lower(col_idx) = 0; model_.col_upper(col_idx) = 1; col_lower_explicit_[col_idx] = 1; } // Binary variable
    else throw std::runtime_error("Unknown bound type in BOUNDS section: " + std::string(btype));
}

template <typename T>
void MpsFormatParser<T>::parse_quadobj(const std::vector<std::string_view>& tokens) {
    // Free-format QUADOBJ line: {<col_name1> <col_name2> <value>} ...
    if (tokens.size() < 3) return; // Invalid line, ignore.
    if ((tokens.size() % 3) != 0)
        throw std::runtime_error("QUADOBJ line does not have a multiple of 3 tokens.");

    for (size_t k = 0; k + 2 < tokens.size(); k += 3) {
        std::string_view cname1 = tokens[k];
        std::string_view cname2 = tokens[k + 1];
        T value = static_cast<T>(std::stod(std::string(tokens[k + 2])));

        auto it1 = col_map_.find(std::string(cname1));
        auto it2 = col_map_.find(std::string(cname2));
        if (it1 == col_map_.end() || it2 == col_map_.end()) {
            throw std::runtime_error(
                "QUADOBJ references unknown column '" +
                std::string(it1 == col_map_.end() ? cname1 : cname2) + "'."
            );
        }

        int i = it1->second;
        int j = it2->second;
        if (i < j) std::swap(i, j);
        Q_triplets_.emplace_back(i, j, value);
    }
}

template <typename T>
void MpsFormatParser<T>::finalize_defaults() {

    T inf = std::numeric_limits<T>::infinity();

    // If objective row was not defined, create a default one.
    if (obj_name_.empty()) {
        obj_name_ = "OBJ";
        RowInfo info;
        info.type = 'N';
        info.idx = -1;
        row_map_[obj_name_] = info;
    }

    // Ensure c is sized correctly for the number of variables.
    if (model_.c.size() < model_.num_cols) {
        int old_size = model_.c.size();
        model_.c.conservativeResize(model_.num_cols);
        model_.c.segment(old_size, model_.num_cols - old_size).setZero(); // Initialize new entries to 0.
    }

    // Ensure col_lower and col_upper are sized correctly for the number of variables.
    if (model_.col_lower.size() < model_.num_cols) {
        int old_size = model_.col_lower.size();
        model_.col_lower.conservativeResize(model_.num_cols);
        model_.col_upper.conservativeResize(model_.num_cols);
        model_.col_lower.segment(old_size, model_.num_cols - old_size).setConstant(static_cast<T>(0)); // Default lower bound is 0.
        model_.col_upper.segment(old_size, model_.num_cols - old_size).setConstant(inf);               // Default upper bound is inf.
    }

    // Ensure rhs_values is sized correctly for the number of constraints.
    if ((int)rhs_values_.size() < model_.num_rows)
        rhs_values_.resize(model_.num_rows, static_cast<T>(0)); // Default RHS is 0.

    // Ensure range_values is sized correctly for the number of constraints.
    if ((int)range_values_.size() < model_.num_rows)
        range_values_.resize(model_.num_rows, static_cast<T>(0)); // Default range is 0 (no range).

    // Allocate row_lower and row_upper with (-inf, inf) for constraints;
    // actual values will be set in finalize_row_bounds().
    model_.row_lower = ParsedModel<T>::Vec::Constant(model_.num_rows, -inf);
    model_.row_upper = ParsedModel<T>::Vec::Constant(model_.num_rows, inf);

    // A negative UP with no other explicit lower-bound entry relaxes the lower bound to -inf
    // (see the "Default choices of the parser" comment above).
    if (static_cast<int>(col_lower_explicit_.size()) < model_.num_cols)
        col_lower_explicit_.resize(model_.num_cols, 0);
    for (int i = 0; i < model_.num_cols; ++i)
        if (!col_lower_explicit_[i] && model_.col_upper(i) < T(0))
            model_.col_lower(i) = -inf;

    // Validate bound consistency.
    for (int i = 0; i < model_.num_cols; ++i)
        if (model_.col_lower(i) > model_.col_upper(i))
            throw std::runtime_error("Inconsistent bounds for variable " + std::to_string(i) + ": lower bound is greater than upper bound.");

}

template <typename T>
void MpsFormatParser<T>::finalize_row_bounds() {
    T inf = std::numeric_limits<T>::infinity();

    // Build inverse map: row index -> row type.
    std::vector<char> row_types(model_.num_rows, '\0'); // Default to null char for undefined rows.
    for (const auto& [name, info] : row_map_)
        if (info.idx >= 0) // Only consider constraint rows.
            row_types[info.idx] = info.type;

    for (int i = 0; i < model_.num_rows; ++i) {
        char type = row_types[i];
        T rhs = rhs_values_[i];
        T range = std::abs(range_values_[i]);
        // E: [rhs - |range|, rhs + |range|]
        // L: [rhs - |range|, rhs]
        // G: [rhs, rhs + |range|)
        // N: free

        if (type == 'E') {
            model_.row_lower(i) = rhs - range;
            model_.row_upper(i) = rhs + range;
        } else if (type == 'L') {
            model_.row_upper(i) = rhs;
            if (range > 0) {
                model_.row_lower(i) = rhs - range;
            }
        } else if (type == 'G') {
            model_.row_lower(i) = rhs;
            if (range > 0) {
                model_.row_upper(i) = rhs + range;
            }
        } else if (type == 'N') {
            model_.row_lower(i) = -inf;
            model_.row_upper(i) = inf;
        } else {
            throw std::runtime_error("Unknown row type in ROWS section: row index = " + std::to_string(i));
        }

        if (model_.row_lower(i) > model_.row_upper(i)) {
            throw std::runtime_error("Inconsistent bounds for constraint " + std::to_string(i) + ": lower bound is greater than upper bound.");
        }
    }
}

template <typename T>
void MpsFormatParser<T>::build_sparse_matrices() {
    model_.A.resize(model_.num_rows, model_.num_cols);
    model_.A.setFromTriplets(A_triplets_.begin(), A_triplets_.end(),
        [](T a, T b) { return a + b; } // Sum duplicates.
    );
    model_.A.makeCompressed();

    if (model_.is_qp) {
        model_.Q.resize(model_.num_cols, model_.num_cols);
        model_.Q.setFromTriplets(Q_triplets_.begin(), Q_triplets_.end(),
            [](T a, T b) { return a + b; } // Sum duplicates.
        );
        model_.Q.makeCompressed();
    }
}

template <typename T>
bool MpsFormatParser<T>::is_comment_or_blank(const std::string& line) {
    for (char c : line) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
            return c == '*'; // Comment line starts with '*'.
        }
    }
    return true; // Blank line.
}

template <typename T>
void MpsFormatParser<T>::split_fixed_by_section(std::string_view line, Section sec,
                                                 std::vector<std::string_view>& out) {
    auto field = [&](int start, int len) -> std::string_view {
        if ((int)line.size() <= start) return {};
        return trim(line.substr(start, std::min(len, (int)line.size() - start)));
    };
    std::string_view F1 = field(1, 2);
    std::string_view F2 = field(4, 8);
    std::string_view F3 = field(14, 8);
    std::string_view F4 = field(24, 12);
    std::string_view F5 = field(39, 8);
    std::string_view F6 = field(49, 12);

    out.clear();
    auto push = [&](std::string_view s) {
        if (!s.empty()) out.push_back(s);
    };

    switch (sec) {
        case Section::ROWS:
            // ROWS: F1 = type, F2 = row name.
            push(F1); push(F2);
            break;
        case Section::COLUMNS:
            // COLUMNS: F1 = "", F2 = col name, F3 = row name, F4 = value, F5 = row name, F6 = value.
            push(F2);
            if (!F3.empty()) {push(F3); push(F4);}
            if (!F5.empty()) {push(F5); push(F6);}
            break;
        case Section::RHS:
            // RHS: F1 = "", F2 = rhs name, F3 = row name, F4 = value, F5 = row name, F6 = value.
            if (F2.empty()) push("RHS");
            else push(F2);
            if (!F3.empty()) {push(F3); push(F4);}
            if (!F5.empty()) {push(F5); push(F6);}
            break;
        case Section::RANGES:
            // RANGES: F1 = "", F2 = range name, F3 = row name, F4 = value, F5 = row name, F6 = value.
            push(F2);
            if (!F3.empty()) {push(F3); push(F4);}
            if (!F5.empty()) {push(F5); push(F6);}
            break;
        case Section::BOUNDS:
            // BOUNDS: F1 = bound type, F2 = bound name, F3 = col name, F4 = value.
            push(F1); push(F2); push(F3);
            if (!F4.empty()) push(F4); // Some bound types do not have a value.
            break;
        case Section::QUADOBJ:
            // QUADOBJ: F1 = "", F2 = col name 1, F3 = col name 2, F4 = value.
            push(F2); push(F3); push(F4);
            break;
        default:
            break; // For other sections, we don't use fixed-format parsing.
    }
}

template <typename T>
void MpsFormatParser<T>::split_ws(std::string_view line, std::vector<std::string_view>& out) {
    out.clear();
    size_t i = 0;
    const size_t n = line.size();
    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i >= n) break;
        size_t start = i;
        while (i < n && !std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        out.push_back(line.substr(start, i - start));
    }
}

template <typename T>
void MpsFormatParser<T>::split_free_by_section(const std::vector<std::string_view>& toks, Section sec,
                                                std::vector<std::string_view>& out) {
    out.clear();
    auto push = [&](std::string_view s) { if (!s.empty()) out.push_back(s); };

    switch (sec) {
        case Section::ROWS:
            // type row
            if (toks.size() >= 2) { push(toks[0]); push(toks[1]); }
            break;

        case Section::COLUMNS:
            // col row val [row val]
            if (toks.size() >= 3) {
                push(toks[0]);
                push(toks[1]); push(toks[2]);
                if (toks.size() >= 5) { push(toks[3]); push(toks[4]); }
            }
            break;

        case Section::RHS:
            // rhsName row val [row val]
            if (toks.size() >= 3) {
                push(toks[0]);
                push(toks[1]); push(toks[2]);
                if (toks.size() >= 5) { push(toks[3]); push(toks[4]); }
            }
            break;

        case Section::RANGES:
            // rangeName row val [row val]
            if (toks.size() >= 3) {
                push(toks[0]);
                push(toks[1]); push(toks[2]);
                if (toks.size() >= 5) { push(toks[3]); push(toks[4]); }
            }
            break;

        case Section::BOUNDS:
            // bndType bndName col [val]
            if (toks.size() >= 3) {
                push(toks[0]); push(toks[1]); push(toks[2]);
                if (toks.size() >= 4) push(toks[3]);
            }
            break;

        case Section::QUADOBJ:
            // col1 col2 val (sometimes name col1 col2 val, handle both)
            if (toks.size() == 3) {
                push(toks[0]); push(toks[1]); push(toks[2]);
            } else if (toks.size() >= 4) {
                // if a name is present, skip it
                push(toks[toks.size()-3]);
                push(toks[toks.size()-2]);
                push(toks[toks.size()-1]);
            }
            break;

        default:
            // NAME/OBJSENSE etc.
            out.assign(toks.begin(), toks.end());
            break;
    }
}

template <typename T>
void MpsFormatParser<T>::tokenize_line(const std::string& line, Section sec) {
    // Format is decided once per file by decide_format_from(). Both branches
    // write the final tokens into the member buffer tokens_ rather than
    // returning a freshly allocated vector.
    if (format_ == Format::FREE) {
        split_free_by_section(ws_tokens_, sec, tokens_); // reuse this line's whitespace split
        return;
    }

    // Otherwise, try fixed-format first, but validate that the number of tokens is correct for the section.
    split_fixed_by_section(line, sec, tokens_);

    auto ok_for_section = [&](const std::vector<std::string_view>& t) {
        switch (sec) {
            case Section::ROWS:    return t.size() == 2;
            case Section::COLUMNS: return t.size() == 3 || t.size() == 5;
            case Section::RHS:     return t.size() == 3 || t.size() == 5;
            case Section::RANGES:  return t.size() == 3 || t.size() == 5;
            case Section::BOUNDS:  return t.size() == 3 || t.size() == 4;
            case Section::QUADOBJ: return t.size() == 3;
            default:               return !t.empty();
        }
    };

    if (ok_for_section(tokens_)) return;

    // Fall back to free, reusing the already-computed whitespace split
    // instead of re-scanning the line.
    split_free_by_section(ws_tokens_, sec, tokens_);
}

template <typename T>
std::string_view MpsFormatParser<T>::trim(std::string_view s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {}; // All whitespace
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}