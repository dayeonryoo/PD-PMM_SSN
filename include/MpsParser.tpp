#pragma once
#include "MpsParser.hpp"

template <typename T>
ParsedModel<T> MpsParser<T>::parse(const std::string& filename) {
    model_ = ParsedModel<T>();
    section_ = Section::NONE;
    row_map_.clear(); col_map_.clear();
    A_triplets_.clear(); Q_triplets_.clear();
    obj_name_.clear(); rhs_name_.clear();
    range_name_.clear(); bound_name_.clear();
    rhs_values_.clear(); range_values_.clear();

    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open MPS file: " + filename);

    std::string line;
    while (std::getline(f, line)) {
        if (is_comment_or_blank(line)) continue;

        // Section header parsing
        std::vector<std::string> head = split_ws(line);
        bool is_header = set_section(head);
        if (section_ == Section::ENDATA) break;
        if (is_header) continue; // Skip section header lines

        // Section content parsing depending on format
        std::vector<std::string> tokens = split_fixed_by_section(line, section_);
        switch (section_) {
            case Section::NAME: parse_name(tokens); break;
            case Section::OBJSENSE: parse_objsense(tokens); break;
            case Section::ROWS: parse_rows(tokens); break;
            case Section::COLUMNS: parse_columns(tokens); break;
            case Section::RHS: parse_rhs(tokens); break;
            case Section::RANGES: parse_ranges(tokens); break;
            case Section::BOUNDS: parse_bounds(tokens); break;
            case Section::QUADOBJ: parse_quadobj(tokens); break;
            default: break; // Ignore lines outside of known sections
        }
    }

    finalize_defaults();
    finalize_row_bounds();
    build_sparse_matrices();

    return model_;
}

template <typename T>
PDPMMdata<T> MpsParser<T>::to_pdpmm(const ParsedModel<T>& model, T eq_tol, T inf_cap) {
    using Vec = typename MpsParser<T>::Vec;
    using SpMat = typename MpsParser<T>::SpMat;
    using Triplet = typename MpsParser<T>::Triplet;

    auto is_inf = [inf_cap](T val) {
        return std::isinf((double)val) || std::abs((double)val) >= (double)inf_cap;
    };

    auto cap_inf = [inf_cap, is_inf](T val) {
        if (is_inf(val)) return (val > 0 ? inf_cap : -inf_cap);
        else return val;
    };

    PDPMMdata<T> pd;
    pd.is_qp = model.is_qp;
    pd.is_min = model.is_min;
    pd.n = model.num_cols;

    pd.c = model.c;
    pd.lx = model.col_lower;
    pd.ux = model.col_upper;
    for (int i = 0; i < pd.n; ++i) {
        pd.lx(i) = cap_inf(pd.lx(i));
        pd.ux(i) = cap_inf(pd.ux(i));
    }

    if (pd.is_qp) pd.Q = model.Q;
    else { pd.Q = SpMat(pd.n, pd.n); pd.Q.setZero(); }

    // If MAX, convert to MIN by negating objective
    if (!pd.is_min) {
        pd.c = -pd.c;
        if (pd.is_qp) pd.Q = -pd.Q;
        pd.is_min = true;
    }

    // Process constraints
    std::vector<int> eq_rows, ineq_rows;
    eq_rows.reserve(model.num_rows);
    ineq_rows.reserve(model.num_rows);

    for (int i = 0; i < model.num_rows; ++i) {
        T lb = cap_inf(model.row_lower(i));
        T ub = cap_inf(model.row_upper(i));
        if (is_inf(lb) && is_inf(ub)) {
            continue; // Skip free constraints
        } else if (std::abs(lb - ub) <= eq_tol) {
            eq_rows.push_back(i);
        } else {
            ineq_rows.push_back(i);
        }
    }

    pd.m = (int)eq_rows.size();
    pd.l = (int)ineq_rows.size();

    // Construct A and b for equality constraints
    pd.A = SpMat(pd.m, pd.n);
    pd.b = Vec(pd.m);
    {
        std::vector<Triplet> triplets;
        triplets.reserve(model.A.nonZeros());
        
        // Row map to track which rows correspond to equality constraints
        std::vector<int> row_map(model.num_rows, -1);
        for (int i = 0; i < eq_rows.size(); ++i) {
            row_map[eq_rows[i]] = i;
            pd.b(i) = model.row_lower(eq_rows[i]);
        }

        for (int col = 0; col < model.A.outerSize(); ++col) {
            for (typename SpMat::InnerIterator it(model.A, col); it; ++it) {
                int r = it.row();
                int loc = row_map[r];
                if (loc != -1) {
                    triplets.emplace_back(loc, col, it.value());
                }
            }
        }
        pd.A.setFromTriplets(triplets.begin(), triplets.end());
        pd.A.makeCompressed();
    }

    // Construct B, lw, uw for inequality constraints
    pd.B = SpMat(pd.l, pd.n);
    pd.lw = Vec(pd.l);
    pd.uw = Vec(pd.l);
    {
        std::vector<Triplet> triplets;
        triplets.reserve(model.A.nonZeros());

        std::vector<int> row_map(model.num_rows, -1);
        for (int i = 0; i < ineq_rows.size(); ++i) {
            int r = ineq_rows[i];
            row_map[r] = i;
            pd.lw(i) = model.row_lower(r);
            pd.uw(i) = model.row_upper(r);
        }

        for (int col = 0; col < model.A.outerSize(); ++col) {
            for (typename SpMat::InnerIterator it(model.A, col); it; ++it) {
                int r = it.row();
                int loc = row_map[r];
                if (loc != -1) {
                    triplets.emplace_back(loc, col, it.value());
                }
            }
        }
        pd.B.setFromTriplets(triplets.begin(), triplets.end());
        pd.B.makeCompressed();
    }

    return pd;
}

template <typename T>
bool MpsParser<T>::set_section(const std::vector<std::string>& tokens) {
    const std::string& sec = tokens[0];
    if (sec == "NAME") {section_ = Section::NAME; return true;}
    else if (sec == "OBJSENSE") {section_ = Section::OBJSENSE; return true;}
    else if (sec == "ROWS") {section_ = Section::ROWS; return true;}
    else if (sec == "COLUMNS") {section_ = Section::COLUMNS; return true;}
    else if (sec == "RHS") {section_ = Section::RHS; return true;}
    else if (sec == "RANGES") {section_ = Section::RANGES; return true;}
    else if (sec == "BOUNDS") {section_ = Section::BOUNDS; return true;}
    else if (sec == "QUADOBJ") {section_ = Section::QUADOBJ; model_.is_qp = true; return true;}
    else if (sec == "ENDATA") {section_ = Section::ENDATA; return true;}
    else return false; // Not a section header
}

template <typename T>
void MpsParser<T>::parse_name(const std::vector<std::string>& tokens) {
    if (tokens.size() >= 2) obj_name_ = tokens[1];
}

template <typename T>
void MpsParser<T>::parse_objsense(const std::vector<std::string>& tokens) {
    if (tokens.size() == 1) sense_ = tokens[0];
    else if (tokens.size() >= 2 && (tokens[0] == "OBJSENSE" || tokens[0] == "'OBJSENSE'")) sense_ = tokens[1];
    else sense_ = tokens.back(); // Take last token as sense

    if (sense_ == "MIN") {
        model_.is_min = true;
    } else if (sense_ == "MAX") {
        model_.is_min = false;
    } else {
        throw std::runtime_error("Unknown optimization sense in OBJSENSE section: " + sense_);
    }
}

template <typename T>
void MpsParser<T>::parse_rows(const std::vector<std::string>& tokens) {
    // Free-format ROWS line: <type> <row_name>
    if (tokens.size() < 2) return; // Invalid line, ignore
    char type = tokens[0].empty() ? 'N' : tokens[0][0]; // Default to 'N' if type is missing
    const std::string& rname = tokens[1];

    RowInfo info;
    info.type = type;

    if (type == 'N') {
        if (obj_name_.empty()) {
            obj_name_ = rname;
            info.idx = -1;
        } else {
            // Non-objective row with type 'N' is not standard, but we can still assign it an index
            info.idx = model_.num_rows++; // Will be treated as a constraint row with no bounds (free) in finalize_row_bounds()
        }
    } else {
        auto it = row_map_.find(rname);
        if (it == row_map_.end()) {
            info.idx = model_.num_rows++; // Assign new index for constraint row
        } else {
            throw std::runtime_error("Duplicate row name in ROWS section: " + rname);
        }
    }
    row_map_[rname] = info;
}

template <typename T>
void MpsParser<T>::parse_columns(const std::vector<std::string>& tokens) {
    // Free-format COLUMNS line: <col_name> {<row_name> <value>} ...
    if (tokens.size() < 3) return; // Invalid line, ignore
    if (tokens.size() >= 2 && (tokens[1] == "MARKER" || tokens[1] == "'MARKER'")) return; // Ignore marker lines

    const std::string& cname = tokens[0];
    int col_idx;
    auto it = col_map_.find(cname);
    if (it == col_map_.end()) {
        col_idx = model_.num_cols++; // Assign new index for variable
        col_map_[cname] = col_idx;
    } else {
        col_idx = it->second;
    }

    // Ensure c is large enough to hold the coefficient for this variable
    int c_size = model_.c.size();
    if (c_size < model_.num_cols) {
        model_.c.conservativeResize(model_.num_cols);
        model_.c.segment(c_size, model_.num_cols - c_size).setZero(); // Initialize new entries to 0
    }

    for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
        const std::string& rname = tokens[i];
        T value = static_cast<T>(std::stod(tokens[i + 1]));

        auto it = row_map_.find(rname);
        if (it == row_map_.end()) {
            throw std::runtime_error("Row name in COLUMNS section not defined in ROWS section: " + rname);
        }
        const RowInfo& info = it->second;
        if (rname == obj_name_) {
            // Objective coefficient
            model_.c(col_idx) += value;
        } else {
            // Constraint matrix entry
            if (info.idx < 0) {
                throw std::runtime_error("Invalid row index for constraint in COLUMNS section: " + rname);
            }
            A_triplets_.emplace_back(info.idx, col_idx, value);
        }
    }
}

template <typename T>
void MpsParser<T>::parse_rhs(const std::vector<std::string>& tokens) {
    // Free-format RHS line: <rhs_name> {<row_name> <value>} ...
    if (tokens.size() < 2) return; // Invalid line, ignore

    // If the first token is empty, use "RHS" as the default name for the RHS set
    const bool has_name = row_map_.find(tokens[0]) == row_map_.end(); // If first token is not a row name, treat it as RHS name
    
    size_t start_idx;
    if (has_name) {
        if (rhs_name_.empty()) rhs_name_ = tokens[0];
        start_idx = 1;
        if (rhs_name_ != tokens[0]) {
            throw std::runtime_error("Multiple RHS sets defined in RHS section: " + tokens[0]);
        }
    } else {
        // No explicit RHS name, use default "RHS"
        if (rhs_name_.empty()) rhs_name_ = "RHS";
        start_idx = 0;
        if (rhs_name_ != "RHS") {
            throw std::runtime_error("Multiple RHS sets defined in RHS section: " + rhs_name_);
        }
    }

    // Ensure rhs_values is large enough to hold the RHS for all constraints
    if ((int)rhs_values_.size() < model_.num_rows) {
        rhs_values_.resize(model_.num_rows, static_cast<T>(0)); // Default RHS is 0
    }

    for (size_t i = start_idx; i + 1 < tokens.size(); i += 2) {
        const std::string& rname = tokens[i];
        T value = static_cast<T>(std::stod(tokens[i + 1]));

        auto it = row_map_.find(rname);
        if (it == row_map_.end()) {
            throw std::runtime_error("Row name in RHS section not defined in ROWS section: " + rname);
        }
        int idx = it->second.idx;
        if (idx < 0) continue; // Ignore RHS for objective row or invalid row

        // Store the RHS value by row index
        rhs_values_[idx] = value; 
    }
}

template <typename T>
void MpsParser<T>::parse_ranges(const std::vector<std::string>& tokens) {
    // Free-format RANGES line: <range_name> {<row_name> <value>} ...
    if (tokens.size() < 3) return; // Invalid line, ignore

    if (range_name_.empty()) range_name_ = tokens[0];
    if (range_name_ != tokens[0]) {
        throw std::runtime_error("Multiple range sets defined in RANGES section: " + tokens[0]);
    }

    // Ensure range_values is large enough to hold the range for all constraints
    if ((int)range_values_.size() < model_.num_rows) {
        range_values_.resize(model_.num_rows, static_cast<T>(0)); // Default range is 0 (no range)
    }

    for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
        const std::string& rname = tokens[i];
        T value = static_cast<T>(std::stod(tokens[i + 1]));

        auto it = row_map_.find(rname);
        if (it == row_map_.end()) {
            throw std::runtime_error("Row name in RANGES section not defined in ROWS section: " + rname);
        }
        int idx = it->second.idx;
        if (idx < 0) continue; // Ignore range for objective row or invalid row

        // Store the range value by row index
        range_values_[idx] = value; 
    }
}

template <typename T>
void MpsParser<T>::parse_bounds(const std::vector<std::string>& tokens) {
    // Free-format BOUNDS line:
    // <bound_type> <bound_name> <col_name> {<value>} ...
    // OR <bound_type> <col_name> {<value>} ... (if bound name is omitted, use default)
    
    if (tokens.size() < 2) return; // Invalid line, ignore

    const std::string& btype = tokens[0];

    // These bound types do not require a value
    const bool needs_value = (btype == "LO" || btype == "UP" || btype == "FX" || btype == "LI" || btype == "UI" || btype == "BV");
    
    std::string bname;
    std::string cname;
    std::string value_str;

    if (tokens.size() >= 3) {
        // tokens[1] could be either bound name or column name.
        // If tokens[1] is an existing column name, then we treat it as column name and use default bound name.
        if (col_map_.find(tokens[1]) != col_map_.end()) {
            bname = "BND"; // Use default bound name
            cname = tokens[1];
            value_str = tokens[2];
        } else {
            bname = tokens[1];
            cname = tokens[2];
            if (tokens.size() >= 4) value_str = tokens[3];
        }
    } else {
        // tokens.size == 2, so we have only bound type and column name, no bound name or value
        bname = "BND"; // Use default bound name
        cname = tokens[1];
    }

    // Enforce that bound name is consistent if provided
    if (bound_name_.empty()) bound_name_ = bname;
    else if (bound_name_ != bname) {
        throw std::runtime_error("Multiple bound sets defined in BOUNDS section: " + bname);
    }

    // Check if value is provided when required
    if (needs_value && value_str.empty()) {
        throw std::runtime_error("Bound type " + btype + " requires a value in BOUNDS section.");
    }

    const T inf = std::numeric_limits<T>::infinity();
    T value = T(0);
    if (!value_str.empty()) value = static_cast<T>(std::stod(value_str));

    // Get or create column index
    int col_idx;
    auto it = col_map_.find(cname);
    if (it == col_map_.end()) {
        col_idx = model_.num_cols++; // Assign new index for variable
        col_map_[cname] = col_idx;
    } else {
        col_idx = it->second;
    }

    // Ensure col_lower and col_upper are large enough to hold bounds for this variable
    int size = model_.col_lower.size();
    if (size < model_.num_cols) {
        model_.col_lower.conservativeResize(model_.num_cols);
        model_.col_upper.conservativeResize(model_.num_cols);
        model_.col_lower.segment(size, model_.num_cols - size).setZero(); // Default lower bound is 0
        model_.col_upper.segment(size, model_.num_cols - size).setConstant(inf);  // Default upper bound is inf
    }

    // Set bounds based on bound type
    if (btype == "LO") model_.col_lower(col_idx) = value; // Lower bound
    else if (btype == "UP") model_.col_upper(col_idx) = value; // Upper bound
    else if (btype == "FX") { model_.col_lower(col_idx) = value; model_.col_upper(col_idx) = value; } // Fixed variable
    else if (btype == "FR") { model_.col_lower(col_idx) = -inf; model_.col_upper(col_idx) = inf; } // Free variable
    else if (btype == "MI") model_.col_lower(col_idx) = -inf; // No lower bound
    else if (btype == "PL") model_.col_upper(col_idx) = inf;  // No upper bound
    else if (btype == "BV") { model_.col_lower(col_idx) = 0; model_.col_upper(col_idx) = 1; } // Binary variable
    else throw std::runtime_error("Unknown bound type in BOUNDS section: " + btype);

    if (model_.col_lower(col_idx) > model_.col_upper(col_idx)) {
        throw std::runtime_error("Inconsistent bounds for variable " + cname + ": lower bound is greater than upper bound.");
    }
}

template <typename T>
void MpsParser<T>::parse_quadobj(const std::vector<std::string>& tokens) {
    // Free-format QUADOBJ line: {<col_name1> <col_name2> <value>} ...
    if (tokens.size() < 3) return; // Invalid line, ignore

    const std::string& cname1 = tokens[0];
    const std::string& cname2 = tokens[1];
    T value = static_cast<T>(std::stod(tokens[2]));

    auto get_col = [&](const std::string& cname) -> int {
        auto it = col_map_.find(cname);
        if (it == col_map_.end()) {
            int idx = model_.num_cols++; // Assign new index for variable
            col_map_[cname] = idx;
            // Ensure c is large enough to hold the coefficient for this variable
            int c_size = model_.c.size();
            if (c_size < model_.num_cols) {
                model_.c.conservativeResize(model_.num_cols);
                model_.c.segment(c_size, model_.num_cols - c_size).setZero(); // Initialize new entries to 0
            }
            return idx;
        } else {
            return it->second; // Column already exists
        }
    };

    int col_idx1 = get_col(cname1);
    int col_idx2 = get_col(cname2);

    if (col_idx1 < col_idx2) std::swap(col_idx1, col_idx2); // Store in lower triangular part
    Q_triplets_.emplace_back(col_idx1, col_idx2, value);
}

template <typename T>
void MpsParser<T>::finalize_defaults() {

    T inf = std::numeric_limits<T>::infinity();

    // If objective row was not defined, create a default one
    if (obj_name_.empty()) {
        obj_name_ = "OBJ";
        RowInfo info;
        info.type = 'N';
        info.idx = -1;
        row_map_[obj_name_] = info;
    }

    // Ensure c is sized correctly for the number of variables
    if (model_.c.size() < model_.num_cols) {
        int old_size = model_.c.size();
        model_.c.conservativeResize(model_.num_cols);
        model_.c.segment(old_size, model_.num_cols - old_size).setZero(); // Initialize new entries to 0
    }

    // Ensure col_lower and col_upper are sized correctly for the number of variables
    if (model_.col_lower.size() < model_.num_cols) {
        int old_size = model_.col_lower.size();
        model_.col_lower.conservativeResize(model_.num_cols);
        model_.col_upper.conservativeResize(model_.num_cols);
        model_.col_lower.segment(old_size, model_.num_cols - old_size).setZero(); // Default lower bound is 0
        model_.col_upper.segment(old_size, model_.num_cols - old_size).setConstant(inf);  // Default upper bound is inf
    }

    // Ensure rhs_values is sized correctly for the number of constraints
    if ((int)rhs_values_.size() < model_.num_rows) {
        rhs_values_.resize(model_.num_rows, static_cast<T>(0)); // Default RHS is 0
    }

    // Ensure range_values is sized correctly for the number of constraints
    if ((int)range_values_.size() < model_.num_rows) {
        range_values_.resize(model_.num_rows, static_cast<T>(0)); // Default range is 0 (no range)
    }

    // Allocate row_lower and row_upper with (-inf, inf) for constraints;
    // actual values will be set in finalize_row_bounds()
    model_.row_lower = ParsedModel<T>::Vec::Constant(model_.num_rows, -inf);
    model_.row_upper = ParsedModel<T>::Vec::Constant(model_.num_rows, inf);
}

template <typename T>
void MpsParser<T>::finalize_row_bounds() {
    T inf = std::numeric_limits<T>::infinity();

    // Build inverse map: row index -> row type
    std::vector<char> row_types(model_.num_rows, '\0'); // Default to null char for undefined rows
    for (const auto& [name, info] : row_map_) {
        if (info.idx >= 0) { // Only consider constraint rows
            row_types[info.idx] = info.type;
        }
    }

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
            throw std::runtime_error(std::string("Unknown row type in ROWS section: ") + type);
        }

        if (model_.row_lower(i) > model_.row_upper(i)) {
            throw std::runtime_error("Inconsistent bounds for constraint " + std::to_string(i) + ": lower bound is greater than upper bound.");
        }
    }
}

template <typename T>
void MpsParser<T>::build_sparse_matrices() {
    model_.A.resize(model_.num_rows, model_.num_cols);
    model_.A.setFromTriplets(A_triplets_.begin(), A_triplets_.end(),
        [](T a, T b) { return a + b; } // Sum duplicates
    );
    model_.A.makeCompressed();

    if (model_.is_qp) {
        model_.Q.resize(model_.num_cols, model_.num_cols);
        model_.Q.setFromTriplets(Q_triplets_.begin(), Q_triplets_.end(),
            [](T a, T b) { return a + b; } // Sum duplicates
        );
        model_.Q.makeCompressed();
    }
}

template <typename T>
bool MpsParser<T>::is_comment_or_blank(const std::string& line) {
    for (char c : line) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
            return c == '*'; // Comment line starts with '*'
        }
    }
    return true; // Blank line
}

template <typename T>
std::vector<std::string> MpsParser<T>::split_fixed_by_section(const std::string& line, Section sec) {
    auto field = [&](int start, int len) -> std::string {
        if ((int)line.size() <= start) return "";
        return trim(line.substr(start, std::min(len, (int)line.size() - start)));
    };
    std::string F1 = field(1, 2);
    std::string F2 = field(4, 8);
    std::string F3 = field(14, 8);
    std::string F4 = field(24, 12);
    std::string F5 = field(39, 8);
    std::string F6 = field(49, 12);

    std::vector<std::string> tokens;
    auto push = [&](const std::string& s) {
        if (!s.empty()) tokens.push_back(s);
    };

    switch (sec) {
        case Section::ROWS:
            // ROWS: F1 = type, F2 = row name
            push(F1); push(F2);
            break;
        case Section::COLUMNS:
            // COLUMNS: F1 = "", F2 = col name, F3 = row name, F4 = value, F5 = row name, F6 = value
            push(F2);
            if (!F3.empty()) {push(F3); push(F4);}
            if (!F5.empty()) {push(F5); push(F6);}
            break;
        case Section::RHS:
            // RHS: F1 = "", F2 = rhs name, F3 = row name, F4 = value, F5 = row name, F6 = value
            if (F2.empty()) push("RHS");
            else push(F2);
            if (!F3.empty()) {push(F3); push(F4);}
            if (!F5.empty()) {push(F5); push(F6);}
            break;
        case Section::RANGES:
            // RANGES: F1 = "", F2 = range name, F3 = row name, F4 = value, F5 = row name, F6 = value
            push(F2);
            if (!F3.empty()) {push(F3); push(F4);}
            if (!F5.empty()) {push(F5); push(F6);}
            break;
        case Section::BOUNDS:
            // BOUNDS: F1 = bound type, F2 = bound name, F3 = col name, F4 = value
            push(F1); push(F2); push(F3);
            if (!F4.empty()) push(F4); // Some bound types do not have a value
            break;
        case Section::QUADOBJ:
            // QUADOBJ: F1 = "", F2 = col name 1, F3 = col name 2, F4 = value
            push(F2); push(F3); push(F4);
            break;
        default:
            break; // For other sections, we don't use fixed-format parsing
    }
    return tokens;
}

template <typename T>
std::vector<std::string> MpsParser<T>::split_ws(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(tok);
    }
    return tokens;
}

template <typename T>
std::string MpsParser<T>::trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return ""; // All whitespace
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}