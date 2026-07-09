#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <limits>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

template <typename T>
struct ParsedModel {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    bool is_qp = false;
    bool is_min = true;

    int num_rows = 0; // number of constraints
    int num_cols = 0; // number of variables

    Vec c; // objective coefficients
    T obj_const = T(0); // objective constant term
    SpMat A; // constraint matrix
    SpMat Q; // quadratic coefficients (for QP; store lower triangular part only)
    Vec row_lower, row_upper; // constraint bounds
    Vec col_lower, col_upper; // variable bounds
};

template <typename T>
struct PDPMMdata {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    int n, m, l;
    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
    T obj_const = T(0);
};

template <typename T>
class MpsParser {
public:
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<T>;

    ParsedModel<T> parse(const std::string& filename);
    PDPMMdata<T> to_pdpmm(const ParsedModel<T>& model, T eq_tol = 1e-12, T inf_cap = std::numeric_limits<T>::infinity());

private:
    enum class Section {
        NONE, NAME, OBJSENSE, ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUADOBJ, ENDATA
    };

    struct RowInfo {
        // For ROWS section
        char type = 'N'; // 'N', 'E', 'L', 'G'
        int idx = -1; // index of constraint rows (objective row has idx = -1)
    };

    Section section_ = Section::NONE;
    std::string sense_ = "MIN";

    std::string obj_name_;
    std::unordered_map<std::string, RowInfo> row_map_;
    std::unordered_map<std::string, int> col_map_;

    std::string rhs_name_;
    std::vector<T> rhs_values_;

    std::string range_name_;
    std::vector<T> range_values_;

    std::string bound_name_;

    std::vector<Triplet> A_triplets_;
    std::vector<Triplet> Q_triplets_;

    ParsedModel<T> model_;

    static bool is_comment_or_blank(const std::string& line);
    static std::vector<std::string> split_ws(const std::string& line);
    static std::vector<std::string> split_free_by_section(const std::string& line, Section sec);
    static std::vector<std::string> split_fixed_by_section(const std::string& line, Section sec);
    static std::vector<std::string> tokenize_line(const std::string& line, Section sec);
    static std::string trim(const std::string& s);
    
    bool set_section(const std::vector<std::string>& tokens);
    void parse_name(const std::vector<std::string>& tokens);
    void parse_objsense(const std::vector<std::string>& tokens);
    void parse_rows(const std::vector<std::string>& tokens);
    void parse_columns(const std::vector<std::string>& tokens);
    void parse_rhs(const std::vector<std::string>& tokens);
    void parse_ranges(const std::vector<std::string>& tokens);
    void parse_bounds(const std::vector<std::string>& tokens);
    void parse_quadobj(const std::vector<std::string>& tokens);

    void finalize_defaults();
    void finalize_row_bounds();
    void build_sparse_matrices();
};

#include "MpsParser.tpp"