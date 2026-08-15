#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <limits>

template <typename T>
struct ParsedModel {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    bool is_qp = false;
    bool is_min = true;

    int num_rows = 0;   // number of constraints
    int num_cols = 0;   // number of variables

    Vec c;              // objective coefficients
    T obj_const = T(0); // objective constant term
    SpMat A;            // constraint matrix
    SpMat Q;            // quadratic coefficients (store lower triangular part only)
    Vec row_lower, row_upper; // constraint bounds
    Vec col_lower, col_upper; // variable bounds
};

template <typename T>
struct KSPQPdata {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    int n, m, l;
    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
    T obj_const = T(0);
};
