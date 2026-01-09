#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

template <typename T>
struct LPdata;

template <typename T>
struct PDPMMdata {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
};

template <typename T>
PDPMMdata<T> lp_to_pdpmm(const LPdata<T>& lp) {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    const int m = static_cast<int>(lp.A.rows());
    const int n = static_cast<int>(lp.A.cols());

    PDPMMdata<T> pd;

    SpMat Q(n, n);
    Q.setZero();
    Q.makeCompressed();
    pd.Q = std::move(Q);

    SpMat A_eq(0, n);
    A_eq.setZero();
    A_eq.makeCompressed();
    pd.A = std::move(A_eq);

    Vec b_eq(0);
    pd.b = std::move(b_eq);

    pd.B = lp.A;
    pd.lw = lp.rhs_lo;
    pd.uw = lp.rhs_hi;

    pd.c = lp.c;
    pd.lx = lp.lb;
    pd.ux = lp.ub;

    return pd;
}