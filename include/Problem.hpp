#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <optional>
#include "Printing.hpp"

// =============================================================
//      min  c^T x + (1/2) x^T Q x,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw
// =============================================================

template <typename T>
class Problem {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
    T tol;
    int max_iter;
    PrintWhen PMM_print_when, SSN_print_when;
    PrintWhat PMM_print_what, SSN_print_what;

    Problem(){}
    Problem(const SpMat& Q_, const SpMat& A_, const SpMat& B_,
            const Vec& c_, const Vec& b_,
            const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_,
            T tol_, int max_iter_,
            PrintWhen PMM_print_when_, PrintWhat PMM_print_what_,
            PrintWhen SSN_print_when_, PrintWhat SSN_print_what_)
    : Q(Q_), A(A_), B(B_), c(c_), b(b_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_),
      tol(tol_), max_iter(max_iter_),
      PMM_print_when(PMM_print_when_), PMM_print_what(PMM_print_what_),
      SSN_print_when(SSN_print_when_), SSN_print_what(SSN_print_what_)
    {}

};