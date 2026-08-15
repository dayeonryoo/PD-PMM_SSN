#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <optional>
#include "printing.hpp"
#include "ksp_qp_types.hpp"

// =============================================================
//      min  c^T x + 0.5 x^T Q x + obj_const,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw,
//  c = n-dim vector, Q = n x n matrix, A = m x n matrix, B = l x n matrix
// =============================================================

template <typename T>
class Problem {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
    T obj_const = T(0);

    int n = 0, m = 0, l = 0;

    T tol = 1e-6;
    int max_iter = 3000;
    double time_limit = 60.0; // in seconds
    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::NONE;

    Problem(){}

    Problem(const SpMat& Q_, const SpMat& A_, const SpMat& B_,
            const Vec& c_, const Vec& b_, T obj_const_,
            const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_,
            T tol_, int max_iter_, double time_limit_,
            PrintWhen when_, PrintWhat what_)
    : Q(Q_), A(A_), B(B_), c(c_), b(b_), obj_const(obj_const_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_),
      tol(tol_), max_iter(max_iter_), time_limit(time_limit_),
      when(when_), what(what_) {}

    Problem(const KSPQPdata<T>& pd, T tol_, int max_iter_, double time_limit_,
            PrintWhen when_, PrintWhat what_)
    : Problem(pd.Q, pd.A, pd.B, pd.c, pd.b, pd.obj_const,
              pd.lx, pd.ux, pd.lw, pd.uw,
              tol_, max_iter_, time_limit_, when_, what_) {
        n = pd.n;
        m = pd.m;
        l = pd.l;
    }
};