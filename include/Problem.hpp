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

    Problem(const SpMat& Q_, const SpMat& A_, const SpMat& B_,
            const Vec& c_, const Vec& b_,
            const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_,
            T tol_ = T(1e-4), int max_iter_ = 200,
            PrintWhen PMM_print_when_ = PrintWhen::END_ONLY,
            PrintWhat PMM_print_what_ = PrintWhat::FULL,
            PrintWhen SSN_print_when_ = PrintWhen::NEVER,
            PrintWhat SSN_print_what_ = PrintWhat::NONE)
    : Q(Q_), A(A_), B(B_), tol(tol_), c(c_), b(b_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_), max_iter(max_iter_),
      PMM_print_when(PMM_print_when_), PMM_print_what(PMM_print_what_),
      SSN_print_when(SSN_print_when_), SSN_print_what(SSN_print_what_)
    {
        // Validate required matrices
        if (Q.rows() == 0 || Q.cols() == 0) {
            throw std::invalid_argument("Matrix Q must be provided with nonzero dimensions.");
        }
        if (A.rows() == 0 || A.cols() == 0) {
            throw std::invalid_argument("Matrix A must be provided with nonzero dimensions.");
        }
        if (B.rows() == 0 || B.cols() == 0) {
            throw std::invalid_argument("Matrix B must be provided with nonzero dimensions.");
        }

        // Dimensions
        int n = Q.rows();
        int m = A.rows();
        int l = B.rows();

        // Validate dimensions
        if (Q.cols() != n) {
            throw std::invalid_argument("Matrix Q must be square (n x n).");
        }
        if (A.cols() != n) {
            throw std::invalid_argument("Matrix A must have n columns.");
        }
        if (B.cols() != n) {
            throw std::invalid_argument("Matrix B must have n columns.");
        }

        // Validate vector dimensions
        if (c.size() != n) {
            throw std::invalid_argument("Vector c must have size n.");
        }
        if (b.size() != m) {
            throw std::invalid_argument("Vector b must have size m.");
        }
        if (lx.size() != n || ux.size() != n) {
            throw std::invalid_argument("Vectors lx and ux must have size n.");
        }
        if (lw.size() != l || uw.size() != l) {
            throw std::invalid_argument("Vectors lw and uw must have size l.");
        }

    }
};