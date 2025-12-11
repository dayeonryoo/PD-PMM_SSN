#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Problem.hpp"
#include "Solution.hpp"
#include "SSN.hpp"


// =============================================================
//      min  c^T x + (1/2) x^T Q x,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw
// =============================================================
// INPUT: Problem
// --------------------------------------------------------------
// A class containing the data of the problem to be solved:
//    .Q       -> n x n sparse quadratic coefficient matrix
//    .A       -> m x n sparse linear equality constraint matrix
//    .B       -> l x n sparse box constraint matrix on Bx
//    .b       -> m-dim right-hand side vector for linear equality constraints
//    .c       -> n-dim coefficient vector
//    .lx      -> n-dim lower bound vector for box constraints on x
//    .ux      -> n-dim upper bound vector for box constraints on x
//    .lw      -> l-dim lower bound vector for box constraints on Bx
//    .uw      -> l-dim upper bound vector for box constraints on Bx
//    .tol     -> tolerance for termination
//    .max_it  -> maximum allowed number of PMM iterations
// =============================================================
// OUTPUT: Solution
// --------------------------------------------------------------
// A class containing the solution of the PMM_SSN solver:
//    .opt     -> Integer indicating the termination status:
//                  0: optimal solution found
//                  1: maximum number of iterations reached
//                  2: termination due to numerical errors
//    .x       -> Optimal primal solution vector
//    .y1      -> Lagrangian multipliers corresponding to Ax = b
//    .y2      -> Lagrangian multipliers corresponding to Bx = w
//    .z       -> Lagrangian multipliers corresponding to box constraints on x
//    .obj_val -> Optimal objective value
//    .PMM_it  -> number of PMM iterations performed to terminate
//    .SSN_it  -> number of SSN iterations performed to terminate
// --------------------------------------------------------------

template <typename T>
class SSN_PMM {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    // Inputs:
    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
    T tol;
    int max_it;

    int n, m, l;

    // SSN parameters
    T mu = 5e1;
    T rho = 1e2;
    int SSN_max_iter = 4000; // 4000
    int SSN_max_in_iter = 3; // 40
    T SSN_tol = tol;
    T reg_limit = 1e6;

    // Outputs:
    int opt;
    Vec x, y1, y2, z;
    T obj_val;
    int PMM_iter, SSN_iter;
    T PMM_tol_achieved, SSN_tol_achieved;

    // Constructor
    SSN_PMM(Problem<T>& problem)
    : Q(problem.Q), A(problem.A), B(problem.B), c(problem.c), b(problem.b),
      lx(problem.lx), ux(problem.ux), lw(problem.lw), uw(problem.uw),
      tol(problem.tol), max_it(problem.max_it)
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
        n = Q.rows();
        m = A.rows();
        l = B.rows();

        // Validate dimensions
        if (Q.cols() != n) {
            throw std::invalid_argument("Matrix Q must be square with size n x n.");
        }
        if (A.cols() != n) {
            throw std::invalid_argument("Matrix A must have n columns.");
        }
        if (B.cols() != n) {
            throw std::invalid_argument("Matrix B must have n columns.");
        }
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

    Vec compute_residual_norms();
    void update_PMM_parameters(const T res_p, const T res_d, const T new_res_p, const T new_res_d);
    Solution<T> solve();
};

#include "SSN_PMM.tpp"