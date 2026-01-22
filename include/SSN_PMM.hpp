#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Problem.hpp"
#include "Solution.hpp"
#include "SSN.hpp"
#include "Printing.hpp"
#include "QInfo.hpp"


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
//                 -1: termination due to numerical errors
//                  0: optimal solution found
//                  1: maximum number of iterations reached
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
    T tol = 0.0;
    int max_iter = 0;

    int n, m, l;
    int N, M;
    QInfo Q_info;
    Vec Q_diag;
    SpMat L;
    SpMat A_tr, B_tr;

    SpMat A_ruiz;
    Vec b_ruiz, lx_ruiz, ux_ruiz;
    Vec D1_diag, D2_diag;
    Vec x_descaled, y1_descaled;

    // PMM parameters
    T mu, rho;

    // SSN parameters
    int SSN_max_iter, SSN_max_in_iter;
    T SSN_tol, reg_limit;

    // Outputs:
    int opt;
    Vec x, y1, y2, z;
    T obj_val;
    int PMM_iter, SSN_iter;
    T PMM_tol_achieved, SSN_tol_achieved;

    // Printing
    PrintWhen PMM_print_when = PrintWhen::NEVER;
    PrintWhen SSN_print_when = PrintWhen::NEVER;
    PrintWhat PMM_print_what = PrintWhat::NONE;
    PrintWhat SSN_print_what = PrintWhat::NONE;
    PrintLabel PMM_print_label = PrintLabel::PMM;

    // Constructors
    SSN_PMM() {}
    SSN_PMM(const Problem<T>& problem)
    : tol(problem.tol), max_iter(problem.max_iter),
      PMM_print_when(problem.PMM_print_when), PMM_print_what(problem.PMM_print_what),
      SSN_print_when(problem.SSN_print_when), SSN_print_what(problem.SSN_print_what)
    {
        get_Q_info(problem.Q);
        determine_dimensions(problem);
        set_default(problem);
        check_dimensions();
        check_infeasibility();

        A_tr = A.transpose();
        B_tr = B.transpose();
    }

    void determine_dimensions(const Problem<T>& problem);
    void get_Q_info(const SpMat& Q);
    void set_L_from_LLT(const SpMat& Q);
    void set_default(const Problem<T>& problem);
    void check_dimensions();
    void check_infeasibility();

    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline T inf_norm(const Vec& v) {
        return v.cwiseAbs().maxCoeff();
    }
    
    Vec compute_residual_norms();
    Vec compute_residual_norms_inf();
    T objective_value();
    void update_PMM_parameters(const T res_p, const T res_d, const T new_res_p, const T new_res_d);
    Solution<T> solve();
};

#include "SSN_PMM.tpp"