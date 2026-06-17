#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Problem.hpp"
#include "Solution.hpp"
#include "SSN.hpp"
#include "Printing.hpp"

// =============================================================
//      min  c^T x + (1/2) x^T Q x + obj_const,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw
// =============================================================
// INPUT: Problem
// --------------------------------------------------------------
// A class containing the data of the problem to be solved:
//    .Q         -> n x n sparse symmetric positive semidefinite quadratic coefficient matrix
//                  (given as a full matrix or a lower triangular matrix)
//    .A         -> m x n sparse linear equality constraint matrix
//    .B         -> l x n sparse box constraint matrix on Bx
//    .b         -> m-dim right-hand side vector for linear equality constraints
//    .c         -> n-dim coefficient vector
//    .lx        -> n-dim lower bound vector for box constraints on x
//    .ux        -> n-dim upper bound vector for box constraints on x
//    .lw        -> l-dim lower bound vector for box constraints on Bx
//    .uw        -> l-dim upper bound vector for box constraints on Bx
//    .obj_const -> constant term in the objective function
//    .tol       -> tolerance for termination
//    .max_it    -> maximum allowed number of PMM iterations
// =============================================================
// OUTPUT: Solution
// --------------------------------------------------------------
// A class containing the solution of the PMM_SSN solver:
//    .opt     -> Integer indicating the termination status:
//                 -3: termination due to dual infeasibility
//                 -2: termination due to primal infeasibility
//                 -1: termination due to numerical errors
//                  0: optimal solution found
//                  1: maximum number of PMM iterations reached
//                  2: maximum number of SSN iterations reached
//                  3: termination due to line search failure
//                  4: termination due to time limit
//    .x       -> Optimal primal solution vector
//    .y1      -> Lagrangian multipliers corresponding to Ax = b
//    .y2      -> Lagrangian multipliers corresponding to Bx = w
//    .z       -> Lagrangian multipliers corresponding to box constraints on x
//    .obj_val -> Optimal objective value
//    .PMM_iter    -> number of PMM iterations performed to terminate
//    .SSN_iter    -> number of SSN iterations performed to terminate
//    .Krylov_iter -> number of Krylov iterations performed to terminate
//    .fact        -> number of factorizations performed to terminate
//    .smw_count   -> number of SMW preconditioner applications performed to terminate
//    .PMM_tol_achieved -> tolerance achieved by PMM
//    .SSN_tol_achieved -> tolerance achieved by SSN
//    .solving_time     -> total time in seconds taken to solve the problem
//    .linesearch_fail  -> number of linesearch failures
//    .Krylov_fail      -> number of Krylov failures
// --------------------------------------------------------------

template <typename T>
class SSN_PMM {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using ResVec = Eigen::Matrix<T, 4, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    // Inputs:
    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
    T obj_const;

    int n, m, l;
    int N, M;
    int Q_info; // 0 = zero; 1 = diagonal; 2 = general
    Vec Q_diag;
    SpMat L, L_tr;
    SpMat A_tr, B_tr;

    SpMat Q_ruiz, A_ruiz, B_ruiz;
    Vec problem_Q_diag, Q_diag_ruiz, c_ruiz, b_ruiz, lx_ruiz, ux_ruiz, lw_ruiz, uw_ruiz;
    Vec D1A_diag, D1B_diag, D2_diag;
    Vec x_sol, y1_sol, y2_sol, z_sol;

    // Pre-allocated scratch vectors for the PMM main loop
    Vec Ax_scratch_, Bx_scratch_, Qx_scratch_;      // current iteration products
    Vec Ax_old_scratch_, Bx_old_scratch_;           // previous iteration products (swapped, not copied)
    Vec Adx_scratch_, Bdx_scratch_;                 // differences
    Vec x_old_scratch_, y2_old_scratch_;            // previous iterates for infeasibility check

    T inf = 1e20;
    T eps_zero = 1e-12; // for checking near-zero values without scaling issues

    // Constant parameters
    T tol = 1e-6;
    int max_iter = 100000000;
    int SSN_max_iter = 100000000;
    int SSN_max_in_iter = 20;
    T eps_limit = 1e-3*tol;
    T mu_limit = 1e5;
    T rho_limit = 1e4;
    T eps_pinf = 5e-2 * tol;
    T eps_dinf = 5e-2 * tol;
    T gamma = 0.95;
    double time_limit = 60.0; // in seconds
    int stagnation = 0;
    int linesearch_fail = 0;
    int consecutive_linesearch_fail = 0;
    
    // Updated parameters
    T mu0 = 1e0;
    T rho0 = 1e0;
    T mu = 1e1;
    T rho = 1e1;
    T eps_bcl = 1e0;
    T SSN_tol = 1e-2;
    
    // Outputs:
    int opt;
    Vec x, y1, y2, z;
    T obj_val;
    int PMM_iter, SSN_iter;
    int Krylov_iter = 0, fact = 0, Krylov_fail = 0;
    T PMM_tol_achieved, SSN_tol_achieved;
    bool ldlt_used = false;

    // Printing
    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::NONE;

    // Constructor
    SSN_PMM(const Problem<T>& problem)
    : tol(problem.tol), max_iter(problem.max_iter), time_limit(problem.time_limit),
      n(problem.n), m(problem.m), l(problem.l), obj_const(problem.obj_const),
      when(problem.when), what(problem.what)
    {
        get_Q_info(problem.Q);
        if (n == 0 && m == 0 && l == 0) {
            determine_dimensions(problem);
        }
        check_dimensions(problem);
        ruiz_scaling(problem, problem_Q_diag);
        set_default(problem);
        initialize_sols();
        check_bounds();
        A_tr = A.transpose();
        B_tr = B.transpose();
    }

    void get_Q_info(const SpMat& Q);
    void determine_dimensions(const Problem<T>& problem);
    void check_dimensions(const Problem<T>& problem);
    void ruiz_scaling(const Problem<T>& problem, const Vec& Q_diag);
    void set_L_from_LLT(const SpMat& Q);
    void set_default(const Problem<T>& problem);
    void compute_initial_penalties();
    void initialize_sols();
    void check_bounds();

    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline T inf_norm(const Vec& v) {
        return v.cwiseAbs().maxCoeff();
    }
    ResVec compute_residual_norms(const Vec& Ax, const Vec& Bx, const Vec& Qx);
    ResVec compute_residual_norms_inf(const Vec& Ax, const Vec& Bx, const Vec& Qx);
    T objective_value(const Vec& x, const Vec& Qx);
    void printable_sol(const Vec& x, const Vec& y1, const Vec& y2, const Vec& z);
    void update_PMM_parameters(const ResVec& res_norms, const ResVec& new_res_norms, int SSN_opt, T SSN_tol_achieved);
    bool primal_infeas(const Vec& cert_y1, const Vec& cert_y2, const Vec& cert_z);
    bool dual_infeas(const Vec& delta_x, const Vec& Adx, const Vec& Bdx);
    Solution<T> solve();
};

#include "SSN_PMM.tpp"