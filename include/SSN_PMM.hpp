#pragma once
#include <limits>
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
//    .pmm_iter    -> number of PMM iterations performed to terminate
//    .ssn_iter    -> number of SSN iterations performed to terminate
//    .krylov_iter -> number of Krylov iterations performed to terminate
//    .fact        -> number of factorizations performed to terminate
//    .smw_count   -> number of SMW preconditioner applications performed to terminate
//    .pmm_tol_achieved -> tolerance achieved by PMM
//    .ssn_tol_achieved -> tolerance achieved by SSN
//    .solving_time     -> total time in seconds taken to solve the problem
//    .linesearch_fail  -> number of linesearch failures
//    .krylov_fail      -> number of Krylov failures
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

    T c_scalar = 1.0; // Global objective scalar
    SpMat Q_ruiz, A_ruiz, B_ruiz;
    Vec problem_Q_diag, Q_diag_ruiz, c_ruiz, b_ruiz, lx_ruiz, ux_ruiz, lw_ruiz, uw_ruiz;
    Vec D1A_diag, D1B_diag, D2_diag;
    Vec D1A_ext, D2_ext; // extended to size N, M
    Vec D1A_ext_inv, D2_ext_inv, D1B_diag_inv; // precomputed reciprocals (cwiseProduct is cheaper than cwiseQuotient)
    Vec c_orig, b_orig, lx_orig, ux_orig, lw_orig, uw_orig; // unscaled problem data for terminataion/infeasibility check
    Vec x_sol, y1_sol, y2_sol, z_sol;

    // Pre-allocated scratch vectors for the PMM main loop
    Vec Ax_scratch_, Bx_scratch_, Qx_scratch_;      // current iteration products
    Vec Ax_old_scratch_, Bx_old_scratch_;           // previous iteration products (swapped, not copied)
    Vec Adx_scratch_, Bdx_scratch_;                 // differences
    Vec x_old_scratch_, y2_old_scratch_;            // previous iterates for infeasibility check

    // Pre-allocated scratch vectors for compute_residual_unscaled_inf_norms
    Vec A_tr_y1_scratch_, B_tr_y2_scratch_;         // size N
    Vec num_scratch_;                               // size N
    Vec proj_K_scratch_, proj_K_unscaled_scratch_;  // size N
    Vec proj_W_scratch_, proj_W_unscaled_scratch_;  // size l
    Vec Ax_unscaled_scratch_;                       // size M
    Vec z_unscaled_scratch_, num_unscaled_scratch_; // size N
    Vec x_unscaled_scratch_;                        // size N
    Vec Bx_unscaled_scratch_, y2_unscaled_scratch_; // size l

    T inf = std::numeric_limits<T>::infinity();
    T eps_zero = T(100) * std::numeric_limits<T>::epsilon(); // ~2.2e-14 for double

    // Constant parameters
    T tol = 1e-6;
    int max_iter = 100000000;
    int ssn_max_iter = 100000000;
    int ssn_max_in_iter = 50;
    T eps_limit = 1e-3 * tol;
    T mu_limit = 1e8;
    T rho_limit = 1e8;
    T eps_pinf = 5e-2 * tol;
    T eps_dinf = 5e-2 * tol;
    T alpha = 0.95;
    double time_limit = 60.0; // in seconds
    int stagnation = 0;
    int linesearch_fail = 0;

    // Updated parameters
    T mu0 = 1e0;
    T rho0 = 1e0;
    T mu = 1e1;
    T rho = 1e1;
    T ssn_tol = 1e-2;
    
    // Outputs:
    int opt;
    Vec x, y1, y2, z;
    T obj_val;
    int pmm_iter, ssn_iter;
    int krylov_iter = 0, fact = 0, krylov_fail = 0;
    T pmm_tol_achieved, ssn_tol_achieved;
    ResVec res_norms;
    ResVec res_norms_scaled;
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
    static void build_reformulated_vecs(int n, int m, int N, int M, T inf,
                                        const Vec& c_in, const Vec& b_in, const Vec& lx_in, const Vec& ux_in,
                                        Vec& c_out, Vec& b_out, Vec& lx_out, Vec& ux_out);
    void set_default(const Problem<T>& problem);
    void initialize_sols();
    void check_bounds();

    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline T inf_norm(const Vec& v) {
        return v.cwiseAbs().maxCoeff();
    }
    ResVec compute_residual_unscaled_inf_norms(const Vec& Ax, const Vec& Bx, const Vec& Qx, ResVec& res_norms_scaled);
    T objective_value(const Vec& x, const Vec& Qx);
    void printable_sol(const Vec& x, const Vec& y1, const Vec& y2, const Vec& z);
    void update_PMM_parameters(const ResVec& res_norms, const ResVec& new_res_norms, int ssn_opt, T ssn_res, int ssn_inner_iters);
    bool primal_infeas(const Vec& cert_y1, const Vec& cert_y2, const Vec& cert_z);
    bool dual_infeas(const Vec& delta_x, const Vec& Adx, const Vec& Bdx);
    Solution<T> solve();
};

#include "SSN_PMM.tpp"