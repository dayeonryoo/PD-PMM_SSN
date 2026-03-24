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
    T obj_const;

    int n, m, l;
    int N, M;
    int Q_info; // 0 = zero; 1 = diagonal; 2 = general
    Vec Q_diag;
    SpMat L, L_tr;
    SpMat A_tr, B_tr;

    SpMat Q_ruiz, A_ruiz, B_ruiz;
    Vec problem_Q_diag, Q_diag_ruiz, c_ruiz, b_ruiz, lx_ruiz, ux_ruiz;
    Vec D1_diag, D2_diag;
    Vec x_descaled, y1_descaled, z_descaled;
    Vec x_sol, y1_sol, z_sol;

    // Constant parameters
    T tol = 1e-4;
    int max_iter = 200;
    int SSN_max_iter = 2000;
    int SSN_max_in_iter = 50;
    T reg_limit = 1e+0/tol;
    T eps_limit = 1e-2*tol;
    T eps_pinf = 1e-8;
    T eps_dinf = 1e-8;
    T gamma = 0.9;
    bool more_rows_than_cols = N < M + l;
    
    // Updated parameters
    T mu0 = 1e0;
    T mu = 1e1;
    T rho = 1e1;
    T eps_bcl = 1e0;
    T SSN_tol = 1e0;
    
    // Outputs:
    int opt;
    Vec x, y1, y2, z;
    T obj_val;
    int PMM_iter, SSN_iter;
    T PMM_tol_achieved, SSN_tol_achieved;

    // Printing
    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::NONE;

    // Constructors
    SSN_PMM() {}
    SSN_PMM(const Problem<T>& problem)
    : tol(problem.tol), max_iter(problem.max_iter),
      n(problem.n), m(problem.m), l(problem.l), obj_const(problem.obj_const),
      when(problem.when), what(problem.what)
    {
        get_Q_info(problem.Q);
        if (n == 0 && m == 0 && l == 0) {
            determine_dimensions(problem);
        }
        check_dimensions(problem);
        ruiz_scaling(problem.Q, problem_Q_diag, problem.A, problem.B, problem.c, problem.b, problem.lx, problem.ux);
        set_default(problem);
        initialize_sols();
        check_bounds();
        A_tr = A.transpose();
        B_tr = B.transpose();
    }

    void get_Q_info(const SpMat& Q);
    void determine_dimensions(const Problem<T>& problem);
    void check_dimensions(const Problem<T>& problem);
    void ruiz_scaling(const SpMat& Q, const Vec& Q_diag, const SpMat& A, const SpMat& B, const Vec& c, const Vec& b, const Vec& lx, const Vec& ux);
    void set_L_from_LLT(const SpMat& Q);
    void set_default(const Problem<T>& problem);
    void initialize_sols();
    void check_bounds();

    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline T inf_norm(const Vec& v) {
        return v.cwiseAbs().maxCoeff();
    }
    Vec compute_residual_norms();
    Vec compute_residual_norms_inf();
    T objective_value(const Vec& x);
    void printable_sol(const Vec& x, const Vec& y1, const Vec& z);
    void update_PMM_parameters(const T res_p, const T res_d, const T new_res_p, const T new_res_d);
    T compute_p(const Vec& x);
    void update_with_bcl(const T p, const Vec& y2_hat, const T res);
    bool primal_infeas_y1(const Vec& delta_y1, T eps_pinf);
    bool primal_infeas_z(const Vec& delta_z, T eps_pinf);
    Solution<T> solve();
};

#include "SSN_PMM.tpp"