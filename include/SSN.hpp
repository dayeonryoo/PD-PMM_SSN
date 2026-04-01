#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Printing.hpp"
#include "SchurOperator.hpp"
#include "SchurPreconditioner.hpp"


template <typename T>
struct SSN_result {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    Vec x;
    Vec y2;
    int opt;
    int iter;
    T tol_achieved;
};

template <typename T>
class SSN {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<T>;

    // Inputs
    const int Q_info;
    const Vec& Q_diag;
    const SpMat& L, A, B;
    const Vec& D1A_diag, D1B_diag, D2_diag;
    const Vec& c, b, lx, ux, lw, uw;
    const T obj_const;
    const int n, m, N, M, l;
    Vec x, y1, y2, z;
    Vec delta_x, delta_y1, delta_y2, delta_z;
    int SSN_max_in_iter;
    T mu, rho, gamma, SSN_tol;
    T eps_pinf, eps_dinf;

    // Useful vectors and matrices
    T inf = 1e20;
    Vec ones_N, ones_M, ones_l;
    const SpMat& A_tr, B_tr, L_tr;
    Vec H_diag, H_diag_inv;
    Vec diag_P_K, diag_P_W;
    BoolArr active_W, inactive_W, active_K;
    int n_active_W, n_inactive_W;
    SpMat B_active_W, B_inactive_W, G, G_tr;

    bool more_rows_than_cols;
    bool do_exact = true;

    // Outputs
    int SSN_in_iter;
    int SSN_iter;
    T SSN_tol_achieved;
    int SSN_opt;
    T obj_val;

    // Backtracking linesearch parameters
    T beta = 0.4995 / 2;
    T delta = 0.995;

    // Conjugate gradient parameters
    T Krylov_tol = 1e-12;
    int Krylov_max_in_iter = 50;

    using CGSolver = Eigen::ConjugateGradient<
        SchurOperator<T>,
        Eigen::Lower | Eigen::Upper,
        SchurPreconditioner<T>
    >;
    CGSolver cg;
    Vec prev_dy_;
    
    // SSN() = default;

    SSN(const int Q_info_, const Vec& Q_diag_, const SpMat& L_, const SpMat& L_tr_,
        const SpMat& A_, const SpMat& B_, const SpMat& A_tr_, const SpMat& B_tr_,
        const Vec& c_, const Vec& b_, const Vec& D1A_diag_, const Vec& D1B_diag_, const Vec& D2_diag_,
        const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_, const T obj_const_,
        int n_, int m_, int N_, int M_, int l_,
        T SSN_tol_, int SSN_max_in_iter_, bool more_rows_than_cols_,
        T eps_pinf_, T eps_dinf_)
    : Q_info(Q_info_), Q_diag(Q_diag_), L(L_), L_tr(L_tr_),
      A(A_), B(B_), A_tr(A_tr_), B_tr(B_tr_),
      c(c_), b(b_), D1A_diag(D1A_diag_), D1B_diag(D1B_diag_), D2_diag(D2_diag_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_), obj_const(obj_const_),
      n(n_), m(m_), N(N_), M(M_), l(l_),
      SSN_tol(SSN_tol_), SSN_max_in_iter(SSN_max_in_iter_),
      more_rows_than_cols(more_rows_than_cols_),
      eps_pinf(eps_pinf_), eps_dinf(eps_dinf_)
    {
        ones_N = Vec::Ones(N);
        ones_M = Vec::Ones(M);
        ones_l = Vec::Ones(l);
        SSN_iter = 0;
        delta_x = Vec::Zero(N);
        delta_y2 = Vec::Zero(l);
    }

    void update_SSN_system(const Vec& x_, const Vec& y1_, const Vec& y2_, const Vec& z_,
                           const Vec& delta_y1_, const Vec& delta_z_,
                           T mu_, T rho_, T gamma_, int SSN_iter_) {
        x = x_;
        y1 = y1_;
        y2 = y2_;
        z = z_;
        mu = mu_;
        rho = rho_;
        gamma = gamma_;
        SSN_iter = SSN_iter_;
        delta_y1 = delta_y1_;
        delta_z = delta_z_;
    }

    static inline T inf_norm(const Vec& v) {
        return v.cwiseAbs().maxCoeff();
    }
    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline Vec compute_dist_box(const Vec& v, const Vec& lower, const Vec& upper) {
        return (v - proj(v, lower, upper));
    }
    T compute_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec Clarke_subgrad_of_proj(const Vec& u, const Vec& lower, const Vec& upper, const bool include_bd);
    bool is_P_unchanged(const Vec& diag_P, const Vec& new_diag_P);
    void split_by_mask(const Vec& u, const BoolArr& mask, Vec& u_sel, Vec& u_unsel);
    void build_B_active_inactive(const SpMat& B, const BoolArr& mask, SpMat& B_active, SpMat& B_inactive);
    SpMat scale_columns(const SpMat& M, const Vec& d);
    Vec retrive_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask);
    SpMat stack_rows(const SpMat& A, const SpMat& B);
    bool form_schur(const SpMat& G);
    Vec solve_using_cg(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv, const BoolArr& active_K, const Vec& r1, const Vec& r2, T mu, T tol, int max_iter, bool update_prec);
    Vec solve_using_schur(const SpMat& G, const SpMat& G_tr, const Vec& H_diag_inv, const Vec& r1, const Vec& r2);
    Vec solve_using_LDLT(const SpMat& G, const Vec& H_diag, const Vec& r1, const Vec& r2);
    T backtracking_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2);
    T exact_line_search_w_Lag(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2);
    T exact_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2);
    bool primal_infeas(const Vec& delta_y1, const Vec& delta_y2, const Vec& delta_z, T eps_pinf);
    bool dual_infeas(const Vec& delta_x, T eps_dinf);
    SSN_result<T> solve_SSN(const T eps);

};

#include "SSN.tpp"