#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Printing.hpp"
#include "QInfo.hpp"


template <typename T>
struct SSN_result {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    Vec x;
    Vec y2;
    int SSN_in_iter;
    T SSN_tol_achieved;
    int SSN_opt;
    T obj_val;
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
    const Vec& D1_diag, D2_diag;
    const Vec& c, b, lx, ux, lw, uw;
    const T obj_const;
    const int n, m, N, M, l;
    Vec x, y1, y2, z;
    Vec y1_sol, z_sol;
    int SSN_max_in_iter;
    T mu, rho, gamma, SSN_tol;
    PrintWhen SSN_print_when;
    PrintWhat SSN_print_what;
    PrintLabel SSN_print_label = PrintLabel::SSN;

    // Useful vectors and matrices
    Vec x_descaled, x_sol;
    Vec ones_N, ones_M, ones_l;
    const SpMat& A_tr, B_tr, L_tr;
    Vec H_diag, H_diag_inv;
    Vec diag_P_K, diag_P_W;
    BoolArr active_W, inactive_W;
    int n_active_W, n_inactive_W;
    SpMat B_active_W, B_inactive_W, G, G_tr;

    // Outputs
    int SSN_in_iter;
    T SSN_tol_achieved;
    int SSN_opt;
    T obj_val;

    // Set the semismooth Newton parameters
    T beta = 0.4995 / 2;
    T delta = 0.995;
    
    // SSN() = default;

    SSN(const int Q_info_, const Vec& Q_diag_, const SpMat& L_, const SpMat& L_tr_,
        const SpMat& A_, const SpMat& B_, const SpMat& A_tr_, const SpMat& B_tr_,
        const Vec& c_, const Vec& b_, const Vec& D1_diag_, const Vec& D2_diag_,
        const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_, const T obj_const_,
        int n_, int m_, int N_, int M_, int l_,
        T SSN_tol_, int SSN_max_in_iter_,
        PrintWhen SSN_print_when_, PrintWhat SSN_print_what_)
    : Q_info(Q_info_), Q_diag(Q_diag_), L(L_), L_tr(L_tr_),
      A(A_), B(B_), A_tr(A_tr_), B_tr(B_tr_),
      c(c_), b(b_), D1_diag(D1_diag_), D2_diag(D2_diag_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_), obj_const(obj_const_),
      n(n_), m(m_), N(N_), M(M_), l(l_),
      SSN_tol(SSN_tol_), SSN_max_in_iter(SSN_max_in_iter_),
      SSN_print_when(SSN_print_when_), SSN_print_what(SSN_print_what_)
    {
        ones_N = Vec::Ones(N);
        ones_M = Vec::Ones(M);
        ones_l = Vec::Ones(l);
    }

    void update_SSN_system(const Vec& x_, const Vec& y1_, const Vec& y2_, const Vec& z_,
                           const Vec& y1_sol_, const Vec& z_sol_, T mu_, T rho_, T gamma_) {
        x = x_;
        y1 = y1_;
        y2 = y2_;
        z = z_;
        y1_sol = y1_sol_;
        z_sol = z_sol_;
        mu = mu_;
        rho = rho_;
        gamma = gamma_; // although it's constant
    }

    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline Vec compute_dist_box(const Vec& v, const Vec& lower, const Vec& upper) {
        return (v - proj(v, lower, upper));
    }
    T get_obj_val(const Vec& x);
    Vec printable_x(const Vec& x);
    T compute_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec Clarke_subgrad_of_proj(const Vec& u, const Vec& lower, const Vec& upper, const bool include_bd);
    bool is_P_unchanged(const Vec& diag_P, const Vec& new_diag_P);
    void split_by_mask(const Vec& u, const BoolArr& mask, Vec& u_sel, Vec& u_unsel);
    void build_B_active_inactive(const SpMat& B, const BoolArr& mask, SpMat& B_active, SpMat& B_inactive);
    void scale_columns(SpMat& M, const Vec& d);
    Vec retrive_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask);
    SpMat stack_rows(const SpMat& A, const SpMat& B);
    Vec solve_via_chol(const SpMat& M, const Vec& r);
    T backtracking_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2);
    SSN_result<T> solve_SSN(const T eps);

};

#include "SSN.tpp"