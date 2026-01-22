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
};

template <typename T>
class SSN {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<T>;

    // Inputs
    QInfo Q_info;
    Vec Q_diag;
    SpMat L, A, B;
    Vec c, b, lx, ux, lw, uw;
    int N, M, l;
    Vec x, y1, y2, z;
    int SSN_max_in_iter;
    T mu, rho, SSN_tol;
    PrintWhen SSN_print_when;
    PrintWhat SSN_print_what;
    PrintLabel SSN_print_label = PrintLabel::SSN;

    // Useful vectors and matrices
    Vec ones_N, ones_M, ones_l;
    SpMat A_tr, B_tr, L_tr;

    // Outputs
    int SSN_in_iter;
    T SSN_tol_achieved;
    int SSN_opt;

    // Set the semismooth Newton parameters
    T beta = 0.4995 / 2;
    T delta = 0.995;
    T eta = 0.1 * SSN_tol;
    T gamma = 0.1;
    
    SSN() {}

    SSN(const QInfo& Q_info_, const Vec& Q_diag_, SpMat& L_,
        const SpMat& A_, const SpMat& B_, const SpMat& A_tr_, const SpMat& B_tr_,
        const Vec& c_, const Vec& b_,
        const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_,
        const Vec& x_, const Vec& y1_, const Vec& y2_, const Vec& z_,
        T mu_, T rho_, int N_, int M_, int l_,
        T SSN_tol_, int SSN_max_in_iter_,
        PrintWhen SSN_print_when_, PrintWhat SSN_print_what_)
    : Q_info(Q_info_), Q_diag(Q_diag_), L(L_), A(A_), B(B_),
      A_tr(A_tr_), B_tr(B_tr_), c(c_), b(b_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_),
      x(x_), y1(y1_), y2(y2_), z(z_),
      mu(mu_), rho(rho_), N(N_), M(M_), l(l_),
      SSN_tol(SSN_tol_), SSN_max_in_iter(SSN_max_in_iter_),
      SSN_print_when(SSN_print_when_), SSN_print_what(SSN_print_what_)
    {
        ones_N = Vec::Ones(N);
        ones_M = Vec::Ones(M);
        ones_l = Vec::Ones(l);
        L_tr = L.transpose();
    }

    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline Vec compute_dist_box(const Vec& v, const Vec& lower, const Vec& upper) {
        return (v - proj(v, lower, upper));
    }
    T compute_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec Clarke_subgrad_of_proj(const Vec& u, const Vec& lower, const Vec& upper);
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