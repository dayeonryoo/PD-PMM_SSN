#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Printing.hpp"

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
    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux;
    Vec lw, uw;
    int n, m, l;
    Vec x, y1, y2, z;
    int SSN_max_in_iter;
    T mu, rho, SSN_tol;
    PrintWhen SSN_print_when;
    PrintWhat SSN_print_what;
    PrintLabel SSN_print_label = PrintLabel::SSN;

    // Useful vectors and matrices
    Vec ones_n, ones_m, ones_l;
    Vec Q_diag;
    SpMat A_tr, B_tr;

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

    SSN(const SpMat& Q_, const SpMat& A_, const SpMat& B_,
        const Vec& c_, const Vec& b_,
        const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_,
        const Vec& x_, const Vec& y1_, const Vec& y2_, const Vec& z_,
        T mu_, T rho_, int n_, int m_, int l_,
        T SSN_tol_, int SSN_max_in_iter_,
        PrintWhen SSN_print_when_ = PrintWhen::NEVER,
        PrintWhat SSN_print_what_ = PrintWhat::NONE)
    : Q(Q_), A(A_), B(B_), c(c_), b(b_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_),
      x(x_), y1(y1_), y2(y2_), z(z_),
      mu(mu_), rho(rho_), n(n_), m(m_), l(l_),
      SSN_tol(SSN_tol_), SSN_max_in_iter(SSN_max_in_iter_),
      SSN_print_when(SSN_print_when_), SSN_print_what(SSN_print_what_)
    {
        ones_n = Vec::Ones(n);
        ones_m = Vec::Ones(m);
        ones_l = Vec::Ones(l);
        Q_diag = Q.diagonal();
        A_tr = A.transpose();
        B_tr = B.transpose();
    }

    Vec proj(const Vec& u, const Vec& lower, const Vec& upper);
    Vec compute_dist_box(const Vec& v, const Vec& lower, const Vec& upper);
    T compute_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec Clarke_subgrad_of_proj(const Vec& u, const Vec& lower, const Vec& upper);
    SpMat build_diag_matrix(const Vec& diag);
    Vec separate_rows(const Vec& u, const BoolArr& mask);
    SpMat separate_rows(const SpMat& M, const BoolArr& mask);
    Vec retrive_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask);
    SpMat stack_rows(const SpMat& A, const SpMat& B);
    Vec solve_via_chol(const SpMat& M, const Vec& r);
    T backtracking_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2);
    SSN_result<T> solve_SSN(const T eps);

};

#include "SSN.tpp"