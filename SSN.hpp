#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

template <typename T>
struct SSN_result {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    Vec x;
    Vec y2;
    int SSN_in_iter;
    T SSN_tol_achieved;
};

template <typename T>
class SSN {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat = Eigen::SparseMatrix<T>;

    // Inputs
    Mat Q, A, B;
    Vec c, b;
    Vec lx, ux;
    Vec lw, uw;
    int n, m, l;
    Vec x, y1, y2, z;
    int SSN_max_in_iter;
    T mu, rho, SSN_tol;

    // Outputs
    int SSN_in_iter;
    bool SSN_tol_achieved;
    
    SSN(const Mat& Q_, const Mat& A_, const Mat& B_,
        const Vec& c_, const Vec& b_,
        const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_,
        const Vec& x_, const Vec& y1_, const Vec& y2_, const Vec& z_,
        T mu_, T rho_, int n_, int m_, int l_,
        T SSN_tol_, int SSN_max_in_iter_)
    : Q(Q_), A(A_), B(B_), c(c_), b(b_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_),
      x(x_), y1(y1_), y2(y2_), z(z_),
      mu(mu_), rho(rho_), n(n_), m(m_), l(l_),
      SSN_tol(SSN_tol_), SSN_max_in_iter(SSN_max_in_iter_)
    {}

    Vec compute_box_proj(const Vec& v, const Vec& lower, const Vec& upper);
    Vec compute_dist_box(const Vec& v, const Vec& lower, const Vec& upper);
    Vec compute_Lagrangian(const Vec& x_new, const Vec& y2_new);
    Vec compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new);
    SSN_result<T> solve_SSN();

};

#include "SSN.tpp"