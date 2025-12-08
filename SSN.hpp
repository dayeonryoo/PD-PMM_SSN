#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

template <typename T>
class SSN {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat = Eigen::SparseMatrix<T>;

    // Inputs
    Mat Q, A, B, A_T, B_T;
    Vec Q_diag;
    Vec c, b;
    Vec lx, ux;
    Vec lw, uw;
    int n, m, l;
    Vec x, y1, y2, z;
    int max_SSN_iters;
    T mu, rho, tol;

    // Outputs
    int iter;
    bool tol_achieved;
    

    SSN(const Mat& Q_, const Mat& A_, const Mat& B_,
        const Vec& c_, const Vec& b_,
        const Vec& lx_, const Vec& ux_,
        const Vec& lw_, const Vec& uw_,
        const Vec& x_, const Vec& y1_, const Vec& y2_, const Vec& z_,
        T mu_, T rho_, T tol_,
        int n_, int m_, int l_,
        int max_SSN_iters_)
    : Q(Q_), A(A_), B(B_),
      c(c_), b(b_),
      lx(lx_), ux(ux_),
      lw(lw_), uw(uw_),
      x(x_), y1(y1_), y2(y2_), z(z_),
      mu(mu_), rho(rho_), tol(tol_),
      n(n_), m(m_), l(l_),
      max_SSN_iters(max_SSN_iters_)
    {}

    void solve(); // Update x, y1, y2, z in place

};

#include "SSN.tpp"