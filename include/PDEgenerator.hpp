#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include "Problem.hpp"

namespace pdegen {

template <typename T>
struct Grid {
    int nc;          // exponent: n = 2^nc + 1
    int n1d;         // nodes per direction
    int np;          // total nodes
    T h;             // mesh size = 1/(n1d-1)

    // node index from (i,j)
    static int idx(int i, int j, int n1d) { return i + j * n1d; }

    explicit Grid(int nc_in)
        : nc(nc_in),
          n1d((1 << nc) + 1),
          np(n1d * n1d),
          h(T(1) / T(n1d - 1)) {}
};

template <typename T>
static inline bool is_boundary_node(int i, int j, int n1d) {
    return (i == 0 || j == 0 || i == n1d - 1 || j == n1d - 1);
}

/**
 * Assemble:
 *  - K: 5-point Laplacian stiffness (Dirichlet handled by row/col modification below)
 *  - M_lump: lumped mass as diagonal (area weights)
 *
 * Dirichlet handling:
 *  - For boundary nodes p: enforce y_p = bc(p) by setting K(p,p)=1, K(p,*)=0, rhs(p)=bc(p).
 */
template <typename T>
static void assemble_poisson_fd_lumped_mass(
    const Grid<T>& g,
    Eigen::SparseMatrix<T>& K,
    Eigen::SparseMatrix<T>& M_lump,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
    T bc_value
) {
    using Trip = Eigen::Triplet<T>;
    std::vector<Trip> Kt;
    std::vector<Trip> Mt;
    Kt.reserve(std::size_t(g.np) * 5);
    Mt.reserve(std::size_t(g.np));

    rhs = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(g.np);

    const T inv_h2 = T(1) / (g.h * g.h);
    const T area   = g.h * g.h;

    for (int j = 0; j < g.n1d; ++j) {
        for (int i = 0; i < g.n1d; ++i) {
            const int p = Grid<T>::idx(i, j, g.n1d);

            // lumped mass diagonal weight (set to 0 on boundary to mimic your MATLAB R(bound)=0 usage)
            if (is_boundary_node<T>(i, j, g.n1d)) {
                Mt.emplace_back(p, p, T(0));
            } else {
                Mt.emplace_back(p, p, area);
            }

            if (is_boundary_node<T>(i, j, g.n1d)) {
                // Dirichlet y = bc_value
                Kt.emplace_back(p, p, T(1));
                rhs(p) = bc_value;
                continue;
            }

            // Interior: -Δ discretization (SPD)
            // K y approx (4 y_p - y_E - y_W - y_N - y_S)/h^2
            Kt.emplace_back(p, p, T(4) * inv_h2);
            Kt.emplace_back(p, Grid<T>::idx(i - 1, j, g.n1d), T(-1) * inv_h2);
            Kt.emplace_back(p, Grid<T>::idx(i + 1, j, g.n1d), T(-1) * inv_h2);
            Kt.emplace_back(p, Grid<T>::idx(i, j - 1, g.n1d), T(-1) * inv_h2);
            Kt.emplace_back(p, Grid<T>::idx(i, j + 1, g.n1d), T(-1) * inv_h2);

            // no source term (g=0) here; boundary effects already in rhs via Dirichlet rows.
        }
    }

    K.resize(g.np, g.np);
    M_lump.resize(g.np, g.np);
    K.setFromTriplets(Kt.begin(), Kt.end());
    M_lump.setFromTriplets(Mt.begin(), Mt.end());
    K.makeCompressed();
    M_lump.makeCompressed();
}

/**
 * Assemble convection–diffusion operator:
 *   D = eps * K + N_upwind
 *
 * Here K is same Laplacian stiffness as above (Dirichlet handled similarly),
 * N_upwind approximates beta·grad with simple first-order upwind on a uniform grid.
 *
 * beta = (beta_x, beta_y) constant (default matches "flow to NE").
 */
template <typename T>
static inline void velocity_field_w(T x1, T x2, T& wx, T& wy) {
    // w = [2 x2 (1 - x1^2), -2 x1 (1 - x2^2)]^T
    wx = T(2) * x2 * (T(1) - x1 * x1);
    wy = T(-2) * x1 * (T(1) - x2 * x2);
}

template <typename T>
static void assemble_convdiff_fd_lumped_mass(
    const Grid<T>& g,
    Eigen::SparseMatrix<T>& D,
    Eigen::SparseMatrix<T>& M_lump,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
    T bc_value,
    T eps
) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Trip  = Eigen::Triplet<T>;
    std::vector<Trip> Dt;
    std::vector<Trip> Mt;
    Dt.reserve(std::size_t(g.np) * 9);
    Mt.reserve(std::size_t(g.np));

    rhs = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(g.np);

    const T inv_h2 = T(1) / (g.h * g.h);
    const T inv_h  = T(1) / g.h;
    const T area   = g.h * g.h;

    for (int j = 0; j < g.n1d; ++j) {
        for (int i = 0; i < g.n1d; ++i) {
            const int p = Grid<T>::idx(i, j, g.n1d);

            // Lumped mass diagonal (0 on boundary to mimic MATLAB's R(bound)=0 behaviour)
            if (is_boundary_node<T>(i, j, g.n1d)) Mt.emplace_back(p, p, T(0));
            else                                   Mt.emplace_back(p, p, area);

            // Dirichlet boundary row: y_p = bc_value
            if (is_boundary_node<T>(i, j, g.n1d)) {
                Dt.emplace_back(p, p, T(1));
                rhs(p) = bc_value;
                continue;
            }

            // -----------------------
            // Diffusion: eps * Laplacian
            // -----------------------
            Dt.emplace_back(p, p, eps * (T(4) * inv_h2));
            Dt.emplace_back(p, Grid<T>::idx(i - 1, j, g.n1d), eps * (T(-1) * inv_h2));
            Dt.emplace_back(p, Grid<T>::idx(i + 1, j, g.n1d), eps * (T(-1) * inv_h2));
            Dt.emplace_back(p, Grid<T>::idx(i, j - 1, g.n1d), eps * (T(-1) * inv_h2));
            Dt.emplace_back(p, Grid<T>::idx(i, j + 1, g.n1d), eps * (T(-1) * inv_h2));

            // -----------------------
            // Convection: w(x) · grad y, first-order upwind at node (i,j)
            // -----------------------
            const T x1 = T(i) * g.h;
            const T x2 = T(j) * g.h;
            T wx, wy;
            velocity_field_w<T>(x1, x2, wx, wy);

            const T wx_p = std::max(wx, T(0));
            const T wx_m = std::max(-wx, T(0));
            const T wy_p = std::max(wy, T(0));
            const T wy_m = std::max(-wy, T(0));

            // Upwind stencil contributions:
            // wx>0:  wx*(y_p - y_W)/h  -> +wx/h * y_p  -wx/h * y_W
            // wx<0:  wx*(y_E - y_p)/h  -> -|wx|/h*y_p +|wx|/h*y_E
            // same in y-direction.
            Dt.emplace_back(p, p, (wx_p + wx_m + wy_p + wy_m) * inv_h);
            Dt.emplace_back(p, Grid<T>::idx(i - 1, j, g.n1d), (-wx_p) * inv_h);
            Dt.emplace_back(p, Grid<T>::idx(i + 1, j, g.n1d), (+wx_m) * inv_h);
            Dt.emplace_back(p, Grid<T>::idx(i, j - 1, g.n1d), (-wy_p) * inv_h);
            Dt.emplace_back(p, Grid<T>::idx(i, j + 1, g.n1d), (+wy_m) * inv_h);
        }
    }

    D.resize(g.np, g.np);
    M_lump.resize(g.np, g.np);
    D.setFromTriplets(Dt.begin(), Dt.end());
    M_lump.setFromTriplets(Mt.begin(), Mt.end());
    D.makeCompressed();
    M_lump.makeCompressed();
}

/**
 * Build the QP using split control:
 *   x = [ y ; u+ ; u- ]   (size 3*np)
 *   w = u = u+ - u-       (size np)
 *
 * Objective:
 *   0.5 (y - yhat)^T M (y - yhat) + 0.5 * alpha2 * u^T M u + 0.5 * alpha1 * sum_i R_i (u_i^+ + u_i^-)
 *
 * Constraints:
 *   A x = b   : PDE  (D_op y - M (u+ - u-) = rhs)
 *   B x = w   : w = u+ - u-
 *   bounds:
 *     y free
 *     u+, u- >= 0
 *     lw <= w <= uw (control bounds)
 */
template <typename T>
static PDPMMdata<T> make_problem_from_mats(
    const Eigen::SparseMatrix<T>& D_op,
    const Eigen::SparseMatrix<T>& M_lump,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& yhat,
    T alpha1,
    T alpha2,
    T u_lower,
    T u_upper,
    T y_lower = -std::numeric_limits<T>::infinity(),
    T y_upper = std::numeric_limits<T>::infinity()
) {
    PDPMMdata<T> pb;
    using SpMat = typename Problem<T>::SpMat;
    using Vec   = typename Problem<T>::Vec;
    using Trip  = Eigen::Triplet<T>;

    const int np = static_cast<int>(rhs.size());
    const int nx = 3 * np; // [y; u+; u-]
    const int nw = np;     // w = u

    // --- Problem dimension
    pb.n = 3 * np;
    pb.m = np;
    pb.l = np;


    // --- Build R (lumped L1 weights) as row-sum of M_lump, but it's diagonal so just diag
    Vec R(np);
    R.setZero();
    // M_lump is diagonal here; extract diag efficiently
    for (int k = 0; k < M_lump.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M_lump, k); it; ++it) {
            if (it.row() == it.col()) R(it.row()) = it.value();
        }
    }

    // --- c vector
    // Tracking: 0.5 y^T M y - (M yhat)^T y + const
    Vec Myhat = M_lump * yhat;

    pb.obj_const = T(0.5) * yhat.dot(Myhat);

    pb.c.resize(nx);
    pb.c.setZero();
    pb.c.segment(0, np) = -Myhat;
    // L1 term becomes linear on u+ and u- with weights 0.5*alpha1*R
    pb.c.segment(np, np)     = (alpha1 / T(2)) * R;
    pb.c.segment(2*np, np)   = (alpha1 / T(2)) * R;

    // --- Q matrix (symmetric PSD)
    // Q = [[ M,     0,         0    ],
    //      [ 0,  alpha2 M, -alpha2 M],
    //      [ 0, -alpha2 M,  alpha2 M]]
    std::vector<Trip> Qt;
    Qt.reserve(std::size_t(1) * (M_lump.nonZeros() * 5));

    // Insert M in y block
    for (int k = 0; k < M_lump.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M_lump, k); it; ++it) {
            const int i = it.row();
            const int j = it.col();
            const T v   = it.value();
            // y-y block
            Qt.emplace_back(i, j, v);
            // u+-u+ block
            Qt.emplace_back(np + i, np + j, alpha2 * v);
            // u--u- block
            Qt.emplace_back(2*np + i, 2*np + j, alpha2 * v);
            // cross u+ u- (negative)
            Qt.emplace_back(np + i, 2*np + j, -alpha2 * v);
            Qt.emplace_back(2*np + i, np + j, -alpha2 * v);
        }
    }

    pb.Q.resize(nx, nx);
    pb.Q.setFromTriplets(Qt.begin(), Qt.end());
    pb.Q.makeCompressed();

    // --- A x = b : PDE in terms of (y,u+,u-)
    // A = [ D_op , -M_lump , +M_lump ], b = rhs
    std::vector<Trip> At;
    At.reserve(D_op.nonZeros() + 2 * M_lump.nonZeros());

    // D_op block on y
    for (int k = 0; k < D_op.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(D_op, k); it; ++it) {
            At.emplace_back(it.row(), it.col(), it.value());
        }
    }
    // -M on u+, +M on u-
    for (int k = 0; k < M_lump.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M_lump, k); it; ++it) {
            const int r = it.row();
            const int c = it.col();
            const T v   = it.value();
            At.emplace_back(r, np + c, -v);
            At.emplace_back(r, 2*np + c, +v);
        }
    }

    pb.A.resize(np, nx);
    pb.A.setFromTriplets(At.begin(), At.end());
    pb.A.makeCompressed();

    pb.b = rhs;

    // --- B x = w : w = u+ - u-
    // B = [ 0 , I , -I ]
    std::vector<Trip> Bt;
    Bt.reserve(2 * np);
    for (int i = 0; i < np; ++i) {
        Bt.emplace_back(i, np + i, T(1));
        Bt.emplace_back(i, 2*np + i, T(-1));
    }
    pb.B.resize(nw, nx);
    pb.B.setFromTriplets(Bt.begin(), Bt.end());
    pb.B.makeCompressed();

    // --- bounds on x: [y; u+; u-]
    pb.lx.resize(nx);
    pb.ux.resize(nx);
    pb.lx.setConstant(-std::numeric_limits<T>::infinity());
    pb.ux.setConstant(std::numeric_limits<T>::infinity());

    // state bounds (default free)
    pb.lx.segment(0, np).setConstant(y_lower);
    pb.ux.segment(0, np).setConstant(y_upper);

    // u+ >= 0, u- >= 0
    pb.lx.segment(np, np).setConstant(T(0));
    pb.lx.segment(2*np, np).setConstant(T(0));
    // ux stays +inf

    // --- bounds on w = u
    pb.lw.resize(nw);
    pb.uw.resize(nw);
    pb.lw.setConstant(u_lower);
    pb.uw.setConstant(u_upper);

    return pb;
}

// ---------------------------
// Problem generators
// ---------------------------

template <typename T>
PDPMMdata<T> make_poisson_L1L2_control(
    int nc, T alpha1, T alpha2,
    T u_lower = T(-2), T u_upper = T(1.5))
{
    Grid<T> g(nc);

    Eigen::SparseMatrix<T> K, M;
    Eigen::Matrix<T, Eigen::Dynamic, 1> rhs;
    assemble_poisson_fd_lumped_mass(g, K, M, rhs, /*bc=*/T(1));

    Eigen::Matrix<T, Eigen::Dynamic, 1> yhat(g.np);
    for (int j = 0; j < g.n1d; ++j)
        for (int i = 0; i < g.n1d; ++i) {
            const int p = Grid<T>::idx(i, j, g.n1d);
            yhat(p) = std::sin(T(M_PI) * T(i) * g.h)
                    * std::sin(T(M_PI) * T(j) * g.h);
        }

    return make_problem_from_mats<T>(K, M, rhs, yhat, alpha1, alpha2, u_lower, u_upper);
}

template <typename T>
PDPMMdata<T> make_poisson_L1L2_control_default() {
    return make_poisson_L1L2_control<T>(7, T(1e-4), T(1e-2));
}

template <typename T>
PDPMMdata<T> make_convdiff_L1L2_control(
    int nc, T alpha1, T alpha2,
    T u_lower = T(-2), T u_upper = T(1.5), T eps = T(0.02))
{
    Grid<T> g(nc);

    Eigen::SparseMatrix<T> D, M;
    Eigen::Matrix<T, Eigen::Dynamic, 1> rhs;
    assemble_convdiff_fd_lumped_mass(g, D, M, rhs, /*bc=*/T(0), eps);

    Eigen::Matrix<T, Eigen::Dynamic, 1> yhat(g.np);
    for (int j = 0; j < g.n1d; ++j)
        for (int i = 0; i < g.n1d; ++i) {
            const int p = Grid<T>::idx(i, j, g.n1d);
            const T dx = T(i) * g.h - T(0.5);
            const T dy = T(j) * g.h - T(0.5);
            yhat(p) = std::exp(T(-64) * (dx*dx + dy*dy));
        }

    return make_problem_from_mats<T>(D, M, rhs, yhat, alpha1, alpha2, u_lower, u_upper);
}

template <typename T>
PDPMMdata<T> make_convdiff_L1L2_control_default() {
    return make_convdiff_L1L2_control<T>(9, T(1e-4), T(1e-2));
}

} // namespace pdegen