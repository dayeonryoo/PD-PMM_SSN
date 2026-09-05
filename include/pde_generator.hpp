#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>
#include <limits>

#include "fem_q1.hpp"
#include "problem.hpp"

/*-----------------------------------------------------------------------
Q1 bilinear finite-element assembly and PDE-constrained QP generation.

The global assembly and boundary-condition routines here are C++
translations of the corresponding IFISS MATLAB functions:
  femq1_diff.m        - IFISS function: DJS; 4 March 2005
  femq1_cd.m          - IFISS function: DJS; 5 March 2005
Copyright (c) 2005 D.J. Silvester, H.C. Elman, A. Ramage

  nonzerobc_input.m  - IFISS function: DJS, JWP; 27 June 2012
Copyright (c) 2012 D.J. Silvester, H.C. Elman, A. Ramage, J.W. Pearson

See also fem_q1.hpp for the element-level kernels (shape/deriv/gauss_*)
and its own IFISS citations.
-----------------------------------------------------------------------*/

namespace pdegen {

// Selects the spatial discretization used to build the PDE operator D_op
// and mass matrix M in the QP generators below. FEM is the default; FD
// uses a standard 5-point Laplacian stiffness with first-order upwind
// convection on the same uniform GridQ1 node layout, so both share
// Dirichlet BC handling (apply_dirichlet_bc / apply_dirichlet_bc_mass)
// and only ever produce a lumped mass matrix.
enum class Discretization { FEM, FD };

template <typename T>
static inline bool is_boundary_node(int i, int j, int n1d) {
    return (i == 0 || j == 0 || i == n1d - 1 || j == n1d - 1);
}

// Composite trapezoidal-rule weight along one axis: edge nodes get half the
// interior weight so that the resulting tensor-product nodal mass sums
// exactly to the domain area (used by the FD lumped mass below).
template <typename T>
static inline T fd_trapezoid_factor(int idx, int n1d) {
    return (idx == 0 || idx == n1d - 1) ? T(0.5) : T(1);
}

/*-----------------------------------------------------------------------
Q1 finite-element mesh on a uniform rectangular tensor-product grid over
the unit square. Element connectivity is generated directly from (ei,ej)
since we assume the mesh is always structured.
-----------------------------------------------------------------------*/
template <typename T>
struct GridQ1 {
    int nc;
    int n1d;    // nodes per direction
    int np;     // total nodes
    int nel1d;  // elements per direction
    int nel;    // total elements
    std::vector<T> x1d; // n1d physical node coordinates along one axis (shared by both axes)

    static int idx(int i, int j, int n1d_in) { return i + j * n1d_in; }

    explicit GridQ1(int nc_in)
        : nc(nc_in),
          n1d((1 << nc_in) + 1),
          np(n1d * n1d),
          nel1d(n1d - 1),
          nel(nel1d * nel1d),
          x1d(fem::uniform_1d_coords<T>(n1d)) {}

    // Global node ids of element (ei,ej)'s 4 vertices, CCW order matching
    // fem::shape's convention: (ei,ej), (ei+1,ej), (ei+1,ej+1), (ei,ej+1).
    std::array<int, 4> element_nodes(int ei, int ej) const {
        return { idx(ei, ej, n1d), idx(ei + 1, ej, n1d),
                 idx(ei + 1, ej + 1, n1d), idx(ei, ej + 1, n1d) };
    }
};

/*-----------------------------------------------------------------------
Q1 diffusion assembly: stiffness A_stiff, consistent mass M_cons, and
source load f_rhs. Translation of femq1_diff.m.

  A_stiff_{ij} = \int_Ω  ∇(phi_i) ⋅ ∇(phi_j)  dΩ
  M_cons_{ij}  = \int_Ω  phi_i * phi_j        dΩ
  f_rhs_i      = \int_Ω  phi_i * source(x)    dΩ

where {phi_i} are the Q1 bilinear nodal basis functions on the element,
evaluated at 2x2 Gauss points (dv.phi, dv.dphidx, dv.dphidy below).
-----------------------------------------------------------------------*/
template <typename T>
struct FemQ1DiffResult {
    Eigen::SparseMatrix<T> A_stiff;
    Eigen::SparseMatrix<T> M_cons;
    Eigen::SparseMatrix<T> M_lump;
    Eigen::Matrix<T, Eigen::Dynamic, 1> f_rhs;
};

template <typename T>
FemQ1DiffResult<T> assemble_femq1_diff(const GridQ1<T>& g) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec   = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Trip  = Eigen::Triplet<T>;

    std::vector<Trip> At, Mt, Mlt;
    At.reserve(std::size_t(g.nel) * 16);
    Mt.reserve(std::size_t(g.nel) * 16);
    Mlt.reserve(std::size_t(g.nel) * 4);
    Vec f_rhs = Vec::Zero(g.np);

    const auto gpts = fem::gauss_2x2<T>();

    for (int ej = 0; ej < g.nel1d; ++ej) {
        for (int ei = 0; ei < g.nel1d; ++ei) {
            const auto nodes = g.element_nodes(ei, ej);
            const T x0 = g.x1d[ei],   x1 = g.x1d[ei + 1];
            const T y0 = g.x1d[ej],   y1 = g.x1d[ej + 1];
            const T xl[4] = { x0, x1, x1, x0 };
            const T yl[4] = { y0, y0, y1, y1 };

            T ae[4][4] = {};
            T me[4][4] = {};
            T me_lump[4] = {};
            T fe[4]      = {};

            for (const auto& gp : gpts) {
                const auto dv = fem::deriv<T>(gp.s, gp.t, xl, yl);
                const T scale = dv.jac * gp.wt;
                const T src   = fem::gauss_source<T>(dv.phi, xl, yl);

                for (int jl = 0; jl < 4; ++jl) {
                    for (int il = 0; il < 4; ++il) {
                        ae[il][jl] += (dv.dphidx[il] * dv.dphidx[jl] +
                                       dv.dphidy[il] * dv.dphidy[jl]) * scale;
                        me[il][jl] += dv.phi[il] * dv.phi[jl] * scale;
                    }
                    fe[jl] += src * dv.phi[jl] * scale;
                }
                for (int il = 0; il < 4; ++il) {
                    me_lump[il] += dv.phi[il] * scale;
                }
            }

            for (int il = 0; il < 4; ++il) {
                f_rhs(nodes[il]) += fe[il];
                Mlt.emplace_back(nodes[il], nodes[il], me_lump[il]);
                for (int jl = 0; jl < 4; ++jl) {
                    At.emplace_back(nodes[il], nodes[jl], ae[il][jl]);
                    Mt.emplace_back(nodes[il], nodes[jl], me[il][jl]);
                }
            }
        }
    }

    FemQ1DiffResult<T> res;
    res.A_stiff.resize(g.np, g.np);
    res.M_cons.resize(g.np, g.np);
    res.M_lump.resize(g.np, g.np);

    res.A_stiff.setFromTriplets(At.begin(), At.end());
    res.M_cons.setFromTriplets(Mt.begin(), Mt.end());
    res.M_lump.setFromTriplets(Mlt.begin(), Mlt.end());

    res.A_stiff.makeCompressed();
    res.M_cons.makeCompressed();
    res.M_lump.makeCompressed();

    res.f_rhs = f_rhs;
    return res;
}

/*---------------------------------------------------------------------
Q1 convection-diffusion assembly: adds convection matrix N_conv to the
diffusion assembly above. Translation of femq1_cd.m. Element Peclet
number / SUPG scaling diagnostics are omitted.

  A_stiff_{ij} = \int_Ω  ∇(phi_i) ⋅ ∇(phi_j)          dΩ
  M_cons_{ij}  = \int_Ω  phi_i * phi_j                dΩ
  N_conv_{ij}  = \int_Ω  phi_i * ( w(x) ⋅ ∇(phi_j) )  dΩ

where w(x) = (wx, wy) is the wind/transport field sampled at each Gauss
point via fem::gauss_transprt. The resulting convection-diffusion
operator is D_op = eps * A_stiff + N_conv.
-------------------------------------------------------------------------*/
template <typename T>
struct FemQ1CdResult {
    Eigen::SparseMatrix<T> A_stiff;
    Eigen::SparseMatrix<T> N_conv;
    Eigen::SparseMatrix<T> M_cons;
    Eigen::SparseMatrix<T> M_lump;
    Eigen::Matrix<T, Eigen::Dynamic, 1> f_rhs;
};

template <typename T, typename WindFn = void (*)(T, T, T&, T&)>
FemQ1CdResult<T> assemble_femq1_cd(const GridQ1<T>& g,
                                    WindFn wind = fem::velocity_field_w_circular<T>) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec   = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Trip  = Eigen::Triplet<T>;

    std::vector<Trip> At, Nt, Mt, Mlt;
    At.reserve(std::size_t(g.nel) * 16);
    Nt.reserve(std::size_t(g.nel) * 16);
    Mt.reserve(std::size_t(g.nel) * 16);
    Mlt.reserve(std::size_t(g.nel) * 4);
    Vec f_rhs = Vec::Zero(g.np);

    const auto gpts = fem::gauss_2x2<T>();

    for (int ej = 0; ej < g.nel1d; ++ej) {
        for (int ei = 0; ei < g.nel1d; ++ei) {
            const auto nodes = g.element_nodes(ei, ej);
            const T x0 = g.x1d[ei],   x1 = g.x1d[ei + 1];
            const T y0 = g.x1d[ej],   y1 = g.x1d[ej + 1];
            const T xl[4] = { x0, x1, x1, x0 };
            const T yl[4] = { y0, y0, y1, y1 };

            T ae[4][4] = {};
            T ne[4][4] = {};
            T me[4][4] = {};
            T me_lump[4] = {};
            T fe[4]      = {};

            for (const auto& gp : gpts) {
                const auto dv = fem::deriv<T>(gp.s, gp.t, xl, yl);
                const T scale = dv.jac * gp.wt;
                const T src   = fem::gauss_source<T>(dv.phi, xl, yl);

                T wx, wy;
                fem::gauss_transprt<T>(dv.phi, xl, yl, wx, wy, wind);

                for (int jl = 0; jl < 4; ++jl) {
                    for (int il = 0; il < 4; ++il) {
                        ae[il][jl] += (dv.dphidx[il] * dv.dphidx[jl] +
                                       dv.dphidy[il] * dv.dphidy[jl]) * scale;
                        me[il][jl] += dv.phi[il] * dv.phi[jl] * scale;
                        ne[il][jl] += (wx * dv.phi[il] * dv.dphidx[jl] +
                                       wy * dv.phi[il] * dv.dphidy[jl]) * scale;
                    }
                    fe[jl] += src * dv.phi[jl] * scale;
                }
                for (int il = 0; il < 4; ++il) {
                    me_lump[il] += dv.phi[il] * scale;
                }
            }

            for (int il = 0; il < 4; ++il) {
                f_rhs(nodes[il]) += fe[il];
                Mlt.emplace_back(nodes[il], nodes[il], me_lump[il]);
                for (int jl = 0; jl < 4; ++jl) {
                    At.emplace_back(nodes[il], nodes[jl], ae[il][jl]);
                    Mt.emplace_back(nodes[il], nodes[jl], me[il][jl]);
                    Nt.emplace_back(nodes[il], nodes[jl], ne[il][jl]);
                }
            }
        }
    }

    FemQ1CdResult<T> res;
    res.A_stiff.resize(g.np, g.np);
    res.N_conv.resize(g.np, g.np);
    res.M_cons.resize(g.np, g.np);
    res.M_lump.resize(g.np, g.np);

    res.A_stiff.setFromTriplets(At.begin(), At.end());
    res.N_conv.setFromTriplets(Nt.begin(), Nt.end());
    res.M_cons.setFromTriplets(Mt.begin(), Mt.end());
    res.M_lump.setFromTriplets(Mlt.begin(), Mlt.end());

    res.A_stiff.makeCompressed();
    res.N_conv.makeCompressed();
    res.M_cons.makeCompressed();
    res.M_lump.makeCompressed();

    res.f_rhs = f_rhs;
    return res;
}

/*-----------------------------------------------------------------------
FD diffusion assembly: standard 5-point Laplacian stiffness A_stiff and
diagonal lumped mass M_lump (composite-trapezoidal area weight per node,
so it sums exactly to the domain area, same invariant as the FEM lumped
mass) on the same uniform GridQ1 node layout used by the FEM path.
Boundary rows of A_stiff are left empty here since apply_dirichlet_bc
fills them in (diagonal = 1) during the shared post-assembly BC step
below.
-----------------------------------------------------------------------*/
template <typename T>
struct FdDiffResult {
    Eigen::SparseMatrix<T> A_stiff;
    Eigen::SparseMatrix<T> M_lump;
    Eigen::Matrix<T, Eigen::Dynamic, 1> f_rhs;
};

template <typename T>
FdDiffResult<T> assemble_fd_diff(const GridQ1<T>& g) {
    using Vec  = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Trip = Eigen::Triplet<T>;

    const T h      = g.x1d[1] - g.x1d[0];
    const T inv_h2 = T(1) / (h * h);
    const T area   = h * h;

    std::vector<Trip> At, Mlt;
    At.reserve(std::size_t(g.np) * 5);
    Mlt.reserve(std::size_t(g.np));

    for (int j = 0; j < g.n1d; ++j) {
        for (int i = 0; i < g.n1d; ++i) {
            const int p = GridQ1<T>::idx(i, j, g.n1d);
            const T node_weight = area * fd_trapezoid_factor<T>(i, g.n1d) * fd_trapezoid_factor<T>(j, g.n1d);
            Mlt.emplace_back(p, p, node_weight);

            if (is_boundary_node<T>(i, j, g.n1d)) continue;

            At.emplace_back(p, p, T(4) * inv_h2);
            At.emplace_back(p, GridQ1<T>::idx(i - 1, j, g.n1d), -inv_h2);
            At.emplace_back(p, GridQ1<T>::idx(i + 1, j, g.n1d), -inv_h2);
            At.emplace_back(p, GridQ1<T>::idx(i, j - 1, g.n1d), -inv_h2);
            At.emplace_back(p, GridQ1<T>::idx(i, j + 1, g.n1d), -inv_h2);
        }
    }

    FdDiffResult<T> res;
    res.A_stiff.resize(g.np, g.np);
    res.M_lump.resize(g.np, g.np);
    res.A_stiff.setFromTriplets(At.begin(), At.end());
    res.M_lump.setFromTriplets(Mlt.begin(), Mlt.end());
    res.A_stiff.makeCompressed();
    res.M_lump.makeCompressed();
    res.f_rhs = Vec::Zero(g.np);
    return res;
}

/*-----------------------------------------------------------------------
FD convection-diffusion assembly: adds a first-order upwind convection
operator N_conv to the FD diffusion assembly above, wind sampled directly
at each grid node. The resulting operator is D_op = eps * A_stiff + N_conv,
matching the FEM composition in assemble_femq1_cd.
-----------------------------------------------------------------------*/
template <typename T>
struct FdCdResult {
    Eigen::SparseMatrix<T> A_stiff;
    Eigen::SparseMatrix<T> N_conv;
    Eigen::SparseMatrix<T> M_lump;
    Eigen::Matrix<T, Eigen::Dynamic, 1> f_rhs;
};

template <typename T, typename WindFn = void (*)(T, T, T&, T&)>
FdCdResult<T> assemble_fd_cd(const GridQ1<T>& g,
                              WindFn wind = fem::velocity_field_w_circular<T>) {
    using Vec  = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Trip = Eigen::Triplet<T>;

    const T h      = g.x1d[1] - g.x1d[0];
    const T inv_h2 = T(1) / (h * h);
    const T inv_h  = T(1) / h;
    const T area   = h * h;

    std::vector<Trip> At, Nt, Mlt;
    At.reserve(std::size_t(g.np) * 5);
    Nt.reserve(std::size_t(g.np) * 5);
    Mlt.reserve(std::size_t(g.np));

    for (int j = 0; j < g.n1d; ++j) {
        for (int i = 0; i < g.n1d; ++i) {
            const int p = GridQ1<T>::idx(i, j, g.n1d);
            const T node_weight = area * fd_trapezoid_factor<T>(i, g.n1d) * fd_trapezoid_factor<T>(j, g.n1d);
            Mlt.emplace_back(p, p, node_weight);

            if (is_boundary_node<T>(i, j, g.n1d)) continue;

            At.emplace_back(p, p, T(4) * inv_h2);
            At.emplace_back(p, GridQ1<T>::idx(i - 1, j, g.n1d), -inv_h2);
            At.emplace_back(p, GridQ1<T>::idx(i + 1, j, g.n1d), -inv_h2);
            At.emplace_back(p, GridQ1<T>::idx(i, j - 1, g.n1d), -inv_h2);
            At.emplace_back(p, GridQ1<T>::idx(i, j + 1, g.n1d), -inv_h2);

            // Upwind: w * dy/dx |_i ~ wx_p*(y_i - y_{i-1})/h - wx_m*(y_{i+1} - y_i)/h,
            // and likewise in y; so both neighbor coefficients carry a minus sign.
            T wx, wy;
            wind(g.x1d[i], g.x1d[j], wx, wy);
            const T wx_p = std::max(wx, T(0)), wx_m = std::max(-wx, T(0));
            const T wy_p = std::max(wy, T(0)), wy_m = std::max(-wy, T(0));

            Nt.emplace_back(p, p, (wx_p + wx_m + wy_p + wy_m) * inv_h);
            Nt.emplace_back(p, GridQ1<T>::idx(i - 1, j, g.n1d), -wx_p * inv_h);
            Nt.emplace_back(p, GridQ1<T>::idx(i + 1, j, g.n1d), -wx_m * inv_h);
            Nt.emplace_back(p, GridQ1<T>::idx(i, j - 1, g.n1d), -wy_p * inv_h);
            Nt.emplace_back(p, GridQ1<T>::idx(i, j + 1, g.n1d), -wy_m * inv_h);
        }
    }

    FdCdResult<T> res;
    res.A_stiff.resize(g.np, g.np);
    res.N_conv.resize(g.np, g.np);
    res.M_lump.resize(g.np, g.np);
    res.A_stiff.setFromTriplets(At.begin(), At.end());
    res.N_conv.setFromTriplets(Nt.begin(), Nt.end());
    res.M_lump.setFromTriplets(Mlt.begin(), Mlt.end());
    res.A_stiff.makeCompressed();
    res.N_conv.makeCompressed();
    res.M_lump.makeCompressed();
    res.f_rhs = Vec::Zero(g.np);
    return res;
}

/*-----------------------------------------------------------------------
Dispatches diffusion / convection-diffusion assembly to FEM or FD based
on `disc`, so the QP generators below only branch once per operator. FD
always uses its lumped mass (there is no FD analogue of the consistent
Q1 mass matrix), so `lump_mass` only affects the FEM path.

D_op feeds into the shared PDE constraint D_op*y - M*u = rhs (see
make_problem_l2_from_mats / make_problem_l1l2_from_mats), which encodes
the FEM weak form K*y = M*u. FD's strong-form Laplacian/convection
assembly instead represents the pointwise equation D_op*y = u (no mass
weighting on u), and FD's stiffness is O(1/h^2) rather than FEM's
O(1) — so passing FD's raw D_op through the same M*u convention would
silently divide the control's influence on the state by an extra O(h^2)
per stage. To reuse the shared constraint assembly unchanged, the FD
operator is mass-scaled here (M_lump * D_op_raw), which is algebraically
equivalent to the strong-form equation (mass is diagonal/invertible) and
also renormalizes FD's stiffness down to FEM's O(1) magnitude.
-----------------------------------------------------------------------*/
template <typename T>
static void assemble_diff_by_discretization(const GridQ1<T>& g, Discretization disc, bool lump_mass,
                                              Eigen::SparseMatrix<T>& D, Eigen::SparseMatrix<T>& M) {
    if (disc == Discretization::FD) {
        auto asm_res = assemble_fd_diff<T>(g);
        D = asm_res.M_lump * asm_res.A_stiff;
        M = asm_res.M_lump;
    } else {
        auto asm_res = assemble_femq1_diff<T>(g);
        D = asm_res.A_stiff;
        M = lump_mass ? asm_res.M_lump : asm_res.M_cons;
    }
}

template <typename T, typename WindFn = void (*)(T, T, T&, T&)>
static void assemble_cd_by_discretization(const GridQ1<T>& g, Discretization disc, bool lump_mass, T eps,
                                           WindFn wind,
                                           Eigen::SparseMatrix<T>& D, Eigen::SparseMatrix<T>& M) {
    if (disc == Discretization::FD) {
        auto asm_res = assemble_fd_cd<T>(g, wind);
        D = asm_res.M_lump * (eps * asm_res.A_stiff + asm_res.N_conv);
        M = asm_res.M_lump;
    } else {
        auto asm_res = assemble_femq1_cd<T>(g, wind);
        D = eps * asm_res.A_stiff + asm_res.N_conv;
        M = lump_mass ? asm_res.M_lump : asm_res.M_cons;
    }
}

/*-----------------------------------------------------------------------
Dirichlet boundary conditions via row/col elimination on the fully
assembled global operator. Translation of nonzerobc_input.m. Applied as
a separate post-assembly step.

For boundary nodes p with prescribed value g_p = bc_values[p]:
  rhs_r  <-  rhs_r - sum_{p in bc_nodes} D_op(r,p) * g_p    for r not in bc_nodes
  D_op(p, :) = D_op(:, p) = 0,  D_op(p, p) = 1              for p in bc_nodes
  rhs_p  <-  g_p                                            for p in bc_nodes

i.e. known boundary columns are folded into the RHS of the interior
equations, then boundary rows/cols are replaced by identity rows so that
solving D_op y = rhs directly yields y_p = g_p at the boundary.
-----------------------------------------------------------------------*/
template <typename T>
void apply_dirichlet_bc(Eigen::SparseMatrix<T>& D_op,
                         Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
                         const std::vector<int>& bc_nodes,
                         const Eigen::Matrix<T, Eigen::Dynamic, 1>& bc_values) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec   = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Trip  = Eigen::Triplet<T>;

    const int np = static_cast<int>(D_op.rows());
    std::vector<char> is_bc(np, 0);
    Vec bc_value_at = Vec::Zero(np);
    for (std::size_t k = 0; k < bc_nodes.size(); ++k) {
        is_bc[bc_nodes[k]] = 1;
        bc_value_at(bc_nodes[k]) = bc_values(static_cast<int>(k));
    }

    // fold known boundary columns into the RHS before elimination
    for (int k = 0; k < D_op.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(D_op, k); it; ++it) {
            const int r = it.row();
            const int c = it.col();
            if (!is_bc[r] && is_bc[c]) {
                rhs(r) -= it.value() * bc_value_at(c);
            }
        }
    }

    // zero boundary rows/cols, keep interior entries, set diagonal = 1
    std::vector<Trip> Dt;
    Dt.reserve(std::size_t(D_op.nonZeros()) + bc_nodes.size());
    for (int k = 0; k < D_op.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(D_op, k); it; ++it) {
            const int r = it.row();
            const int c = it.col();
            if (!is_bc[r] && !is_bc[c]) Dt.emplace_back(r, c, it.value());
        }
    }
    for (int p : bc_nodes) Dt.emplace_back(p, p, T(1));

    D_op.setFromTriplets(Dt.begin(), Dt.end());
    D_op.makeCompressed();

    for (int p : bc_nodes) rhs(p) = bc_value_at(p);
}

// Zeroes boundary rows/cols of a mass matrix (no RHS coupling).
template <typename T>
void apply_dirichlet_bc_mass(Eigen::SparseMatrix<T>& M, const std::vector<int>& bc_nodes) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Trip  = Eigen::Triplet<T>;

    const int np = static_cast<int>(M.rows());
    std::vector<char> is_bc(np, 0);
    for (int p : bc_nodes) is_bc[p] = 1;

    std::vector<Trip> Mt;
    Mt.reserve(std::size_t(M.nonZeros()));
    for (int k = 0; k < M.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M, k); it; ++it) {
            const int r = it.row();
            const int c = it.col();
            if (!is_bc[r] && !is_bc[c]) Mt.emplace_back(r, c, it.value());
        }
    }
    M.setFromTriplets(Mt.begin(), Mt.end());
    M.makeCompressed();
}

// Collects the boundary node ids of a GridQ1.
template <typename T>
std::vector<int> fem_boundary_nodes(const GridQ1<T>& g) {
    std::vector<int> bc_nodes;
    bc_nodes.reserve(std::size_t(4) * g.n1d);
    for (int j = 0; j < g.n1d; ++j)
        for (int i = 0; i < g.n1d; ++i)
            if (is_boundary_node<T>(i, j, g.n1d))
                bc_nodes.push_back(GridQ1<T>::idx(i, j, g.n1d));
    return bc_nodes;
}

/*-----------------------------------------------------------------------
L1/L2-regularized PDE-constrained QP, using split control:
    x = [ y ; u+ ; u- ]   (size 3*np)
    w = u = u+ - u-       (size np)

Objective:
    0.5 (y - yhat)^T M (y - yhat) + 0.5 * alpha2 * u^T M u + 0.5 * alpha1 * sum_i R_i (u+_i + u-_i)

Constraints:
    A x = b   : PDE  (D_op y - M (u+ - u-) = rhs)
    B x = w   : w = u+ - u-
    bounds:
        y free
        u+, u- >= 0
        lw <= w <= uw

(Gondzio, Pougkakiotis & Pearson 2022) 
-----------------------------------------------------------------------*/
template <typename T>
static KSPQPdata<T> make_problem_l1l2_from_mats(
    const Eigen::SparseMatrix<T>& D_op,
    const Eigen::SparseMatrix<T>& M,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& yhat,
    T alpha1,
    T alpha2,
    T u_lower,
    T u_upper,
    T y_lower = -std::numeric_limits<T>::infinity(),
    T y_upper = std::numeric_limits<T>::infinity()
) {
    KSPQPdata<T> pb;
    using SpMat = typename Problem<T>::SpMat;
    using Vec   = typename Problem<T>::Vec;
    using Trip  = Eigen::Triplet<T>;

    const int np = static_cast<int>(rhs.size());
    const int nx = 3 * np; // [y; u+; u-]
    const int nw = np;     // w = u

    // --- Problem dimension ---
    pb.n = 3 * np;
    pb.m = np;
    pb.l = np;


    // --- Build R (lumped L1 weights) as row-sum of M ---
    Vec R(np);
    R.setZero();
    for (int k = 0; k < M.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M, k); it; ++it) {
            R(it.row()) += it.value();
        }
    }

    // --- c vector ---
    // c = 0.5 y^T M y - (M yhat)^T y + const
    Vec Myhat = M * yhat;

    pb.obj_const = T(0.5) * yhat.dot(Myhat);

    pb.c.resize(nx);
    pb.c.setZero();
    pb.c.segment(0, np) = -Myhat;
    // L1 term becomes linear on u+ and u- with weights 0.5*alpha1*R
    pb.c.segment(np, np)     = (alpha1 / T(2)) * R;
    pb.c.segment(2*np, np)   = (alpha1 / T(2)) * R;

    // --- Q matrix (symmetric PSD) ---
    // Q = [[ M,     0,         0    ],
    //      [ 0,  alpha2 M, -alpha2 M],
    //      [ 0, -alpha2 M,  alpha2 M]]
    std::vector<Trip> Qt;
    Qt.reserve(std::size_t(1) * (M.nonZeros() * 5));

    // Insert M in y block
    for (int k = 0; k < M.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M, k); it; ++it) {
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

    // --- A x = b : PDE in terms of (y, u+, u-) ---
    // A = [ D_op , -M , +M ], b = rhs
    std::vector<Trip> At;
    At.reserve(D_op.nonZeros() + 2 * M.nonZeros());

    // D_op block on y
    for (int k = 0; k < D_op.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(D_op, k); it; ++it) {
            At.emplace_back(it.row(), it.col(), it.value());
        }
    }
    // -M on u+, +M on u-
    for (int k = 0; k < M.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M, k); it; ++it) {
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

    // --- B x = w = u+ - u- ---
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

    // --- bounds on x: [y; u+; u-] ---
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

    // --- bounds on w = u ---
    pb.lw.resize(nw);
    pb.uw.resize(nw);
    pb.lw.setConstant(u_lower);
    pb.uw.setConstant(u_upper);

    return pb;
}

/*-----------------------------------------------------------------------
L2-regularized PDE-constrained QP:

  x = [y; u]
  min  0.5 (y - yhat)^T M (y - yhat) + 0.5 * beta * u^T M u
  s.t. D_op y - M u = rhs
       y_lower <= y <= y_upper
       u_lower <= u <= u_upper

(Pearson & Gondzio 2017)

Note: B is an empty 0 x n matrix, i.e. l = 0.
-----------------------------------------------------------------------*/
template <typename T>
static KSPQPdata<T> make_problem_l2_from_mats(
    const Eigen::SparseMatrix<T>& D_op,
    const Eigen::SparseMatrix<T>& M,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& yhat,
    T beta,
    T y_lower = -std::numeric_limits<T>::infinity(),
    T y_upper =  std::numeric_limits<T>::infinity(),
    T u_lower = -std::numeric_limits<T>::infinity(),
    T u_upper =  std::numeric_limits<T>::infinity()
) {
    KSPQPdata<T> pb;
    using SpMat = typename Problem<T>::SpMat;
    using Vec   = typename Problem<T>::Vec;
    using Trip  = Eigen::Triplet<T>;

    const int np = static_cast<int>(rhs.size());
    const int nx = 2 * np; // [y; u]

    pb.n = nx;
    pb.m = np;
    pb.l = 0;

    Vec Myhat = M * yhat;
    pb.obj_const = T(0.5) * yhat.dot(Myhat);

    pb.c.resize(nx);
    pb.c.setZero();
    pb.c.segment(0, np) = -Myhat;
    // c on u is zero.

    // Q = [[M      0   ],
    //      [0, beta * M]]
    std::vector<Trip> Qt;
    Qt.reserve(std::size_t(2) * M.nonZeros());
    for (int k = 0; k < M.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M, k); it; ++it) {
            const int i = it.row();
            const int j = it.col();
            const T v   = it.value();
            Qt.emplace_back(i, j, v);
            Qt.emplace_back(np + i, np + j, beta * v);
        }
    }
    pb.Q.resize(nx, nx);
    pb.Q.setFromTriplets(Qt.begin(), Qt.end());
    pb.Q.makeCompressed();

    // A = [D_op, -M], b = rhs
    std::vector<Trip> At;
    At.reserve(D_op.nonZeros() + M.nonZeros());
    for (int k = 0; k < D_op.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(D_op, k); it; ++it) {
            At.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int k = 0; k < M.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(M, k); it; ++it) {
            At.emplace_back(it.row(), np + it.col(), -it.value());
        }
    }
    pb.A.resize(np, nx);
    pb.A.setFromTriplets(At.begin(), At.end());
    pb.A.makeCompressed();
    pb.b = rhs;

    // No B x = w block.
    pb.B.resize(0, nx);
    pb.B.makeCompressed();
    pb.lw.resize(0);
    pb.uw.resize(0);

    // Bounds on x = [y; u]
    pb.lx.resize(nx);
    pb.ux.resize(nx);
    pb.lx.segment(0, np).setConstant(y_lower);
    pb.ux.segment(0, np).setConstant(y_upper);
    pb.lx.segment(np, np).setConstant(u_lower);
    pb.ux.segment(np, np).setConstant(u_upper);

    return pb;
}


// ===== QP generators =====

/*-----------------------------------------------------------------------
2D Poisson control problem (control-constrained).
    Ω = [0,1]^2, y = 0 on boundary,
    D = -Δ,
    yhat = exp(-64((x1 - 0.5)^2 + (x2 - 0.5)^2)).
-----------------------------------------------------------------------*/
template <typename T>
KSPQPdata<T> make_poisson_l2_control(
    int nc, T beta,
    T y_lower = -std::numeric_limits<T>::infinity(),
    T y_upper =  std::numeric_limits<T>::infinity(),
    T u_lower = -std::numeric_limits<T>::infinity(),
    T u_upper =  std::numeric_limits<T>::infinity(),
    bool lump_mass = false,
    Discretization disc = Discretization::FEM)
{
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    GridQ1<T> g(nc);

    Vec yhat(g.np);
    for (int j = 0; j < g.n1d; ++j)
        for (int i = 0; i < g.n1d; ++i) {
            const int p = GridQ1<T>::idx(i, j, g.n1d);
            const T dx = g.x1d[i] - T(0.5);
            const T dy = g.x1d[j] - T(0.5);
            yhat(p) = std::exp(T(-64) * (dx * dx + dy * dy));
        }

    Eigen::SparseMatrix<T> D, M;
    assemble_diff_by_discretization<T>(g, disc, lump_mass, D, M);
    Vec rhs = Vec::Zero(g.np);

    const std::vector<int> bc_nodes = fem_boundary_nodes<T>(g);
    const Vec bc_values = Vec::Zero(static_cast<int>(bc_nodes.size()));
    apply_dirichlet_bc<T>(D, rhs, bc_nodes, bc_values);
    apply_dirichlet_bc_mass<T>(M, bc_nodes);

    return make_problem_l2_from_mats<T>(D, M, rhs, yhat, beta, y_lower, y_upper, u_lower, u_upper);
}

/*-----------------------------------------------------------------------
2D Poisson control problem (state-constrained).
    Ω = [0,1]^2, y = yhat on boundary,
    D = -Δ,
    yhat = sin(pi x1) sin(pi x2).
-----------------------------------------------------------------------*/
template <typename T>
KSPQPdata<T> make_poisson_l2_state_control(
    int nc, T beta,
    T y_lower = -std::numeric_limits<T>::infinity(),
    T y_upper =  std::numeric_limits<T>::infinity(),
    T u_lower = -std::numeric_limits<T>::infinity(),
    T u_upper =  std::numeric_limits<T>::infinity(),
    bool lump_mass = false,
    Discretization disc = Discretization::FEM)
{
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    GridQ1<T> g(nc);

    Vec yhat(g.np);
    for (int j = 0; j < g.n1d; ++j)
        for (int i = 0; i < g.n1d; ++i) {
            const int p = GridQ1<T>::idx(i, j, g.n1d);
            yhat(p) = std::sin(T(M_PI) * g.x1d[i]) * std::sin(T(M_PI) * g.x1d[j]);
        }

    Eigen::SparseMatrix<T> D, M;
    assemble_diff_by_discretization<T>(g, disc, lump_mass, D, M);
    Vec rhs = Vec::Zero(g.np);

    const std::vector<int> bc_nodes = fem_boundary_nodes<T>(g);
    Vec bc_values(static_cast<int>(bc_nodes.size()));
    for (std::size_t k = 0; k < bc_nodes.size(); ++k)
        bc_values(static_cast<int>(k)) = yhat(bc_nodes[k]);

    apply_dirichlet_bc<T>(D, rhs, bc_nodes, bc_values);
    apply_dirichlet_bc_mass<T>(M, bc_nodes);

    return make_problem_l2_from_mats<T>(D, M, rhs, yhat, beta, y_lower, y_upper, u_lower, u_upper);
}

/*-----------------------------------------------------------------------
2D convection-diffusion control problem.
    Ω = [0,1]^2, y = 0 on boundary,
    D = -eps * Δ + w ⋅ ∇ (constant wind w = [-1/sqrt(2), 1/sqrt(2)]^T),
    yhat = exp(-64((x1 - 0.5)^2 + (x2 - 0.5)^2)).
-----------------------------------------------------------------------*/
template <typename T>
KSPQPdata<T> make_convdiff_l2_control(
    int nc, T beta,
    T y_lower = -std::numeric_limits<T>::infinity(),
    T y_upper =  std::numeric_limits<T>::infinity(),
    T u_lower = -std::numeric_limits<T>::infinity(),
    T u_upper =  std::numeric_limits<T>::infinity(),
    T eps = T(0.01),
    bool lump_mass = false,
    Discretization disc = Discretization::FEM)
{
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    GridQ1<T> g(nc);

    Vec yhat(g.np);
    for (int j = 0; j < g.n1d; ++j)
        for (int i = 0; i < g.n1d; ++i) {
            const int p = GridQ1<T>::idx(i, j, g.n1d);
            const T dx = g.x1d[i] - T(0.5);
            const T dy = g.x1d[j] - T(0.5);
            yhat(p) = std::exp(T(-64) * (dx * dx + dy * dy));
        }

    Eigen::SparseMatrix<T> D, M;
    assemble_cd_by_discretization<T>(g, disc, lump_mass, eps, fem::velocity_field_w_constant<T>, D, M);
    Vec rhs = Vec::Zero(g.np);

    const std::vector<int> bc_nodes = fem_boundary_nodes<T>(g);
    const Vec bc_values = Vec::Zero(static_cast<int>(bc_nodes.size()));
    apply_dirichlet_bc<T>(D, rhs, bc_nodes, bc_values);
    apply_dirichlet_bc_mass<T>(M, bc_nodes);

    return make_problem_l2_from_mats<T>(D, M, rhs, yhat, beta, y_lower, y_upper, u_lower, u_upper);
}

/*-----------------------------------------------------------------------
L1/L2-regularized Poisson-constrained QP.
    Ω = (0,1)^2, y = 1 on boundary,
    D = -Δ,
    yhat = sin(pi x1) sin(pi x2),
    u_lower <= u <= u_upper,
    y_lower <= y <= y_upper (free by default).
-----------------------------------------------------------------------*/
template <typename T>
KSPQPdata<T> make_poisson_l1l2_control(
    int nc, T alpha1, T alpha2,
    T u_lower = T(-2), T u_upper = T(1.5),
    T y_lower = -std::numeric_limits<T>::infinity(),
    T y_upper =  std::numeric_limits<T>::infinity(),
    bool lump_mass = false,
    Discretization disc = Discretization::FEM)
{
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    GridQ1<T> g(nc);

    Vec yhat(g.np);
    for (int j = 0; j < g.n1d; ++j)
        for (int i = 0; i < g.n1d; ++i) {
            const int p = GridQ1<T>::idx(i, j, g.n1d);
            yhat(p) = std::sin(T(M_PI) * g.x1d[i]) * std::sin(T(M_PI) * g.x1d[j]);
        }

    Eigen::SparseMatrix<T> D, M;
    assemble_diff_by_discretization<T>(g, disc, lump_mass, D, M);
    Vec rhs = Vec::Zero(g.np);

    const std::vector<int> bc_nodes = fem_boundary_nodes<T>(g);
    const Vec bc_values = Vec::Constant(static_cast<int>(bc_nodes.size()), T(1));
    apply_dirichlet_bc<T>(D, rhs, bc_nodes, bc_values);
    apply_dirichlet_bc_mass<T>(M, bc_nodes);

    return make_problem_l1l2_from_mats<T>(D, M, rhs, yhat, alpha1, alpha2, u_lower, u_upper,
                                           y_lower, y_upper);
}

/*-----------------------------------------------------------------------
L1/L2-regularized convection-diffusion-constrained QP.
    Ω = (0,1)^2, y = 0 on boundary,
    D = -eps * Δ + w ⋅ ∇ (circular wind w),
    yhat = exp(-64((x1 - 0.5)^2 + (x2 - 0.5)^2)),
    u_lower <= u <= u_upper,
    y_lower <= y <= y_upper (free by default).
-----------------------------------------------------------------------*/
template <typename T>
KSPQPdata<T> make_convdiff_l1l2_control(
    int nc, T alpha1, T alpha2,
    T u_lower = T(-2), T u_upper = T(1.5), T eps = T(0.02),
    T y_lower = -std::numeric_limits<T>::infinity(),
    T y_upper =  std::numeric_limits<T>::infinity(),
    bool lump_mass = false,
    Discretization disc = Discretization::FEM)
{
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    GridQ1<T> g(nc);

    Vec yhat(g.np);
    for (int j = 0; j < g.n1d; ++j)
        for (int i = 0; i < g.n1d; ++i) {
            const int p = GridQ1<T>::idx(i, j, g.n1d);
            const T dx = g.x1d[i] - T(0.5);
            const T dy = g.x1d[j] - T(0.5);
            yhat(p) = std::exp(T(-64) * (dx * dx + dy * dy));
        }

    Eigen::SparseMatrix<T> D, M;
    assemble_cd_by_discretization<T>(g, disc, lump_mass, eps, fem::velocity_field_w_circular<T>, D, M);
    Vec rhs = Vec::Zero(g.np);

    const std::vector<int> bc_nodes = fem_boundary_nodes<T>(g);
    const Vec bc_values = Vec::Zero(static_cast<int>(bc_nodes.size()));
    apply_dirichlet_bc<T>(D, rhs, bc_nodes, bc_values);
    apply_dirichlet_bc_mass<T>(M, bc_nodes);

    return make_problem_l1l2_from_mats<T>(D, M, rhs, yhat, alpha1, alpha2, u_lower, u_upper,
                                           y_lower, y_upper);
}

} // namespace pdegen
