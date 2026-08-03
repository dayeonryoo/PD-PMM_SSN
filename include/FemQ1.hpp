#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// -----------------------------------------------------------------------
// The Q1 element kernels in this file (shape, deriv, gauss_transprt,
// gauss_source, gauss_2x2) are C++ translations of the corresponding IFISS
// MATLAB functions:
//   shape.m           - IFISS function: DJS; 4 March 2005
//   deriv.m           - IFISS function: DJS; 4 March 2005
//   gauss_transprt.m  - IFISS function: DJS; 4 March 2005
//   gauss_source.m    - IFISS function: DJS; 4 March 2005
// Copyright (c) 2005 D.J. Silvester, H.C. Elman, A. Ramage
// -----------------------------------------------------------------------

namespace pdegen {
namespace fem {

// Convection field w = [2 x2 (1 - x1^2), -2 x1 (1 - x2^2)]^T ,
// (circular wind). Used by gauss_transprt below.
template <typename T>
inline void velocity_field_w_circular(T x1, T x2, T& wx, T& wy) {
    wx = T(2) * x2 * (T(1) - x1 * x1);
    wy = T(-2) * x1 * (T(1) - x2 * x2);
}

// Constant convection field w = [-1/sqrt(2), 1/sqrt(2)]^T.
// Used by gauss_transprt below.
template <typename T>
inline void velocity_field_w_constant(T /*x1*/, T /*x2*/, T& wx, T& wy) {
    wx = T(-1) / std::sqrt(T(2));
    wy =  T(1) / std::sqrt(T(2));
}

// -----------------------------------------------------------------------
// Q1 bilinear shape functions at reference point (s,t) in [-1,1]^2.
// Vertex order: (-1,-1), (1,-1), (1,1), (-1,1) (counter-clockwise).
// Translation of shape.m.
// -----------------------------------------------------------------------
template <typename T>
struct ShapeVal {
    T phi[4];
    T dphids[4];
    T dphidt[4];
};

template <typename T>
inline ShapeVal<T> shape(T s, T t) {
    const T one = T(1);
    ShapeVal<T> sv;
    sv.phi[0] =  T(0.25) * (s - one) * (t - one);
    sv.phi[1] = -T(0.25) * (s + one) * (t - one);
    sv.phi[2] =  T(0.25) * (s + one) * (t + one);
    sv.phi[3] = -T(0.25) * (s - one) * (t + one);

    sv.dphids[0] =  T(0.25) * (t - one);
    sv.dphids[1] = -T(0.25) * (t - one);
    sv.dphids[2] =  T(0.25) * (t + one);
    sv.dphids[3] = -T(0.25) * (t + one);

    sv.dphidt[0] =  T(0.25) * (s - one);
    sv.dphidt[1] = -T(0.25) * (s + one);
    sv.dphidt[2] =  T(0.25) * (s + one);
    sv.dphidt[3] = -T(0.25) * (s - one);
    return sv;
}

// -----------------------------------------------------------------------
// Jacobian and physical-space derivatives of the Q1 shape functions for one
// element, at reference point (s,t). xl/yl are the 4 vertex physical
// coordinates in the same order as shape()'s vertex convention.
// Translation of deriv.m, specialised to a single element/point.
//
// dphidx/dphidy here are true, normalised physical derivatives.
// -----------------------------------------------------------------------
template <typename T>
struct DerivVal {
    T phi[4];
    T dphidx[4];
    T dphidy[4];
    T jac;
};

template <typename T>
inline DerivVal<T> deriv(T s, T t, const T xl[4], const T yl[4]) {
    const ShapeVal<T> sv = shape<T>(s, t);

    T dxds = T(0), dxdt = T(0), dyds = T(0), dydt = T(0);
    for (int k = 0; k < 4; ++k) {
        dxds += xl[k] * sv.dphids[k];
        dxdt += xl[k] * sv.dphidt[k];
        dyds += yl[k] * sv.dphids[k];
        dydt += yl[k] * sv.dphidt[k];
    }

    DerivVal<T> dv;
    dv.jac = dxds * dydt - dxdt * dyds;
    if (dv.jac <= T(0)) {
        throw std::runtime_error("fem::deriv: singular or inverted element Jacobian");
    }
    const T invjac = T(1) / dv.jac;
    for (int k = 0; k < 4; ++k) {
        dv.phi[k]    = sv.phi[k];
        dv.dphidx[k] = (sv.dphids[k] * dydt - sv.dphidt[k] * dyds) * invjac;
        dv.dphidy[k] = (-sv.dphids[k] * dxdt + sv.dphidt[k] * dxds) * invjac;
    }
    return dv;
}

// -----------------------------------------------------------------------
// 2x2 Gauss-Legendre quadrature rule on [-1,1]^2 (weights are 1 each).
// Matches the Gauss point setup duplicated in femq1_diff.m/femq1_cd.m.
// -----------------------------------------------------------------------
template <typename T>
struct GaussPt2x2 {
    T s, t, wt;
};

template <typename T>
inline std::array<GaussPt2x2<T>, 4> gauss_2x2() {
    const T gpt = T(1) / std::sqrt(T(3));
    return {{
        {-gpt, -gpt, T(1)},
        { gpt, -gpt, T(1)},
        { gpt,  gpt, T(1)},
        {-gpt,  gpt, T(1)},
    }};
}

// -----------------------------------------------------------------------
// Interpolates physical (x,y) at a point from precomputed shape values and
// element vertex coordinates.
// -----------------------------------------------------------------------
template <typename T>
inline void interpolate_xy(const T phi[4], const T xl[4], const T yl[4], T& xx, T& yy) {
    xx = T(0);
    yy = T(0);
    for (int k = 0; k < 4; ++k) {
        xx += phi[k] * xl[k];
        yy += phi[k] * yl[k];
    }
}

// -----------------------------------------------------------------------
// Translation of gauss_transprt.m: interpolate physical (x,y), then evaluate
// the convection field. WindFn defaults to the circular wind; pass
// velocity_field_w_constant<T> to use the constant wind instead.
// -----------------------------------------------------------------------
template <typename T, typename WindFn = void (*)(T, T, T&, T&)>
inline void gauss_transprt(const T phi[4], const T xl[4], const T yl[4], T& wx, T& wy,
                            WindFn wind = velocity_field_w_circular<T>) {
    T xx, yy;
    interpolate_xy<T>(phi, xl, yl, xx, yy);
    wind(xx, yy, wx, wy);
}

// -----------------------------------------------------------------------
// Translation of gauss_source.m/specific_rhs.m: currently zero forcing.
// -----------------------------------------------------------------------
template <typename T>
inline T gauss_source(const T phi[4], const T xl[4], const T yl[4]) {
    T xx, yy;
    interpolate_xy<T>(phi, xl, yl, xx, yy);
    (void)xx;
    (void)yy;
    return T(0);
}

// -----------------------------------------------------------------------
// 1D node coordinates for a uniform tensor-product grid on [0,1] with n1d nodes.
// -----------------------------------------------------------------------
template <typename T>
inline std::vector<T> uniform_1d_coords(int n1d) {
    std::vector<T> x1d(n1d);
    for (int i = 0; i < n1d; ++i) x1d[i] = T(i) / T(n1d - 1);
    return x1d;
}

} // namespace fem
} // namespace pdegen
