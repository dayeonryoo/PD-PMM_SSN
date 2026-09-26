"""
Q1 bilinear element kernels (shape functions, Jacobian/derivatives, quadrature).

The kernels in this file (shape, deriv, gauss_transprt, gauss_source,
gauss_2x2) are Python translations of the corresponding IFISS MATLAB
functions:
  shape.m           - IFISS function: DJS; 4 March 2005
  deriv.m           - IFISS function: DJS; 4 March 2005
  gauss_transprt.m  - IFISS function: DJS; 4 March 2005
  gauss_source.m    - IFISS function: DJS; 4 March 2005
Copyright (c) 2005 D.J. Silvester, H.C. Elman, A. Ramage

Every routine here is vectorised over a leading "element" axis: element
vertex coordinates are passed as arrays of shape (..., 4) and the whole
mesh is processed in one call, which keeps assembly practical at the grid
sizes used by the benchmarks (nc = 10 is ~1e6 elements).

See pde_generator.py for the global assembly and the QP generators built
on top of these kernels.
"""

import numpy as np

# Number of vertices of a Q1 (bilinear quadrilateral) element.
NUM_LOCAL_NODES = 4


# -----------------------------------------------------------------------
# Convection field.
# -----------------------------------------------------------------------

def velocity_field_w_constant(x1, x2):
    """Constant convection field w = [-1/sqrt(2), 1/sqrt(2)]^T.

    Returns (wx, wy) broadcast to the shape of the inputs, so it can be
    evaluated at a single point or at a whole array of quadrature points.
    """
    shape = np.broadcast(np.asarray(x1, dtype=float),
                         np.asarray(x2, dtype=float)).shape
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    return np.full(shape, -inv_sqrt2), np.full(shape, inv_sqrt2)


# -----------------------------------------------------------------------
# Q1 bilinear shape functions at reference point (s,t) in [-1,1]^2.
# Vertex order: (-1,-1), (1,-1), (1,1), (-1,1) (counter-clockwise).
# Translation of shape.m.
# -----------------------------------------------------------------------

def shape(s, t):
    """Returns (phi, dphids, dphidt), each a length-4 array."""
    one = 1.0
    phi = np.array([
         0.25 * (s - one) * (t - one),
        -0.25 * (s + one) * (t - one),
         0.25 * (s + one) * (t + one),
        -0.25 * (s - one) * (t + one),
    ])
    dphids = np.array([
         0.25 * (t - one),
        -0.25 * (t - one),
         0.25 * (t + one),
        -0.25 * (t + one),
    ])
    dphidt = np.array([
         0.25 * (s - one),
        -0.25 * (s + one),
         0.25 * (s + one),
        -0.25 * (s - one),
    ])
    return phi, dphids, dphidt


# -----------------------------------------------------------------------
# Jacobian and physical-space derivatives of the Q1 shape functions, at
# reference point (s,t). xl/yl hold the 4 vertex physical coordinates in
# the same order as shape()'s vertex convention, with an arbitrary number
# of leading element axes.
# Translation of deriv.m.
#
# dphidx/dphidy here are true, normalised physical derivatives.
# -----------------------------------------------------------------------

def deriv(s, t, xl, yl):
    """Returns (phi, dphidx, dphidy, jac).

    phi has shape (4,) (it does not depend on the element); dphidx and
    dphidy have shape (..., 4) and jac has shape (...), matching the
    leading axes of xl/yl.
    """
    phi, dphids, dphidt = shape(s, t)

    xl = np.asarray(xl, dtype=float)
    yl = np.asarray(yl, dtype=float)

    dxds = xl @ dphids
    dxdt = xl @ dphidt
    dyds = yl @ dphids
    dydt = yl @ dphidt

    jac = dxds * dydt - dxdt * dyds
    if np.any(jac <= 0.0):
        raise ValueError("fem_q1.deriv: singular or inverted element Jacobian")

    invjac = 1.0 / jac
    dphidx = (dphids * dydt[..., None] - dphidt * dyds[..., None]) * invjac[..., None]
    dphidy = (-dphids * dxdt[..., None] + dphidt * dxds[..., None]) * invjac[..., None]
    return phi, dphidx, dphidy, jac


# -----------------------------------------------------------------------
# 2x2 Gauss-Legendre quadrature rule on [-1,1]^2 (weights are 1 each).
# Matches the Gauss point setup duplicated in femq1_diff.m/femq1_cd.m.
# -----------------------------------------------------------------------

def gauss_2x2():
    """Returns a list of (s, t, wt) triples."""
    gpt = 1.0 / np.sqrt(3.0)
    return [
        (-gpt, -gpt, 1.0),
        ( gpt, -gpt, 1.0),
        ( gpt,  gpt, 1.0),
        (-gpt,  gpt, 1.0),
    ]


# -----------------------------------------------------------------------
# Interpolates physical (x,y) at a point from precomputed shape values and
# element vertex coordinates.
# -----------------------------------------------------------------------

def interpolate_xy(phi, xl, yl):
    """Returns (xx, yy), each with the leading element shape of xl/yl."""
    return np.asarray(xl, dtype=float) @ phi, np.asarray(yl, dtype=float) @ phi


# -----------------------------------------------------------------------
# Translation of gauss_transprt.m: interpolate physical (x,y), then evaluate
# the convection field. wind defaults to the constant wind.
# -----------------------------------------------------------------------

def gauss_transprt(phi, xl, yl, wind=velocity_field_w_constant):
    xx, yy = interpolate_xy(phi, xl, yl)
    return wind(xx, yy)


# -----------------------------------------------------------------------
# Translation of gauss_source.m/specific_rhs.m: currently zero forcing.
# -----------------------------------------------------------------------

def gauss_source(phi, xl, yl):
    xx, _ = interpolate_xy(phi, xl, yl)
    return np.zeros(np.shape(xx))


# -----------------------------------------------------------------------
# 1D node coordinates for a uniform tensor-product grid on [0,1] with n1d nodes.
# -----------------------------------------------------------------------

def uniform_1d_coords(n1d):
    return np.arange(n1d, dtype=float) / float(n1d - 1)
