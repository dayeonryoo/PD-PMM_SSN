#include "fem_q1.hpp"

#include <gtest/gtest.h>

#include <cmath>

using pdegen::fem::deriv;
using pdegen::fem::gauss_2x2;
using pdegen::fem::gauss_source;
using pdegen::fem::gauss_transprt;
using pdegen::fem::interpolate_xy;
using pdegen::fem::shape;
using pdegen::fem::uniform_1d_coords;
using pdegen::fem::velocity_field_w_circular;
using pdegen::fem::velocity_field_w_constant;

namespace {

// Reference-square vertex order matches shape()'s convention:
// (-1,-1), (1,-1), (1,1), (-1,1).
constexpr double kRefS[4] = {-1.0, 1.0, 1.0, -1.0};
constexpr double kRefT[4] = {-1.0, -1.0, 1.0, 1.0};

}  // namespace

TEST(Shape, IsOneAtOwnVertexAndZeroAtOthers) {
  for (int v = 0; v < 4; ++v) {
    const auto sv = shape<double>(kRefS[v], kRefT[v]);
    for (int k = 0; k < 4; ++k) {
      EXPECT_NEAR(sv.phi[k], k == v ? 1.0 : 0.0, 1e-12) << "vertex " << v << " phi[" << k << "]";
    }
  }
}

TEST(Shape, PartitionOfUnityHoldsAwayFromVertices) {
  for (double s : {-0.7, 0.0, 0.3}) {
    for (double t : {-0.4, 0.0, 0.6}) {
      const auto sv = shape<double>(s, t);
      double sum = 0.0;
      for (int k = 0; k < 4; ++k) sum += sv.phi[k];
      EXPECT_NEAR(sum, 1.0, 1e-12) << "s=" << s << " t=" << t;
    }
  }
}

TEST(Shape, CenterValuesAreOneQuarter) {
  const auto sv = shape<double>(0.0, 0.0);
  for (int k = 0; k < 4; ++k) EXPECT_NEAR(sv.phi[k], 0.25, 1e-12);
  EXPECT_NEAR(sv.dphids[0], -0.25, 1e-12);
  EXPECT_NEAR(sv.dphids[1], 0.25, 1e-12);
  EXPECT_NEAR(sv.dphids[2], 0.25, 1e-12);
  EXPECT_NEAR(sv.dphids[3], -0.25, 1e-12);
  EXPECT_NEAR(sv.dphidt[0], -0.25, 1e-12);
  EXPECT_NEAR(sv.dphidt[1], -0.25, 1e-12);
  EXPECT_NEAR(sv.dphidt[2], 0.25, 1e-12);
  EXPECT_NEAR(sv.dphidt[3], 0.25, 1e-12);
}

// Affine map from the reference square [-1,1]^2 onto the unit square [0,1]^2:
// x = (s+1)/2, y = (t+1)/2. Under this map jac == 1/4 everywhere and physical
// derivatives are exactly 2x the reference derivatives.
TEST(Deriv, AffineUnitSquareMapGivesConstantJacobianAndScaledDerivatives) {
  const double xl[4] = {0.0, 1.0, 1.0, 0.0};
  const double yl[4] = {0.0, 0.0, 1.0, 1.0};

  for (double s : {-0.6, 0.0, 0.5}) {
    for (double t : {-0.3, 0.0, 0.8}) {
      const auto sv = shape<double>(s, t);
      const auto dv = deriv<double>(s, t, xl, yl);

      EXPECT_NEAR(dv.jac, 0.25, 1e-12) << "s=" << s << " t=" << t;
      for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(dv.phi[k], sv.phi[k], 1e-12);
        EXPECT_NEAR(dv.dphidx[k], 2.0 * sv.dphids[k], 1e-12);
        EXPECT_NEAR(dv.dphidy[k], 2.0 * sv.dphidt[k], 1e-12);
      }
    }
  }
}

TEST(Deriv, ThrowsOnInvertedElement) {
  // Swapping two vertices flips orientation and makes the Jacobian <= 0.
  const double xl[4] = {1.0, 0.0, 1.0, 0.0};
  const double yl[4] = {0.0, 0.0, 1.0, 1.0};
  EXPECT_THROW(deriv<double>(0.0, 0.0, xl, yl), std::runtime_error);
}

TEST(Deriv, ThrowsOnExactlyZeroJacobian) {
  const double xl[4] = {0.0, 0.0, 0.0, 0.0};
  const double yl[4] = {0.0, 0.0, 0.0, 0.0};
  EXPECT_THROW(deriv<double>(0.0, 0.0, xl, yl), std::runtime_error);
}

TEST(InterpolateXy, RecoversElementCenterOnUnitSquare) {
  const double xl[4] = {0.0, 1.0, 1.0, 0.0};
  const double yl[4] = {0.0, 0.0, 1.0, 1.0};
  const auto sv = shape<double>(0.0, 0.0);

  double xx = 0.0, yy = 0.0;
  interpolate_xy<double>(sv.phi, xl, yl, xx, yy);
  EXPECT_NEAR(xx, 0.5, 1e-12);
  EXPECT_NEAR(yy, 0.5, 1e-12);
}

TEST(Gauss2x2, PointsAndWeightsMatchStandardRule) {
  const auto pts = gauss_2x2<double>();
  const double gpt = 1.0 / std::sqrt(3.0);

  double weight_sum = 0.0;
  for (const auto& p : pts) {
    EXPECT_NEAR(std::fabs(p.s), gpt, 1e-12);
    EXPECT_NEAR(std::fabs(p.t), gpt, 1e-12);
    EXPECT_DOUBLE_EQ(p.wt, 1.0);
    weight_sum += p.wt;
  }
  // Sum of weights must equal the area of the reference domain [-1,1]^2.
  EXPECT_DOUBLE_EQ(weight_sum, 4.0);
}

TEST(Uniform1dCoords, SpansZeroToOneInclusive) {
  const auto x = uniform_1d_coords<double>(5);
  ASSERT_EQ(x.size(), 5u);
  const double expected[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
  for (int i = 0; i < 5; ++i) EXPECT_NEAR(x[i], expected[i], 1e-12);
}

TEST(Uniform1dCoords, HandlesTwoPointDegenerateCase) {
  const auto x = uniform_1d_coords<double>(2);
  ASSERT_EQ(x.size(), 2u);
  EXPECT_NEAR(x[0], 0.0, 1e-12);
  EXPECT_NEAR(x[1], 1.0, 1e-12);
}

TEST(Uniform1dCoords, ZeroPointsReturnsEmptyVector) {
  const auto x = uniform_1d_coords<double>(0);
  EXPECT_TRUE(x.empty());
}

// ===================== velocity fields =====================

TEST(VelocityFieldWCircular, MatchesClosedFormAtSeveralPoints) {
  double wx, wy;
  velocity_field_w_circular<double>(0.0, 0.0, wx, wy);
  EXPECT_NEAR(wx, 0.0, 1e-12);
  EXPECT_NEAR(wy, 0.0, 1e-12);  // origin: both components vanish

  velocity_field_w_circular<double>(0.5, 0.5, wx, wy);
  EXPECT_NEAR(wx, 0.75, 1e-12);   // 2*0.5*(1-0.25)
  EXPECT_NEAR(wy, -0.75, 1e-12);  // -2*0.5*(1-0.25)

  velocity_field_w_circular<double>(1.0, 0.0, wx, wy);
  EXPECT_NEAR(wx, 0.0, 1e-12);  // 1-x1^2 = 0
  EXPECT_NEAR(wy, -2.0, 1e-12);
}

TEST(VelocityFieldWConstant, IgnoresItsInputEverywhere) {
  double wx, wy;
  velocity_field_w_constant<double>(0.0, 0.0, wx, wy);
  EXPECT_NEAR(wx, -1.0 / std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(wy, 1.0 / std::sqrt(2.0), 1e-12);

  // Wildly different (x1,x2) must produce the identical output.
  velocity_field_w_constant<double>(123.0, -456.0, wx, wy);
  EXPECT_NEAR(wx, -1.0 / std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(wy, 1.0 / std::sqrt(2.0), 1e-12);
}

// ===================== gauss_transprt / gauss_source =====================

TEST(GaussTransprt, InterpolatesLocationThenAppliesDefaultCircularWind) {
  const double xl[4] = {0.0, 1.0, 1.0, 0.0};
  const double yl[4] = {0.0, 0.0, 1.0, 1.0};
  const auto sv = shape<double>(0.0, 0.0);  // element center -> (x,y) = (0.5, 0.5)

  double wx, wy;
  gauss_transprt<double>(sv.phi, xl, yl, wx, wy);
  EXPECT_NEAR(wx, 0.75, 1e-12);
  EXPECT_NEAR(wy, -0.75, 1e-12);
}

TEST(GaussTransprt, UsesExplicitlyPassedWindFunction) {
  const double xl[4] = {0.0, 1.0, 1.0, 0.0};
  const double yl[4] = {0.0, 0.0, 1.0, 1.0};
  const auto sv = shape<double>(0.0, 0.0);

  double wx, wy;
  gauss_transprt<double>(sv.phi, xl, yl, wx, wy, velocity_field_w_constant<double>);
  EXPECT_NEAR(wx, -1.0 / std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(wy, 1.0 / std::sqrt(2.0), 1e-12);
}

TEST(GaussSource, IsIdenticallyZero) {
  const double xl[4] = {0.0, 1.0, 1.0, 0.0};
  const double yl[4] = {0.0, 0.0, 1.0, 1.0};
  const auto sv = shape<double>(0.2, -0.4);
  EXPECT_DOUBLE_EQ(gauss_source<double>(sv.phi, xl, yl), 0.0);
}
