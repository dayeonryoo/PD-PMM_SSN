#include "pde_generator.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace pdegen;

namespace {

constexpr double kTight = 1e-12;
constexpr double kLoose = 1e-9;  // for sums accumulated over many elements

// Single-element (nc=0) grid:
// node 0 = (0,0), node 1 = (1,0), node 2 = (0,1), node 3 = (1,1)
// in GridQ1's global order.
// The standard bilinear-Q1 unit-square local stiffness/mass matrices
// (verified by direct integration of the shape functions
// phi0=(1-x)(1-y), phi1=x(1-y), phi2=xy, phi3=(1-x)y over [0,1]^2)
// are permuted from local order (0,0),(1,0),(1,1),(0,1) into GridQ1's node numbering.
constexpr double kExpectedAStiff0[4][4] = {
    {4.0 / 6, -1.0 / 6, -1.0 / 6, -2.0 / 6},
    {-1.0 / 6, 4.0 / 6, -2.0 / 6, -1.0 / 6},
    {-1.0 / 6, -2.0 / 6, 4.0 / 6, -1.0 / 6},
    {-2.0 / 6, -1.0 / 6, -1.0 / 6, 4.0 / 6},
};
constexpr double kExpectedMCons0[4][4] = {
    {4.0 / 36, 2.0 / 36, 2.0 / 36, 1.0 / 36},
    {2.0 / 36, 4.0 / 36, 1.0 / 36, 2.0 / 36},
    {2.0 / 36, 1.0 / 36, 4.0 / 36, 2.0 / 36},
    {1.0 / 36, 2.0 / 36, 2.0 / 36, 4.0 / 36},
};

Eigen::SparseMatrix<double> DenseArrayToSparse(const double (&arr)[4][4]) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      if (arr[i][j] != 0.0) trips.emplace_back(i, j, arr[i][j]);
  Eigen::SparseMatrix<double> m(4, 4);
  m.setFromTriplets(trips.begin(), trips.end());
  return m;
}

}  // namespace

// ===================== GridQ1 =====================

TEST(GridQ1, SingleElementGridHasFourCornerNodes) {
  GridQ1<double> g(0);
  EXPECT_EQ(g.n1d, 2);
  EXPECT_EQ(g.np, 4);
  EXPECT_EQ(g.nel1d, 1);
  EXPECT_EQ(g.nel, 1);
  ASSERT_EQ(g.x1d.size(), 2u);
  EXPECT_DOUBLE_EQ(g.x1d[0], 0.0);
  EXPECT_DOUBLE_EQ(g.x1d[1], 1.0);

  const auto nodes = g.element_nodes(0, 0);
  EXPECT_EQ(nodes, (std::array<int, 4>{0, 1, 3, 2}));
}

TEST(GridQ1, FourByFourElementGridHasExpectedCounts) {
  GridQ1<double> g(2);  // n1d = 2^2 + 1 = 5
  EXPECT_EQ(g.n1d, 5);
  EXPECT_EQ(g.np, 25);
  EXPECT_EQ(g.nel1d, 4);
  EXPECT_EQ(g.nel, 16);
  const double expected_x1d[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
  for (int i = 0; i < 5; ++i) EXPECT_NEAR(g.x1d[i], expected_x1d[i], kTight);
}

TEST(GridQ1, ElementNodesUseCcwConventionMatchingShape) {
  GridQ1<double> g(2);  // n1d = 5
  const auto nodes = g.element_nodes(1, 1);
  // idx(i,j) = i + 5*j
  EXPECT_EQ(nodes, (std::array<int, 4>{6, 7, 12, 11}));
}

TEST(GridQ1, ElementNodesForLastElementAtFarCorner) {
  GridQ1<double> g(2);  // n1d = 5, nel1d = 4, valid ei,ej in [0,3]
  const auto nodes = g.element_nodes(3, 3);
  // idx(i,j) = i + 5*j: (3,3)=18, (4,3)=19, (4,4)=24, (3,4)=23
  EXPECT_EQ(nodes, (std::array<int, 4>{18, 19, 24, 23}));
}

// ===================== boundary node collection =====================

TEST(IsBoundaryNode, FlagsOnlyGridEdges) {
  const int n1d = 3;
  EXPECT_TRUE(is_boundary_node<double>(0, 0, n1d));
  EXPECT_TRUE(is_boundary_node<double>(2, 1, n1d));
  EXPECT_TRUE(is_boundary_node<double>(1, 2, n1d));
  EXPECT_FALSE(is_boundary_node<double>(1, 1, n1d));  // sole interior node
}

TEST(FemBoundaryNodes, MatchesFourTimesN1dMinusFourOnSmallGrid) {
  GridQ1<double> g(1);  // n1d = 3, np = 9, one interior node
  const auto bc = fem_boundary_nodes<double>(g);
  EXPECT_EQ(bc.size(), 8u);  // all nodes except the single interior one

  const int interior = GridQ1<double>::idx(1, 1, g.n1d);
  EXPECT_EQ(std::count(bc.begin(), bc.end(), interior), 0);
  for (int p = 0; p < g.np; ++p) {
    if (p == interior) continue;
    EXPECT_EQ(std::count(bc.begin(), bc.end(), p), 1) << "node " << p;
  }
}

TEST(FemBoundaryNodes, AllNodesAreBoundaryOnSmallestGrid) {
  GridQ1<double> g(0);  // n1d = 2, np = 4
  const auto bc = fem_boundary_nodes<double>(g);
  EXPECT_EQ(bc.size(), static_cast<std::size_t>(g.np));
  for (int p = 0; p < g.np; ++p) {
    EXPECT_EQ(std::count(bc.begin(), bc.end(), p), 1) << "node " << p;
  }
}

// ===================== assemble_femq1_diff =====================

TEST(AssembleFemq1Diff, SingleElementMatchesHandDerivedUnitSquareMatrices) {
  GridQ1<double> g(0);
  const auto res = assemble_femq1_diff<double>(g);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NEAR(res.A_stiff.coeff(i, j), kExpectedAStiff0[i][j], kTight)
          << "A_stiff(" << i << "," << j << ")";
      EXPECT_NEAR(res.M_cons.coeff(i, j), kExpectedMCons0[i][j], kTight)
          << "M_cons(" << i << "," << j << ")";
    }
  }
  EXPECT_EQ(res.f_rhs.size(), 4);
  EXPECT_DOUBLE_EQ(res.f_rhs.norm(), 0.0);  // gauss_source is identically zero
}

TEST(AssembleFemq1Diff, StiffnessRowsSumToZeroAndMassSumsToDomainArea) {
  // grad(constant) = 0 (row sums of A_stiff vanish),
  // sum_ij M_ij = integral of (sum_i phi_i)^2 = integral of 1 = area(Omega).
  GridQ1<double> g(2);  // 4x4 elements
  const auto res = assemble_femq1_diff<double>(g);

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(res.A_stiff.cols());
  Eigen::VectorXd row_sums = res.A_stiff * ones;
  for (int i = 0; i < row_sums.size(); ++i) EXPECT_NEAR(row_sums(i), 0.0, kLoose);

  EXPECT_NEAR(res.M_cons.sum(), 1.0, kLoose);  // unit square area
}

// ===================== assemble_femq1_cd =====================

TEST(AssembleFemq1Cd, DiffusionAndMassBlocksMatchAssembleFemq1Diff) {
  GridQ1<double> g(1);
  const auto diff_res = assemble_femq1_diff<double>(g);
  const auto cd_res = assemble_femq1_cd<double>(g);

  EXPECT_TRUE(cd_res.A_stiff.isApprox(diff_res.A_stiff, kTight));
  EXPECT_TRUE(cd_res.M_cons.isApprox(diff_res.M_cons, kTight));
}

TEST(AssembleFemq1Cd, ConvectionRowsSumToZero) {
  // N_ij = integral phi_i (w . grad phi_j); summing over j gives
  // integral phi_i * w . grad(sum_j phi_j) = integral phi_i * w . grad(1) = 0,
  // for any wind field, independent of mesh resolution.
  GridQ1<double> g(2);
  const auto res = assemble_femq1_cd<double>(g);  // default circular wind
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(res.N_conv.cols());
  Eigen::VectorXd row_sums = res.N_conv * ones;
  for (int i = 0; i < row_sums.size(); ++i) EXPECT_NEAR(row_sums(i), 0.0, kLoose);
}

// ===================== apply_dirichlet_bc / apply_dirichlet_bc_mass =====================

TEST(ApplyDirichletBc, FoldsBoundaryColumnIntoRhsAndSetsIdentityRow) {
  Eigen::SparseMatrix<double> D = DenseArrayToSparse(kExpectedAStiff0);
  Eigen::VectorXd rhs(4);
  rhs << 1.0, 2.0, 3.0, 4.0;

  const std::vector<int> bc_nodes = {0};
  Eigen::VectorXd bc_values(1);
  bc_values << 5.0;

  apply_dirichlet_bc<double>(D, rhs, bc_nodes, bc_values);

  // rhs_r <- rhs_r - D_orig(r,0) * 5, for r != 0
  EXPECT_NEAR(rhs(0), 5.0, kTight);
  EXPECT_NEAR(rhs(1), 2.0 - kExpectedAStiff0[1][0] * 5.0, kTight);
  EXPECT_NEAR(rhs(2), 3.0 - kExpectedAStiff0[2][0] * 5.0, kTight);
  EXPECT_NEAR(rhs(3), 4.0 - kExpectedAStiff0[3][0] * 5.0, kTight);

  // Row/col 0 becomes an identity row; the interior 3x3 block is untouched.
  for (int j = 0; j < 4; ++j) EXPECT_NEAR(D.coeff(0, j), j == 0 ? 1.0 : 0.0, kTight);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(D.coeff(i, 0), i == 0 ? 1.0 : 0.0, kTight);
  for (int i = 1; i < 4; ++i)
    for (int j = 1; j < 4; ++j)
      EXPECT_NEAR(D.coeff(i, j), kExpectedAStiff0[i][j], kTight);
}

TEST(ApplyDirichletBcMass, ZeroesBoundaryRowsAndColumnsEntirely) {
  Eigen::SparseMatrix<double> M = DenseArrayToSparse(kExpectedMCons0);
  apply_dirichlet_bc_mass<double>(M, {0, 2});

  for (int j = 0; j < 4; ++j) {
    EXPECT_NEAR(M.coeff(0, j), 0.0, kTight);
    EXPECT_NEAR(M.coeff(2, j), 0.0, kTight);
    EXPECT_NEAR(M.coeff(j, 0), 0.0, kTight);
    EXPECT_NEAR(M.coeff(j, 2), 0.0, kTight);
  }
  // Interior rows/cols {1,3} are untouched.
  EXPECT_NEAR(M.coeff(1, 1), kExpectedMCons0[1][1], kTight);
  EXPECT_NEAR(M.coeff(1, 3), kExpectedMCons0[1][3], kTight);
  EXPECT_NEAR(M.coeff(3, 1), kExpectedMCons0[3][1], kTight);
  EXPECT_NEAR(M.coeff(3, 3), kExpectedMCons0[3][3], kTight);
}

TEST(ApplyDirichletBc, EmptyBcNodesIsNoOp) {
  Eigen::SparseMatrix<double> D = DenseArrayToSparse(kExpectedAStiff0);
  Eigen::VectorXd rhs(4);
  rhs << 1.0, 2.0, 3.0, 4.0;
  const Eigen::VectorXd rhs_before = rhs;
  const std::vector<int> bc_nodes;
  const Eigen::VectorXd bc_values;

  apply_dirichlet_bc<double>(D, rhs, bc_nodes, bc_values);

  EXPECT_TRUE(rhs.isApprox(rhs_before, kTight));
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(D.coeff(i, j), kExpectedAStiff0[i][j], kTight);
}

TEST(ApplyDirichletBc, AllNodesAsBoundaryProducesIdentity) {
  Eigen::SparseMatrix<double> D = DenseArrayToSparse(kExpectedAStiff0);
  Eigen::VectorXd rhs(4);
  rhs << 1.0, 2.0, 3.0, 4.0;
  const std::vector<int> bc_nodes = {0, 1, 2, 3};
  Eigen::VectorXd bc_values(4);
  bc_values << 10.0, 20.0, 30.0, 40.0;

  apply_dirichlet_bc<double>(D, rhs, bc_nodes, bc_values);

  EXPECT_TRUE(rhs.isApprox(bc_values, kTight));
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(D.coeff(i, j), i == j ? 1.0 : 0.0, kTight);
}

TEST(ApplyDirichletBcMass, EmptyBcNodesIsNoOp) {
  Eigen::SparseMatrix<double> M = DenseArrayToSparse(kExpectedMCons0);
  apply_dirichlet_bc_mass<double>(M, /*bc_nodes=*/{});
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(M.coeff(i, j), kExpectedMCons0[i][j], kTight);
}

TEST(ApplyDirichletBcMass, AllNodesAsBoundaryProducesZeroMatrix) {
  Eigen::SparseMatrix<double> M = DenseArrayToSparse(kExpectedMCons0);
  apply_dirichlet_bc_mass<double>(M, {0, 1, 2, 3});
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(M.coeff(i, j), 0.0, kTight);
}

// ===================== make_problem_l2_from_mats =====================

TEST(MakeProblemL2FromMats, MatchesHandDerivedTwoNodeExample) {
  // np=2, D_op = diag(2,3), M = I (a lumped mass matrix is diagonal).
  Eigen::SparseMatrix<double> D(2, 2), M(2, 2);
  D.insert(0, 0) = 2.0;
  D.insert(1, 1) = 3.0;
  M.insert(0, 0) = 1.0;
  M.insert(1, 1) = 1.0;
  D.makeCompressed();
  M.makeCompressed();

  Eigen::VectorXd rhs(2);
  rhs << 5.0, 7.0;
  Eigen::VectorXd yhat(2);
  yhat << 1.0, 2.0;

  const double beta = 4.0;
  auto pb = make_problem_l2_from_mats<double>(D, M, rhs, yhat, beta, -1.0, 1.0, -2.0, 2.0);

  EXPECT_EQ(pb.n, 4);
  EXPECT_EQ(pb.m, 2);
  EXPECT_EQ(pb.l, 0);
  EXPECT_NEAR(pb.obj_const, 2.5, kTight);  // 0.5 * (1*1 + 2*2)

  Eigen::VectorXd expected_c(4);
  expected_c << -1.0, -2.0, 0.0, 0.0;
  EXPECT_TRUE(pb.c.isApprox(expected_c, kTight));

  const double expected_Q[4][4] = {
      {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, beta, 0}, {0, 0, 0, beta}};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(pb.Q.coeff(i, j), expected_Q[i][j], kTight);

  // A = [D_op, -M]
  const double expected_A[2][4] = {{2, 0, -1, 0}, {0, 3, 0, -1}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(pb.A.coeff(i, j), expected_A[i][j], kTight);
  EXPECT_TRUE(pb.b.isApprox(rhs, kTight));

  EXPECT_EQ(pb.B.rows(), 0);
  EXPECT_EQ(pb.lw.size(), 0);
  EXPECT_EQ(pb.uw.size(), 0);

  Eigen::VectorXd expected_lx(4), expected_ux(4);
  expected_lx << -1.0, -1.0, -2.0, -2.0;
  expected_ux << 1.0, 1.0, 2.0, 2.0;
  EXPECT_TRUE(pb.lx.isApprox(expected_lx, kTight));
  EXPECT_TRUE(pb.ux.isApprox(expected_ux, kTight));
}

TEST(MakeProblemL2FromMats, HandlesNonSymmetricMassMatrix) {
  Eigen::SparseMatrix<double> D(2, 2), M(2, 2);
  D.insert(0, 0) = 2.0;
  D.insert(1, 1) = 3.0;
  M.insert(0, 0) = 2.0;
  M.insert(0, 1) = 1.0;
  M.insert(1, 0) = 3.0;
  M.insert(1, 1) = 4.0;
  D.makeCompressed();
  M.makeCompressed();

  Eigen::VectorXd rhs(2);
  rhs << 5.0, 7.0;
  Eigen::VectorXd yhat(2);
  yhat << 1.0, 2.0;

  const double beta = 4.0;
  auto pb = make_problem_l2_from_mats<double>(D, M, rhs, yhat, beta);

  // Myhat = M*yhat = [2*1+1*2, 3*1+4*2] = [4, 11]
  EXPECT_NEAR(pb.obj_const, 13.0, kTight);  // 0.5*(1*4 + 2*11)

  Eigen::VectorXd expected_c(4);
  expected_c << -4.0, -11.0, 0.0, 0.0;
  EXPECT_TRUE(pb.c.isApprox(expected_c, kTight));

  // y-y block is M verbatim (row i, col j preserved); u-u block is beta*M.
  const double expected_Q[4][4] = {
      {2, 1, 0, 0}, {3, 4, 0, 0}, {0, 0, beta * 2, beta * 1}, {0, 0, beta * 3, beta * 4}};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(pb.Q.coeff(i, j), expected_Q[i][j], kTight);

  // A = [D_op, -M]: column j of the -M block holds -M(row, j), not -M(j, row).
  const double expected_A[2][4] = {{2, 0, -2, -1}, {0, 3, -3, -4}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(pb.A.coeff(i, j), expected_A[i][j], kTight);
}

// ===================== make_problem_l1l2_from_mats =====================

TEST(MakeProblemL1l2FromMats, MatchesHandDerivedTwoNodeExample) {
  // np=2, D_op = diag(2,3), M = I (a lumped mass matrix is diagonal).
  Eigen::SparseMatrix<double> D(2, 2), M(2, 2);
  D.insert(0, 0) = 2.0;
  D.insert(1, 1) = 3.0;
  M.insert(0, 0) = 1.0;
  M.insert(1, 1) = 1.0;
  D.makeCompressed();
  M.makeCompressed();

  Eigen::VectorXd rhs(2);
  rhs << 5.0, 7.0;
  Eigen::VectorXd yhat(2);
  yhat << 1.0, 2.0;

  const double alpha1 = 2.0, alpha2 = 3.0;
  auto pb = make_problem_l1l2_from_mats<double>(D, M, rhs, yhat, alpha1, alpha2, -2.0, 1.5);

  EXPECT_EQ(pb.n, 6);
  EXPECT_EQ(pb.m, 2);
  EXPECT_EQ(pb.l, 2);
  EXPECT_NEAR(pb.obj_const, 2.5, kTight);

  // R = row-sum of M = [1, 1] (M is diagonal identity here).
  Eigen::VectorXd expected_c(6);
  expected_c << -1.0, -2.0, 1.0, 1.0, 1.0, 1.0;  // (alpha1/2)*R = 1 on both u+/u- blocks
  EXPECT_TRUE(pb.c.isApprox(expected_c, kTight));

  // Q = [[M,0,0],[0,a2 M,-a2 M],[0,-a2 M,a2 M]], all blocks diagonal here.
  double expected_Q[6][6] = {};
  expected_Q[0][0] = 1.0;
  expected_Q[1][1] = 1.0;
  expected_Q[2][2] = alpha2;
  expected_Q[3][3] = alpha2;
  expected_Q[4][4] = alpha2;
  expected_Q[5][5] = alpha2;
  expected_Q[2][4] = -alpha2;
  expected_Q[4][2] = -alpha2;
  expected_Q[3][5] = -alpha2;
  expected_Q[5][3] = -alpha2;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) EXPECT_NEAR(pb.Q.coeff(i, j), expected_Q[i][j], kTight);

  // A = [D_op, -M, +M]
  const double expected_A[2][6] = {{2, 0, -1, 0, 1, 0}, {0, 3, 0, -1, 0, 1}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 6; ++j) EXPECT_NEAR(pb.A.coeff(i, j), expected_A[i][j], kTight);
  EXPECT_TRUE(pb.b.isApprox(rhs, kTight));

  // B = [0, I, -I]
  const double expected_B[2][6] = {{0, 0, 1, 0, -1, 0}, {0, 0, 0, 1, 0, -1}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 6; ++j) EXPECT_NEAR(pb.B.coeff(i, j), expected_B[i][j], kTight);

  const double inf = std::numeric_limits<double>::infinity();
  const double expected_lx[6] = {-inf, -inf, 0.0, 0.0, 0.0, 0.0};
  const double expected_ux[6] = {inf, inf, inf, inf, inf, inf};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(pb.lx(i), expected_lx[i]) << "lx(" << i << ")";
    EXPECT_EQ(pb.ux(i), expected_ux[i]) << "ux(" << i << ")";
  }

  Eigen::VectorXd expected_lw(2), expected_uw(2);
  expected_lw << -2.0, -2.0;
  expected_uw << 1.5, 1.5;
  EXPECT_TRUE(pb.lw.isApprox(expected_lw, kTight));
  EXPECT_TRUE(pb.uw.isApprox(expected_uw, kTight));
}

TEST(MakeProblemL1l2FromMats, HandlesNonSymmetricMassMatrix) {
  Eigen::SparseMatrix<double> D(2, 2), M(2, 2);
  D.insert(0, 0) = 2.0;
  D.insert(1, 1) = 3.0;
  M.insert(0, 0) = 2.0;
  M.insert(0, 1) = 1.0;
  M.insert(1, 0) = 3.0;
  M.insert(1, 1) = 4.0;
  D.makeCompressed();
  M.makeCompressed();

  Eigen::VectorXd rhs(2);
  rhs << 5.0, 7.0;
  Eigen::VectorXd yhat(2);
  yhat << 1.0, 2.0;

  const double alpha1 = 2.0, alpha2 = 3.0;
  auto pb = make_problem_l1l2_from_mats<double>(D, M, rhs, yhat, alpha1, alpha2, -2.0, 1.5);

  // R = row-sum of M = [2+1, 3+4] = [3, 7]
  Eigen::VectorXd expected_c(6);
  expected_c << -4.0, -11.0, 3.0, 7.0, 3.0, 7.0;
  EXPECT_TRUE(pb.c.isApprox(expected_c, kTight));
  EXPECT_NEAR(pb.obj_const, 13.0, kTight);

  double expected_Q[6][6] = {};
  expected_Q[0][0] = 2;
  expected_Q[0][1] = 1;
  expected_Q[1][0] = 3;
  expected_Q[1][1] = 4;
  expected_Q[2][2] = alpha2 * 2;
  expected_Q[2][3] = alpha2 * 1;
  expected_Q[3][2] = alpha2 * 3;
  expected_Q[3][3] = alpha2 * 4;
  expected_Q[4][4] = alpha2 * 2;
  expected_Q[4][5] = alpha2 * 1;
  expected_Q[5][4] = alpha2 * 3;
  expected_Q[5][5] = alpha2 * 4;
  expected_Q[2][4] = -alpha2 * 2;  // from M(0,0)
  expected_Q[2][5] = -alpha2 * 1;  // from M(0,1)
  expected_Q[3][4] = -alpha2 * 3;  // from M(1,0)
  expected_Q[3][5] = -alpha2 * 4;  // from M(1,1)
  expected_Q[4][2] = -alpha2 * 2;  // from M(0,0)
  expected_Q[4][3] = -alpha2 * 1;  // from M(0,1)
  expected_Q[5][2] = -alpha2 * 3;  // from M(1,0)
  expected_Q[5][3] = -alpha2 * 4;  // from M(1,1)
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) EXPECT_NEAR(pb.Q.coeff(i, j), expected_Q[i][j], kTight);

  // A = [D_op, -M, +M]
  const double expected_A[2][6] = {{2, 0, -2, -1, 2, 1}, {0, 3, -3, -4, 3, 4}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 6; ++j) EXPECT_NEAR(pb.A.coeff(i, j), expected_A[i][j], kTight);
}

// ===================== QP generators (integration smoke tests) =====================

namespace {

// After Dirichlet elimination, each boundary node's row in A must be 
// an identity row on its own y-column and zero everywhere else.
void ExpectBoundaryRowsAreIdentity(const KSPQPdata<double>& pb, const GridQ1<double>& g,
                                    const std::vector<int>& bc_nodes) {
  for (int p : bc_nodes) {
    for (int j = 0; j < pb.n; ++j) {
      const double expected = (j == p) ? 1.0 : 0.0;
      EXPECT_NEAR(pb.A.coeff(p, j), expected, kTight) << "row " << p << " col " << j;
    }
  }
  const int interior = GridQ1<double>::idx(1, 1, g.n1d);
  EXPECT_NE(pb.A.coeff(interior, interior), 0.0) << "interior diagonal should retain D_op";
}

}  // namespace

TEST(MakePoissonL2Control, DimensionsAndZeroBoundaryValues) {
  GridQ1<double> g(1);  // np = 9
  const auto bc_nodes = fem_boundary_nodes<double>(g);
  auto pb = make_poisson_l2_control<double>(1, 4.0);

  EXPECT_EQ(pb.n, 18);
  EXPECT_EQ(pb.m, 9);
  EXPECT_EQ(pb.l, 0);
  ExpectBoundaryRowsAreIdentity(pb, g, bc_nodes);
  for (int p : bc_nodes) EXPECT_NEAR(pb.b(p), 0.0, kTight);
}

TEST(MakePoissonL2StateControl, BoundaryValuesMatchYhat) {
  GridQ1<double> g(1);
  const auto bc_nodes = fem_boundary_nodes<double>(g);
  auto pb = make_poisson_l2_state_control<double>(1, 4.0);

  ExpectBoundaryRowsAreIdentity(pb, g, bc_nodes);
  for (int p : bc_nodes) {
    const int i = p % g.n1d, j = p / g.n1d;
    const double expected = std::sin(M_PI * g.x1d[i]) * std::sin(M_PI * g.x1d[j]);
    EXPECT_NEAR(pb.b(p), expected, kTight);
  }
}

TEST(MakeConvdiffL2Control, DimensionsAndZeroBoundaryValues) {
  GridQ1<double> g(1);
  const auto bc_nodes = fem_boundary_nodes<double>(g);
  auto pb = make_convdiff_l2_control<double>(1, 4.0);

  EXPECT_EQ(pb.n, 18);
  EXPECT_EQ(pb.m, 9);
  ExpectBoundaryRowsAreIdentity(pb, g, bc_nodes);
  for (int p : bc_nodes) EXPECT_NEAR(pb.b(p), 0.0, kTight);
}

TEST(MakePoissonL1l2Control, DimensionsAndConstantOneBoundaryValues) {
  GridQ1<double> g(1);
  const auto bc_nodes = fem_boundary_nodes<double>(g);
  auto pb = make_poisson_l1l2_control<double>(1, 2.0, 3.0);

  EXPECT_EQ(pb.n, 27);
  EXPECT_EQ(pb.m, 9);
  EXPECT_EQ(pb.l, 9);
  ExpectBoundaryRowsAreIdentity(pb, g, bc_nodes);
  for (int p : bc_nodes) EXPECT_NEAR(pb.b(p), 1.0, kTight);

  Eigen::VectorXd expected_lw = Eigen::VectorXd::Constant(9, -2.0);
  Eigen::VectorXd expected_uw = Eigen::VectorXd::Constant(9, 1.5);
  EXPECT_TRUE(pb.lw.isApprox(expected_lw, kTight));
  EXPECT_TRUE(pb.uw.isApprox(expected_uw, kTight));
}

TEST(MakeConvdiffL1l2Control, DimensionsAndZeroBoundaryValues) {
  GridQ1<double> g(1);
  const auto bc_nodes = fem_boundary_nodes<double>(g);
  auto pb = make_convdiff_l1l2_control<double>(1, 2.0, 3.0);

  EXPECT_EQ(pb.n, 27);
  EXPECT_EQ(pb.m, 9);
  EXPECT_EQ(pb.l, 9);
  ExpectBoundaryRowsAreIdentity(pb, g, bc_nodes);
  for (int p : bc_nodes) EXPECT_NEAR(pb.b(p), 0.0, kTight);
}
