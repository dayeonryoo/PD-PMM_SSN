#include "schur_operator.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <vector>

namespace {

Eigen::SparseMatrix<double> DenseToSparse(const Eigen::MatrixXd& dense) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < dense.rows(); ++i)
    for (int j = 0; j < dense.cols(); ++j)
      if (dense(i, j) != 0.0) trips.emplace_back(i, j, dense(i, j));
  Eigen::SparseMatrix<double> sp(dense.rows(), dense.cols());
  sp.setFromTriplets(trips.begin(), trips.end());
  sp.makeCompressed();
  return sp;
}

// Small 3x4 sparse G (3 equality rows, 4 primal variables).
Eigen::MatrixXd MakeDenseG() {
  Eigen::MatrixXd G(3, 4);
  G << 1.0, 0.0, 2.0, 0.0,
       0.0, 1.0, 0.0, 3.0,
       1.0, 1.0, 0.0, 0.0;
  return G;
}

}  // namespace

// ===================== rows/cols =====================

TEST(SchurOperator, RowsAndColsEqualGRowCount) {
  const auto G_dense = MakeDenseG();
  const auto G = DenseToSparse(G_dense);
  const auto G_tr = DenseToSparse(G_dense.transpose());
  Eigen::VectorXd H_diag_inv = Eigen::VectorXd::Ones(4);

  SchurOperator<double> S(G, G_tr, H_diag_inv, /*mu=*/2.0);

  EXPECT_EQ(S.rows(), 3);
  EXPECT_EQ(S.cols(), 3);
}

// ===================== operator* correctness =====================

TEST(SchurOperator, MatvecMatchesDenseForUniformHDiagInv) {
  const auto G_dense = MakeDenseG();
  const auto G = DenseToSparse(G_dense);
  const auto G_tr = DenseToSparse(G_dense.transpose());
  Eigen::VectorXd H_diag_inv = Eigen::VectorXd::Ones(4);
  const double mu = 2.0;

  SchurOperator<double> S(G, G_tr, H_diag_inv, mu);

  Eigen::VectorXd v(3);
  v << 1.0, -2.0, 0.5;

  const Eigen::VectorXd got = S * v;
  const Eigen::MatrixXd expected_dense =
      G_dense * H_diag_inv.asDiagonal() * G_dense.transpose() + (1.0 / mu) * Eigen::MatrixXd::Identity(3, 3);
  const Eigen::VectorXd expected = expected_dense * v;

  EXPECT_TRUE(got.isApprox(expected, 1e-12));
}

TEST(SchurOperator, MatvecMatchesDenseForNonUniformHDiagInv) {
  const auto G_dense = MakeDenseG();
  const auto G = DenseToSparse(G_dense);
  const auto G_tr = DenseToSparse(G_dense.transpose());
  Eigen::VectorXd H_diag_inv(4);
  H_diag_inv << 0.5, 2.0, 1.0, 4.0;
  const double mu = 3.0;

  SchurOperator<double> S(G, G_tr, H_diag_inv, mu);

  Eigen::VectorXd v(3);
  v << 2.0, 1.0, -1.0;

  const Eigen::VectorXd got = S * v;
  const Eigen::MatrixXd expected_dense =
      G_dense * H_diag_inv.asDiagonal() * G_dense.transpose() + (1.0 / mu) * Eigen::MatrixXd::Identity(3, 3);
  const Eigen::VectorXd expected = expected_dense * v;

  EXPECT_TRUE(got.isApprox(expected, 1e-12));
}

TEST(SchurOperator, MatvecIsLinearInInputVector) {
  const auto G_dense = MakeDenseG();
  const auto G = DenseToSparse(G_dense);
  const auto G_tr = DenseToSparse(G_dense.transpose());
  Eigen::VectorXd H_diag_inv(4);
  H_diag_inv << 1.0, 2.0, 3.0, 0.5;

  SchurOperator<double> S(G, G_tr, H_diag_inv, /*mu=*/1.5);

  Eigen::VectorXd v1(3), v2(3);
  v1 << 1.0, 0.0, -1.0;
  v2 << 0.5, 2.0, 1.0;
  const double a = 2.0, b = -3.0;

  const Eigen::VectorXd lhs = S * (a * v1 + b * v2);
  const Eigen::VectorXd rhs = a * (S * v1) + b * (S * v2);

  EXPECT_TRUE(lhs.isApprox(rhs, 1e-12));
}

TEST(SchurOperator, HandlesZeroRowGGracefully) {
  Eigen::SparseMatrix<double> G(0, 4), G_tr(4, 0);
  Eigen::VectorXd H_diag_inv = Eigen::VectorXd::Ones(4);

  SchurOperator<double> S(G, G_tr, H_diag_inv, /*mu=*/1.0);

  EXPECT_EQ(S.rows(), 0);
  EXPECT_EQ(S.cols(), 0);

  Eigen::VectorXd v(0);
  const Eigen::VectorXd got = S * v;
  EXPECT_EQ(got.size(), 0);
}

// ===================== integration with Eigen's CG =====================

TEST(SchurOperator, WorksAsMatrixFreeOperatorInsideEigenConjugateGradient) {
  // A well-conditioned SPD system so CG converges quickly and unambiguously.
  Eigen::MatrixXd G_dense(3, 3);
  G_dense << 2.0, 0.0, 0.0,
             0.0, 2.0, 0.0,
             0.0, 0.0, 2.0;
  const auto G = DenseToSparse(G_dense);
  const auto G_tr = DenseToSparse(G_dense.transpose());
  Eigen::VectorXd H_diag_inv = Eigen::VectorXd::Ones(3);
  const double mu = 1.0;

  SchurOperator<double> S(G, G_tr, H_diag_inv, mu);

  Eigen::VectorXd rhs(3);
  rhs << 1.0, 2.0, 3.0;

  Eigen::ConjugateGradient<SchurOperator<double>, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
  cg.compute(S);
  const Eigen::VectorXd x = cg.solve(rhs);

  EXPECT_EQ(cg.info(), Eigen::Success);

  const Eigen::MatrixXd expected_dense =
      G_dense * H_diag_inv.asDiagonal() * G_dense.transpose() + (1.0 / mu) * Eigen::MatrixXd::Identity(3, 3);
  const Eigen::VectorXd expected_x = expected_dense.colPivHouseholderQr().solve(rhs);

  EXPECT_TRUE(x.isApprox(expected_x, 1e-8));
}
