#include "schur_preconditioner.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <limits>
#include <random>
#include <vector>

namespace {

using Prec = SchurPreconditioner<double>;
using SpMat = Eigen::SparseMatrix<double>;
using RowMajorSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

constexpr double kTol = 1e-9;

SpMat DenseToSparse(const Eigen::MatrixXd& dense) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < dense.rows(); ++i)
    for (int j = 0; j < dense.cols(); ++j)
      if (dense(i, j) != 0.0) trips.emplace_back(i, j, dense(i, j));
  SpMat sp(dense.rows(), dense.cols());
  sp.setFromTriplets(trips.begin(), trips.end());
  sp.makeCompressed();
  return sp;
}

RowMajorSpMat DenseToSparseRowMajor(const Eigen::MatrixXd& dense) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < dense.rows(); ++i)
    for (int j = 0; j < dense.cols(); ++j)
      if (dense(i, j) != 0.0) trips.emplace_back(i, j, dense(i, j));
  RowMajorSpMat sp(dense.rows(), dense.cols());
  sp.setFromTriplets(trips.begin(), trips.end());
  sp.makeCompressed();
  return sp;
}

BoolArr ToBoolArr(const std::vector<bool>& v) {
  BoolArr arr(static_cast<int>(v.size()));
  for (std::size_t i = 0; i < v.size(); ++i) arr(static_cast<int>(i)) = v[i];
  return arr;
}

// P = G * diag(active_K ? 1/H_diag : 0) * G^T + (1/mu) I, computed independently via dense linear algebra.
Eigen::MatrixXd DenseSchurComplement(const Eigen::MatrixXd& G_dense, const Eigen::VectorXd& H_diag,
                                      const std::vector<bool>& active_K, double mu) {
  const int n = static_cast<int>(H_diag.size());
  Eigen::VectorXd e_diag = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; ++i)
    if (active_K[i]) e_diag(i) = 1.0 / H_diag(i);
  const int s = static_cast<int>(G_dense.rows());
  return G_dense * e_diag.asDiagonal() * G_dense.transpose() + (1.0 / mu) * Eigen::MatrixXd::Identity(s, s);
}

// A fixed tiny problem shape:
//   N = 3 primal columns.
//   M_rows = 1 "always active" equality row: A_row = [1, 1, 1].
//   l = 2 candidate inequality (W) rows: B_row0 = [1,0,0], B_row1 = [0,1,0].
struct Fixture {
  Eigen::MatrixXd A_row = (Eigen::MatrixXd(1, 3) << 1.0, 1.0, 1.0).finished();
  Eigen::MatrixXd B_rows = (Eigen::MatrixXd(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished();
  Eigen::VectorXd H_diag = (Eigen::VectorXd(3) << 2.0, 3.0, 4.0).finished();
  double mu = 5.0;
  double rho = 3.0;  // distinct from mu so a mu/rho argument swap would be caught by assertions

  RowMajorSpMat B_rm() const { return DenseToSparseRowMajor(B_rows); }

  // G = [A; B_active_W]
  Eigen::MatrixXd StackG(const std::vector<bool>& active_w) const {
    std::vector<Eigen::MatrixXd> rows{A_row};
    for (std::size_t i = 0; i < active_w.size(); ++i)
      if (active_w[i]) rows.push_back(B_rows.row(static_cast<int>(i)));
    Eigen::MatrixXd G(rows.size(), 3);
    for (std::size_t i = 0; i < rows.size(); ++i) G.row(static_cast<int>(i)) = rows[i];
    return G;
  }
};

}  // namespace

// Test-only peer granting direct access to SchurPreconditioner's private scratch buffers.
struct SchurPreconditionerTestPeer {
  static Prec::Vec& smw_tmp(Prec& p) { return p.smw_tmp_; }
  static Prec::Vec& smw_ldlt_padded(Prec& p) { return p.smw_ldlt_padded_; }
  static Prec::Mat& Y_all(Prec& p) { return p.Y_all_; }
  static Prec::Vec& r_pad(Prec& p) { return p.r_pad_; }
};

// NOTE: SchurPreconditioner::arm()/setData() store pointers to their arguments.

// ===================== direct-factorization correctness =====================

TEST(DirectFactorization, SolveMatchesDenseCholeskyWhenAllKActive) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});  // just the equality row
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/false);
  prec.compute(0);

  ASSERT_EQ(prec.info(), Eigen::Success);
  Eigen::VectorXd b(1);
  b << 3.0;
  const Eigen::VectorXd got = prec.solve(b);

  const Eigen::MatrixXd P = DenseSchurComplement(G_dense, f.H_diag, active_k, f.mu);
  const Eigen::VectorXd expected = P.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(DirectFactorization, SolveMatchesDenseWhenSomeKInactive) {
  Fixture f;
  const std::vector<bool> active_k = {true, false, true};  // column 1 inactive
  const Eigen::MatrixXd G_dense = f.StackG({true, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({true, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/false);
  prec.compute(0);

  Eigen::VectorXd b(2);
  b << 1.0, -2.0;
  const Eigen::VectorXd got = prec.solve(b);

  const Eigen::MatrixXd P = DenseSchurComplement(G_dense, f.H_diag, active_k, f.mu);
  const Eigen::VectorXd expected = P.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(DirectFactorization, LdltAndCholeskyPathsAgreeOnIdenticalData) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const Eigen::MatrixXd G_dense = f.StackG({true, true});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({true, true});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec_chol;
  prec_chol.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/false);
  prec_chol.compute(0);

  Prec prec_ldlt;
  prec_ldlt.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/true);
  prec_ldlt.compute(0);

  Eigen::VectorXd b(3);
  b << 1.0, 2.0, 3.0;
  const Eigen::VectorXd got_chol = prec_chol.solve(b);
  const Eigen::VectorXd got_ldlt = prec_ldlt.solve(b);

  EXPECT_TRUE(got_chol.isApprox(got_ldlt, kTol));
}

TEST(DirectFactorization, MuOnlyChangeMatchesFreshDenseRecomputationAtNewMu) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, /*mu=*/5.0, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Same G/H_diag/active_K/rho, only mu changes, and neither size nor pattern changed:
  // this should take factorize_by_chol's diagonal-shift-only fast path (numeric_dirty_ stays false).
  const double mu2 = 8.0;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, mu2, f.rho, /*rebuild=*/false, /*prec_pattern_changed=*/false, false);
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 2);  // mu change alone still triggers a (cheap) refactorization
  EXPECT_FALSE(prec.used_smw());    // rank==0 (no active-set delta) correctly rejects SMW

  Eigen::VectorXd b(1);
  b << 4.0;
  const Eigen::VectorXd got = prec.solve(b);

  const Eigen::MatrixXd P2 = DenseSchurComplement(G_dense, f.H_diag, active_k, mu2);
  const Eigen::VectorXd expected = P2.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(DirectFactorization, MuOnlyChangeMatchesFreshDenseRecomputationAtNewMuLdlt) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, /*mu=*/5.0, f.rho, true, true, /*use_ldlt=*/true);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Same G/H_diag/active_K/rho, only mu changes, and neither size nor pattern changed:
  // this should take factorize_by_ldlt's in-place-block-overwrite fast path (numeric_dirty_ stays false).
  const double mu2 = 8.0;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, mu2, f.rho, /*rebuild=*/false, /*prec_pattern_changed=*/false,
           /*use_ldlt=*/true);
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 2);  // mu change alone still triggers a (cheap) refactorization
  EXPECT_FALSE(prec.used_smw());    // rank==0 (no active-set delta) correctly rejects SMW

  Eigen::VectorXd b(1);
  b << 4.0;
  const Eigen::VectorXd got = prec.solve(b);

  const Eigen::MatrixXd P2 = DenseSchurComplement(G_dense, f.H_diag, active_k, mu2);
  const Eigen::VectorXd expected = P2.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(DirectFactorization, CholRebuildsOnRhoOnlyChangeWithoutPatternChange) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Same G/active_K/active_W/mu as the last build -- only rho changed, standing in for H_diag's
  // active-K entries drifting (H_diag(i) = Q_diag(i) + 1/rho there). Unlike a mu-only change,
  // E = 1/H_diag is nonlinear in rho, so this must fully rebuild G E G^T rather than diagonal-shift.
  const Eigen::VectorXd H_diag2 = (Eigen::VectorXd(3) << 6.0, 7.0, 9.0).finished();
  const double rho2 = 7.0;
  prec.arm(G, G_tr, H_diag2, active_K, active_W, B_rm, f.mu, rho2, /*rebuild=*/false,
           /*prec_pattern_changed=*/false, /*use_ldlt=*/false);
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 2);  // rho change alone still triggers a refactorization
  EXPECT_FALSE(prec.used_smw());    // rank==0 (no active-set delta) correctly rejects SMW

  Eigen::VectorXd b(1);
  b << 4.0;
  const Eigen::VectorXd got = prec.solve(b);

  const Eigen::MatrixXd P2 = DenseSchurComplement(G_dense, H_diag2, active_k, f.mu);
  const Eigen::VectorXd expected = P2.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(DirectFactorization, LdltPatchesTopLeftBlockOnRhoOnlyChangeWithoutPatternChange) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/true);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Same shape as the chol test above, but LDLT's -H_act sits directly on P_hat's diagonal (no
  // matrix product), so a rho-only change should be a cheap in-place patch, not a full triplet
  // rebuild -- verified indirectly here via correctness; the patch-vs-rebuild code path itself is
  // exercised by construction (numeric_dirty_ is false: prec_pattern_changed/size are unchanged).
  const Eigen::VectorXd H_diag2 = (Eigen::VectorXd(3) << 6.0, 7.0, 9.0).finished();
  const double rho2 = 7.0;
  prec.arm(G, G_tr, H_diag2, active_K, active_W, B_rm, f.mu, rho2, /*rebuild=*/false,
           /*prec_pattern_changed=*/false, /*use_ldlt=*/true);
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_FALSE(prec.used_smw());

  Eigen::VectorXd b(1);
  b << 4.0;
  const Eigen::VectorXd got = prec.solve(b);

  const Eigen::MatrixXd P2 = DenseSchurComplement(G_dense, H_diag2, active_k, f.mu);
  const Eigen::VectorXd expected = P2.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(DirectFactorization, FirstBuildNeverUsesSmwAndIncrementsFactCount) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
}

// ===================== SMW branch / edge cases =====================

TEST(SmwBranch, SmwActivatesOnSingleKFlipWithEverythingElseRetained) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});  // s stays 1 throughout
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_k1 = ToBoolArr({true, true, true});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_k1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Flip column 2 from active to inactive; everything else (G, active_W) retained.
  const std::vector<bool> active_k2 = {true, true, false};
  const BoolArr active_K2 = ToBoolArr(active_k2);
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);  // no full refactorization needed
  EXPECT_EQ(prec.smw_last_rank(), 1);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::None);

  Eigen::VectorXd b(1);
  b << 2.5;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G_dense, f.H_diag, active_k2, f.mu);
  const Eigen::VectorXd expected = P2.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(SmwBranch, RejectsSmwWhenRhoChangedSinceSnapshotEvenWithNonzeroRankDelta) {
  // Companion to SmwActivatesOnSingleKFlipWithEverythingElseRetained: same single-K-flip delta
  // (which alone would be SMW-eligible, rank=1), but rho also changed since the snapshot. The
  // low-rank update only recomputes H_diag-derived values for the flipped index (column 2); it
  // implicitly reuses P_old (factorized against the snapshot's rho) for everything else -- e.g.
  // column 0/1's contribution to the capacitance math. A rho drift silently invalidates that
  // reuse, so this must be rejected at the gate and fall back to a full, correct rebuild.
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});  // s stays 1 throughout
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_k1 = ToBoolArr({true, true, true});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_k1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Same single-K-flip delta as the companion test, but rho also changed (standing in for
  // H_diag's active-K entries drifting on top of the flip).
  const std::vector<bool> active_k2 = {true, true, false};
  const BoolArr active_K2 = ToBoolArr(active_k2);
  const Eigen::VectorXd H_diag2 = (Eigen::VectorXd(3) << 6.0, 7.0, 9.0).finished();
  const double rho2 = 7.0;
  prec.arm(G, G_tr, H_diag2, active_K2, active_W, B_rm, f.mu, rho2, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);  // full rebuild, not a low-rank patch
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::RhoChangedSinceSnapshot);

  Eigen::VectorXd b(1);
  b << 2.5;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G_dense, H_diag2, active_k2, f.mu);
  const Eigen::VectorXd expected = P2.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(SmwBranch, SmwHandlesSingleWRowAddition) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const BoolArr active_K = ToBoolArr(active_k);
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_W1 = ToBoolArr({true, false});
  const BoolArr active_W2 = ToBoolArr({true, true});

  const Eigen::MatrixXd G1_dense = f.StackG({true, false});
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, f.H_diag, active_K, active_W1, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Row 1 becomes newly active (added at the tail, after the already-active row 0).
  const Eigen::MatrixXd G2_dense = f.StackG({true, true});
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, f.H_diag, active_K, active_W2, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 1);

  Eigen::VectorXd b(3);
  b << 1.0, -1.0, 2.0;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, f.H_diag, active_k, f.mu);
  const Eigen::VectorXd expected = P2.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(SmwBranch, SmwHandlesSingleWRowDeletion) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const BoolArr active_K = ToBoolArr(active_k);
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_W1 = ToBoolArr({true, true});
  const BoolArr active_W2 = ToBoolArr({true, false});

  const Eigen::MatrixXd G1_dense = f.StackG({true, true});
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, f.H_diag, active_K, active_W1, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Row 1 is deactivated (deleted from the tail).
  const Eigen::MatrixXd G2_dense = f.StackG({true, false});
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, f.H_diag, active_K, active_W2, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 1);

  Eigen::VectorXd b(2);
  b << 0.5, 1.5;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, f.H_diag, active_k, f.mu);
  const Eigen::VectorXd expected = P2.colPivHouseholderQr().solve(b);
  EXPECT_TRUE(got.isApprox(expected, kTol));
}

TEST(SmwBranch, FallsBackToFullRebuildWhenDeltaRankExceedsThreshold) {
  // N candidate W rows (one variable each, diagonal-like); 
  // none active initially, all N become active at once: rank = N = 51 > kSmwRankThreshold(50).
  const int N = 51;
  Eigen::MatrixXd A_row = Eigen::MatrixXd::Ones(1, N);
  Eigen::MatrixXd B_rows = Eigen::MatrixXd::Identity(N, N);
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag = Eigen::VectorXd::LinSpaced(N, 2.0, 2.0 + N - 1);
  const double mu = 5.0;
  const double rho = 3.0;
  const BoolArr active_K = ToBoolArr(std::vector<bool>(N, true));
  const BoolArr active_W1 = ToBoolArr(std::vector<bool>(N, false));
  const BoolArr active_W2 = ToBoolArr(std::vector<bool>(N, true));

  const SpMat G1 = DenseToSparse(A_row);
  const SpMat G1_tr = DenseToSparse(A_row.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, H_diag, active_K, active_W1, B_rm, mu, rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  Eigen::MatrixXd G2_dense(N + 1, N);
  G2_dense.row(0) = A_row.row(0);
  G2_dense.bottomRows(N) = B_rows;
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, H_diag, active_K, active_W2, B_rm, mu, rho,
           /*rebuild=*/true, /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_rank(), N);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::RankZeroOrExceedsThreshold);
}

TEST(SmwBranch, ForceFullRebuildBypassesEligibleSmw) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_K1 = ToBoolArr({true, true, true});
  const BoolArr active_K2 = ToBoolArr({true, true, false});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // A single K flip would normally be SMW-eligible but force_rebuild=true should bypass it entirely.
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false, /*force_rebuild=*/true);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::ForcedRebuild);
}

TEST(SmwBranch, RecordSmwRebuildSuppressesSmwAfterFailStreak) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_K1 = ToBoolArr({true, true, true});
  const BoolArr active_K2 = ToBoolArr({true, true, false});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  for (int i = 0; i < 5; ++i) prec.record_smw_rebuild();
  EXPECT_TRUE(prec.smw_suppressed());

  // Otherwise-eligible single K flip should now be rejected as Suppressed.
  prec.set_data(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
               /*prec_pattern_changed=*/false);
  prec.set_use_ldlt(false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::Suppressed);
}

TEST(SmwBranch, ResetSmwFailStreakReenablesSmw) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_K1 = ToBoolArr({true, true, true});
  const BoolArr active_K2 = ToBoolArr({true, true, false});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Below the suppression threshold: the snapshot from the build above is not wiped.
  prec.record_smw_rebuild();
  ASSERT_FALSE(prec.smw_suppressed());
  prec.reset_smw_fail_streak();

  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
}

TEST(SmwBranch, ReleaseClearsStateAndForcesFullRebuildNext) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_K1 = ToBoolArr({true, true, true});
  const BoolArr active_K2 = ToBoolArr({true, true, false});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  prec.release();

  // Even an otherwise-SMW-eligible delta can't use SMW: release() cleared the snapshot and
  // initialized_, so build() skips try_build_smw() entirely and does a full factorization.
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/true, false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
}

TEST(SmwBranch, FactorizationMethodSwitchForcesFullRebuildInsteadOfSmw) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_K1 = ToBoolArr({true, true, true});
  const BoolArr active_K2 = ToBoolArr({true, true, false});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K1, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);
  ASSERT_FALSE(prec.used_ldlt_at_last_fact());

  // Same otherwise-SMW-eligible K flip, but switching factorization methods should force 
  // a full (LDLT) rebuild instead.
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, /*use_ldlt=*/true);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::FactorizationMethodChanged);
  EXPECT_TRUE(prec.used_ldlt_at_last_fact());
}

TEST(SmwBranch, SetDataForcesRebuildOnSizeChange) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const BoolArr active_K = ToBoolArr(active_k);
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_W1 = ToBoolArr({false, false});
  const BoolArr active_W2 = ToBoolArr({true, false});

  const Eigen::MatrixXd G1_dense = f.StackG({false, false});
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, f.H_diag, active_K, active_W1, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // rebuild=false is passed explicitly, but G's row count changed (one W row activated) --
  // setData()'s internal size_changed detection must still force a build.
  const Eigen::MatrixXd G2_dense = f.StackG({true, false});
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, f.H_diag, active_K, active_W2, B_rm, f.mu, f.rho, /*rebuild=*/false,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_EQ(prec.fact_count(), 1);  // the rebuild that did happen went through SMW, not a full refactorization
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.smw_count(), 1);
}

// ===================== finish_factorization (shared factorize_by_chol/ldlt tail) =====================
// finish_factorization() is a private helper so it is tested indirectly through arm()/compute()/solve().

TEST(FinishFactorization, ComputeSkipsRefactorizationWhenNothingChangesCholesky) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Re-arm with identical data and rebuild=false: finish_factorization must have left
  // mu_at_last_fact_ == mu, so compute() sees mu_changed == false and skips build() entirely.
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, /*rebuild=*/false,
           /*prec_pattern_changed=*/false, /*use_ldlt=*/false);
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 1);
}

TEST(FinishFactorization, ComputeSkipsRefactorizationWhenNothingChangesLdlt) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/true);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, /*rebuild=*/false,
           /*prec_pattern_changed=*/false, /*use_ldlt=*/true);
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 1);
}

TEST(FinishFactorization, LdltFullRebuildAcrossSizeChangeMatchesDenseOnBothFactorizations) {
  Fixture f;
  const Eigen::MatrixXd G1_dense = f.StackG({false, false});  // s = 1
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());
  const std::vector<bool> active_k1 = {true, true, true};  // n_act = 3
  const BoolArr active_K1 = ToBoolArr(active_k1);
  const BoolArr active_W1 = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G1, G1_tr, f.H_diag, active_K1, active_W1, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/true);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  Eigen::VectorXd b1(1);
  b1 << 3.0;
  const Eigen::VectorXd got1 = prec.solve(b1);
  const Eigen::MatrixXd P1 = DenseSchurComplement(G1_dense, f.H_diag, active_k1, f.mu);
  EXPECT_TRUE(got1.isApprox(P1.colPivHouseholderQr().solve(b1), kTol));

  const Eigen::MatrixXd G2_dense = f.StackG({true, false});  // s = 2
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());
  const std::vector<bool> active_k2 = {true, false, true};  // n_act = 2
  const BoolArr active_K2 = ToBoolArr(active_k2);
  const BoolArr active_W2 = ToBoolArr({true, false});

  prec.arm(G2, G2_tr, f.H_diag, active_K2, active_W2, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/true, /*use_ldlt=*/true, /*force_rebuild=*/true);
  prec.compute(0);
  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);

  Eigen::VectorXd b2(2);
  b2 << 1.0, -2.0;
  const Eigen::VectorXd got2 = prec.solve(b2);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, f.H_diag, active_k2, f.mu);
  EXPECT_TRUE(got2.isApprox(P2.colPivHouseholderQr().solve(b2), kTol));
}

TEST(FinishFactorization, SmwAfterLdltFullRebuildMatchesDense) {
  // Companion to SmwActivatesOnSingleKFlipWithEverythingElseRetained, but for the LDLT branch:
  // solve_smw() reads n_act_ from the snapshot taken by finish_factorization() after the initial
  // full LDLT build, so this exercises that n_act_ (not just s_current_) survives the LDLT path.
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});  // s stays 1 throughout
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_k1 = ToBoolArr({true, true, true});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_k1, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/true);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  const std::vector<bool> active_k2 = {true, true, false};
  const BoolArr active_K2 = ToBoolArr(active_k2);
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, /*use_ldlt=*/true);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);  // no full refactorization needed
  EXPECT_EQ(prec.smw_last_rank(), 1);

  Eigen::VectorXd b(1);
  b << 2.5;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G_dense, f.H_diag, active_k2, f.mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), kTol));
}

TEST(FinishFactorization, ScratchBuffersResetCorrectlyAcrossProblemSizeChange) {
  // Across try_build_smw() calls, a size change (here N = number of primal columns, 3 -> 5)
  // between two epochs on the same Prec instance must fully re-zero the buffers.
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(
      (Eigen::MatrixXd(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished());
  const Eigen::VectorXd H_diag = (Eigen::VectorXd(3) << 2.0, 3.0, 4.0).finished();
  const double mu = 5.0;
  const double rho = 3.0;
  const Eigen::MatrixXd A_row = (Eigen::MatrixXd(1, 3) << 1.0, 1.0, 1.0).finished();
  const Eigen::MatrixXd B_rows = (Eigen::MatrixXd(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished();

  Prec prec;

  // ---- Epoch 1 (N = 3): full LDLT build, then a q=2 W-row-addition SMW delta ----
  const Eigen::MatrixXd G1a_dense = A_row;
  const SpMat G1a = DenseToSparse(G1a_dense);
  const SpMat G1a_tr = DenseToSparse(G1a_dense.transpose());
  const std::vector<bool> active_k1 = {true, true, true};
  const BoolArr active_K1 = ToBoolArr(active_k1);
  const BoolArr active_W1a = ToBoolArr({false, false});

  prec.arm(G1a, G1a_tr, H_diag, active_K1, active_W1a, B_rm, mu, rho, true, true, /*use_ldlt=*/true);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  Eigen::MatrixXd G1b_dense(3, 3);
  G1b_dense.row(0) = A_row.row(0);
  G1b_dense.bottomRows(2) = B_rows;
  const SpMat G1b = DenseToSparse(G1b_dense);
  const SpMat G1b_tr = DenseToSparse(G1b_dense.transpose());
  const BoolArr active_W1b = ToBoolArr({true, true});

  prec.arm(G1b, G1b_tr, H_diag, active_K1, active_W1b, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, /*use_ldlt=*/true);
  prec.compute(0);
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 2);

  Eigen::VectorXd b1(3);
  b1 << 1.0, -0.5, 2.0;
  const Eigen::VectorXd got1 = prec.solve(b1);
  const Eigen::MatrixXd P1 = DenseSchurComplement(G1b_dense, H_diag, active_k1, mu);
  EXPECT_TRUE(got1.isApprox(P1.colPivHouseholderQr().solve(b1), kTol));

  // ---- Epoch 2 (N = 5, a different problem): forced full LDLT rebuild, then a q=2 delta ----
  const Eigen::MatrixXd A_row2 = (Eigen::MatrixXd(1, 5) << 1.0, 1.0, 1.0, 1.0, 1.0).finished();
  const Eigen::MatrixXd B_rows2 =
      (Eigen::MatrixXd(2, 5) << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0).finished();
  const RowMajorSpMat B_rm2 = DenseToSparseRowMajor(B_rows2);
  const Eigen::VectorXd H_diag2 = (Eigen::VectorXd(5) << 1.0, 2.0, 3.0, 4.0, 5.0).finished();
  const double mu2 = 4.0;
  const double rho2 = 6.0;
  const std::vector<bool> active_k2 = {true, true, true, true, true};
  const BoolArr active_K2 = ToBoolArr(active_k2);
  const BoolArr active_W2a = ToBoolArr({false, false});

  const Eigen::MatrixXd G2a_dense = A_row2;
  const SpMat G2a = DenseToSparse(G2a_dense);
  const SpMat G2a_tr = DenseToSparse(G2a_dense.transpose());

  prec.arm(G2a, G2a_tr, H_diag2, active_K2, active_W2a, B_rm2, mu2, rho2, /*rebuild=*/true,
           /*prec_pattern_changed=*/true, /*use_ldlt=*/true, /*force_rebuild=*/true);
  prec.compute(0);
  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);

  Eigen::MatrixXd G2b_dense(3, 5);
  G2b_dense.row(0) = A_row2.row(0);
  G2b_dense.bottomRows(2) = B_rows2;
  const SpMat G2b = DenseToSparse(G2b_dense);
  const SpMat G2b_tr = DenseToSparse(G2b_dense.transpose());
  const BoolArr active_W2b = ToBoolArr({true, true});

  prec.arm(G2b, G2b_tr, H_diag2, active_K2, active_W2b, B_rm2, mu2, rho2, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, /*use_ldlt=*/true);
  prec.compute(0);
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_rank(), 2);

  Eigen::VectorXd b2(3);
  b2 << 0.5, 1.5, -1.0;
  const Eigen::VectorXd got2 = prec.solve(b2);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2b_dense, H_diag2, active_k2, mu2);
  EXPECT_TRUE(got2.isApprox(P2.colPivHouseholderQr().solve(b2), kTol));
}

// ===================== consume_fact_count_delta() / should_retry_after_failure() =====================

TEST(ConsumeFactCountDelta, ConsumeFactCountDeltaReturnsBuildsSinceLastArmAndResetsBaseline) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  EXPECT_EQ(prec.consume_fact_count_delta(), 1);
  // No new arm()/build() since the last sample: delta is 0, and it stays 0 on repeated calls.
  EXPECT_EQ(prec.consume_fact_count_delta(), 0);
  EXPECT_EQ(prec.consume_fact_count_delta(), 0);

  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/true, false, /*force_rebuild=*/true);
  prec.compute(0);
  EXPECT_EQ(prec.consume_fact_count_delta(), 1);
}

TEST(ShouldRetryAfterFailure, ShouldRetryAfterFailureReturnsFalseWhenLastBuildWasFullRebuild) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_FALSE(prec.used_smw());

  EXPECT_FALSE(prec.should_retry_after_failure());
  EXPECT_FALSE(prec.smw_suppressed());  // no failure was recorded, since used_smw() was false
}

TEST(ShouldRetryAfterFailure, ShouldRetryAfterFailureReturnsTrueAndRecordsFailureWhenLastBuildUsedSmw) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_K1 = ToBoolArr({true, true, true});
  const BoolArr active_K2 = ToBoolArr({true, true, false});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);
  ASSERT_TRUE(prec.used_smw());

  // Each call both reports true (retry warranted) and records a failure via record_smw_rebuild();
  // five in a row should reach the suppression threshold exactly like calling it directly.
  for (int i = 0; i < 5; ++i) EXPECT_TRUE(prec.should_retry_after_failure());
  EXPECT_TRUE(prec.smw_suppressed());
}

// ===================== SMW: cumulative multi-step updates & threshold boundaries =====================

TEST(SmwCumulativeUpdates, ConsecutiveSmwUpdatesAccumulateRankAgainstOriginalSnapshot) {
  // Successive successful SMW calls all diff against the snapshot from the last full rebuild, not
  // against each other -- so smw_last_rank() after N consecutive single-K-flip SMW calls reads N
  // (cumulative), and every intermediate solve() must still match dense ground truth for the
  // current active set even though the cached factorization is still the original one.
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});  // s stays 1 throughout
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  const BoolArr active_K0 = ToBoolArr({true, true, true});
  prec.arm(G, G_tr, f.H_diag, active_K0, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  Eigen::VectorXd b(1);
  b << 2.5;

  // Step 1: flip column 2 off. Cumulative rank vs. the original (all-active) snapshot = 1.
  const std::vector<bool> k1 = {true, true, false};
  const BoolArr active_K1 = ToBoolArr(k1);
  prec.arm(G, G_tr, f.H_diag, active_K1, active_W, B_rm, f.mu, f.rho, true, false, false);
  prec.compute(0);
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 1);
  EXPECT_TRUE(prec.solve(b).isApprox(
      DenseSchurComplement(G_dense, f.H_diag, k1, f.mu).colPivHouseholderQr().solve(b), kTol));

  // Step 2: also flip column 1 off. Cumulative rank vs. the original snapshot = 2 (not 1 again).
  const std::vector<bool> k2 = {true, false, false};
  const BoolArr active_K2 = ToBoolArr(k2);
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, true, false, false);
  prec.compute(0);
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 2);
  EXPECT_TRUE(prec.solve(b).isApprox(
      DenseSchurComplement(G_dense, f.H_diag, k2, f.mu).colPivHouseholderQr().solve(b), kTol));

  // Step 3: also flip column 0 off (every column now inactive). Cumulative rank = 3.
  const std::vector<bool> k3 = {false, false, false};
  const BoolArr active_K3 = ToBoolArr(k3);
  prec.arm(G, G_tr, f.H_diag, active_K3, active_W, B_rm, f.mu, f.rho, true, false, false);
  prec.compute(0);
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 3);
  EXPECT_TRUE(prec.solve(b).isApprox(
      DenseSchurComplement(G_dense, f.H_diag, k3, f.mu).colPivHouseholderQr().solve(b), kTol));
}

TEST(SmwCumulativeUpdates, SmwSucceedsWhenDeltaRankExactlyEqualsThreshold) {
  // Companion to FallsBackToFullRebuildWhenDeltaRankExceedsThreshold: 
  // the boundary case (rank == kSmwRankThreshold, 50) must still succeed via SMW -- only rank > threshold rejects.
  const int N = 50;
  Eigen::MatrixXd A_row = Eigen::MatrixXd::Ones(1, N);
  Eigen::MatrixXd B_rows = Eigen::MatrixXd::Identity(N, N);
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag = Eigen::VectorXd::LinSpaced(N, 2.0, 2.0 + N - 1);
  const double mu = 5.0;
  const double rho = 3.0;
  const BoolArr active_K = ToBoolArr(std::vector<bool>(N, true));
  const BoolArr active_W1 = ToBoolArr(std::vector<bool>(N, false));
  const BoolArr active_W2 = ToBoolArr(std::vector<bool>(N, true));

  const SpMat G1 = DenseToSparse(A_row);
  const SpMat G1_tr = DenseToSparse(A_row.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, H_diag, active_K, active_W1, B_rm, mu, rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  Eigen::MatrixXd G2_dense(N + 1, N);
  G2_dense.row(0) = A_row.row(0);
  G2_dense.bottomRows(N) = B_rows;
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, H_diag, active_K, active_W2, B_rm, mu, rho,
           /*rebuild=*/true, /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), N);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::None);

  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(N + 1, 1.0, 2.0);
  const Eigen::VectorXd got = prec.solve(b);
  const std::vector<bool> active_k_std(N, true);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, H_diag, active_k_std, mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), kTol));
}

TEST(SmwCumulativeUpdates, SmwRejectsZeroRankDeltaWithExplicitReason) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Identical active_K/active_W/G, but rebuild=true forces build() to run try_build_smw() anyway:
  // h=p=q=0 so the rank==0 sub-case of RankZeroOrExceedsThreshold fires specifically.
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_rank(), 0);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::RankZeroOrExceedsThreshold);
}

TEST(SmwCumulativeUpdates, SmwRejectsWithNoSnapshotAfterFailStreakClearsSnapshot) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_K1 = ToBoolArr({true, true, true});
  const BoolArr active_K2 = ToBoolArr({true, true, false});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  for (int i = 0; i < 5; ++i) prec.record_smw_rebuild();
  ASSERT_TRUE(prec.smw_suppressed());
  prec.reset_smw_fail_streak();
  ASSERT_FALSE(prec.smw_suppressed());

  // When the fail-streak hit its threshold, record_smw_rebuild() cleared G_old_.
  // So although has_snapshot_ is still true even with the streak reset (only release() clears that),
  // (!has_snapshot_ || G_old_.rows() == 0) hits its second half and rejects the SMW update.
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::NoSnapshot);
}

TEST(SmwCumulativeUpdates, SmwSuppressedExactlyAtFailStreakThresholdNotBelow) {
  Prec prec;
  for (int i = 0; i < 4; ++i) {
    prec.record_smw_rebuild();
    EXPECT_FALSE(prec.smw_suppressed());
  }
  prec.record_smw_rebuild();  // 5th call reaches kMaxSmwFailStreak.
  EXPECT_TRUE(prec.smw_suppressed());
}

// ===================== degenerate active-set configurations =====================

TEST(DegenerateActiveSet, SolveMatchesDenseWhenAllKInactiveCholesky) {
  Fixture f;
  const std::vector<bool> active_k = {false, false, false};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/false);
  prec.compute(0);
  ASSERT_EQ(prec.info(), Eigen::Success);

  Eigen::VectorXd b(1);
  b << 3.0;
  const Eigen::VectorXd got = prec.solve(b);
  // active_K all false: P reduces to exactly (1/mu) I.
  const Eigen::MatrixXd P = DenseSchurComplement(G_dense, f.H_diag, active_k, f.mu);
  EXPECT_TRUE(got.isApprox(P.colPivHouseholderQr().solve(b), kTol));
}

TEST(DegenerateActiveSet, SolveMatchesDenseWhenAllKInactiveLdlt) {
  Fixture f;
  const std::vector<bool> active_k = {false, false, false};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, /*use_ldlt=*/true);
  prec.compute(0);
  ASSERT_EQ(prec.info(), Eigen::Success);

  Eigen::VectorXd b(1);
  b << 3.0;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P = DenseSchurComplement(G_dense, f.H_diag, active_k, f.mu);
  EXPECT_TRUE(got.isApprox(P.colPivHouseholderQr().solve(b), kTol));
}

TEST(DegenerateActiveSet, SmwHandlesMultipleSimultaneousKFlipsInOneDelta) {
  Fixture f;
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_k1 = ToBoolArr({true, true, true});

  Prec prec;
  prec.arm(G, G_tr, f.H_diag, active_k1, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Flip columns 1 and 2 off simultaneously in a single delta (p = 2),
  // as opposed to two separate single-flip SMW calls.
  const std::vector<bool> active_k2 = {true, false, false};
  const BoolArr active_K2 = ToBoolArr(active_k2);
  prec.arm(G, G_tr, f.H_diag, active_K2, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 2);

  Eigen::VectorXd b(1);
  b << 2.5;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G_dense, f.H_diag, active_k2, f.mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), kTol));
}

TEST(DegenerateActiveSet, SmwHandlesDeletingAllActiveWRowsAtOnce) {
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const BoolArr active_K = ToBoolArr(active_k);
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_W1 = ToBoolArr({true, true});
  const BoolArr active_W2 = ToBoolArr({false, false});

  const Eigen::MatrixXd G1_dense = f.StackG({true, true});
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, f.H_diag, active_K, active_W1, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Both currently-active W rows deactivated at once (h = 2):
  //  s drops back down to just the equality row (s_new == M_rows_).
  const Eigen::MatrixXd G2_dense = f.StackG({false, false});
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, f.H_diag, active_K, active_W2, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 2);

  Eigen::VectorXd b(1);
  b << 1.5;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, f.H_diag, active_k, f.mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), kTol));
}

TEST(DegenerateActiveSet, SmwHandlesSimultaneousKFlipAndWRowAddAndDelete) {
  Fixture f;
  const BoolArr active_K1 = ToBoolArr({true, true, true});
  const std::vector<bool> active_k2 = {true, true, false};
  const RowMajorSpMat B_rm = f.B_rm();
  const BoolArr active_W1 = ToBoolArr({true, false});
  const BoolArr active_W2 = ToBoolArr({false, true});

  const Eigen::MatrixXd G1_dense = f.StackG({true, false});
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, f.H_diag, active_K1, active_W1, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // All three delta types at once: row 0 deactivated (h=1), row 1 activated (q=1), column 2
  // flipped off (p=1) -- h>0 && p>0 && q>0 simultaneously, not tested individually elsewhere.
  const Eigen::MatrixXd G2_dense = f.StackG({false, true});
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  const BoolArr active_K2 = ToBoolArr(active_k2);
  prec.arm(G2, G2_tr, f.H_diag, active_K2, active_W2, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 3);

  Eigen::VectorXd b(2);
  b << 1.0, -1.5;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, f.H_diag, active_k2, f.mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), kTol));
}

TEST(DegenerateActiveSet, SmwFallsBackWhenCapacitanceMatrixIsSingular) {
  // Two candidate W rows with identical coefficients, both newly activated in the same delta (q=2):
  // their V_plus_ columns are identical, so the capacitance matrix's added-row block is
  // exactly rank-1 up to the (1/mu) regularization -- with mu large enough that regularization is
  // negligible relative to the sqrt(eps) rank-detection threshold, the block is detected as singular
  // and try_build_smw() falls back to a full rebuild.
  Eigen::MatrixXd A_row(1, 3);
  A_row << 1.0, 1.0, 1.0;
  Eigen::MatrixXd B_rows(2, 3);
  B_rows << 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0;  // both candidate W rows are identical
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 1e10;  // 1/mu negligible relative to the sqrt(eps) rank-detection threshold
  const double rho = 3.0;
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W1 = ToBoolArr({false, false});
  const BoolArr active_W2 = ToBoolArr({true, true});

  const Eigen::MatrixXd G1_dense = A_row;
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, H_diag, active_K, active_W1, B_rm, mu, rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  Eigen::MatrixXd G2_dense(3, 3);
  G2_dense.row(0) = A_row.row(0);
  G2_dense.bottomRows(2) = B_rows;
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, H_diag, active_K, active_W2, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::SingularCapacitance);
}

TEST(DegenerateActiveSet, SmwSucceedsFromLegitimatelyEmptyZeroByZeroSnapshot) {
  // M_rows = 0 (a 0x3 "A"): no equality constraints. Two candidate W rows, both inactive at the
  // first build, so G is a genuine 0x0 matrix -- not a snapshot wiped by record_smw_rebuild()'s
  // fail-streak path. A subsequent W-row activation must still be able to use SMW against this
  // legitimately-empty snapshot instead of being falsely rejected as NoSnapshot.
  Eigen::MatrixXd B_rows(2, 3);
  B_rows << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0;
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 5.0;
  const double rho = 3.0;
  const std::vector<bool> active_k = {true, true, true};
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W1 = ToBoolArr({false, false});

  const Eigen::MatrixXd G1_dense(0, 3);
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, H_diag, active_K, active_W1, B_rm, mu, rho, /*rebuild=*/true, /*prec_pattern_changed=*/true,
           /*use_ldlt=*/false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);
  ASSERT_EQ(prec.info(), Eigen::Success);

  // Activate one W row: G grows from 0x3 to 1x3. This arm() call also lets set_data() finally
  // pin down M_rows_ = 1 - 1 = 0 (G.rows() was 0 on the first call, so the auto-detect skipped it).
  const BoolArr active_W2 = ToBoolArr({true, false});
  const Eigen::MatrixXd G2_dense = B_rows.row(0);
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, H_diag, active_K, active_W2, B_rm, mu, rho, /*rebuild=*/true, /*prec_pattern_changed=*/false,
           /*use_ldlt=*/false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_NE(prec.smw_last_reject_reason(), Prec::SmwRejectReason::NoSnapshot);
  EXPECT_EQ(prec.fact_count(), 1);  // still just the one full factorization; the update was via SMW

  Eigen::VectorXd b(1);
  b << 4.0;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, H_diag, active_k, mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), kTol));
}

// ===================== public-API edge cases: alternate constructor, release() =====================

TEST(PublicApiEdgeCases, MissingDataReasonWhenSetDataNeverCalled) {
  // The 5-arg constructor sets G_/G_tr_/H_diag_/active_K_/mu_ directly, bypassing set_data()/
  // arm() entirely -- so active_W_/B_rm_ stay null and M_rows_ stays -1 forever. A full
  // factorization only needs G_/H_diag_/active_K_/mu_, so this is a legitimate way to use
  // the class purely as a direct Cholesky/LDLT preconditioner with SMW permanently disabled.
  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);

  Prec prec(G, G_tr, f.H_diag, active_K, f.mu);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::None);  // never attempted yet

  Eigen::VectorXd b(1);
  b << 3.0;
  const Eigen::MatrixXd P = DenseSchurComplement(G_dense, f.H_diag, active_k, f.mu);
  EXPECT_TRUE(prec.solve(b).isApprox(P.colPivHouseholderQr().solve(b), kTol));

  // rebuild_ is now correctly cleared after a successful factorization, so a second compute()
  // call with nothing changed is a no-op -- no redundant refactorization.
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::None);
  EXPECT_TRUE(prec.solve(b).isApprox(P.colPivHouseholderQr().solve(b), kTol));
}

TEST(PublicApiEdgeCases, ReleaseIsSafeOnFreshObjectAndIdempotent) {
  Prec prec;
  prec.release();  // release() before ever arm()/compute() must not crash.
  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 0);

  Fixture f;
  const std::vector<bool> active_k = {true, true, true};
  const Eigen::MatrixXd G_dense = f.StackG({false, false});
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W = ToBoolArr({false, false});
  const RowMajorSpMat B_rm = f.B_rm();

  prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true, false);
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 1);

  Eigen::VectorXd b(1);
  b << 3.0;
  const Eigen::MatrixXd P = DenseSchurComplement(G_dense, f.H_diag, active_k, f.mu);
  EXPECT_TRUE(prec.solve(b).isApprox(P.colPivHouseholderQr().solve(b), kTol));

  prec.release();
  prec.release();  // calling release() twice in a row must also not crash.
  EXPECT_EQ(prec.fact_count(), 1);  // release() doesn't reset fact_count_ (cumulative counter)
}

// ===================== scratch-buffer leakage (state-poisoning) =====================
// try_build_smw() reuses several private scratch buffers (smw_tmp_, smw_ldlt_padded_, Y_all_,
// r_pad_) across calls via zero_resize(), which only re-zeroes a buffer when its *size*
// changes. When two consecutive SMW updates have identical rank/shape (so every buffer is
// reused at the same size), correctness depends entirely on the class's own logic overwriting
// every entry that matters -- not on zero_resize() clearing anything. These tests deliberately
// poison the buffers with NaN/Inf between two same-shape SMW calls (simulating reused/garbage
// heap memory) and prove the second call's result is still correct.

namespace {

// N = 4 primal columns; 1 "always active" equality row; 3 candidate W rows (one variable each).
struct LeakageFixture {
  Eigen::MatrixXd A_row = (Eigen::MatrixXd(1, 4) << 1.0, 1.0, 1.0, 1.0).finished();
  Eigen::MatrixXd B_rows =
      (Eigen::MatrixXd(3, 4) << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
          .finished();
  Eigen::VectorXd H_diag = (Eigen::VectorXd(4) << 2.0, 3.0, 4.0, 5.0).finished();
  double mu = 5.0;
  double rho = 3.0;  // distinct from mu so a mu/rho argument swap would be caught by assertions

  RowMajorSpMat B_rm() const { return DenseToSparseRowMajor(B_rows); }

  Eigen::MatrixXd StackG(const std::vector<bool>& active_w) const {
    std::vector<Eigen::MatrixXd> rows{A_row};
    for (std::size_t i = 0; i < active_w.size(); ++i)
      if (active_w[i]) rows.push_back(B_rows.row(static_cast<int>(i)));
    Eigen::MatrixXd G(rows.size(), 4);
    for (std::size_t i = 0; i < rows.size(); ++i) G.row(static_cast<int>(i)) = rows[i];
    return G;
  }
};

// Y_all_ is resized (not zero-filled) then written column-by-column across its *entire* column
// range every call, and r_pad_/smw_ldlt_padded_ are cleared via an unconditional setZero() (not
// zero_resize()) every call -- so all three are expected to come out clean regardless of prior
// content, even when zero_resize() itself would have no-opped on a same-size reuse.
void PoisonFullyOverwrittenBuffers(Prec& prec, bool use_ldlt) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  Prec::Mat& y_all = SchurPreconditionerTestPeer::Y_all(prec);
  y_all.setConstant(inf);

  Prec::Vec& rpad = SchurPreconditionerTestPeer::r_pad(prec);
  rpad.setConstant(nan);

  if (use_ldlt) {
    Prec::Vec& padded = SchurPreconditionerTestPeer::smw_ldlt_padded(prec);
    for (int i = 0; i < padded.size(); ++i) padded(i) = (i % 2 == 0) ? inf : nan;
  }
}

// smw_tmp_ is different: zero_resize() no-ops on a same-size reuse, and compute_y_all() only
// ever writes/clears the single (or few) entries it touches per basis vector -- every other
// entry is assumed to already be zero, an invariant maintained purely by the *previous* call
// having correctly reset its own touched entries, not by any defensive clear on entry. Poisoning
// the whole vector (as "reused garbage heap memory" would look) violates that assumption.
void PoisonSmwTmpScratch(Prec& prec) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  Prec::Vec& tmp = SchurPreconditionerTestPeer::smw_tmp(prec);
  for (int i = 0; i < tmp.size(); ++i) tmp(i) = (i % 2 == 0) ? nan : inf;
}

// Ingredients for delta #2: same rank/shape as delta #1 (h=1, p=1), but deactivating the *other*
// W row and flipping a *different* K column, still measured against the Epoch A snapshot (an
// SMW-only update never calls snapshot_state()).
struct LeakageDelta2 {
  std::vector<bool> active_k_d2 = {true, true, true, false};
  BoolArr active_K_D2 = ToBoolArr(active_k_d2);
  BoolArr active_W_D2 = ToBoolArr({true, false, false});
  Eigen::MatrixXd G_D2_dense;
  SpMat G_D2, G_D2_tr;

  explicit LeakageDelta2(const LeakageFixture& f) {
    G_D2_dense = f.StackG({true, false, false});
    G_D2 = DenseToSparse(G_D2_dense);
    G_D2_tr = DenseToSparse(G_D2_dense.transpose());
  }
};

// Drives `prec` through Epoch A (full rebuild) then delta #1 (a rank-2 SMW update), verified
// against dense ground truth. `prec` is not copyable/movable (it holds a non-relocatable
// std::variant of Eigen factorizations), so it's taken and left armed by reference; `f`/`B_rm`
// must outlive `prec` (arm() stores pointers, though not across this call -- delta #1's own G/
// active_K/active_W locals are re-pointed-away-from by the next arm() call before anything would
// dereference them again).
void ArmThroughEpochAAndDelta1(Prec& prec, const LeakageFixture& f, const RowMajorSpMat& B_rm,
                                bool use_ldlt) {
  // Epoch A: full rebuild. All K active; W rows 0 and 1 active, row 2 inactive.
  const BoolArr active_K_A = ToBoolArr({true, true, true, true});
  const BoolArr active_W_A = ToBoolArr({true, true, false});
  const Eigen::MatrixXd G_A_dense = f.StackG({true, true, false});
  const SpMat G_A = DenseToSparse(G_A_dense);
  const SpMat G_A_tr = DenseToSparse(G_A_dense.transpose());

  prec.arm(G_A, G_A_tr, f.H_diag, active_K_A, active_W_A, B_rm, f.mu, f.rho, true, true, use_ldlt);
  prec.compute(0);
  EXPECT_EQ(prec.fact_count(), 1);

  // Delta #1 (vs. Epoch A snapshot): deactivate W row 0 (h=1) and flip K column 2 off (p=1).
  const std::vector<bool> active_k_d1 = {true, true, false, true};
  const BoolArr active_K_D1 = ToBoolArr(active_k_d1);
  const BoolArr active_W_D1 = ToBoolArr({false, true, false});
  const Eigen::MatrixXd G_D1_dense = f.StackG({false, true, false});
  const SpMat G_D1 = DenseToSparse(G_D1_dense);
  const SpMat G_D1_tr = DenseToSparse(G_D1_dense.transpose());

  prec.arm(G_D1, G_D1_tr, f.H_diag, active_K_D1, active_W_D1, B_rm, f.mu, f.rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, use_ldlt);
  prec.compute(0);
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 2);

  Eigen::VectorXd b1(2);
  b1 << 1.0, -0.5;
  const Eigen::VectorXd got1 = prec.solve(b1);
  const Eigen::MatrixXd P1 = DenseSchurComplement(G_D1_dense, f.H_diag, active_k_d1, f.mu);
  EXPECT_TRUE(got1.isApprox(P1.colPivHouseholderQr().solve(b1), kTol));
}

}  // namespace

// Poisons only the buffers that are unconditionally fully overwritten every call (Y_all_,
// r_pad_, smw_ldlt_padded_). Delta #2 is expected to complete cleanly via SMW, proving those
// buffers' correctness never depended on zero_resize()'s size-triggered clear.
static void RunFullyOverwrittenBufferScenario(bool use_ldlt) {
  LeakageFixture f;
  const RowMajorSpMat B_rm = f.B_rm();
  Prec prec;
  ArmThroughEpochAAndDelta1(prec, f, B_rm, use_ldlt);
  PoisonFullyOverwrittenBuffers(prec, use_ldlt);

  const LeakageDelta2 d2(f);
  prec.arm(d2.G_D2, d2.G_D2_tr, f.H_diag, d2.active_K_D2, d2.active_W_D2, B_rm, f.mu, f.rho,
           /*rebuild=*/true, /*prec_pattern_changed=*/false, use_ldlt);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);  // still no full refactorization
  EXPECT_EQ(prec.smw_last_rank(), 2);

  Eigen::VectorXd b2(2);
  b2 << 0.5, 1.5;
  const Eigen::VectorXd got2 = prec.solve(b2);
  ASSERT_TRUE(got2.allFinite()) << "solve() returned non-finite values -- a poisoned scratch "
                                    "buffer leaked into the result instead of being overwritten.";
  const Eigen::MatrixXd P2 = DenseSchurComplement(d2.G_D2_dense, f.H_diag, d2.active_k_d2, f.mu);
  EXPECT_TRUE(got2.isApprox(P2.colPivHouseholderQr().solve(b2), kTol));
}

TEST(ScratchBufferLeakage, PoisonedBuffersFullyOverwrittenBeforeNextSmwCallCholesky) {
  RunFullyOverwrittenBufferScenario(/*use_ldlt=*/false);
}

TEST(ScratchBufferLeakage, PoisonedBuffersFullyOverwrittenBeforeNextSmwCallLdlt) {
  RunFullyOverwrittenBufferScenario(/*use_ldlt=*/true);
}

// Poisons smw_tmp_ too (all four buffers named in the audit, matching "reused/garbage heap
// memory"). Finding: unlike the three buffers above, smw_tmp_ has no defensive re-zeroing on
// entry to compute_y_all() -- it relies entirely on the *previous* call having restored its
// touched entries to zero. compute_y_all() now asserts this invariant on entry (see
// schur_preconditioner.hpp), so in a build with assertions enabled the violation aborts loudly
// instead of falling through. This project's default CMakeLists.txt configure (CMAKE_BUILD_TYPE
// defaults to Release, i.e. -DNDEBUG) compiles that assert out, in which case the corrupted
// entries currently propagate into the capacitance matrix and (via NaN contaminating
// FullPivLU's rank computation) get flagged as SingularCapacitance, so try_build_smw() safely
// falls back to a full rebuild -- an accidental side effect of NaN/rank interaction, not a
// designed safeguard. Either way, no corrupted state may reach the caller undetected, so this
// test checks whichever of the two is actually compiled in.
static void RunSmwTmpPoisoningScenario(bool use_ldlt) {
  LeakageFixture f;
  const RowMajorSpMat B_rm = f.B_rm();
  Prec prec;
  ArmThroughEpochAAndDelta1(prec, f, B_rm, use_ldlt);
  PoisonFullyOverwrittenBuffers(prec, use_ldlt);
  PoisonSmwTmpScratch(prec);

  const LeakageDelta2 d2(f);
  auto trigger_delta2 = [&] {
    prec.arm(d2.G_D2, d2.G_D2_tr, f.H_diag, d2.active_K_D2, d2.active_W_D2, B_rm, f.mu, f.rho,
             /*rebuild=*/true, /*prec_pattern_changed=*/false, use_ldlt);
    prec.compute(0);
  };

#ifdef NDEBUG
  trigger_delta2();
  // classify_active_set_delta() itself never touches smw_tmp_, so the rank classification is
  // unaffected by the poisoning even when the later capacitance stage is.
  EXPECT_EQ(prec.smw_last_rank(), 2);

  Eigen::VectorXd b2(2);
  b2 << 0.5, 1.5;
  const Eigen::VectorXd got2 = prec.solve(b2);
  ASSERT_TRUE(got2.allFinite()) << "solve() returned non-finite values -- poisoned smw_tmp_ "
                                    "state leaked all the way through to the caller.";
  const Eigen::MatrixXd P2 = DenseSchurComplement(d2.G_D2_dense, f.H_diag, d2.active_k_d2, f.mu);
  EXPECT_TRUE(got2.isApprox(P2.colPivHouseholderQr().solve(b2), kTol));
#else
  EXPECT_DEATH(trigger_delta2(), "smw_tmp_ zero-invariant violated");
#endif
}

TEST(ScratchBufferLeakage, PoisonedSmwTmpNeverLeaksIncorrectResultsCholesky) {
  RunSmwTmpPoisoningScenario(/*use_ldlt=*/false);
}

TEST(ScratchBufferLeakage, PoisonedSmwTmpNeverLeaksIncorrectResultsLdlt) {
  RunSmwTmpPoisoningScenario(/*use_ldlt=*/true);
}

// ===================== snapshot desynchronization (rapid active-set oscillation) =====================
// classify_active_set_delta() always diffs the *current* active_K/active_W against
// active_K_old_/active_W_old_ (the last full-rebuild snapshot), never against the previous
// SMW call's state. These tests rapidly flip active_K/active_W back and forth -- including
// exact returns to the snapshot's own state -- to prove that classification never drifts.

TEST(SnapshotDesync, RapidActiveSetOscillationNeverDriftsFromSnapshotClassification) {
  Fixture f;
  const RowMajorSpMat B_rm = f.B_rm();

  const std::vector<bool> k_base = {true, true, true};
  const std::vector<bool> w_base = {false, false};
  const BoolArr active_K_base = ToBoolArr(k_base);
  const BoolArr active_W_base = ToBoolArr(w_base);
  const Eigen::MatrixXd G_base_dense = f.StackG(w_base);
  const SpMat G_base = DenseToSparse(G_base_dense);
  const SpMat G_base_tr = DenseToSparse(G_base_dense.transpose());

  Prec prec;
  prec.arm(G_base, G_base_tr, f.H_diag, active_K_base, active_W_base, B_rm, f.mu, f.rho, true, true,
           /*use_ldlt=*/false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  auto step = [&](const std::vector<bool>& k, const std::vector<bool>& w) {
    const BoolArr active_K = ToBoolArr(k);
    const BoolArr active_W = ToBoolArr(w);
    const Eigen::MatrixXd G_dense = f.StackG(w);
    const SpMat G = DenseToSparse(G_dense);
    const SpMat G_tr = DenseToSparse(G_dense.transpose());
    prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
             /*prec_pattern_changed=*/false, /*use_ldlt=*/false);
    prec.compute(0);
    const Eigen::VectorXd b =
        Eigen::VectorXd::LinSpaced(static_cast<int>(G_dense.rows()), 1.0, 2.0);
    const Eigen::VectorXd got = prec.solve(b);
    const Eigen::MatrixXd P = DenseSchurComplement(G_dense, f.H_diag, k, f.mu);
    EXPECT_TRUE(got.isApprox(P.colPivHouseholderQr().solve(b), kTol));
  };

  // Step 1: flip K col 2 off. rank=1 vs. the snapshot.
  step({true, true, false}, {false, false});
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 1);

  // Step 2: revert K col 2 -- exactly back to the snapshot's own state. Must be rejected as
  // rank 0 (not silently treated as "still rank 1" or some other stale value) and force a
  // fresh full rebuild, which becomes the new snapshot.
  step(k_base, w_base);
  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_rank(), 0);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::RankZeroOrExceedsThreshold);

  // Step 3: flip a *different* K column (col 1) off. rank=1 vs. the new snapshot.
  step({true, false, true}, {false, false});
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_rank(), 1);

  // Step 4: revert col 1 and activate W row 0 in the same call. Net delta vs. the snapshot is
  // just the W row (K matches the snapshot again): rank=1.
  step({true, true, true}, {true, false});
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_rank(), 1);

  // Step 5: deactivate W row 0 again (back to matching the snapshot) and flip K col 0 off in
  // the same call. rank=1 (only the K flip contributes; the W row is back to its snapshot
  // state).
  step({false, true, true}, {false, false});
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_rank(), 1);

  // Step 6: revert everything -- exactly back to the (second) snapshot's state again. Must be
  // rejected as rank 0 a second time, proving the zero-delta detection doesn't "wear out" or
  // drift after repeated oscillation.
  step(k_base, w_base);
  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 3);
  EXPECT_EQ(prec.smw_last_rank(), 0);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::RankZeroOrExceedsThreshold);
}

TEST(SnapshotDesync, LongDeterministicOscillationMatchesIndependentlyComputedRankAtEveryStep) {
  Fixture f;
  const RowMajorSpMat B_rm = f.B_rm();

  // Each state is (K0, K1, K2, W0, W1). Includes a revisited non-snapshot state (s1 == s3) and
  // two exact returns to the live snapshot (s8 == s0, s14 == s8), each of which must force a
  // full rebuild (rank 0) rather than silently drifting.
  struct State {
    bool k0, k1, k2, w0, w1;
  };
  const std::vector<State> states = {
      {true, true, true, false, false},    // s0:  baseline / Epoch A full build
      {true, true, false, false, false},   // s1
      {true, false, false, false, false},  // s2
      {true, true, false, false, false},   // s3  == s1 (revisit)
      {true, true, false, true, false},    // s4
      {true, true, false, true, true},     // s5
      {true, true, true, true, true},      // s6
      {true, true, true, false, true},     // s7
      {true, true, true, false, false},    // s8  == s0 (exact snapshot revisit)
      {false, true, true, false, false},   // s9
      {false, false, true, false, false},  // s10
      {false, false, true, false, true},   // s11
      {true, false, true, false, true},    // s12
      {true, true, true, false, true},     // s13
      {true, true, true, false, false},    // s14 == s8 (exact snapshot revisit)
      {true, false, true, true, false},    // s15
  };

  State snapshot = states[0];
  int last_fact_count = 0;

  Prec prec;
  for (std::size_t i = 0; i < states.size(); ++i) {
    const State& s = states[i];
    const std::vector<bool> active_k = {s.k0, s.k1, s.k2};
    const std::vector<bool> active_w = {s.w0, s.w1};
    const BoolArr active_K = ToBoolArr(active_k);
    const BoolArr active_W = ToBoolArr(active_w);
    const Eigen::MatrixXd G_dense = f.StackG(active_w);
    const SpMat G = DenseToSparse(G_dense);
    const SpMat G_tr = DenseToSparse(G_dense.transpose());

    const int expected_rank = (s.k0 != snapshot.k0) + (s.k1 != snapshot.k1) +
                               (s.k2 != snapshot.k2) + (s.w0 != snapshot.w0) +
                               (s.w1 != snapshot.w1);

    if (i == 0) {
      prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, true, true,
               /*use_ldlt=*/false);
    } else {
      prec.arm(G, G_tr, f.H_diag, active_K, active_W, B_rm, f.mu, f.rho, /*rebuild=*/true,
               /*prec_pattern_changed=*/false, /*use_ldlt=*/false);
    }
    prec.compute(0);

    if (i == 0 || expected_rank == 0) {
      // Baseline, or an exact return to the current snapshot: must force a full rebuild.
      EXPECT_FALSE(prec.used_smw()) << "step " << i;
      EXPECT_GT(prec.fact_count(), last_fact_count) << "step " << i;
      if (i > 0) {
        EXPECT_EQ(prec.smw_last_rank(), 0) << "step " << i;
        EXPECT_EQ(prec.smw_last_reject_reason(),
                  Prec::SmwRejectReason::RankZeroOrExceedsThreshold)
            << "step " << i;
      }
      snapshot = s;
    } else {
      EXPECT_TRUE(prec.used_smw()) << "step " << i;
      EXPECT_EQ(prec.fact_count(), last_fact_count) << "step " << i;
      EXPECT_EQ(prec.smw_last_rank(), expected_rank) << "step " << i;
    }
    last_fact_count = prec.fact_count();

    const Eigen::VectorXd b =
        Eigen::VectorXd::LinSpaced(static_cast<int>(G_dense.rows()), 1.0, 2.0);
    const Eigen::VectorXd got = prec.solve(b);
    const Eigen::MatrixXd P = DenseSchurComplement(G_dense, f.H_diag, active_k, f.mu);
    EXPECT_TRUE(got.isApprox(P.colPivHouseholderQr().solve(b), kTol)) << "step " << i;
  }
}

// ===================== zero-row Schur complement (0x0 P / trivial P_hat row-block) =====================
// s = G.rows() can legitimately be 0 (M_rows_ == 0 and no active W rows). solve_direct()'s
// LDLT .tail(0), finalize_smw_success()'s z_new_.resize(s_new) with s_new == 0, and
// solve_smw()'s empty-range head()/tail() assignments are all untested at s == 0 elsewhere in
// this file -- DegenerateActiveSet.SmwSucceedsFromLegitimatelyEmptyZeroByZeroSnapshot builds a
// 0-row G but never calls solve() on it directly, and never drives an *SMW update* down to
// exactly 0 rows (only up, from 0).

TEST(ZeroRowSchurComplement, DirectSolveOnZeroRowGCholesky) {
  Eigen::MatrixXd B_rows(2, 3);
  B_rows << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0;
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 5.0;
  const double rho = 3.0;
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W = ToBoolArr({false, false});

  const Eigen::MatrixXd G_dense(0, 3);
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());

  Prec prec;
  prec.arm(G, G_tr, H_diag, active_K, active_W, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/true, /*use_ldlt=*/false);
  prec.compute(0);

  EXPECT_EQ(prec.info(), Eigen::Success);
  EXPECT_EQ(prec.fact_count(), 1);

  const Eigen::VectorXd b(0);
  const Eigen::VectorXd got = prec.solve(b);
  EXPECT_EQ(got.size(), 0);
}

TEST(ZeroRowSchurComplement, DirectSolveOnZeroRowGLdlt) {
  // For LDLT, s=0 does NOT make P_hat empty -- it collapses to just the -H_act block (n_act x
  // n_act), since the G_act/(1/mu)I blocks vanish with s=0. This exercises factoring and
  // solving that non-trivial-but-zero-row-space P_hat.
  Eigen::MatrixXd B_rows(2, 3);
  B_rows << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0;
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 5.0;
  const double rho = 3.0;
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W = ToBoolArr({false, false});

  const Eigen::MatrixXd G_dense(0, 3);
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());

  Prec prec;
  prec.arm(G, G_tr, H_diag, active_K, active_W, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/true, /*use_ldlt=*/true);
  prec.compute(0);

  EXPECT_EQ(prec.info(), Eigen::Success);
  EXPECT_EQ(prec.fact_count(), 1);

  const Eigen::VectorXd b(0);
  const Eigen::VectorXd got = prec.solve(b);
  EXPECT_EQ(got.size(), 0);
}

TEST(ZeroRowSchurComplement, SmwDeletionLandsExactlyOnZeroRows) {
  // Epoch A: M_rows_ pins to 0 on this very first arm() call (G.rows()=1, active_W.count()=1).
  Eigen::MatrixXd B_rows(2, 3);
  B_rows << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0;
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 5.0;
  const double rho = 3.0;
  const BoolArr active_K = ToBoolArr({true, true, true});
  const BoolArr active_W1 = ToBoolArr({true, false});

  const Eigen::MatrixXd G1_dense = B_rows.row(0);
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, H_diag, active_K, active_W1, B_rm, mu, rho, true, true, /*use_ldlt=*/false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  // Deactivate the sole active W row (h=1): s_new = s_old(1) - h(1) + q(0) = 0. This is the
  // only path in the suite that drives finalize_smw_success()/solve_smw() to s_new == 0.
  const BoolArr active_W2 = ToBoolArr({false, false});
  const Eigen::MatrixXd G2_dense(0, 3);
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, H_diag, active_K, active_W2, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);  // no full refactorization
  EXPECT_EQ(prec.smw_last_rank(), 1);

  const Eigen::VectorXd b(0);
  const Eigen::VectorXd got = prec.solve(b);
  EXPECT_EQ(got.size(), 0);
}

TEST(ZeroRowSchurComplement, SmwCumulativeAdditionsFromZeroRowSnapshot) {
  // Companion to DegenerateActiveSet.SmwSucceedsFromLegitimatelyEmptyZeroByZeroSnapshot, taken
  // one step further: a *second* SMW addition still measured against the same s_old_==0
  // snapshot (an SMW-only update never calls snapshot_state()), so Y_all_/V_plus_ get built
  // with zero rows twice in a row, landing at a genuinely non-trivial s_new == 2.
  Eigen::MatrixXd B_rows(2, 3);
  B_rows << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0;
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 5.0;
  const double rho = 3.0;
  const std::vector<bool> active_k = {true, true, true};
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W0 = ToBoolArr({false, false});

  const Eigen::MatrixXd G0_dense(0, 3);
  const SpMat G0 = DenseToSparse(G0_dense);
  const SpMat G0_tr = DenseToSparse(G0_dense.transpose());

  Prec prec;
  prec.arm(G0, G0_tr, H_diag, active_K, active_W0, B_rm, mu, rho, true, true, /*use_ldlt=*/false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);
  ASSERT_EQ(prec.info(), Eigen::Success);

  // Step 1: activate row 0 (q=1 vs. the 0-row snapshot).
  const BoolArr active_W1 = ToBoolArr({true, false});
  const Eigen::MatrixXd G1_dense = B_rows.row(0);
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  prec.arm(G1, G1_tr, H_diag, active_K, active_W1, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);
  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 1);

  // Step 2: still measured against the *original* 0-row snapshot -- activate row 1 too,
  // cumulative q=2.
  const BoolArr active_W2 = ToBoolArr({true, true});
  const Eigen::MatrixXd G2_dense = B_rows;
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, H_diag, active_K, active_W2, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_rank(), 2);

  Eigen::VectorXd b(2);
  b << 1.5, -0.5;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, H_diag, active_k, mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), kTol));
}

// ===================== near-singular capacitance: realistic (non-exact) perturbations =====================
// Companion to DegenerateActiveSet.SmwFallsBackWhenCapacitanceMatrixIsSingular, which only
// covers exactly-identical rows. Real data is rarely exactly degenerate -- these check that the
// sqrt(eps)-thresholded rank detection in factorize_capacitance() behaves sensibly on
// realistic near-degenerate data: a perturbation far below the threshold should still read as
// singular, and one comfortably above it should not be a false-positive rejection.

TEST(NearSingularCapacitance, SubEpsilonRowPerturbationStillReadsAsSingular) {
  Eigen::MatrixXd A_row(1, 3);
  A_row << 1.0, 1.0, 1.0;
  // Row 1 differs from row 0 by 1e-15 in a *second* column (not a scalar rescale of the same
  // column -- scaling column 0 alone would keep the two rows exactly parallel, hence exactly
  // rank-1, for *any* scale factor, never actually testing the threshold). sqrt(eps) ~= 1.49e-8
  // for double, so 1e-15 is ~7 orders of magnitude below it.
  Eigen::MatrixXd B_rows(2, 3);
  B_rows << 1.0, 0.0,     0.0,
            1.0, 1e-15,   0.0;
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 1e10;  // 1/mu negligible relative to the sqrt(eps) rank-detection threshold
  const double rho = 3.0;
  const std::vector<bool> active_k = {true, true, true};
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W1 = ToBoolArr({false, false});
  const BoolArr active_W2 = ToBoolArr({true, true});

  const Eigen::MatrixXd G1_dense = A_row;
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, H_diag, active_K, active_W1, B_rm, mu, rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  Eigen::MatrixXd G2_dense(3, 3);
  G2_dense.row(0) = A_row.row(0);
  G2_dense.bottomRows(2) = B_rows;
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, H_diag, active_K, active_W2, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_FALSE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 2);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::SingularCapacitance);

  // The fallback full rebuild must still be correct, but not at the file's usual kTol=1e-9: with
  // mu=1e10 driving the capacitance threshold check, P2 itself (what the fallback directly
  // factorizes) is extremely ill-conditioned by construction (cond(P2) ~= 1.75e10, verified via
  // its singular values -- its smallest one is ~1e-10, coming straight from the 1/mu
  // regularization, independent of the row perturbation). Empirically the sparse-factorization
  // solve agrees with the dense QR oracle to a relative error of ~2.3e-7, consistent with that
  // conditioning amplifying ordinary double-precision rounding (~1e-16 * 1.75e10 ~= 1.75e-6,
  // same order of magnitude) -- not a bug. 1e-4 is a real correctness bound (three orders of
  // magnitude of headroom above the observed error) rather than a "doesn't crash" placeholder.
  Eigen::VectorXd b(3);
  b << 1.0, -1.0, 2.0;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, H_diag, active_k, mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), 1e-4));
}

TEST(NearSingularCapacitance, AboveThresholdRowPerturbationSucceedsViaSmw) {
  Eigen::MatrixXd A_row(1, 3);
  A_row << 1.0, 1.0, 1.0;
  // Same non-parallel perturbation shape as the sub-epsilon test above. Because the capacitance
  // matrix's determinant scales roughly as the square of this perturbation (the two W rows
  // enter symmetrically), the effective crossover empirically sits between 1e-4 (still
  // singular) and 1e-3 (full rank) here -- comfortably above the raw sqrt(eps) ~= 1.49e-8
  // threshold, but nowhere near as far above it as that raw number alone would suggest.
  Eigen::MatrixXd B_rows(2, 3);
  B_rows << 1.0, 0.0,   0.0,
            1.0, 1e-3,  0.0;
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B_rows);
  Eigen::VectorXd H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 1e10;
  const double rho = 3.0;
  const std::vector<bool> active_k = {true, true, true};
  const BoolArr active_K = ToBoolArr(active_k);
  const BoolArr active_W1 = ToBoolArr({false, false});
  const BoolArr active_W2 = ToBoolArr({true, true});

  const Eigen::MatrixXd G1_dense = A_row;
  const SpMat G1 = DenseToSparse(G1_dense);
  const SpMat G1_tr = DenseToSparse(G1_dense.transpose());

  Prec prec;
  prec.arm(G1, G1_tr, H_diag, active_K, active_W1, B_rm, mu, rho, true, true, false);
  prec.compute(0);
  ASSERT_EQ(prec.fact_count(), 1);

  Eigen::MatrixXd G2_dense(3, 3);
  G2_dense.row(0) = A_row.row(0);
  G2_dense.bottomRows(2) = B_rows;
  const SpMat G2 = DenseToSparse(G2_dense);
  const SpMat G2_tr = DenseToSparse(G2_dense.transpose());

  prec.arm(G2, G2_tr, H_diag, active_K, active_W2, B_rm, mu, rho, /*rebuild=*/true,
           /*prec_pattern_changed=*/false, false);
  prec.compute(0);

  EXPECT_TRUE(prec.used_smw());
  EXPECT_EQ(prec.fact_count(), 1);
  EXPECT_EQ(prec.smw_last_reject_reason(), Prec::SmwRejectReason::None);

  Eigen::VectorXd b(3);
  b << 1.0, -1.0, 2.0;
  const Eigen::VectorXd got = prec.solve(b);
  const Eigen::MatrixXd P2 = DenseSchurComplement(G2_dense, H_diag, active_k, mu);
  EXPECT_TRUE(got.isApprox(P2.colPivHouseholderQr().solve(b), kTol));
}

// ===================== randomized Cholesky/LDLT cross-check =====================
// DirectFactorization.LdltAndCholeskyPathsAgreeOnIdenticalData only exercises the tiny
// 3-column Fixture (s up to 2). A larger, seeded-random system stresses factorize_by_chol's
// G*E*G_tr assembly against factorize_by_ldlt's augmented P_hat assembly together, at a scale
// where a subtle indexing/sign bug in one path but not the other is more likely to surface.
// The seed is a fixed literal for full reproducibility across runs/platforms.

TEST(RandomizedCholLdltConsistency, FiftyColumnRandomSystemCholeskyAndLdltAgree) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::uniform_real_distribution<double> h_diag_dist(0.5, 5.0);
  std::uniform_real_distribution<double> prob(0.0, 1.0);
  std::uniform_real_distribution<double> rhs_dist(-10.0, 10.0);

  const int N = 50;      // primal columns
  const int M_rows = 8;  // equality rows
  const int l = 25;      // candidate W rows

  Eigen::MatrixXd A(M_rows, N);
  for (int i = 0; i < M_rows; ++i)
    for (int j = 0; j < N; ++j) A(i, j) = unit(rng);

  Eigen::MatrixXd B(l, N);
  for (int i = 0; i < l; ++i)
    for (int j = 0; j < N; ++j) B(i, j) = unit(rng);
  const RowMajorSpMat B_rm = DenseToSparseRowMajor(B);

  Eigen::VectorXd H_diag(N);
  for (int i = 0; i < N; ++i) H_diag(i) = h_diag_dist(rng);

  std::vector<bool> active_k_std(N);
  for (int i = 0; i < N; ++i) active_k_std[i] = prob(rng) < 0.7;
  const BoolArr active_K = ToBoolArr(active_k_std);

  std::vector<bool> active_w_std(l);
  for (int i = 0; i < l; ++i) active_w_std[i] = prob(rng) < 0.5;
  const BoolArr active_W = ToBoolArr(active_w_std);

  std::vector<Eigen::MatrixXd> rows;
  for (int i = 0; i < M_rows; ++i) rows.push_back(A.row(i));
  for (int i = 0; i < l; ++i)
    if (active_w_std[i]) rows.push_back(B.row(i));
  Eigen::MatrixXd G_dense(static_cast<int>(rows.size()), N);
  for (std::size_t i = 0; i < rows.size(); ++i) G_dense.row(static_cast<int>(i)) = rows[i];
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());

  const double mu = 2.0;
  const double rho = 3.0;

  Prec prec_chol;
  prec_chol.arm(G, G_tr, H_diag, active_K, active_W, B_rm, mu, rho, true, true, /*use_ldlt=*/false);
  prec_chol.compute(0);
  ASSERT_EQ(prec_chol.info(), Eigen::Success);

  Prec prec_ldlt;
  prec_ldlt.arm(G, G_tr, H_diag, active_K, active_W, B_rm, mu, rho, true, true, /*use_ldlt=*/true);
  prec_ldlt.compute(0);
  ASSERT_EQ(prec_ldlt.info(), Eigen::Success);

  const Eigen::MatrixXd P = DenseSchurComplement(G_dense, H_diag, active_k_std, mu);
  const int s = static_cast<int>(G_dense.rows());

  for (int trial = 0; trial < 3; ++trial) {
    Eigen::VectorXd b(s);
    for (int i = 0; i < s; ++i) b(i) = rhs_dist(rng);

    const Eigen::VectorXd got_chol = prec_chol.solve(b);
    const Eigen::VectorXd got_ldlt = prec_ldlt.solve(b);
    EXPECT_LE((got_chol - got_ldlt).lpNorm<Eigen::Infinity>(), 1e-10) << "trial " << trial;

    const Eigen::VectorXd expected = P.colPivHouseholderQr().solve(b);
    EXPECT_TRUE(got_chol.isApprox(expected, 1e-7)) << "chol trial " << trial;
    EXPECT_TRUE(got_ldlt.isApprox(expected, 1e-7)) << "ldlt trial " << trial;
  }
}
