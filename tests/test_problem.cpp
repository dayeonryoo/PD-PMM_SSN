#include "problem.hpp"

#include <gtest/gtest.h>

namespace {

KSPQPdata<double> MakeTinyKspqpData() {
  KSPQPdata<double> pd;
  pd.n = 2;
  pd.m = 1;
  pd.l = 1;

  pd.Q = Eigen::SparseMatrix<double>(2, 2);
  pd.Q.insert(0, 0) = 3.0;
  pd.Q.insert(1, 1) = 4.0;
  pd.Q.makeCompressed();

  pd.A = Eigen::SparseMatrix<double>(1, 2);
  pd.A.insert(0, 0) = 1.0;
  pd.A.insert(0, 1) = 1.0;
  pd.A.makeCompressed();

  pd.B = Eigen::SparseMatrix<double>(1, 2);
  pd.B.insert(0, 0) = 1.0;
  pd.B.makeCompressed();

  pd.c.resize(2);
  pd.c << 1.0, 2.0;
  pd.b.resize(1);
  pd.b << 5.0;
  pd.obj_const = 7.0;

  pd.lx.resize(2);
  pd.lx << -1.0, -2.0;
  pd.ux.resize(2);
  pd.ux << 1.0, 2.0;
  pd.lw.resize(1);
  pd.lw << -3.0;
  pd.uw.resize(1);
  pd.uw << 3.0;

  return pd;
}

}  // namespace

// ===================== KSPQPdata constructor =====================

TEST(Problem, KspqpDataConstructorCopiesFieldsAndDimensions) {
  const auto pd = MakeTinyKspqpData();
  Problem<double> pb(pd, 1e-8, 500, 30.0, PrintWhen::ALWAYS, PrintWhat::FULL);

  EXPECT_EQ(pb.n, 2);
  EXPECT_EQ(pb.m, 1);
  EXPECT_EQ(pb.l, 1);

  EXPECT_TRUE(pb.Q.isApprox(pd.Q));
  EXPECT_TRUE(pb.A.isApprox(pd.A));
  EXPECT_TRUE(pb.B.isApprox(pd.B));
  EXPECT_TRUE(pb.c.isApprox(pd.c));
  EXPECT_TRUE(pb.b.isApprox(pd.b));
  EXPECT_DOUBLE_EQ(pb.obj_const, 7.0);
  EXPECT_TRUE(pb.lx.isApprox(pd.lx));
  EXPECT_TRUE(pb.ux.isApprox(pd.ux));
  EXPECT_TRUE(pb.lw.isApprox(pd.lw));
  EXPECT_TRUE(pb.uw.isApprox(pd.uw));

  EXPECT_DOUBLE_EQ(pb.tol, 1e-8);
  EXPECT_EQ(pb.max_iter, 500);
  EXPECT_DOUBLE_EQ(pb.time_limit, 30.0);
  EXPECT_EQ(pb.when, PrintWhen::ALWAYS);
  EXPECT_EQ(pb.what, PrintWhat::FULL);
}

// ===================== direct constructor =====================

TEST(Problem, DirectConstructorDefaultsDimensionsToZeroNotGarbage) {
  using SpMat = Eigen::SparseMatrix<double>;
  using Vec = Eigen::VectorXd;

  SpMat Q(2, 2), A(0, 2), B(0, 2);
  Vec c(2), b(0), lx(2), ux(2), lw(0), uw(0);
  c << 1.0, 2.0;
  lx << -1.0, -1.0;
  ux << 1.0, 1.0;

  Problem<double> pb(Q, A, B, c, b, 0.0, lx, ux, lw, uw,
                      1e-6, 3000, 60.0, PrintWhen::NEVER, PrintWhat::NONE);

  EXPECT_EQ(pb.n, 0);
  EXPECT_EQ(pb.m, 0);
  EXPECT_EQ(pb.l, 0);
}

// ===================== default constructor =====================

TEST(Problem, DefaultConstructorHasZeroSizedMatricesAndDefaultTuningParams) {
  Problem<double> pb;

  EXPECT_EQ(pb.n, 0);
  EXPECT_EQ(pb.m, 0);
  EXPECT_EQ(pb.l, 0);
  EXPECT_EQ(pb.Q.rows(), 0);
  EXPECT_EQ(pb.Q.cols(), 0);
  EXPECT_EQ(pb.A.rows(), 0);
  EXPECT_EQ(pb.B.rows(), 0);
  EXPECT_EQ(pb.c.size(), 0);
  EXPECT_EQ(pb.b.size(), 0);
  EXPECT_EQ(pb.lx.size(), 0);
  EXPECT_EQ(pb.ux.size(), 0);
  EXPECT_EQ(pb.lw.size(), 0);
  EXPECT_EQ(pb.uw.size(), 0);

  EXPECT_DOUBLE_EQ(pb.obj_const, 0.0);
  EXPECT_DOUBLE_EQ(pb.tol, 1e-6);
  EXPECT_EQ(pb.max_iter, 3000);
  EXPECT_DOUBLE_EQ(pb.time_limit, 60.0);
  EXPECT_EQ(pb.when, PrintWhen::NEVER);
  EXPECT_EQ(pb.what, PrintWhat::NONE);
}

// ===================== degenerate / edge-case dimensions =====================

TEST(Problem, KspqpDataConstructorHandlesFullyEmptyZeroDimensionProblem) {
  KSPQPdata<double> pd;
  pd.n = 0;
  pd.m = 0;
  pd.l = 0;

  pd.Q = Eigen::SparseMatrix<double>(0, 0);
  pd.A = Eigen::SparseMatrix<double>(0, 0);
  pd.B = Eigen::SparseMatrix<double>(0, 0);
  pd.c.resize(0);
  pd.b.resize(0);
  pd.obj_const = 0.0;
  pd.lx.resize(0);
  pd.ux.resize(0);
  pd.lw.resize(0);
  pd.uw.resize(0);

  Problem<double> pb(pd, 1e-6, 3000, 60.0, PrintWhen::NEVER, PrintWhat::NONE);

  EXPECT_EQ(pb.n, 0);
  EXPECT_EQ(pb.m, 0);
  EXPECT_EQ(pb.l, 0);
  EXPECT_EQ(pb.Q.rows(), 0);
  EXPECT_EQ(pb.Q.cols(), 0);
  EXPECT_EQ(pb.c.size(), 0);
  EXPECT_EQ(pb.lx.size(), 0);
}

TEST(Problem, KspqpDataConstructorHandlesEqualityOnlyProblemWithZeroInequalityRows) {
  // l == 0: no inequality (B) block at all -- must not crash or misassign m/l.
  KSPQPdata<double> pd;
  pd.n = 2;
  pd.m = 1;
  pd.l = 0;

  pd.Q = Eigen::SparseMatrix<double>(2, 2);
  pd.A = Eigen::SparseMatrix<double>(1, 2);
  pd.A.insert(0, 0) = 1.0;
  pd.A.insert(0, 1) = 1.0;
  pd.A.makeCompressed();
  pd.B = Eigen::SparseMatrix<double>(0, 2);

  pd.c.resize(2);
  pd.c << 1.0, 1.0;
  pd.b.resize(1);
  pd.b << 2.0;
  pd.obj_const = 0.0;

  pd.lx.resize(2);
  pd.lx << -10.0, -10.0;
  pd.ux.resize(2);
  pd.ux << 10.0, 10.0;
  pd.lw.resize(0);
  pd.uw.resize(0);

  Problem<double> pb(pd, 1e-6, 3000, 60.0, PrintWhen::NEVER, PrintWhat::NONE);

  EXPECT_EQ(pb.m, 1);
  EXPECT_EQ(pb.l, 0);
  EXPECT_EQ(pb.B.rows(), 0);
  EXPECT_EQ(pb.lw.size(), 0);
  EXPECT_EQ(pb.uw.size(), 0);
}

TEST(Problem, KspqpDataConstructorHandlesInequalityOnlyProblemWithZeroEqualityRows) {
  // m == 0: no equality (A) block at all.
  KSPQPdata<double> pd;
  pd.n = 2;
  pd.m = 0;
  pd.l = 1;

  pd.Q = Eigen::SparseMatrix<double>(2, 2);
  pd.A = Eigen::SparseMatrix<double>(0, 2);
  pd.B = Eigen::SparseMatrix<double>(1, 2);
  pd.B.insert(0, 0) = 1.0;
  pd.B.makeCompressed();

  pd.c.resize(2);
  pd.c << 1.0, 1.0;
  pd.b.resize(0);
  pd.obj_const = 0.0;

  pd.lx.resize(2);
  pd.lx << -10.0, -10.0;
  pd.ux.resize(2);
  pd.ux << 10.0, 10.0;
  pd.lw.resize(1);
  pd.lw << -5.0;
  pd.uw.resize(1);
  pd.uw << 5.0;

  Problem<double> pb(pd, 1e-6, 3000, 60.0, PrintWhen::NEVER, PrintWhat::NONE);

  EXPECT_EQ(pb.m, 0);
  EXPECT_EQ(pb.l, 1);
  EXPECT_EQ(pb.A.rows(), 0);
  EXPECT_EQ(pb.b.size(), 0);
}

TEST(Problem, DirectConstructorCopiesObjConstAndBoxBoundVectors) {
  using SpMat = Eigen::SparseMatrix<double>;
  using Vec = Eigen::VectorXd;

  SpMat Q(2, 2), A(1, 2), B(1, 2);
  Vec c(2), b(1), lx(2), ux(2), lw(1), uw(1);
  c << 1.0, 2.0;
  b << 3.0;
  lx << -1.0, -2.0;
  ux << 1.0, 2.0;
  lw << -4.0;
  uw << 4.0;

  Problem<double> pb(Q, A, B, c, b, /*obj_const=*/9.5, lx, ux, lw, uw,
                      1e-6, 3000, 60.0, PrintWhen::NEVER, PrintWhat::NONE);

  EXPECT_DOUBLE_EQ(pb.obj_const, 9.5);
  EXPECT_TRUE(pb.c.isApprox(c));
  EXPECT_TRUE(pb.b.isApprox(b));
  EXPECT_TRUE(pb.lx.isApprox(lx));
  EXPECT_TRUE(pb.ux.isApprox(ux));
  EXPECT_TRUE(pb.lw.isApprox(lw));
  EXPECT_TRUE(pb.uw.isApprox(uw));
}

TEST(Problem, DirectConstructorHandlesFullyEmptyMatricesAndVectors) {
  using SpMat = Eigen::SparseMatrix<double>;
  using Vec = Eigen::VectorXd;

  SpMat Q(0, 0), A(0, 0), B(0, 0);
  Vec c(0), b(0), lx(0), ux(0), lw(0), uw(0);

  Problem<double> pb(Q, A, B, c, b, 0.0, lx, ux, lw, uw,
                      1e-6, 3000, 60.0, PrintWhen::NEVER, PrintWhat::NONE);

  EXPECT_EQ(pb.n, 0);
  EXPECT_EQ(pb.m, 0);
  EXPECT_EQ(pb.l, 0);
  EXPECT_EQ(pb.Q.rows(), 0);
  EXPECT_EQ(pb.c.size(), 0);
  EXPECT_EQ(pb.lx.size(), 0);
}
