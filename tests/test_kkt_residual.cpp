#include "ksp_qp.hpp"
#include "pde_generator.hpp"

#include <gtest/gtest.h>

namespace {

using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<double>;

constexpr double kSolverTol = 1e-6; // passed to the solver
constexpr double kKktTol    = 1e-6; // residual margine tolerance

// Independently re-derives the four KKT residual blocks of the QP
//     min  c^T x + 0.5 x^T Q x + obj_const,  s.t. A x = b,  B x = w,
//          lx <= x <= ux,  lw <= w <= uw
// from the original Problem data and a Solution -- a true outside-in regression check.

void ExpectFullQpKktResidualSmall(const Problem<double>& pb,
                                  const Solution<double>& sol,
                                  double tol = kKktTol) {
  ASSERT_EQ(sol.x.size(), pb.n);

  // Qx via Q's lower-triangular-only storage.
  Vec Qx = pb.Q.selfadjointView<Eigen::Lower>() * sol.x;

  // 1. Stationarity / dual residual: c + Qx - A^T y1 - B^T y2 + z.
  Vec r_dual = Qx + pb.c + sol.z;
  if (pb.m > 0) r_dual -= SpMat(pb.A.transpose()) * sol.y1;
  if (pb.l > 0) r_dual -= SpMat(pb.B.transpose()) * sol.y2;
  EXPECT_LT(r_dual.cwiseAbs().maxCoeff(), tol) << "dual/stationarity residual too large";

  // 2. Primal equality residual: A x - b.
  if (pb.m > 0) {
    Vec r_primal = pb.A * sol.x - pb.b;
    EXPECT_LT(r_primal.cwiseAbs().maxCoeff(), tol) << "primal equality residual too large";
  }

  // 3. Box complementarity on x: x - proj(x + z, lx, ux).
  Vec proj_x = (sol.x + sol.z).cwiseMax(pb.lx).cwiseMin(pb.ux);
  Vec r_box_x = sol.x - proj_x;
  EXPECT_LT(r_box_x.cwiseAbs().maxCoeff(), tol) << "x-box complementarity residual too large";

  // 4. Box complementarity on Bx: Bx - proj(Bx - y2, lw, uw).
  if (pb.l > 0) {
    Vec Bx = pb.B * sol.x;
    Vec proj_w = (Bx - sol.y2).cwiseMax(pb.lw).cwiseMin(pb.uw);
    Vec r_box_w = Bx - proj_w;
    EXPECT_LT(r_box_w.cwiseAbs().maxCoeff(), tol) << "Bx-box complementarity residual too large";
  }
}

// Builds Problem<double>, runs KSP_QP<double>::solve() end-to-end, asserts Optimal termination,
// then checks the outer KKT residual.
void SolveAndCheckKkt(const KSPQPdata<double>& data, double tol = kKktTol) {
  Problem<double> pb(data, kSolverTol, /*max_iter=*/3000, /*time_limit=*/60.0,
                      PrintWhen::NEVER, PrintWhat::NONE);
  KSP_QP<double> solver(pb);
  ASSERT_FALSE(solver.setup_failed);
  Solution<double> sol = solver.solve();
  ASSERT_EQ(sol.opt, TerminationStatus::Optimal);
  ExpectFullQpKktResidualSmall(pb, sol, tol);
}

}  // namespace

using namespace pdegen;

TEST(KktResidual, PoissonL2ControlSatisfiesKkt) {
  SolveAndCheckKkt(make_poisson_l2_control<double>(2, 1e-2));
}

TEST(KktResidual, PoissonL2StateControlSatisfiesKkt) {
  SolveAndCheckKkt(make_poisson_l2_state_control<double>(2, 1.0, -0.1, 0.002));
}

TEST(KktResidual, ConvDiffL2ControlSatisfiesKkt) {
  SolveAndCheckKkt(make_convdiff_l2_control<double>(2, 0.1, 0.0, 0.2, -0.75, 0.75));
}

TEST(KktResidual, PoissonL1L2ControlSatisfiesKkt) {
  // Exercises the l > 0 / Bx-complementarity residual branch (block 4 above).
  SolveAndCheckKkt(make_poisson_l1l2_control<double>(2, 1e-2, 1e-2));
}

TEST(KktResidual, ConvDiffL1L2ControlSatisfiesKkt) {
  // Exercises the l > 0 / Bx-complementarity residual branch with a non-symmetric A.
  SolveAndCheckKkt(make_convdiff_l1l2_control<double>(2, 1e-2, 1e-2));
}