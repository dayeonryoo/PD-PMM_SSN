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

// ===================== Discretization::FD =====================
// Mirrors the FEM cases above but with the FD discretization. FD's PDE
// operator is mass-scaled in assemble_diff_by_discretization /
// assemble_cd_by_discretization specifically so these solve to the same
// KKT residual tolerance as the FEM path -- see PoissonL2ControlFdIsNotDegenerate
// below for the regression this guards against.

TEST(KktResidual, PoissonL2ControlFdSatisfiesKkt) {
  const double inf = std::numeric_limits<double>::infinity();
  SolveAndCheckKkt(make_poisson_l2_control<double>(2, 1e-2, -inf, inf, -inf, inf, false,
                                                    Discretization::FD));
}

TEST(KktResidual, PoissonL2StateControlFdSatisfiesKkt) {
  const double inf = std::numeric_limits<double>::infinity();
  SolveAndCheckKkt(make_poisson_l2_state_control<double>(2, 1.0, -0.1, 0.002, -inf, inf, false,
                                                          Discretization::FD));
}

TEST(KktResidual, ConvDiffL2ControlFdSatisfiesKkt) {
  SolveAndCheckKkt(make_convdiff_l2_control<double>(2, 0.1, 0.0, 0.2, -0.75, 0.75, 0.01, false,
                                                     Discretization::FD));
}

TEST(KktResidual, PoissonL1L2ControlFdSatisfiesKkt) {
  const double inf = std::numeric_limits<double>::infinity();
  SolveAndCheckKkt(make_poisson_l1l2_control<double>(2, 1e-2, 1e-2, -2.0, 1.5, -inf, inf, false,
                                                       Discretization::FD));
}

TEST(KktResidual, ConvDiffL1L2ControlFdSatisfiesKkt) {
  const double inf = std::numeric_limits<double>::infinity();
  SolveAndCheckKkt(make_convdiff_l1l2_control<double>(2, 1e-2, 1e-2, -2.0, 1.5, 0.02, -inf, inf,
                                                        false, Discretization::FD));
}

// Regression test for the bug this FD-scaling fix addresses: with a raw
// (unscaled) FD stiffness matrix in the D_op*y = M*u constraint, FD's
// O(1/h^2) operator combined with the O(h^2) mass on the control made the
// state's response to control collapse as O(h^4), so the trivial x=0
// solution was (numerically) KKT-optimal for a control-constrained,
// weakly-regularized tracking problem -- i.e. the solver would "converge"
// in a single iteration to a degenerate, physically wrong answer. Assert
// the control actually engages (uses a meaningful fraction of its box)
// instead of collapsing to its lower bound everywhere.
TEST(KktResidual, PoissonL2ControlFdIsNotDegenerate) {
  const double inf = std::numeric_limits<double>::infinity();
  const double u_upper = 300.0;
  auto pb = make_poisson_l2_control<double>(/*nc=*/5, /*beta=*/1e-6, -inf, inf, 0.0, u_upper,
                                             false, Discretization::FD);
  Problem<double> prob(pb, kSolverTol, /*max_iter=*/3000, /*time_limit=*/60.0,
                        PrintWhen::NEVER, PrintWhat::NONE);
  KSP_QP<double> solver(prob);
  ASSERT_FALSE(solver.setup_failed);
  Solution<double> sol = solver.solve();
  ASSERT_EQ(sol.opt, TerminationStatus::Optimal);
  ExpectFullQpKktResidualSmall(prob, sol);

  const int np = pb.n / 2;
  const double u_max = sol.x.segment(np, np).maxCoeff();
  // The degenerate bug produced u_max on the order of 1e-9 to 1e-10; a
  // correctly-scaled solve uses a large fraction of the [0, 300] box.
  EXPECT_GT(u_max, 1.0) << "control collapsed near zero -- FD state/control coupling regressed";
}