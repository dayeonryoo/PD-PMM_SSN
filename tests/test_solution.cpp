#include "solution.hpp"

#include <gtest/gtest.h>

#include <sstream>

namespace {

// Redirects std::cout into an internal buffer for the lifetime of the object;
// str() returns everything written so far.
class CoutCapture {
 public:
  CoutCapture() : old_buf_(std::cout.rdbuf(buf_.rdbuf())) {}
  ~CoutCapture() { std::cout.rdbuf(old_buf_); }
  std::string str() const { return buf_.str(); }

 private:
  std::ostringstream buf_;
  std::streambuf* old_buf_;
};

Solution<double> MakeSolution(TerminationStatus opt) {
  Eigen::VectorXd x(2), y1(1), y2(1), z(2);
  x << 1.0, 2.0;
  y1 << 3.0;
  y2 << 4.0;
  z << 5.0, 6.0;
  return Solution<double>(opt, x, y1, y2, z,
                           /*obj_val=*/42.0,
                           /*pmm_iter=*/7, /*ssn_iter=*/11, /*krylov_iter=*/13,
                           /*fact=*/2, /*smw_count=*/1,
                           /*pmm_tol_achieved=*/1e-7, /*ssn_tol_achieved=*/1e-8,
                           /*setup_time=*/0.25, /*solve_time=*/1.75,
                           /*linesearch_fail=*/1, /*krylov_fail=*/0);
}

}  // namespace

// ===================== constructor =====================

TEST(Solution, ConstructorComputesRunTimeAsSetupPlusSolve) {
  const auto sol = MakeSolution(TerminationStatus::Optimal);
  EXPECT_DOUBLE_EQ(sol.run_time, 0.25 + 1.75);
}

TEST(Solution, ConstructorCopiesAllVectorAndScalarFields) {
  const auto sol = MakeSolution(TerminationStatus::Optimal);

  Eigen::VectorXd expected_x(2);
  expected_x << 1.0, 2.0;
  EXPECT_TRUE(sol.x.isApprox(expected_x));
  EXPECT_DOUBLE_EQ(sol.y1(0), 3.0);
  EXPECT_DOUBLE_EQ(sol.y2(0), 4.0);

  EXPECT_DOUBLE_EQ(sol.obj_val, 42.0);
  EXPECT_EQ(sol.pmm_iter, 7);
  EXPECT_EQ(sol.ssn_iter, 11);
  EXPECT_EQ(sol.krylov_iter, 13);
  EXPECT_EQ(sol.fact, 2);
  EXPECT_EQ(sol.smw_count, 1);
  EXPECT_DOUBLE_EQ(sol.pmm_tol_achieved, 1e-7);
  EXPECT_DOUBLE_EQ(sol.ssn_tol_achieved, 1e-8);
  EXPECT_EQ(sol.linesearch_fail, 1);
  EXPECT_EQ(sol.krylov_fail, 0);
}

// ===================== print_summary() =====================

TEST(Solution, PrintSummaryOptimalCasePrintsObjectiveAndIterationCounts) {
  const auto sol = MakeSolution(TerminationStatus::Optimal);

  CoutCapture capture;
  sol.print_summary();
  const std::string out = capture.str();

  EXPECT_NE(out.find("Objective value"), std::string::npos);
  EXPECT_NE(out.find("pmm_iter"), std::string::npos);
  EXPECT_NE(out.find("ssn_iter"), std::string::npos);
}

TEST(Solution, PrintSummaryPrimalInfeasibleCaseOmitsObjectiveBlock) {
  const auto sol = MakeSolution(TerminationStatus::PrimalInfeasible);

  CoutCapture capture;
  sol.print_summary();
  const std::string out = capture.str();

  EXPECT_NE(out.find("primal infeasible"), std::string::npos);
  EXPECT_EQ(out.find("Objective value"), std::string::npos);
}

TEST(Solution, PrintSummaryDualInfeasibleCaseOmitsObjectiveBlock) {
  const auto sol = MakeSolution(TerminationStatus::DualInfeasible);

  CoutCapture capture;
  sol.print_summary();
  const std::string out = capture.str();

  EXPECT_NE(out.find("dual infeasible"), std::string::npos);
  EXPECT_EQ(out.find("Objective value"), std::string::npos);
}

// ===================== degenerate / edge-case opt values =====================

TEST(Solution, PrintSummaryNumericalErrorCasePrintsNoNumbers) {
  // opt == NumericalError prints only the one-line message and return immediately
  // without objective/iteration/tolerance numbers.
  const auto sol = MakeSolution(TerminationStatus::NumericalError);

  CoutCapture capture;
  sol.print_summary();
  const std::string out = capture.str();

  EXPECT_NE(out.find("numerical error"), std::string::npos);
  EXPECT_EQ(out.find("Objective value"), std::string::npos);
  EXPECT_EQ(out.find("pmm_iter"), std::string::npos);
  EXPECT_EQ(out.find("linesearch_fail"), std::string::npos);
  EXPECT_EQ(out.find("run_time"), std::string::npos);
}

TEST(Solution, PrintSummaryNonOptimalTerminationCodesStillPrintObjectiveBlock) {
  // opt in {MaxPmmIterations, MaxSsnIterations, TimeLimit, Interrupted} all share the same
  // "best iterate found so far" branch as opt == Optimal.
  for (TerminationStatus opt : {TerminationStatus::MaxPmmIterations, TerminationStatus::MaxSsnIterations,
                                 TerminationStatus::TimeLimit, TerminationStatus::Interrupted}) {
    const auto sol = MakeSolution(opt);

    CoutCapture capture;
    sol.print_summary();
    const std::string out = capture.str();

    EXPECT_NE(out.find("Objective value"), std::string::npos) << "opt = " << static_cast<int>(opt);
    const std::string expected_status_line = "Termination status (opt): " + std::string(to_string(opt)) +
                                              " (" + std::to_string(static_cast<int>(opt)) + ")";
    EXPECT_NE(out.find(expected_status_line), std::string::npos) << "opt = " << static_cast<int>(opt);
  }
}

TEST(Solution, PrintSummaryHandlesEmptySolutionVectors) {
  // n == m == l == 0: degenerate empty problem's solution.
  Eigen::VectorXd x(0), y1(0), y2(0), z(0);
  Solution<double> sol(TerminationStatus::Optimal, x, y1, y2, z, /*obj_val=*/0.0, /*pmm_iter=*/0, /*ssn_iter=*/0,
                        /*krylov_iter=*/0, /*fact=*/0, /*smw_count=*/0,
                        /*pmm_tol_achieved=*/0.0, /*ssn_tol_achieved=*/0.0,
                        /*setup_time=*/0.0, /*solve_time=*/0.0,
                        /*linesearch_fail=*/0, /*krylov_fail=*/0);

  CoutCapture capture;
  sol.print_summary();
  const std::string out = capture.str();

  EXPECT_NE(out.find("n = 0, m = 0, l = 0"), std::string::npos);
}

TEST(Solution, ConstructorHandlesZeroTimesAndZeroIterationCounts) {
  Eigen::VectorXd x(0), y1(0), y2(0), z(0);
  Solution<double> sol(TerminationStatus::Optimal, x, y1, y2, z, 0.0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0);

  EXPECT_DOUBLE_EQ(sol.run_time, 0.0);
  EXPECT_EQ(sol.x.size(), 0);
  EXPECT_EQ(sol.pmm_iter, 0);
  EXPECT_EQ(sol.fact, 0);
  EXPECT_EQ(sol.smw_count, 0);
}

TEST(Solution, ConstructorHandlesNegativeObjValueAndTolerances) {
  Eigen::VectorXd x(1), y1(0), y2(0), z(1);
  x << -3.5;
  z << 0.0;
  Solution<double> sol(TerminationStatus::Optimal, x, y1, y2, z, /*obj_val=*/-100.25, 1, 1, 1, 1, 0,
                        /*pmm_tol_achieved=*/-1e-9, /*ssn_tol_achieved=*/-1e-9,
                        0.0, 0.0, 0, 0);

  EXPECT_DOUBLE_EQ(sol.obj_val, -100.25);
  EXPECT_DOUBLE_EQ(sol.pmm_tol_achieved, -1e-9);
}
