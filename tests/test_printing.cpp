#include "printing.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <sstream>

namespace {

class CoutCapture {
 public:
  CoutCapture() : old_buf_(std::cout.rdbuf(buf_.rdbuf())) {}
  ~CoutCapture() { std::cout.rdbuf(old_buf_); }
  std::string str() const { return buf_.str(); }

 private:
  std::ostringstream buf_;
  std::streambuf* old_buf_;
};

int CountOccurrences(const std::string& haystack, const std::string& needle) {
  int count = 0;
  std::size_t pos = 0;
  while ((pos = haystack.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

}  // namespace

// ===================== time_diff_s =====================

TEST(TimeDiffS, ComputesElapsedSecondsBetweenTimePoints) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto t1 = t0 + std::chrono::milliseconds(250);
  EXPECT_NEAR(time_diff_s(t0, t1), 0.25, 1e-6);
}

TEST(TimeDiffS, ReturnsZeroForIdenticalTimePoints) {
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_DOUBLE_EQ(time_diff_s(t0, t0), 0.0);
}

TEST(TimeDiffS, ReturnsNegativeWhenEndPrecedesStart) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto t1 = t0 - std::chrono::milliseconds(100);
  EXPECT_NEAR(time_diff_s(t0, t1), -0.1, 1e-6);
}

// ===================== print_header =====================

TEST(PrintHeader, NeverOrNoneProducesNoOutput) {
  {
    CoutCapture capture;
    print_header(PrintWhen::NEVER, PrintWhat::FULL);
    EXPECT_TRUE(capture.str().empty());
  }
  {
    CoutCapture capture;
    print_header(PrintWhen::ALWAYS, PrintWhat::NONE);
    EXPECT_TRUE(capture.str().empty());
  }
}

TEST(PrintHeader, MinimalOmitsKrylovAndObjectiveColumns) {
  CoutCapture capture;
  print_header(PrintWhen::ALWAYS, PrintWhat::MINIMAL);
  const std::string out = capture.str();

  EXPECT_NE(out.find("PrimalRes"), std::string::npos);
  EXPECT_EQ(out.find("Krylov"), std::string::npos);
  EXPECT_EQ(out.find("Objective"), std::string::npos);
}

TEST(PrintHeader, FullIncludesObjectiveColumn) {
  CoutCapture capture;
  print_header(PrintWhen::ALWAYS, PrintWhat::FULL);
  const std::string out = capture.str();

  EXPECT_NE(out.find("Objective"), std::string::npos);
}

TEST(PrintHeader, FullOmitsKrylovAndKrylovFailColumns) {
  CoutCapture capture;
  print_header(PrintWhen::ALWAYS, PrintWhat::FULL);
  const std::string out = capture.str();

  EXPECT_EQ(out.find("Krylov"), std::string::npos);
  EXPECT_EQ(out.find("k.f."), std::string::npos);
}

TEST(PrintHeader, SsnIncludesKrylovFactAndKrylovFailColumns) {
  CoutCapture capture;
  print_header(PrintWhen::ALWAYS, PrintWhat::SSN);
  const std::string out = capture.str();

  EXPECT_NE(out.find("Krylov"), std::string::npos);
  EXPECT_NE(out.find("fact"), std::string::npos);
  EXPECT_NE(out.find("k.f."), std::string::npos);
}

TEST(PrintHeader, TuningIncludesKrylovFactAndKrylovFailColumns) {
  CoutCapture capture;
  print_header(PrintWhen::ALWAYS, PrintWhat::TUNING);
  const std::string out = capture.str();

  EXPECT_NE(out.find("Krylov"), std::string::npos);
  EXPECT_NE(out.find("fact"), std::string::npos);
  EXPECT_NE(out.find("k.f."), std::string::npos);
}

TEST(PrintHeader, MinimalOmitsKrylovFailColumnToo) {
  CoutCapture capture;
  print_header(PrintWhen::ALWAYS, PrintWhat::MINIMAL);
  const std::string out = capture.str();

  EXPECT_EQ(out.find("mu"), std::string::npos);
  EXPECT_EQ(out.find("k.f."), std::string::npos);
}

// ===================== print =====================

TEST(Print, NeverOrNoneSuppressesOutput) {
  Eigen::VectorXd res_norms(4);
  res_norms << 1.0, 2.0, 3.0, 4.0;
  {
    CoutCapture capture;
    print<double>(PrintWhen::NEVER, PrintWhat::FULL, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0,
                  0.0, 0, 0);
    EXPECT_TRUE(capture.str().empty());
  }
  {
    CoutCapture capture;
    print<double>(PrintWhen::ALWAYS, PrintWhat::NONE, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0,
                  0.0, 0, 0);
    EXPECT_TRUE(capture.str().empty());
  }
}

TEST(Print, Every10SkipsNonMultipleIterations) {
  Eigen::VectorXd res_norms(4);
  res_norms << 1.0, 2.0, 3.0, 4.0;

  {
    CoutCapture capture;
    print<double>(PrintWhen::EVERY10, PrintWhat::MINIMAL, /*pmm_iter=*/3, 1, 1, 1, 0.0, res_norms,
                  0.0, 0.0, 0.0, 0.0, 0, 0);
    EXPECT_TRUE(capture.str().empty());
  }
  {
    CoutCapture capture;
    print<double>(PrintWhen::EVERY10, PrintWhat::MINIMAL, /*pmm_iter=*/10, 1, 1, 1, 0.0, res_norms,
                  0.0, 0.0, 0.0, 0.0, 0, 0);
    EXPECT_FALSE(capture.str().empty());
  }
}

TEST(Print, HighlightsMaxResidualWithAnsiBold) {
  Eigen::VectorXd res_norms(4);
  res_norms << 1.0, 2.0, 9.0, 4.0;  // index 2 is the unique max

  CoutCapture capture;
  print<double>(PrintWhen::ALWAYS, PrintWhat::MINIMAL, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0,
                0.0, 0, 0);
  const std::string out = capture.str();

  EXPECT_EQ(CountOccurrences(out, "\033[1m"), 1);
}

TEST(Print, EmptyResNormsLeavesResidualColumnsBlank) {
  Eigen::VectorXd res_norms(0);

  CoutCapture capture;
  print<double>(PrintWhen::ALWAYS, PrintWhat::MINIMAL, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0,
                0.0, 0, 0);
  const std::string out = capture.str();

  EXPECT_FALSE(out.empty());
  EXPECT_EQ(CountOccurrences(out, "\033[1m"), 0);
}

TEST(Print, SingleElementResNormsBoldsTheSoleEntry) {
  Eigen::VectorXd res_norms(1);
  res_norms << 5.0;

  CoutCapture capture;
  print<double>(PrintWhen::ALWAYS, PrintWhat::MINIMAL, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0,
                0.0, 0, 0);
  const std::string out = capture.str();

  EXPECT_EQ(CountOccurrences(out, "\033[1m"), 1);
}

TEST(Print, TiedMaxResidualsAreAllBolded) {
  Eigen::VectorXd res_norms(4);
  res_norms << 9.0, 2.0, 9.0, 4.0;  // two entries tie for the max

  CoutCapture capture;
  print<double>(PrintWhen::ALWAYS, PrintWhat::MINIMAL, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0,
                0.0, 0, 0);
  const std::string out = capture.str();

  EXPECT_EQ(CountOccurrences(out, "\033[1m"), 2);
}

TEST(Print, AllZeroResNormsBoldsEveryEntry) {
  // Degenerate case: every entry equals the (zero) max, so every one is bolded.
  Eigen::VectorXd res_norms(4);
  res_norms << 0.0, 0.0, 0.0, 0.0;

  CoutCapture capture;
  print<double>(PrintWhen::ALWAYS, PrintWhat::MINIMAL, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0,
                0.0, 0, 0);
  const std::string out = capture.str();

  EXPECT_EQ(CountOccurrences(out, "\033[1m"), 4);
}

TEST(Print, EveryTenPrintsAtIterationZero) {
  // 0 % 10 == 0, so iteration 0 must print just like any other multiple of 10.
  Eigen::VectorXd res_norms(4);
  res_norms << 1.0, 2.0, 3.0, 4.0;

  CoutCapture capture;
  print<double>(PrintWhen::EVERY10, PrintWhat::MINIMAL, /*pmm_iter=*/0, 1, 1, 1, 0.0, res_norms,
                0.0, 0.0, 0.0, 0.0, 0, 0);
  EXPECT_FALSE(capture.str().empty());
}

TEST(Print, ShowPmmIterFalseBlanksPmmColumn) {
  Eigen::VectorXd res_norms(4);
  res_norms << 1.0, 2.0, 3.0, 4.0;

  CoutCapture capture;
  print<double>(PrintWhen::ALWAYS, PrintWhat::MINIMAL, /*pmm_iter=*/7, 1, 1, 1, 0.0, res_norms,
                0.0, 0.0, 0.0, 0.0, 0, 0, /*show_pmm_iter=*/false);
  const std::string out = capture.str();

  EXPECT_EQ(out.find('7'), std::string::npos);
}

TEST(Print, SsnAndTuningIncludeKrylovFailValue) {
  Eigen::VectorXd res_norms(4);
  res_norms << 1.0, 2.0, 3.0, 4.0;

  CoutCapture capture;
  print<double>(PrintWhen::ALWAYS, PrintWhat::SSN, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0, 0.0,
                0, /*krylov_fail=*/42);
  const std::string out = capture.str();

  EXPECT_NE(out.find("42"), std::string::npos);
}

TEST(Print, NegativeAndZeroResNormsDoNotCrashAndBoldOnlyTheMax) {
  Eigen::VectorXd res_norms(4);
  res_norms << -5.0, 0.0, -1.0, -8.0;  // max is 0.0 at index 1

  CoutCapture capture;
  print<double>(PrintWhen::ALWAYS, PrintWhat::MINIMAL, 1, 1, 1, 1, 0.0, res_norms, 0.0, 0.0, 0.0,
                0.0, 0, 0);
  const std::string out = capture.str();

  EXPECT_EQ(CountOccurrences(out, "\033[1m"), 1);
}
