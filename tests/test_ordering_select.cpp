#include "ordering_select.hpp"

#include <gtest/gtest.h>

#include <Eigen/Sparse>
#include <vector>

namespace {

using SpMat = Eigen::SparseMatrix<double>;
using Idx = SpMat::StorageIndex;

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

// n x n symmetric tridiagonal SPD matrix (diagonal 4, off-diagonal -1).
SpMat MakeTridiagonal(int n) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, 4.0);
  for (int i = 0; i + 1 < n; ++i) {
    trips.emplace_back(i, i + 1, -1.0);
    trips.emplace_back(i + 1, i, -1.0);
  }
  SpMat A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  A.makeCompressed();
  return A;
}

// (path_len+1) x (path_len+1): a path 0-1-...-(path_len-1), plus one extra "hub" vertex at index
// path_len connected to every path vertex (degree path_len, versus the path's own degree ~2-3).
SpMat MakePathPlusHub(int path_len) {
  const int n = path_len + 1;
  const int hub = path_len;
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, 4.0);
  for (int i = 0; i + 1 < path_len; ++i) {
    trips.emplace_back(i, i + 1, -1.0);
    trips.emplace_back(i + 1, i, -1.0);
  }
  for (int i = 0; i < path_len; ++i) {
    trips.emplace_back(i, hub, -1.0);
    trips.emplace_back(hub, i, -1.0);
  }
  SpMat A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  A.makeCompressed();
  return A;
}

// Adjacency list of an n-vertex path 0-1-...-(n-1).
std::vector<std::vector<int>> MakePathAdjacency(int n) {
  std::vector<std::vector<int>> adj(n);
  for (int i = 0; i + 1 < n; ++i) {
    adj[i].push_back(i + 1);
    adj[i + 1].push_back(i);
  }
  return adj;
}

// Adjacency list of a star: vertex 0 is the center, 1..num_leaves are leaves.
std::vector<std::vector<int>> MakeStarAdjacency(int num_leaves) {
  std::vector<std::vector<int>> adj(num_leaves + 1);
  for (int leaf = 1; leaf <= num_leaves; ++leaf) {
    adj[0].push_back(leaf);
    adj[leaf].push_back(0);
  }
  return adj;
}

// Adjacency list of a side x side 4-neighbor grid, row-major (vertex (r,c) = r*side+c); vertex 0
// is corner (0,0).
std::vector<std::vector<int>> MakeGridAdjacency(int side) {
  std::vector<std::vector<int>> adj(side * side);
  auto idx = [side](int r, int c) { return r * side + c; };
  for (int r = 0; r < side; ++r)
    for (int c = 0; c < side; ++c) {
      if (r + 1 < side) { adj[idx(r, c)].push_back(idx(r + 1, c)); adj[idx(r + 1, c)].push_back(idx(r, c)); }
      if (c + 1 < side) { adj[idx(r, c)].push_back(idx(r, c + 1)); adj[idx(r, c + 1)].push_back(idx(r, c)); }
    }
  return adj;
}

}  // namespace

// ===================== Part 1: size screen =====================

TEST(IsSmallProblem, BelowThresholdIsTrue) {
  const SpMat A = MakeTridiagonal(5);
  EXPECT_TRUE(ordering_select::is_small_problem(A, /*threshold=*/10));
}

TEST(IsSmallProblem, AtThresholdIsFalse) {
  const SpMat A = MakeTridiagonal(5);
  EXPECT_FALSE(ordering_select::is_small_problem(A, /*threshold=*/5));
}

TEST(IsSmallProblem, AboveThresholdIsFalse) {
  const SpMat A = MakeTridiagonal(5);
  EXPECT_FALSE(ordering_select::is_small_problem(A, /*threshold=*/3));
}

TEST(SelectOrdering, TinyMatrixPicksAmdViaSizeScreenWithoutProbing) {
  const SpMat A = MakeTridiagonal(5);
  const auto decision = ordering_select::select_ordering<SpMat, Idx>(A); // default threshold 20000

  EXPECT_EQ(decision.winner, "AMD");
  EXPECT_EQ(decision.screen, ordering_select::DecisionScreen::kSizeThreshold);
  EXPECT_FALSE(decision.winner_was_probed());
  EXPECT_EQ(decision.amd.nnz_l, -1);
  EXPECT_EQ(decision.amd.anz, -1);
  EXPECT_EQ(decision.amd.analyze_seconds, -1.0);
}

// ===================== Part 2: AMD fill screen =====================

TEST(LowerTriangleNnz, CountsDiagonalAndBelow) {
  // 3x3: full diagonal (3) + one symmetric off-diagonal pair -> 1 strictly-lower entry.
  Eigen::MatrixXd dense(3, 3);
  dense << 1, 1, 0,
           1, 2, 0,
           0, 0, 3;
  const SpMat A = DenseToSparse(dense);
  EXPECT_EQ(ordering_select::lower_triangle_nnz(A), 4); // 3 diagonal + 1 strictly-lower
}

TEST(ProbeAmd, ZeroFillOnTridiagonalMatrix) {
  const SpMat A = MakeTridiagonal(5);
  const auto stats = ordering_select::probe_amd<SpMat, Idx>(A);

  // AMD should peel a tridiagonal matrix from an endpoint with no fill-in: nnz_l should equal
  // the input's own strictly-lower nnz (n-1 = 4).
  EXPECT_EQ(stats.nnz_l, 4);
  EXPECT_EQ(stats.anz, 9); // 5 diagonal + 4 strictly-lower
  EXPECT_GE(stats.analyze_seconds, 0.0);
}

TEST(AmdIsGoodEnough, FillRatioBelowThresholdAccepts) {
  ordering_select::AmdProbeStats s;
  s.nnz_l = 100;
  s.anz = 30; // nnz_l < 5*anz (100 < 150)
  EXPECT_TRUE(ordering_select::amd_is_good_enough(s));
}

TEST(AmdIsGoodEnough, FillRatioAboveThresholdRejects) {
  ordering_select::AmdProbeStats s;
  s.nnz_l = 1000;
  s.anz = 100; // nnz_l/anz = 10, not < 5
  EXPECT_FALSE(ordering_select::amd_is_good_enough(s));
}

// ===================== Part 3: BFS structural screen =====================

TEST(BfsFrom, PathGraphLevelsAreAllSingletons) {
  const auto adj = MakePathAdjacency(10);
  const auto levels = ordering_select::bfs_from(adj, /*source=*/0);

  ASSERT_EQ(levels.level_sizes.size(), 10u);
  for (int sz : levels.level_sizes) EXPECT_EQ(sz, 1);
  EXPECT_EQ(levels.farthest_vertex, 9);
  EXPECT_EQ(levels.reached, 10);
}

TEST(BfsProfile, PathGraphHasLowGrowthSlope) {
  const auto adj = MakePathAdjacency(10);
  const auto prof = ordering_select::bfs_profile(adj);

  EXPECT_EQ(prof.eccentricity, 9);
  EXPECT_EQ(prof.max_level_size, 1);
  EXPECT_EQ(prof.fit_status, ordering_select::GrowthFitStatus::kFitted);
  // Verified by direct computation: OLS slope of log(cumulative ball size) vs log(radius) for a
  // 10-vertex path (|B(r)| = r+1) is ~0.7432.
  EXPECT_NEAR(prof.growth_slope, 0.7432, 1e-3);
}

TEST(BfsProfile, StarGraphIsClassifiedAsTooShallow) {
  const auto adj = MakeStarAdjacency(10); // 1 center + 10 leaves, no hub-drop applied
  const auto prof = ordering_select::bfs_profile(adj);

  // A star's diameter is exactly 2 regardless of which vertex the double-sweep starts from (a
  // structural property of diameter-2 graphs): too few (radius, ball-size) points to fit a slope.
  EXPECT_EQ(prof.eccentricity, 2);
  EXPECT_EQ(prof.fit_status, ordering_select::GrowthFitStatus::kTooShallow);
  EXPECT_NEAR(prof.max_level_fraction, 9.0 / 11.0, 1e-9);
}

TEST(BuildBfsAdjacency, DropsEngineeredHubVertex) {
  // 20-vertex path (degree ~2-3) plus one extra vertex connected to all 20 (degree 20). Median
  // degree over the 21 vertices is 3 (two path endpoints have degree 2, eighteen interior path
  // vertices have degree 3, the hub has degree 20) -- verified by direct computation. A
  // multiplier of 5.0 (hub_floor = 5*3 = 15) clears the hub's degree (20 > 15) without touching
  // any path vertex (3 is not > 15).
  const SpMat H = MakePathPlusHub(20);

  const auto raw = ordering_select::build_bfs_adjacency(H, /*hubs_to_drop=*/0);
  EXPECT_TRUE(raw.dropped_hub_original_index.empty());
  const auto raw_prof = ordering_select::bfs_profile(raw.adjacency);
  EXPECT_EQ(raw_prof.eccentricity, 2); // masked: the hub gives every vertex a 2-hop shortcut

  const auto dropped = ordering_select::build_bfs_adjacency(H, /*hubs_to_drop=*/1,
                                                             /*hub_degree_multiplier=*/5.0);
  ASSERT_EQ(dropped.dropped_hub_original_index.size(), 1u);
  EXPECT_EQ(dropped.dropped_hub_original_index[0], 20); // the hub's original index
  const auto dropped_prof = ordering_select::bfs_profile(dropped.adjacency);
  EXPECT_EQ(dropped_prof.eccentricity, 19); // recovered: the true path length
  EXPECT_EQ(dropped_prof.fit_status, ordering_select::GrowthFitStatus::kFitted);
  EXPECT_LT(dropped_prof.growth_slope, 1.0); // still path-like, not mesh-like
}

TEST(BfsProfile, SmallGridGraphHasHigherGrowthSlopeThanPath) {
  const auto grid_adj = MakeGridAdjacency(10);
  const auto grid_prof = ordering_select::bfs_profile(grid_adj);
  const auto path_prof = ordering_select::bfs_profile(MakePathAdjacency(10));

  EXPECT_EQ(grid_prof.eccentricity, 18); // corner-to-corner Manhattan distance, (10-1)+(10-1)
  EXPECT_EQ(grid_prof.fit_status, ordering_select::GrowthFitStatus::kFitted);
  // 2D area growth measurably outpaces the path's 1D growth, even with finite-size/boundary
  // effects pulling the fitted slope below the idealized asymptotic exponent of 2.
  EXPECT_GT(grid_prof.growth_slope, path_prof.growth_slope);
  EXPECT_NEAR(grid_prof.growth_slope, 1.321, 1e-2);
}

TEST(BfsProfile, DisconnectedGraphIgnoresOtherComponent) {
  auto solo = MakePathAdjacency(5);

  // The appended component (size 3) is smaller than the original (size 5), so it stays the
  // largest and this still isolates it -- see SeedsFromLargestComponentEvenWhenItIsNotFirstByIndex
  // below for the case where the appended piece is the *bigger* one instead.
  auto with_extra_component = MakePathAdjacency(5);
  with_extra_component.resize(8);
  with_extra_component[5].push_back(6);
  with_extra_component[6].push_back(5);
  with_extra_component[6].push_back(7);
  with_extra_component[7].push_back(6);

  const auto solo_prof = ordering_select::bfs_profile(solo);
  const auto extra_prof = ordering_select::bfs_profile(with_extra_component);

  EXPECT_EQ(solo_prof.n_total, 5);
  EXPECT_EQ(extra_prof.n_total, 8); // only n_total differs...
  EXPECT_EQ(solo_prof.eccentricity, extra_prof.eccentricity);
  EXPECT_EQ(solo_prof.n_reached, extra_prof.n_reached); // ...the largest component is unaffected
  EXPECT_EQ(solo_prof.max_level_size, extra_prof.max_level_size);
  EXPECT_DOUBLE_EQ(solo_prof.growth_slope, extra_prof.growth_slope);
}

TEST(LargestComponentSeed, PicksVertexInBiggestComponentRegardlessOfIndexOrder) {
  // A 4-vertex component at indices 0-3, then a 9-vertex (bigger) component at indices 4-12.
  // A "first non-isolated vertex" rule would return 0; the correct seed must come from the
  // 9-vertex component instead.
  auto adj = MakePathAdjacency(4);
  adj.resize(13);
  for (int i = 4; i < 12; ++i) {
    adj[i].push_back(i + 1);
    adj[i + 1].push_back(i);
  }

  const int seed = ordering_select::largest_component_seed(adj);
  ASSERT_GE(seed, 4);
  EXPECT_LE(seed, 12);
}

TEST(LargestComponentSeed, ReturnsNegativeOneWhenEveryVertexIsIsolated) {
  const std::vector<std::vector<int>> adj(5); // 5 vertices, no edges at all
  EXPECT_EQ(ordering_select::largest_component_seed(adj), -1);
}

TEST(BfsProfile, SeedsFromLargestComponentEvenWhenItIsNotFirstByIndex) {
  // A small fragment at the lowest indices, with the graph's real (here mesh-like-by-construction)
  // giant component sitting behind it.
  auto adj = MakePathAdjacency(3); // indices 0-2, a small decoy component
  adj.resize(13);
  for (int i = 3; i < 12; ++i) { // 10-vertex path at indices 3-12, the graph's real component
    adj[i].push_back(i + 1);
    adj[i + 1].push_back(i);
  }

  const auto prof = ordering_select::bfs_profile(adj);
  const auto big_alone_prof = ordering_select::bfs_profile(MakePathAdjacency(10));

  EXPECT_EQ(prof.n_total, 13);
  EXPECT_EQ(prof.n_reached, 10); // the 10-vertex component, not the 3-vertex decoy
  EXPECT_EQ(prof.eccentricity, big_alone_prof.eccentricity);
  EXPECT_DOUBLE_EQ(prof.growth_slope, big_alone_prof.growth_slope);
}

TEST(BfsPredictsMeshLike, TooShallowIsAlwaysFalse) {
  ordering_select::BfsStructuralProfile p;
  p.fit_status = ordering_select::GrowthFitStatus::kTooShallow;
  p.eccentricity = 100;         // otherwise would look mesh-like
  p.max_level_fraction = 0.01;  // otherwise would look mesh-like
  p.growth_slope = 2.0;         // otherwise would look mesh-like
  EXPECT_FALSE(ordering_select::bfs_predicts_mesh_like(p));
}

TEST(BfsPredictsMeshLike, HighMaxLevelFractionIsFalseRegardlessOfSlope) {
  ordering_select::BfsStructuralProfile p;
  p.fit_status = ordering_select::GrowthFitStatus::kFitted;
  p.eccentricity = 100;
  p.max_level_fraction = 0.9; // wide/bushy: expander-like despite a plausible slope
  p.growth_slope = 2.0;
  EXPECT_FALSE(ordering_select::bfs_predicts_mesh_like(p));
}

TEST(BfsPredictsMeshLike, TooShallowEccentricityIsFalse) {
  ordering_select::BfsStructuralProfile p;
  p.fit_status = ordering_select::GrowthFitStatus::kFitted;
  p.eccentricity = 5; // below kMeshLikeEccentricityThreshold
  p.max_level_fraction = 0.01;
  p.growth_slope = 2.0;
  EXPECT_FALSE(ordering_select::bfs_predicts_mesh_like(p));
}

TEST(BfsPredictsMeshLike, InRangeEverythingIsTrue) {
  ordering_select::BfsStructuralProfile p;
  p.fit_status = ordering_select::GrowthFitStatus::kFitted;
  p.eccentricity = 100;
  p.max_level_fraction = 0.01;
  p.growth_slope = 2.0;
  EXPECT_TRUE(ordering_select::bfs_predicts_mesh_like(p));
}

// ===================== Cascade wiring =====================

TEST(SelectOrdering, OverriddenThresholdForcesScreen2OnSmallMatrix) {
  const SpMat A = MakeTridiagonal(5);
  ordering_select::OrderingSelectConfig cfg;
  cfg.small_problem_threshold = 0;

  const auto decision = ordering_select::select_ordering<SpMat, Idx>(A, cfg);

  EXPECT_EQ(decision.screen, ordering_select::DecisionScreen::kAmdFillRatio);
  EXPECT_EQ(decision.winner, "AMD");
  EXPECT_TRUE(decision.winner_was_probed());
  EXPECT_GE(decision.amd.nnz_l, 0);
}

// select_ordering()'s amd_solver_inout lets a caller probe Part 2 in place on its own solver
// instead of an internal throwaway, so that -- when winner_was_probed() is true -- it can go on
// using that exact solver directly. Guard against a regression that silently re-runs
// analyzePattern() (which would still "work" numerically) by never calling analyzePattern() again
// here: factorize()/solve() must succeed straight off the probed solver.
TEST(SelectOrdering, AmdSolverInoutIsUsableDirectlyWithoutReanalyzing) {
  const SpMat A = MakeTridiagonal(50);
  ordering_select::OrderingSelectConfig cfg;
  cfg.small_problem_threshold = 0; // force past Part 1 so Part 2 actually probes amd_solver_inout

  auto amd_candidate = ordering_select::make_solver<SpMat, /*IsLdlt=*/true>("AMD");
  const auto decision =
      ordering_select::select_ordering<SpMat, Idx>(A, cfg, amd_candidate.get());

  ASSERT_EQ(decision.winner, "AMD");
  ASSERT_TRUE(decision.winner_was_probed());

  Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(50, 1.0, 50.0);
  amd_candidate->factorize(A); // no analyzePattern() call here -- select_ordering() already did it
  ASSERT_EQ(amd_candidate->info(), Eigen::Success);
  const Eigen::VectorXd x = amd_candidate->solve(rhs);

  auto reference = ordering_select::make_solver<SpMat, /*IsLdlt=*/true>("AMD");
  reference->analyzePattern(A);
  reference->factorize(A);
  const Eigen::VectorXd x_ref = reference->solve(rhs);

  EXPECT_TRUE(x.isApprox(x_ref, 1e-12));
}

#ifdef KSP_QP_HAVE_METIS
TEST(SelectOrdering, ImpossibleAmdThresholdsForceBfsScreen) {
  const SpMat H = MakePathPlusHub(20);
  ordering_select::OrderingSelectConfig cfg;
  cfg.small_problem_threshold = 0;
  cfg.fill_input_ratio = 0;
  cfg.hubs_to_drop = 1;
  cfg.hub_degree_multiplier = 5.0;

  const auto decision = ordering_select::select_ordering<SpMat, Idx>(H, cfg);

  EXPECT_EQ(decision.screen, ordering_select::DecisionScreen::kBfsStructural);
  ASSERT_EQ(decision.bfs.dropped_hub_vertices.size(), 1u);
  EXPECT_EQ(decision.bfs.dropped_hub_vertices[0], 20);
  EXPECT_EQ(decision.bfs.eccentricity, 19);

  // Cross-check against calling the three Part-3 functions directly on the same matrix/config.
  const auto adj = ordering_select::build_bfs_adjacency(H, cfg.hubs_to_drop, cfg.hub_degree_multiplier);
  const auto prof = ordering_select::bfs_profile(adj.adjacency, cfg.min_eccentricity_for_slope_fit);
  const bool expect_metis = ordering_select::bfs_predicts_mesh_like(
      prof, cfg.mesh_eccentricity_threshold, cfg.expander_max_level_fraction,
      cfg.mesh_growth_slope_min, cfg.mesh_growth_slope_max);
  EXPECT_EQ(decision.winner, expect_metis ? "METIS" : "AMD");
}

TEST(MakeSolver, AmdAndMetisAgreeNumericallyOnSolve) {
  const SpMat A = MakeTridiagonal(50);
  Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(50, 1.0, 50.0);

  auto amd_solver = ordering_select::make_solver<SpMat, /*IsLdlt=*/true>("AMD");
  amd_solver->analyzePattern(A);
  amd_solver->factorize(A);
  ASSERT_EQ(amd_solver->info(), Eigen::Success);
  const Eigen::VectorXd x_amd = amd_solver->solve(rhs);

  auto metis_solver = ordering_select::make_solver<SpMat, /*IsLdlt=*/true>("METIS");
  metis_solver->analyzePattern(A);
  metis_solver->factorize(A);
  ASSERT_EQ(metis_solver->info(), Eigen::Success);
  const Eigen::VectorXd x_metis = metis_solver->solve(rhs);

  EXPECT_TRUE(x_amd.isApprox(x_metis, 1e-10));
}
#else
// Only compiled (and only meaningful) in a -DKSP_QP_BUILD_METIS=OFF build; needs at least one
// such CI/manual build to actually execute.
TEST(SelectOrdering, MetisUnavailableIsHonestlyReported) {
  const SpMat H = MakePathPlusHub(20);
  ordering_select::OrderingSelectConfig cfg;
  cfg.small_problem_threshold = 0;
  cfg.fill_input_ratio = 0;

  const auto decision = ordering_select::select_ordering<SpMat, Idx>(H, cfg);

  EXPECT_EQ(decision.screen, ordering_select::DecisionScreen::kMetisUnavailable);
  EXPECT_EQ(decision.winner, "AMD");
  EXPECT_TRUE(decision.winner_was_probed());
}
#endif
