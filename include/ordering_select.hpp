#pragma once
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#ifdef KSP_QP_HAVE_METIS
#include <iostream>  // Eigen's MetisSupport.h uses std::cerr without including this itself
#include <Eigen/MetisSupport>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Cheap ordering selection between AMD and (if KSP_QP_HAVE_METIS) METIS's nested dissection for
// a symmetric sparse matrix, via the cascade of screens in select_ordering(): an O(1) size screen,
// an O(nnz) CHOLMOD-style fill screen, and an O(nnz) BFS structural screen -- each tier only runs
// if the previous one didn't already decide. See ksp_qp.tpp's set_L_from_LLT and
// schur_preconditioner.hpp for the call sites.

namespace ordering_select {

struct Candidate {
    std::string name;
    long long nnz_l;
    double analyze_seconds;
};

// Exposes SimplicialLDLT's protected symbolic-factor pattern after analyzePattern() alone
// (before factorize()) -- Eigen's public matrixL()/nonZeros() assert m_factorizationIsOk, but
// the symbolic nnz is already final once analyzePattern() returns.
template <typename SpMat, typename Ordering>
struct FillProbe : public Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Ordering> {
    using Base = Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Ordering>;
    long long nnz_l() const { return static_cast<long long>(Base::m_matrix.nonZeros()); }
};

template <typename SpMat, typename Ordering>
Candidate probe(const std::string& name, const SpMat& A_sym) {
    FillProbe<SpMat, Ordering> probe;
    const auto t0 = std::chrono::steady_clock::now();
    probe.analyzePattern(A_sym);
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return Candidate{name, probe.nnz_l(), t};
}

// =================================================================================================
// Ordering-selection cascade.
//
// Three tiers, each only run if the previous one didn't already decide:
//
//   1. Size screen (O(1)): below kSmallProblemThreshold rows, just use AMD.
//   2. AMD fill screen (O(nnz)): probe AMD alone to decide if AMD's result is good enough
//      (lnz < 5*anz, i.e. low fill ratio).
//   3. BFS structural screen (O(nnz)): diagnostic on A_sym's sparsity graph to decide if the
//      graph is mesh-like (use nested dissection) or expander-like (use AMD). Drop the diagonal
//      and a handful of high-degree hub vertices, find a pseudo-peripheral vertex via the
//      double-sweep  BFS heuristic (George & Liu 1979), and read off that BFS's eccentricity 
//      (widest level-set)  and the log-log growth slope of its cumulative ball size.
// =================================================================================================

// ------ Part 1: O(1) size screen ------

// Below this row count, AMD's own analyzePattern() is already cheap.
inline constexpr Eigen::Index kSmallProblemThreshold = 20000;

template <typename SpMat>
bool is_small_problem(const SpMat& A_sym, Eigen::Index threshold = kSmallProblemThreshold) {
    return A_sym.rows() < threshold;
}

// ------ Part 2: O(nnz) CHOLMOD-style fill screen ------

// Counts entries on-or-below the diagonal (row >= col).
template <typename SpMat>
long long lower_triangle_nnz(const SpMat& A_sym) {
    long long anz = 0;
    for (Eigen::Index k = 0; k < A_sym.outerSize(); ++k)
        for (typename SpMat::InnerIterator it(A_sym, k); it; ++it)
            if (it.row() >= it.col()) ++anz;
    return anz;
}

struct AmdProbeStats {
    long long nnz_l = -1;   // predicted nnz(L) under AMD
    long long anz   = -1;   // nnz of the lower triangle of the input A_sym
    double analyze_seconds = -1.0;
};

// One real AMD analyzePattern() -- the only ordering ever actually probed by this cascade.
template <typename SpMat, typename Index>
AmdProbeStats probe_amd(const SpMat& A_sym) {
    const Candidate c = probe<SpMat, Eigen::AMDOrdering<Index>>("AMD", A_sym);
    return AmdProbeStats{c.nnz_l, lower_triangle_nnz(A_sym), c.analyze_seconds};
}

inline constexpr long long kFillInputRatio = 5;

inline bool amd_is_good_enough(const AmdProbeStats& s, long long fill_input_ratio = kFillInputRatio) {
    return s.nnz_l < fill_input_ratio * s.anz;
}

// ------ Part 3: O(nnz) BFS structural screen ------

struct BfsLevels {
    std::vector<int> level_sizes; // level_sizes[r] = |L_r|, the r-th BFS ring from the source
    int start_vertex    = -1;
    int farthest_vertex = -1;     // a vertex in the deepest level reached
    int reached         = 0;      // vertices reached (== component size when the graph is connected)
};

// Single BFS pass from `source`, O(reachable vertices + their incident edges).
inline BfsLevels bfs_from(const std::vector<std::vector<int>>& adjacency, int source) {
    const int n = static_cast<int>(adjacency.size());
    std::vector<int> dist(n, -1);
    dist[source] = 0;

    BfsLevels levels;
    levels.start_vertex    = source;
    levels.farthest_vertex = source;
    levels.reached         = 1;
    levels.level_sizes.push_back(1);

    std::vector<int> frontier{source};
    while (!frontier.empty()) {
        std::vector<int> next;
        for (int u : frontier)
            for (int v : adjacency[u])
                if (dist[v] == -1) { dist[v] = dist[u] + 1; next.push_back(v); }
        if (!next.empty()) {
            levels.level_sizes.push_back(static_cast<int>(next.size()));
            levels.reached += static_cast<int>(next.size());
            levels.farthest_vertex = next.back(); // any max-distance vertex is a valid next seed
        }
        frontier.swap(next);
    }
    return levels;
}

enum class GrowthFitStatus { kFitted, kTooShallow };

struct BfsStructuralProfile {
    int n_total         = 0; // adjacency.size(), regardless of connectivity
    int n_reached       = 0; // vertices in v1's connected component
    int eccentricity    = 0; // deepest level reached from v1 in the second BFS sweep
    int max_level_size  = 0; // widest level (max |L_r|) in the second BFS sweep
    double max_level_fraction = 0.0; // max_level_size / n_reached

    GrowthFitStatus fit_status = GrowthFitStatus::kTooShallow;
    double growth_slope = 0.0; // OLS slope of log(cumulative ball size) vs log(radius)

    int v0 = -1, v1 = -1; // v0: arbitrary seed; v1: pseudo-peripheral vertex found from v0
    std::vector<int> dropped_hub_vertices;
};

// Eccentricity <= this is treated as a strong expander-like signal directly.
inline constexpr int kMinEccentricityForSlopeFit = 2;

// Returns a vertex in `adjacency`'s largest connected component (by vertex count), or -1 if every
// vertex is isolated. O(n + nnz), one full iterative traversal (explicit stack, no recursion).
inline int largest_component_seed(const std::vector<std::vector<int>>& adjacency) {
    const int n = static_cast<int>(adjacency.size());
    std::vector<char> visited(n, 0);
    std::vector<int> stack;
    int best_vertex = -1, best_size = 0;
    for (int v = 0; v < n; ++v) {
        if (visited[v] || adjacency[v].empty()) continue;
        const int component_start = v;
        int size = 0;
        stack.assign(1, v);
        visited[v] = 1;
        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            ++size;
            for (int w : adjacency[u])
                if (!visited[w]) { visited[w] = 1; stack.push_back(w); }
        }
        if (size > best_size) { best_size = size; best_vertex = component_start; }
    }
    return best_vertex;
}

// Double-sweep pseudo-peripheral-vertex heuristic (George & Liu, 1979): BFS from v0, take the farthest
// vertex found as v1, BFS again from v1, and record statistics of that second sweep.
// v0 is any vertex in the largest connected component.
inline BfsStructuralProfile bfs_profile(const std::vector<std::vector<int>>& adjacency,
                                         int min_eccentricity_for_slope_fit = kMinEccentricityForSlopeFit) {
    BfsStructuralProfile prof;
    prof.n_total = static_cast<int>(adjacency.size());
    if (prof.n_total == 0) return prof;

    const int v0 = largest_component_seed(adjacency);
    if (v0 == -1) return prof; // every vertex isolated -- nothing to traverse.

    const BfsLevels sweep1 = bfs_from(adjacency, v0);
    const int v1 = sweep1.farthest_vertex;
    const BfsLevels sweep2 = bfs_from(adjacency, v1);

    prof.v0 = v0;
    prof.v1 = v1;
    prof.v2 = sweep2.farthest_vertex;
    prof.n_reached    = sweep2.reached;
    prof.eccentricity = static_cast<int>(sweep2.level_sizes.size()) - 1;
    prof.max_level_size = *std::max_element(sweep2.level_sizes.begin(), sweep2.level_sizes.end());
    prof.max_level_fraction =
        prof.n_reached > 0 ? static_cast<double>(prof.max_level_size) / prof.n_reached : 0.0;

    if (prof.eccentricity <= min_eccentricity_for_slope_fit) {
        prof.fit_status = GrowthFitStatus::kTooShallow;
        return prof;
    }

    // OLS fit of log(cumulative ball size |B(v1,r)|) against log(r) for r = 1..eccentricity.
    std::vector<double> log_r, log_b;
    log_r.reserve(prof.eccentricity);
    log_b.reserve(prof.eccentricity);
    long long cumulative = sweep2.level_sizes[0];
    for (int r = 1; r <= prof.eccentricity; ++r) {
        cumulative += sweep2.level_sizes[r];
        log_r.push_back(std::log(static_cast<double>(r)));
        log_b.push_back(std::log(static_cast<double>(cumulative)));
    }
    const int m = static_cast<int>(log_r.size());
    const double mean_x = std::accumulate(log_r.begin(), log_r.end(), 0.0) / m;
    const double mean_y = std::accumulate(log_b.begin(), log_b.end(), 0.0) / m;
    double cov = 0.0, var = 0.0;
    for (int i = 0; i < m; ++i) {
        cov += (log_r[i] - mean_x) * (log_b[i] - mean_y);
        var += (log_r[i] - mean_x) * (log_r[i] - mean_x);
    }
    prof.fit_status   = GrowthFitStatus::kFitted;
    prof.growth_slope = (var > 0.0) ? (cov / var) : 0.0;
    return prof;
}

struct BfsAdjacencyResult {
    std::vector<std::vector<int>> adjacency;         // over the KEPT vertices only, reindexed
    std::vector<int> kept_original_index;            // kept_original_index[new_idx] = original_idx
    std::vector<int> dropped_hub_original_index;     // original indices dropped as hubs, ascending
};

// Top-K-by-degree considered for dropping...
inline constexpr int kHubVerticesToDrop = 8;
// ...but only if its degree exceeds this multiplier times the graph's median degree, so a
// genuinely uniform mesh (max degree ~= median degree) loses nothing.
inline constexpr double kHubDegreeMultiplier = 20.0;

// Builds the BFS adjacency graph for A_sym (a full, already-symmetric sparse matrix), dropping the
// diagonal (self-loops never affect BFS distances) and up to `hubs_to_drop` of the highest-degree
// vertices whose degree exceeds `hub_degree_multiplier` times the median degree. O(nnz(A_sym)).
template <typename SpMat>
BfsAdjacencyResult build_bfs_adjacency(const SpMat& A_sym, int hubs_to_drop = kHubVerticesToDrop,
                                        double hub_degree_multiplier = kHubDegreeMultiplier) {
    const int n = static_cast<int>(A_sym.rows());
    BfsAdjacencyResult result;
    if (n == 0) return result;

    std::vector<int> degree(n, 0);
    for (Eigen::Index k = 0; k < A_sym.outerSize(); ++k)
        for (typename SpMat::InnerIterator it(A_sym, k); it; ++it)
            if (it.row() != it.col()) ++degree[it.col()]; // A_sym symmetric: col-count == true degree

    std::vector<int> sorted_degree = degree;
    std::nth_element(sorted_degree.begin(), sorted_degree.begin() + n / 2, sorted_degree.end());
    const double hub_floor = hub_degree_multiplier * std::max(static_cast<double>(sorted_degree[n / 2]), 1.0);

    std::vector<int> by_degree(n);
    std::iota(by_degree.begin(), by_degree.end(), 0);
    const int k = std::min(hubs_to_drop, n);
    std::partial_sort(by_degree.begin(), by_degree.begin() + k, by_degree.end(),
                       [&](int a, int b) { return degree[a] != degree[b] ? degree[a] > degree[b] : a < b; });

    std::vector<bool> is_dropped(n, false);
    for (int i = 0; i < k; ++i) {
        const int v = by_degree[i];
        if (degree[v] > hub_floor) {
            is_dropped[v] = true;
            result.dropped_hub_original_index.push_back(v);
        }
    }
    std::sort(result.dropped_hub_original_index.begin(), result.dropped_hub_original_index.end());

    std::vector<int> new_index(n, -1);
    for (int i = 0; i < n; ++i)
        if (!is_dropped[i]) {
            new_index[i] = static_cast<int>(result.kept_original_index.size());
            result.kept_original_index.push_back(i);
        }

    result.adjacency.resize(result.kept_original_index.size());
    for (Eigen::Index col = 0; col < A_sym.outerSize(); ++col) {
        if (is_dropped[col]) continue;
        for (typename SpMat::InnerIterator it(A_sym, col); it; ++it) {
            const int r = static_cast<int>(it.row()), c = static_cast<int>(it.col());
            if (r == c || is_dropped[r]) continue;
            result.adjacency[new_index[c]].push_back(new_index[r]);
        }
    }
    return result;
}

inline constexpr int    kMeshLikeEccentricityThreshold = 25;
inline constexpr double kExpanderLikeMaxLevelFraction  = 0.5;
inline constexpr double kMeshLikeGrowthSlopeMin        = 1.5;
inline constexpr double kMeshLikeGrowthSlopeMax        = 5.0;

inline bool bfs_predicts_mesh_like(const BfsStructuralProfile& p,
                                    int mesh_eccentricity_threshold = kMeshLikeEccentricityThreshold,
                                    double expander_max_level_fraction = kExpanderLikeMaxLevelFraction,
                                    double mesh_growth_slope_min = kMeshLikeGrowthSlopeMin,
                                    double mesh_growth_slope_max = kMeshLikeGrowthSlopeMax) {
    if (p.fit_status == GrowthFitStatus::kTooShallow) return false;         // expander-like signal
    if (p.max_level_fraction >= expander_max_level_fraction) return false;  // wide/bushy => expander-like
    if (p.eccentricity < mesh_eccentricity_threshold) return false;         // too shallow to look mesh-like
    return p.growth_slope >= mesh_growth_slope_min && p.growth_slope <= mesh_growth_slope_max;
}

// ------ Top-level cascade ------

enum class DecisionScreen {
    kSizeThreshold,    // Part 1 decided: n < kSmallProblemThreshold, AMD used unconditionally
    kAmdFillRatio,     // Part 2 decided: AMD's own fill ratio already looked good enough
    kBfsStructural,    // Part 3 decided: BFS structural read broke the Part-2 ambiguity
    kMetisUnavailable, // Part 2 said "not good enough" but METIS isn't compiled in to escalate to
};

inline const char* screen_name(DecisionScreen s) {
    switch (s) {
        case DecisionScreen::kSizeThreshold:    return "size";
        case DecisionScreen::kAmdFillRatio:     return "amd_fill_ratio";
        case DecisionScreen::kBfsStructural:    return "bfs_structural";
        case DecisionScreen::kMetisUnavailable: return "metis_unavailable";
    }
    return "unknown";
}

struct OrderingSelectConfig {
    Eigen::Index small_problem_threshold = kSmallProblemThreshold;
    long long fill_input_ratio = kFillInputRatio;
    int hubs_to_drop = kHubVerticesToDrop;
    double hub_degree_multiplier = kHubDegreeMultiplier;
    int min_eccentricity_for_slope_fit = kMinEccentricityForSlopeFit;
    int mesh_eccentricity_threshold = kMeshLikeEccentricityThreshold;
    double expander_max_level_fraction = kExpanderLikeMaxLevelFraction;
    double mesh_growth_slope_min = kMeshLikeGrowthSlopeMin;
    double mesh_growth_slope_max = kMeshLikeGrowthSlopeMax;
};

// Which tier decided, and its diagnostics.
// Note: METIS's own analyzePattern() is never run anywhere in this cascade.
struct Decision {
    std::string winner;
    DecisionScreen screen = DecisionScreen::kSizeThreshold;
    AmdProbeStats amd;         // sentinel (-1) fields iff screen == kSizeThreshold
    BfsStructuralProfile bfs;  // default-constructed iff screen != kBfsStructural

    // True iff `winner`'s own analyzePattern() was actually run by this cascade (Part 2's probe).
    bool winner_was_probed() const {
        return winner == "AMD" && screen != DecisionScreen::kSizeThreshold;
    }
};

template <typename SpMat, typename Index>
Decision select_ordering(const SpMat& A_sym, const OrderingSelectConfig& cfg = OrderingSelectConfig()) {
    Decision result;

    if (is_small_problem(A_sym, cfg.small_problem_threshold)) {
        result.winner = "AMD";
        result.screen = DecisionScreen::kSizeThreshold;
        return result;
    }

    result.amd = probe_amd<SpMat, Index>(A_sym);
    if (amd_is_good_enough(result.amd, cfg.fill_input_ratio)) {
        result.winner = "AMD";
        result.screen = DecisionScreen::kAmdFillRatio;
        return result;
    }

#ifdef KSP_QP_HAVE_METIS
    // Eigen::MetisOrdering passes StorageIndex buffers straight to METIS_NodeND's idx_t*
    // parameters with no size check; METIS is built here with IDXTYPEWIDTH=32 (CMakeLists.txt)
    // specifically to match. This is the compile-time tripwire against that ever drifting.
    static_assert(sizeof(Index) == sizeof(std::int32_t),
                  "METIS was built with IDXTYPEWIDTH=32 (see CMakeLists.txt); Eigen's "
                  "StorageIndex must be 32-bit or MetisOrdering will silently corrupt memory.");
    const BfsAdjacencyResult adj = build_bfs_adjacency(A_sym, cfg.hubs_to_drop, cfg.hub_degree_multiplier);
    result.bfs = bfs_profile(adj.adjacency, cfg.min_eccentricity_for_slope_fit);
    result.bfs.dropped_hub_vertices = adj.dropped_hub_original_index;
    result.screen = DecisionScreen::kBfsStructural;
    result.winner = bfs_predicts_mesh_like(result.bfs, cfg.mesh_eccentricity_threshold,
                                            cfg.expander_max_level_fraction, cfg.mesh_growth_slope_min,
                                            cfg.mesh_growth_slope_max)
                        ? "METIS"
                        : "AMD";
#else
    // Part 2 said "not good enough," but there's nothing to escalate to -- not a genuine judgment
    // that AMD is fine, just the only option available.
    result.winner = "AMD";
    result.screen = DecisionScreen::kMetisUnavailable;
#endif
    return result;
}

// Runs the analyzePattern()+factorize() retry loop (regularizing a meaningfully negative pivot)
// with the given Ordering. Returns the (correctly un-permuted) lower-triangular L such that
// L*L^T approximates the original symmetric matrix.
template <typename SpMat, typename Ordering, typename T>
SpMat factorize_L_with_ordering(SpMat& Q_reg, const std::vector<int>& diag_idx,
                                 const Eigen::Matrix<T, Eigen::Dynamic, 1>& Q_diag, T Q_scale,
                                 T delta_noise, int max_attempts, T& delta, bool& accepted,
                                 bool& clamped, Eigen::Matrix<T, Eigen::Dynamic, 1>& D) {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Ordering> ldlt;
    ldlt.analyzePattern(Q_reg); // sparsity pattern is fixed across retries below; analyze once

    accepted = false;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        ldlt.factorize(Q_reg);
        if (ldlt.info() == Eigen::Success) {
            D = ldlt.vectorD();
            if (D.minCoeff() >= -delta_noise) {
                accepted = true;
                break;
            }
        }
        delta = (delta == T(0)) ? std::sqrt(std::numeric_limits<T>::epsilon()) * Q_scale : delta * T(10);
        for (int k = 0; k < static_cast<int>(diag_idx.size()); ++k)
            Q_reg.valuePtr()[diag_idx[k]] = Q_diag(k) + delta;
    }

    if (!accepted) return SpMat();

    clamped = (D.minCoeff() < T(0));
    Vec D_sqrt = D.cwiseMax(T(0)).cwiseSqrt();
    auto P = ldlt.permutationP();
    SpMat L_D = ldlt.matrixL();
    return (P.transpose() * L_D) * D_sqrt.asDiagonal();
}

// ------ Type-erased solver, for call sites that keep a solver alive across many calls ------
//
// This interface covers exactly analyzePattern/factorize/info/solve which are needed for
// schur_preconditioner.hpp's Schur/KKT preconditioner.
template <typename SpMat>
class ISymmetricSolver {
public:
    using Scalar = typename SpMat::Scalar;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    virtual ~ISymmetricSolver() = default;
    virtual void analyzePattern(const SpMat& P) = 0;
    virtual void factorize(const SpMat& P) = 0;
    virtual Eigen::ComputationInfo info() const = 0;
    virtual Vec solve(const Vec& rhs) const = 0;
    virtual Mat solve(const Mat& rhs) const = 0;
};

template <typename SpMat, typename Ordering, bool IsLdlt>
class SymmetricSolverImpl : public ISymmetricSolver<SpMat> {
public:
    using Base = ISymmetricSolver<SpMat>;
    using Vec = typename Base::Vec;
    using Mat = typename Base::Mat;
    using EigenSolver = std::conditional_t<IsLdlt,
                                            Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Ordering>,
                                            Eigen::SimplicialLLT<SpMat, Eigen::Lower, Ordering>>;

    void analyzePattern(const SpMat& P) override { solver_.analyzePattern(P); }
    void factorize(const SpMat& P) override { solver_.factorize(P); }
    Eigen::ComputationInfo info() const override { return solver_.info(); }
    Vec solve(const Vec& rhs) const override { return solver_.solve(rhs); }
    Mat solve(const Mat& rhs) const override { return solver_.solve(rhs); }

private:
    EigenSolver solver_;
};

// Constructs the wrapper for the named winner ("AMD"/"METIS").
// IsLdlt selects SimplicialLDLT (true) vs SimplicialLLT (false).
template <typename SpMat, bool IsLdlt>
std::unique_ptr<ISymmetricSolver<SpMat>> make_solver(const std::string& ordering_name) {
    using Idx = typename SpMat::StorageIndex;
    if (ordering_name == "AMD")
        return std::make_unique<SymmetricSolverImpl<SpMat, Eigen::AMDOrdering<Idx>, IsLdlt>>();
#ifdef KSP_QP_HAVE_METIS
    if (ordering_name == "METIS")
        return std::make_unique<SymmetricSolverImpl<SpMat, Eigen::MetisOrdering<Idx>, IsLdlt>>();
#endif
    throw std::logic_error("ordering_select::make_solver: unknown ordering name '" + ordering_name + "'");
}

} // namespace ordering_select
