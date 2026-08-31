#pragma once
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#ifdef KSP_QP_HAVE_METIS
#include <Eigen/MetisSupport>
#endif
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// CHOLMOD-like ordering selection: try each fill-reducing ordering Eigen can offer for a
// symmetric sparse matrix (AMD, Natural, and -- if KSP_QP_HAVE_METIS -- METIS's nested
// dissection), measure the resulting fill via analyzePattern() alone (cheap, near-linear;
// no factorize() needed -- Eigen already knows L's exact symbolic nnz once analyzePattern()
// returns), and report the winner by least fill. See include/ksp_qp.tpp's set_L_from_LLT for
// the one call site currently using this.
namespace ordering_select {

struct Candidate {
    std::string name;
    long long nnz_l;
    double analyze_seconds;
};

struct Result {
    std::vector<Candidate> candidates;
    std::string winner;
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

// Trial-runs analyzePattern() with each available ordering on A_sym (a full, already-symmetric
// sparse matrix -- not just the lower triangle), measuring wall time and resulting fill without
// ever factorizing. Returns per-candidate stats and the name of the winner (least nnz_l).
template <typename SpMat, typename Index>
Result try_orderings(const SpMat& A_sym) {
    Result result;
    result.candidates.push_back(probe<SpMat, Eigen::AMDOrdering<Index>>("AMD", A_sym));
    result.candidates.push_back(probe<SpMat, Eigen::NaturalOrdering<Index>>("Natural", A_sym));
#ifdef KSP_QP_HAVE_METIS
    // Eigen::MetisOrdering passes StorageIndex buffers straight to METIS_NodeND's idx_t*
    // parameters with no size check; METIS is built here with IDXTYPEWIDTH=32 (CMakeLists.txt)
    // specifically to match. This is the compile-time tripwire against that ever drifting.
    static_assert(sizeof(Index) == sizeof(std::int32_t),
                  "METIS was built with IDXTYPEWIDTH=32 (see CMakeLists.txt); Eigen's "
                  "StorageIndex must be 32-bit or MetisOrdering will silently corrupt memory.");
    result.candidates.push_back(probe<SpMat, Eigen::MetisOrdering<Index>>("METIS", A_sym));
#endif

    result.winner = result.candidates.front().name;
    long long best_nnz = result.candidates.front().nnz_l;
    for (const auto& c : result.candidates) {
        if (c.nnz_l < best_nnz) { best_nnz = c.nnz_l; result.winner = c.name; }
    }
    return result;
}

// Runs the real analyzePattern()+factorize() retry loop (escalating regularization on a
// meaningfully negative pivot) with the given Ordering, mirroring KSP_QP<T>::set_L_from_LLT's
// original single-ordering logic. Q_reg is mutated in place across retries (its diagonal is
// re-patched via diag_idx on each escalation) exactly as before. Returns the (correctly
// un-permuted) lower-triangular L such that L*L^T approximates the original symmetric matrix;
// writes back delta/accepted/clamped/D for the caller's post-processing (null-row scrubbing,
// residual verification) to use unchanged.
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

} // namespace ordering_select
