#pragma once
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#ifdef KSP_QP_HAVE_METIS
#include <iostream>  // Eigen's MetisSupport.h uses std::cerr without including this itself
#include <Eigen/MetisSupport>
#endif
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// CHOLMOD-like ordering selection: try each fill-reducing ordering Eigen can offer for a
// symmetric sparse matrix (AMD, and -- if KSP_QP_HAVE_METIS -- METIS's nested dissection),
// measure the resulting fill via analyzePattern() alone (cheap, near-linear; no factorize()
// needed -- Eigen already knows L's exact symbolic nnz once analyzePattern() returns), and
// report the winner by least fill. See include/ksp_qp.tpp's set_L_from_LLT and
// include/schur_preconditioner.hpp for the call sites using this.
//
// NaturalOrdering (no reordering) is deliberately not a candidate here, not merely a bad choice
// among others: on a genuinely indefinite input (e.g. schur_preconditioner.hpp's
// P_hat = [-H_act, G_act^T; G_act, (1/mu)I]), Natural ordering processes variables in raw index
// order, which for that matrix means eliminating the entire negative-diagonal -H_act block
// before ever touching the positive (1/mu)I block -- a long run of same-signed pivots. Eigen's
// SimplicialLDLT has no Bunch-Kaufman-style dynamic pivoting, so a degenerate pivot produced
// along such a sequence isn't caught; it was observed in practice to segfault inside
// factorize_preordered() at large scale (nc=8, 462k-row P_hat) despite Natural having reported
// the *least* predicted fill of the three candidates tried at the time -- i.e. nnz(L) alone
// can't detect this risk. Do not re-add it without also handling that.
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
    // TEMPORARY (isolated AMD-vs-METIS runtime experiment, revert before committing): when set,
    // skip probing the other ordering entirely so the forced one's own factorize/solve time is
    // measured with zero trial overhead, isolating "is this ordering itself faster" from "is
    // paying for the trial-and-compare worth it".
    const char* forced = std::getenv("KSP_QP_FORCE_ORDERING");

    Result result;
    if (!forced || std::string(forced) == "AMD")
        result.candidates.push_back(probe<SpMat, Eigen::AMDOrdering<Index>>("AMD", A_sym));
#ifdef KSP_QP_HAVE_METIS
    // Eigen::MetisOrdering passes StorageIndex buffers straight to METIS_NodeND's idx_t*
    // parameters with no size check; METIS is built here with IDXTYPEWIDTH=32 (CMakeLists.txt)
    // specifically to match. This is the compile-time tripwire against that ever drifting.
    static_assert(sizeof(Index) == sizeof(std::int32_t),
                  "METIS was built with IDXTYPEWIDTH=32 (see CMakeLists.txt); Eigen's "
                  "StorageIndex must be 32-bit or MetisOrdering will silently corrupt memory.");
    if (!forced || std::string(forced) == "METIS")
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

// ------ Type-erased solver, for call sites that keep a solver alive across many calls ------
//
// set_L_from_LLT (above) is a one-shot local factorization, so picking the ordering and
// constructing the concrete Eigen solver type in the same expression is enough. A call site
// that instead holds a solver as long-lived state (e.g. schur_preconditioner.hpp's Schur/KKT
// preconditioner, factorized on nearly every SSN iteration but only re-analyzed on an
// active-set change) needs to store "whichever ordering won" behind a stable type. Rather than
// a variant over {AMD,METIS} x {LDLT,LLT} (4 concrete alternatives), wrap whichever one is
// live behind this interface -- it covers exactly analyzePattern/factorize/info/solve, the
// only methods such a call site needs (no vectorD/permutationP/matrixL, unlike
// factorize_L_with_ordering above, which needs those to extract L).
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

// Constructs the wrapper for the named winner ("AMD"/"METIS", the same names try_orderings()
// returns). IsLdlt selects SimplicialLDLT (true) vs SimplicialLLT (false).
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
