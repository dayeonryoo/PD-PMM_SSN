#pragma once
#include <cassert>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <vector>
#include <stdexcept>

template <typename T>
class SchurPreconditioner {
public:
    using Scalar = T;
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    using StorageIndex = Eigen::Index;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    SchurPreconditioner() = default;

    SchurPreconditioner(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const BoolArr& active_K, T mu)
        : G_(&G), G_tr_(&G_tr), H_diag_(&H_diag), active_K_(&active_K), mu_(mu) {}

    // prec_pattern_changed must be true whenever G's sparsity structure or active_K changes:
    // P = G E G^T + (1/mu) I, where E_ii = 1/H_diag_i if active_K_i else 0.
    // Both G's column structure and which entries of E are nonzero determine P's sparsity.
    // G E G^T is independent of mu (active_K entries have H_diag = 1/rho, not mu).
    // So when only mu changes, we skip the expensive G*E*G^T product and only update
    // the (1/mu)I diagonal in P_ before refactorizing.
    void setData(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const BoolArr& active_K, T mu, bool rebuild, bool prec_pattern_changed) {
        G_ = &G;
        G_tr_ = &G_tr;
        H_diag_ = &H_diag;
        active_K_ = &active_K;
        mu_ = mu;
        bool size_changed = (P_.rows() != G.rows()) || (P_.cols() != G.rows());
        if (!pattern_analyzed_ || prec_pattern_changed || size_changed) {
            pattern_dirty_ = true;
        }
        if (prec_pattern_changed || size_changed) {
            base_dirty_ = true;
        }
        // A size change means llt_'s stored factorization has the wrong dimension;
        // rebuild regardless of whether the caller considers the change "significant".
        rebuild_ = rebuild || size_changed;
    }

    template <typename MatrixType>
    SchurPreconditioner& compute(const MatrixType&) {
        // Also rebuild when only mu changed: G E G^T is reused, only diagonal shifts.
        bool mu_changed = initialized_ && (mu_ != mu_at_last_fact_);
        if (!initialized_ || rebuild_ || mu_changed) {
            build();
            initialized_ = true;
        }
        return *this;
    }

    template <typename Rhs>
    Vec solve(const Eigen::MatrixBase<Rhs>& b) const {
        if (info_ != Eigen::Success)
            throw std::runtime_error("SchurPreconditioner solve called after failed factorization.");
        return llt_.solve(b);
    }

    Eigen::ComputationInfo info() const { return info_; }
    int fact_count() const { return fact_count_; }

private:
    // Build P = G E G^T + (1/mu) I.
    // If base_dirty_: recompute P_base_ = G E G^T (expensive sparse product), then build P_.
    // If only mu changed: update the (1/mu)I diagonal of P_ in-place (cheap), then refactorize.
    void build() {
        const SpMat& G = *G_;
        const SpMat& G_tr = *G_tr_;
        const Vec& H_diag = *H_diag_;
        const BoolArr& active_K = *active_K_;

        const Eigen::Index s = G.rows();
        const Eigen::Index n = G.cols();

        assert(G_tr.rows() == n);
        assert(G_tr.cols() == s);
        assert(H_diag.size() == n);
        assert(active_K.size() == n);

        if (base_dirty_) {
            // Rebuild G E G^T: E_ii = 1/H_diag(i) if active_K(i), else 0.
            // active_K(i) implies diag_P_K(i)=1, so H_diag(i) = 1/rho (independent of mu).
            std::vector<Triplet> trips;
            trips.reserve(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                if (active_K(i))
                    trips.emplace_back(i, i, T(1) / H_diag(i));
            }
            SpMat E(n, n);
            E.setFromTriplets(trips.begin(), trips.end());
            E.makeCompressed();

            P_base_ = G * E * G_tr;
            base_dirty_ = false;

            P_ = P_base_;
            for (Eigen::Index i = 0; i < s; ++i)
                P_.coeffRef(i, i) += T(1) / mu_;
            P_.makeCompressed();
        } else {
            // Only mu changed: shift the (1/mu)I diagonal by delta.
            // P_base_ is unchanged; all diagonal entries of P_ already exist from prior build.
            const T delta = T(1) / mu_ - T(1) / mu_at_last_fact_;
            for (Eigen::Index i = 0; i < s; ++i)
                P_.coeffRef(i, i) += delta;
        }

        // Symbolic analysis only when P's sparsity pattern may have changed.
        if (pattern_dirty_) {
            llt_.analyzePattern(P_);
            pattern_analyzed_ = true;
            pattern_dirty_ = false;
            // std::cout << "Symbolic factoriztion.\n";
        }
        // std::cout << "Numeric factorization.\n";
        llt_.factorize(P_);
        info_ = llt_.info();
        fact_count_++;
        mu_at_last_fact_ = mu_;
    }

    const SpMat* G_ = nullptr;
    const SpMat* G_tr_ = nullptr;
    const Vec* H_diag_ = nullptr;
    const BoolArr* active_K_ = nullptr;
    T mu_ = T(1);

    bool rebuild_ = true;
    bool initialized_ = false;
    bool pattern_analyzed_ = false;
    bool pattern_dirty_ = true;
    bool base_dirty_ = true;       // true when G E G^T must be recomputed
    T mu_at_last_fact_ = T(-1);    // mu used in the last factorization

    SpMat P_base_;                 // cached G E G^T without the (1/mu)I shift
    SpMat P_;
    Eigen::SimplicialLDLT<SpMat> llt_;
    Eigen::ComputationInfo info_ = Eigen::Success;
    int fact_count_ = 0;
};
