#pragma once
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
    void setData(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const BoolArr& active_K, T mu, bool rebuild, bool prec_pattern_changed) {
        G_ = &G;
        G_tr_ = &G_tr;
        H_diag_ = &H_diag;
        active_K_ = &active_K;
        mu_ = mu;
        rebuild_ = rebuild;

        bool size_changed = (P_.rows() != G.rows()) || (P_.cols() != G.rows());
        if (!pattern_analyzed_ || prec_pattern_changed || size_changed) {
            pattern_dirty_ = true;
        }
    }

    template <typename MatrixType>
    SchurPreconditioner& compute(const MatrixType&) {
        if (!initialized_ || rebuild_) {
            build();
            initialized_ = true;
        }
        return *this;
    }

    template <typename Rhs>
    Vec solve(const Eigen::MatrixBase<Rhs>& b) const {
        if (info_ != Eigen::Success) {
            throw std::runtime_error("SchurPreconditioner solve called after failed factorization.");
        }
        return llt_.solve(b);
    }

    Eigen::ComputationInfo info() const { return info_; }

private:
    // Build a preconditioner P = G E G^T + (1/mu) I
    // active_K(i) is true when diag_P_K(i) = 1
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

        // Build E diagonal as a sparse diagonal matrix
        std::vector<Triplet> trips;
        trips.reserve(n);

        for (Eigen::Index i = 0; i < n; ++i) {
            if (active_K(i)) {
                trips.emplace_back(i, i, T(1) / H_diag(i));
            }
        }
        SpMat E(n, n);
        E.setFromTriplets(trips.begin(), trips.end());
        E.makeCompressed();

        // Build (1/mu) I as a sparse diagonal matrix
        // std::vector<Triplet> Itrips;
        // Itrips.reserve(s);
        // for (Eigen::Index i = 0; i < s; ++i) {
        //     Itrips.emplace_back(i, i, T(1) / mu_);
        // }
        // SpMat Ishift(s, s);
        // Ishift.setFromTriplets(Itrips.begin(), Itrips.end());

        // // Build P = G E G^T + (1/mu) I
        // P_ = G * E * G_tr + Ishift;
        P_ = G * E * G_tr;
        for (Eigen::Index i = 0; i < s; ++i) {
            P_.coeffRef(i, i) += T(1) / mu_;
        }
        P_.makeCompressed();

        // Symbolic analysis only when P's sparsity pattern may have changed.
        if (pattern_dirty_) {
            llt_.analyzePattern(P_);
            pattern_analyzed_ = true;
            pattern_dirty_ = false;
        }
        llt_.factorize(P_);
        info_ = llt_.info();

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

    SpMat P_;
    Eigen::SimplicialLLT<SpMat> llt_;
    Eigen::ComputationInfo info_ = Eigen::Success;
};