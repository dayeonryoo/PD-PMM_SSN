#pragma once
#include <cassert>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/LU>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <variant>

template <typename T>
class SchurPreconditioner {
public:
    using Scalar = T;
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    using StorageIndex = Eigen::Index;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using SpMat = Eigen::SparseMatrix<T>;
    using RowMajorSpMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using Triplet = Eigen::Triplet<T>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    // ------ Setup ------

    SchurPreconditioner() = default;

    SchurPreconditioner(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const BoolArr& active_K, T mu)
        : G_(&G), G_tr_(&G_tr), H_diag_(&H_diag), active_K_(&active_K), mu_(mu) {}

    void setData(const SpMat& G, const SpMat& G_tr, const Vec& H_diag,
                 const BoolArr& active_K, const BoolArr& active_W,
                 const RowMajorSpMat& B_rm,
                 T mu, bool rebuild, bool prec_pattern_changed) {
        // Store data pointers and set flags for refactorization.
        G_        = &G;
        G_tr_     = &G_tr;
        H_diag_   = &H_diag;
        active_K_ = &active_K;
        active_W_ = &active_W;
        B_rm_     = &B_rm; // row-major B matrix
        mu_       = mu;

        // Detect if refactorization is needed.
        bool size_changed = (s_current_ != static_cast<int>(G.rows()));
        if (!pattern_analyzed_ || prec_pattern_changed || size_changed)
            pattern_dirty_ = true;

        // Detect if G E G^T needs to be recomputed.
        if (prec_pattern_changed || size_changed)
            base_dirty_ = true;

        // Detect if P needs to be rebuild.
        rebuild_ = rebuild || size_changed;

        // M_rows_ = number of equality-constraint (A) rows = G.rows() - n_active_W; constant.
        if (M_rows_ < 0 && static_cast<int>(G.rows()) > 0)
            M_rows_ = static_cast<int>(G.rows()) - static_cast<int>(active_W.count());
    }

    // ------ Factorization ------

    template <typename MatrixType>
    SchurPreconditioner& compute(const MatrixType&) {
        bool mu_changed = initialized_ && (mu_ != mu_at_last_fact_);
        if (!initialized_ || rebuild_ || mu_changed) {
            build();
            initialized_ = true;
        }
        return *this;
    }

    // ------ Application ------

    template <typename Rhs>
    const Vec& solve(const Eigen::MatrixBase<Rhs>& b) const {
        if (info_ != Eigen::Success)
            throw std::runtime_error("SchurPreconditioner solve called after failed factorization.");

        if (!use_smw_) {
            if (use_ldlt_) {
                // Solve hat_P [w1; w2] = [0; b]; return w2 = P^{-1} b.
                const int s = static_cast<int>(b.size());
                ldlt_rhs_.resize(n_act_ + s);
                ldlt_rhs_.head(n_act_).setZero();
                ldlt_rhs_.tail(s) = b;
                direct_result_ = std::get<LdltSolver>(active_solver_).ldlt.solve(ldlt_rhs_).tail(s);
            } else {
                direct_result_ = std::get<CholSolver>(active_solver_).llt.solve(b);
            }
            return direct_result_;
        }

        // ---- SMW Application Phase ----

        r_pad_.head(M_rows_) = b.head(M_rows_);
        for (int i = 0; i < static_cast<int>(retained_old_rows_.size()); ++i)
            r_pad_(retained_old_rows_[i]) = b(retained_new_rows_[i]);

        // u_base = P_old^-1 r_pad
        if (use_ldlt_) {
            const int s = r_pad_.size();
            ldlt_rhs_.resize(n_act_ + s);
            ldlt_rhs_.head(n_act_).setZero();
            ldlt_rhs_.tail(s) = r_pad_;
            u_base_ = std::get<LdltSolver>(active_solver_).ldlt.solve(ldlt_rhs_).tail(s);
        } else {
            u_base_ = std::get<CholSolver>(active_solver_).llt.solve(r_pad_);
        }

        // Lambda = g_all - V_all^T u_base.
        //   E_-^T u_base
        for (int k = 0; k < h_; ++k)
            Lambda_all_(k) = -u_base_(deleted_old_rows_[k]);
        //   U^T u_base 
        for (int j = 0; j < p_; ++j) {
            T val = T(0);
            for (typename SpMat::InnerIterator it(G_old_, delta_K_idx_[j]); it; ++it)
                val += it.value() * u_base_(it.row());
            Lambda_all_(h_ + j) = -val;
        }
        //   V_+^T u_base - r_2
        for (int j = 0; j < q_; ++j)
            Lambda_all_(h_ + p_ + j) = b(added_new_rows_[j]);
        if (q_ > 0)
            Lambda_all_.tail(q_).noalias() -= V_plus_.transpose() * u_base_;

        // Capacitance solve
        lambda_work_ = S_lambda_lu_.solve(Lambda_all_);
        Lambda_all_  = lambda_work_;

        // z_base = u_base - Y_all * Lambda_all
        z_base_  = u_base_;
        z_base_.noalias() -= Y_all_ * Lambda_all_;

        // Assemble z_new.
        z_new_.head(M_rows_) = z_base_.head(M_rows_);
        for (int i = 0; i < static_cast<int>(retained_old_rows_.size()); ++i)
            z_new_(retained_new_rows_[i]) = z_base_(retained_old_rows_[i]);
        for (int j = 0; j < q_; ++j)
            z_new_(added_new_rows_[j]) = Lambda_all_(h_ + p_ + j);

        return z_new_;
    }

    // ------ Diagnostics & SMW control ------

    Eigen::ComputationInfo info() const { return info_; }
    int fact_count() const { return fact_count_; }
    int smw_count()  const { return smw_count_; }
    bool used_smw()  const { return use_smw_; }

    void force_full_rebuild() { skip_smw_ = true; base_dirty_ = true; }
    void record_smw_rebuild() {
        smw_fail_streak_++;
        smw_fail_total_++;
        if (smw_fail_streak_ >= kMaxSmwFailStreak) {
            SpMat().swap(G_old_);
            H_diag_old_.resize(0);
            active_K_old_.resize(0);
            active_W_old_.resize(0);
        }
    }
    void reset_smw_fail_streak() { smw_fail_streak_ = 0; }
    bool smw_suppressed() const {
        if (smw_fail_streak_ >= kMaxSmwFailStreak) return true;
        else return false;
    }
    void set_use_ldlt(bool flag) { use_ldlt_ = flag; }

    // Free all factorization/SMW state once the caller has permanently switched away from PCG.
    void release() {
        active_solver_.template emplace<std::monostate>();

        SpMat().swap(G_old_);
        H_diag_old_.resize(0);
        active_K_old_.resize(0);
        active_W_old_.resize(0);

        std::vector<int>().swap(deleted_old_rows_);
        std::vector<int>().swap(retained_old_rows_);
        std::vector<int>().swap(retained_new_rows_);
        std::vector<int>().swap(added_new_rows_);
        std::vector<int>().swap(added_W_src_);
        std::vector<int>().swap(delta_K_idx_);
        std::vector<int>().swap(ldlt_act_idx_);
        std::vector<Triplet>().swap(ldlt_build_trips_);
        std::vector<Triplet>().swap(chol_build_trips_);
        V_plus_.resize(0, 0);
        Y_all_.resize(0, 0);
        S_lambda_lu_ = Eigen::FullPivLU<Mat>();

        r_pad_.resize(0);
        u_base_.resize(0);
        Lambda_all_.resize(0);
        lambda_work_.resize(0);
        z_base_.resize(0);
        z_new_.resize(0);
        direct_result_.resize(0);
        ldlt_rhs_.resize(0);

        G_ = nullptr; G_tr_ = nullptr; H_diag_ = nullptr; active_K_ = nullptr; active_W_ = nullptr; B_rm_ = nullptr;

        initialized_      = false;
        rebuild_          = true;
        pattern_analyzed_ = false;
        pattern_dirty_    = true;
        base_dirty_       = true;
        use_smw_          = false;
        skip_smw_         = false;
    }

private:
    // ------ Formation and Cholesky factorization ------

    void build() {
        use_smw_ = false;

        // Case 1. Skip build() and reuse the cached factorization via low-rank SMW update.
        if (initialized_ && info_ == Eigen::Success && try_build_smw())
            return;
        V_plus_.resize(0, 0);
        Y_all_.resize(0, 0);

        // Case 2. Full factorization.
        if (use_ldlt_) {
            if (!std::holds_alternative<LdltSolver>(active_solver_))
                active_solver_.template emplace<LdltSolver>();
            factorize_by_ldlt();
        } else {
            if (!std::holds_alternative<CholSolver>(active_solver_))
                active_solver_.template emplace<CholSolver>();
            factorize_by_chol();
        }
    }

    // Build P_hat = [-H_act, G_act^T; G_act, (1/mu)I] and factorize with LDLT.
    void factorize_by_ldlt() {
        const SpMat&   G        = *G_;
        const Vec&     H_diag   = *H_diag_;
        const BoolArr& active_K = *active_K_;

        const Eigen::Index s = G.rows();
        const Eigen::Index n = G.cols();

        // Collect active K indices.
        ldlt_act_idx_.clear();
        ldlt_act_idx_.reserve(n);
        for (Eigen::Index i = 0; i < n; ++i)
            if (active_K(i)) ldlt_act_idx_.push_back(static_cast<int>(i));
        const int n_act = static_cast<int>(ldlt_act_idx_.size());

        // Build P_hat of size (n_act + s) x (n_act + s).
        ldlt_build_trips_.clear();
        ldlt_build_trips_.reserve(n_act + 2 * static_cast<int>(G.nonZeros()) + static_cast<int>(s));

        // Top-left block: -H_act (diagonal, n_act x n_act).
        for (int k = 0; k < n_act; ++k)
            ldlt_build_trips_.emplace_back(k, k, -H_diag(ldlt_act_idx_[k]));

        // Off-diagonal blocks: G_act (s x n_act) and G_act^T (n_act x s).
        for (int k = 0; k < n_act; ++k)
            for (typename SpMat::InnerIterator it(G, ldlt_act_idx_[k]); it; ++it) {
                const int row = static_cast<int>(it.row());
                ldlt_build_trips_.emplace_back(n_act + row, k,          it.value()); // G_act
                ldlt_build_trips_.emplace_back(k,           n_act + row, it.value()); // G_act^T
            }

        // Bottom-right block: (1/mu) I_s.
        for (Eigen::Index i = 0; i < s; ++i)
            ldlt_build_trips_.emplace_back(n_act + i, n_act + i, T(1) / mu_);

        auto& sol = std::get<LdltSolver>(active_solver_);

        sol.P_hat.resize(n_act + s, n_act + s);
        sol.P_hat.setFromTriplets(ldlt_build_trips_.begin(), ldlt_build_trips_.end());
        sol.P_hat.makeCompressed();

        if (pattern_dirty_) {
            sol.ldlt.analyzePattern(sol.P_hat);
            pattern_analyzed_ = true;
            pattern_dirty_ = false;
        }
        sol.ldlt.factorize(sol.P_hat);
        info_ = sol.ldlt.info();
        fact_count_++;
        mu_at_last_fact_       = mu_;
        use_ldlt_at_last_fact_ = true;
        n_act_     = n_act;
        s_current_ = static_cast<int>(s);
        base_dirty_ = false;

        if (!smw_suppressed()) snapshot_state();
    }

    // Build P = G E G^T + (1/mu) I (or shift its mu diagonal), then factorize with Cholesky.
    void factorize_by_chol() {
        const SpMat&   G        = *G_;
        const SpMat&   G_tr     = *G_tr_;
        const Vec&     H_diag   = *H_diag_;
        const BoolArr& active_K = *active_K_;

        const Eigen::Index s = G.rows();
        const Eigen::Index n = G.cols();

        assert(G_tr.rows() == n);
        assert(G_tr.cols() == s);
        assert(H_diag.size() == n);
        assert(active_K.size() == n);

        auto& sol = std::get<CholSolver>(active_solver_);

        if (base_dirty_) {
            // Rebuild G E G^T.
            assert(H_diag.minCoeff() > T(0));
            chol_build_trips_.clear();
            chol_build_trips_.reserve(n);
            for (Eigen::Index i = 0; i < n; ++i)
                if (active_K(i))
                    chol_build_trips_.emplace_back(i, i, T(1) / H_diag(i));
            SpMat E(n, n);
            E.setFromTriplets(chol_build_trips_.begin(), chol_build_trips_.end());
            E.makeCompressed();

            sol.P = G * E * G_tr;
            base_dirty_ = false;

            for (Eigen::Index i = 0; i < s; ++i)
                sol.P.coeffRef(i, i) += T(1) / mu_;
            sol.P.makeCompressed();
        } else {
            // G E G^T unchanged; only mu changed: shift the (1/mu) I diagonal by delta.
            assert(sol.P.rows() == s && sol.P.cols() == s);
            const T delta = (mu_at_last_fact_ - mu_) / (mu_ * mu_at_last_fact_);
            for (Eigen::Index i = 0; i < s; ++i)
                sol.P.coeffRef(i, i) += delta;
        }

        if (pattern_dirty_) {
            sol.llt.analyzePattern(sol.P);
            pattern_analyzed_ = true;
            pattern_dirty_ = false;
        }
        sol.llt.factorize(sol.P);
        info_ = sol.llt.info();
        fact_count_++;
        mu_at_last_fact_       = mu_;
        use_ldlt_at_last_fact_ = false;
        s_current_ = static_cast<int>(s);

        if (!smw_suppressed())
            snapshot_state();
    }

    // ------ SMW low-rank update ------

    // SMW Setup Phase. Returns true and arms use_smw_ iff 0 < h+p+q <= threshold.
    bool try_build_smw() {
        if (skip_smw_) { skip_smw_ = false; return false; }
        if (smw_fail_streak_ >= kMaxSmwFailStreak) return false;
        if (!active_W_ || !B_rm_ || M_rows_ < 0) return false;
        // If the factorization method changed since last full rebuild, do refactorization instead of SMW.
        if (use_ldlt_ != use_ldlt_at_last_fact_) return false; 
        if (active_W_old_.size() == 0 || active_K_old_.size() == 0) return false;

        const int s_old = static_cast<int>(G_old_.rows());
        if (s_old == 0) return false;
        const int N = static_cast<int>(G_old_.cols());
        const int l = static_cast<int>(active_W_->size());

        // Classify W and K changes
        deleted_old_rows_.clear();
        retained_old_rows_.clear();
        retained_new_rows_.clear();
        added_new_rows_.clear();
        added_W_src_.clear();
        delta_K_idx_.clear();

        {
            int old_pos = M_rows_, new_pos = M_rows_;
            for (int i = 0; i < l; ++i) {
                const bool in_old = active_W_old_(i), in_new = (*active_W_)(i);
                if (in_old && in_new) {
                    retained_old_rows_.push_back(old_pos++);
                    retained_new_rows_.push_back(new_pos++);
                } else if (in_old) {
                    deleted_old_rows_.push_back(old_pos++);
                } else if (in_new) {
                    added_new_rows_.push_back(new_pos++);
                    added_W_src_.push_back(i);
                }
            }
        }
        for (int i = 0; i < N; ++i)
            if (active_K_old_(i) != (*active_K_)(i))
                delta_K_idx_.push_back(i);

        const int h = static_cast<int>(deleted_old_rows_.size());
        const int q = static_cast<int>(added_new_rows_.size());
        const int p = static_cast<int>(delta_K_idx_.size());
        const int rank = h + p + q;
        const int threshold = 5;
        if (rank == 0 || rank > threshold) return false;

        h_ = h; p_ = p; q_ = q; s_old_ = s_old;

        // ---- Build V_plus_ (s_old × q) ----
        // For each added W constraint j:
        //   V_plus_[:,j] = sum_{i in B_j, active_K[i]} (b_ji / H_diag[i]) * G_old_[:,i].
        V_plus_.setZero(s_old, q);
        {
            Vec e_new_b = Vec::Zero(N);
            std::vector<int> touched;
            touched.reserve(32);
            for (int j = 0; j < q; ++j) {
                touched.clear();
                for (typename RowMajorSpMat::InnerIterator it(*B_rm_, added_W_src_[j]); it; ++it) {
                    const int col = it.col();
                    if ((*active_K_)(col)) {
                        e_new_b(col) = it.value() / (*H_diag_)(col);
                        touched.push_back(col);
                    }
                }
                for (int col : touched)
                    for (typename SpMat::InnerIterator git(G_old_, col); git; ++git)
                        V_plus_(git.row(), j) += e_new_b(col) * git.value();
                for (int col : touched) e_new_b(col) = T(0);
            }
        }

        // ---- Build M_sub (rank × rank) ----
        Mat M_sub = Mat::Zero(rank, rank);

        // Block 2: -C^-1 (diagonal p×p); C_jj = E_new[idx] - E_old[idx]
        for (int j = 0; j < p; ++j) {
            const int idx = delta_K_idx_[j];
            M_sub(h + j, h + j) = (*active_K_)(idx) ? -(*H_diag_)(idx) : H_diag_old_(idx);
        }

        // Block 3: W_+ = B_+ E_new B_+^T + (1/mu) I  (q×q).
        {
            Vec e_new_b = Vec::Zero(N);
            std::vector<int> touched;
            touched.reserve(32);
            for (int j = 0; j < q; ++j) {
                touched.clear();
                for (typename RowMajorSpMat::InnerIterator it(*B_rm_, added_W_src_[j]); it; ++it) {
                    const int col = it.col();
                    if ((*active_K_)(col)) {
                        e_new_b(col) = it.value() / (*H_diag_)(col);
                        touched.push_back(col);
                    }
                }
                for (int k = j; k < q; ++k) {
                    T val = (j == k) ? T(1) / mu_ : T(0);
                    for (typename RowMajorSpMat::InnerIterator it(*B_rm_, added_W_src_[k]); it; ++it)
                        val += it.value() * e_new_b(it.col());  // e_new_b is 0 at non-active_K
                    M_sub(h + p + j, h + p + k) = val;
                    if (j != k) M_sub(h + p + k, h + p + j) = val;
                }
                for (int col : touched) e_new_b(col) = T(0);
            }
        }

        // ---- Compute Y_all_ = P_old^-1 V_all (s_old × rank), column by column ----
        // E_- solves: toggle one entry at a time (no full setZero).
        // U solves: toggle G_old_ column nonzeros.
        // V_+ solves: dense block multi-RHS.
        Y_all_.resize(s_old, rank);
        {
            Vec tmp = Vec::Zero(s_old);

            // Helper: P_old^-1 v via LLT, or (P_hat_old^-1 [0;v]).tail via LDLT.
            auto solve_base_vec = [&](const Vec& rhs) -> Vec {
                if (use_ldlt_) {
                    Vec padded = Vec::Zero(n_act_ + s_old);
                    padded.tail(s_old) = rhs;
                    return std::get<LdltSolver>(active_solver_).ldlt.solve(padded).tail(s_old);
                } else {
                    return std::get<CholSolver>(active_solver_).llt.solve(rhs);
                }
            };

            // Cols 0..h-1: P_old^-1 e_{del_k}
            for (int k = 0; k < h; ++k) {
                if (k > 0) tmp(deleted_old_rows_[k - 1]) = T(0);
                tmp(deleted_old_rows_[k]) = T(1);
                Y_all_.col(k) = solve_base_vec(tmp);
            }
            if (h > 0) tmp(deleted_old_rows_[h - 1]) = T(0);

            // Cols h..h+p-1: P_old^-1 G_old_col_j 
            for (int j = 0; j < p; ++j) {
                for (typename SpMat::InnerIterator it(G_old_, delta_K_idx_[j]); it; ++it)
                    tmp(it.row()) = it.value();
                Y_all_.col(h + j) = solve_base_vec(tmp);
                for (typename SpMat::InnerIterator it(G_old_, delta_K_idx_[j]); it; ++it)
                    tmp(it.row()) = T(0);
            }

            // Cols h+p..rank-1: P_old^-1 V_plus_ 
            if (q > 0) {
                if (use_ldlt_) {
                    Mat padded_V = Mat::Zero(n_act_ + s_old, q);
                    padded_V.bottomRows(s_old) = V_plus_;
                    Y_all_.rightCols(q) = std::get<LdltSolver>(active_solver_).ldlt.solve(padded_V).bottomRows(s_old);
                } else {
                    Y_all_.rightCols(q) = std::get<CholSolver>(active_solver_).llt.solve(V_plus_);
                }
            }
        }

        // ---- Capacitance matrix S_Lambda = M_sub - V_all^T Y_all ----
        Mat S_Lambda = M_sub;

        // Rows 0..h-1: -(E_-)^T Y_all_ = -Y_all_.row(del_k)
        for (int k = 0; k < h; ++k)
            S_Lambda.row(k) -= Y_all_.row(deleted_old_rows_[k]);

        // Rows h..h+p-1: -(G_old_col_j)^T Y_all_
        for (int j = 0; j < p; ++j)
            for (typename SpMat::InnerIterator it(G_old_, delta_K_idx_[j]); it; ++it)
                S_Lambda.row(h + j) -= it.value() * Y_all_.row(it.row());

        // Rows h+p..rank-1: -V_plus_^T Y_all_
        if (q > 0)
            S_Lambda.bottomRows(q).noalias() -= V_plus_.transpose() * Y_all_;

        S_lambda_lu_.setThreshold(std::sqrt(std::numeric_limits<T>::epsilon()));
        S_lambda_lu_.compute(S_Lambda);
        if (S_lambda_lu_.rank() < rank)
            return false;  // near-singular capacitance matrix; fall back to full rebuild

        // ---- Pre-allocate hot-loop vectors ----
        const int s_new = s_old - h + q;
        r_pad_.setZero(s_old);
        u_base_.resize(s_old);
        Lambda_all_.resize(rank);
        lambda_work_.resize(rank);
        z_base_.resize(s_old);
        z_new_.resize(s_new);
        s_current_ = s_new;
        use_smw_   = true;
        smw_count_++;
        return true;
    }

    // Helper: snapshot last full-rebuild state for next SMW attempt
    void snapshot_state() {
        G_old_        = *G_;
        H_diag_old_   = *H_diag_;
        active_K_old_ = *active_K_;
        if (active_W_) active_W_old_ = *active_W_;
    }

    // ------ Setup: external data ------
    const SpMat*         G_        = nullptr;
    const SpMat*         G_tr_     = nullptr;
    const Vec*           H_diag_   = nullptr;
    const BoolArr*       active_K_ = nullptr;
    const BoolArr*       active_W_ = nullptr;
    const RowMajorSpMat* B_rm_     = nullptr;
    T mu_ = T(1);
    int M_rows_    = -1;  // number of equality-constraint rows
    int s_current_ = -1;  // current row count of G

    // ------ Formation state ------
    bool initialized_      = false;
    bool rebuild_          = true;
    bool pattern_analyzed_ = false;
    bool pattern_dirty_    = true;
    bool base_dirty_       = true;
    T    mu_at_last_fact_  = T(-1);

    // ------ Cholesky factorization ------
    bool use_ldlt_ = false;
    bool use_ldlt_at_last_fact_ = false;

    struct CholSolver {
        SpMat P;
        Eigen::SimplicialLLT<SpMat> llt;
    };
    struct LdltSolver {
        SpMat P_hat;
        Eigen::SimplicialLDLT<SpMat> ldlt;
    };
    std::variant<std::monostate, CholSolver, LdltSolver> active_solver_;

    int n_act_ = 0;
    Eigen::ComputationInfo info_ = Eigen::Success;
    int fact_count_ = 0;

    std::vector<int> ldlt_act_idx_;
    std::vector<Triplet> ldlt_build_trips_;
    std::vector<Triplet> chol_build_trips_;

    // ------ SMW low-rank update ------
    // Control & failure tracking
    bool skip_smw_        = false;
    bool use_smw_         = false;
    int  smw_count_       = 0;
    int  smw_fail_streak_ = 0;
    int  smw_fail_total_  = 0;
    static constexpr int    kMaxSmwFailStreak = 5;

    // Snapshots from last full rebuild (input to try_build_smw)
    SpMat   G_old_;
    Vec     H_diag_old_;
    BoolArr active_K_old_;
    BoolArr active_W_old_;

    // Setup results (output of try_build_smw, consumed by solve)
    int s_old_ = 0, h_ = 0, p_ = 0, q_ = 0;
    std::vector<int> deleted_old_rows_;
    std::vector<int> retained_old_rows_;
    std::vector<int> retained_new_rows_;
    std::vector<int> added_new_rows_;
    std::vector<int> added_W_src_;
    std::vector<int> delta_K_idx_;
    Mat V_plus_;
    Mat Y_all_;
    Eigen::FullPivLU<Mat> S_lambda_lu_;

    // ------ Application: solve-time working storage ------
    mutable Vec r_pad_, u_base_, Lambda_all_, lambda_work_, z_base_, z_new_, direct_result_;
    mutable Vec ldlt_rhs_;
};
