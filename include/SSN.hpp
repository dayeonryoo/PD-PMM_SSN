#pragma once
#include <string>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
#include "Printing.hpp"
#include "SchurOperator.hpp"
#include "SchurPreconditioner.hpp"

// TIMER: master switch for per-step SSN-loop timing instrumentation.
// Set to 1 (here, or via -DSSN_ENABLE_TIMERS=1) to accumulate and print a phase-by-phase
// wall-clock breakdown of solve_ssn() to stderr; 0 compiles the timers out entirely (no overhead).
#ifndef SSN_ENABLE_TIMERS
#define SSN_ENABLE_TIMERS 1
#endif

#if SSN_ENABLE_TIMERS
#include <chrono>
#include <cstdio>

// TIMER: RAII scoped timer; accumulates elapsed wall-clock seconds into `acc` on scope exit.
struct SsnScopedTimer {
    std::chrono::steady_clock::time_point t0;
    double& acc;
    explicit SsnScopedTimer(double& acc_) : t0(std::chrono::steady_clock::now()), acc(acc_) {}
    ~SsnScopedTimer() { acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
#define SSN_TIMER_BLOCK(acc) SsnScopedTimer _ssn_scoped_timer(acc)
#else
#define SSN_TIMER_BLOCK(acc) do {} while (0)
#endif


template <typename T>
class SSN {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using RowMajorSpMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<T>;

    struct Breakpoint { T t; T slope_change; };

    // Inputs
    const int Q_info;
    const Vec& Q_diag;
    const SpMat& L, A, B;
    const Vec& D2_ext_inv, D1B_diag_inv; // for unscaling the SSN termination criterion
    const Vec& c, b, lx, ux, lw, uw;
    const int n, m, N, M, l;
    Vec x, y1, y2, z;
    Vec delta_x, delta_y1, delta_y2, delta_z;
    int ssn_max_in_iter;
    T mu, rho, alpha, ssn_tol;
    T eps_pinf, eps_dinf;

    // Useful vectors and matrices
    T inf = std::numeric_limits<T>::infinity();
    T eps_zero      = T(100)  * std::numeric_limits<T>::epsilon();  // relative tolerance for boundary/slope checks.
    T eps_direction = std::sqrt(std::numeric_limits<T>::epsilon()); // threshold for skipping near-zero Newton step components.
    Vec ones_N, ones_M, ones_l;

    const SpMat& A_tr, B_tr;
    Vec H_diag, H_diag_inv;
    Vec diag_P_K, diag_P_W;
    BoolArr active_W, inactive_W, active_K;
    int n_active_W, n_inactive_W;
    SpMat B_active_W, B_inactive_W, G, G_tr;

    RowMajorSpMat B_rm;              // Row-major B for rebuilding G.
    std::vector<Triplet> G_A_trips_; // A's contribution to G, computed once since A is const.

    // Scratch triplet buffers for rebuild_G() and solve_using_ldlt().
    std::vector<Triplet> B_act_trips_, B_inact_trips_, G_trips_;
    std::vector<Triplet> ldlt_trip_;

    // Printing
    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::NONE;

    // Outputs
    int opt, iter, ssn_iter;
    T tol_achieved, obj_val;
    int krylov_iter = 0, fact = 0, smw_count = 0;
    int linesearch_fail = 0, krylov_fail = 0;
    bool krylov_converged = true;

#if SSN_ENABLE_TIMERS
    // TIMER: wall-clock seconds per SSN-loop phase for the current SSN iteration; reset at the top
    // of each iteration and printed to stderr at the end of that same iteration. See solve_ssn() in
    // SSN.tpp for what each phase covers.
    double timer_prep         = 0.0; // subgrad_dist + pk_update + rebuild_g + rhs_build + choose_ldlt
    double timer_linear_solve = 0.0; // solve_using_cg() + iterative refinement + recovering dy2
    double timer_prec_setup   = 0.0; // subset of timer_linear_solve: preconditioner setData()/compute()
    double timer_prec_assembly  = 0.0; // subset of timer_prec_setup: SchurPreconditioner sparse matrix assembly (P_hat / G*E*G_tr)
    double timer_prec_analyze   = 0.0; // subset of timer_prec_setup: SchurPreconditioner::analyzePattern()
    double timer_prec_factorize = 0.0; // subset of timer_prec_setup: SchurPreconditioner::factorize()
    double timer_krylov_solve = 0.0; // subset of timer_linear_solve: cg.solve()/solveWithGuess()
    double timer_ldlt_analyze   = 0.0; // subset of timer_linear_solve: solve_using_ldlt()'s ldlt_.analyzePattern()
    double timer_ldlt_factorize = 0.0; // subset of timer_linear_solve: solve_using_ldlt()'s ldlt_.factorize()
    double timer_ldlt_solve     = 0.0; // subset of timer_linear_solve: solve_using_ldlt()'s ldlt_.solve()
    double timer_linesearch   = 0.0; // exact_line_search() (incl. gradient-descent retry)
    double timer_state_update = 0.0; // x_cur_/y2_cur_/Ax_ssn_/Bx_ssn_ update + gradient/termination check
#endif

    // Conjugate gradient parameters
    T krylov_tol = 1e-12;
    int krylov_max_in_iter = 100;

    // Iterative refinement of the augmented system solve.
    int refine_max_iter = 3;
    T refine_rel_tol = 1e-10;
    T refine_abs_tol = 1e-12;

    bool use_ldlt = false;        // Factorize a preconditioner via LDLT.
    int ldlt_decisions_made_ = 0; // choose_ldlt() call count; locked after the first 3.
    bool ldlt_used = false;       // PCG on normal eqn failed so LDLT on KKT system was used at least once.

    using CGSolver = Eigen::ConjugateGradient<
        SchurOperator<T>,
        Eigen::Lower | Eigen::Upper,
        SchurPreconditioner<T>
    >;
    CGSolver cg;

    using MINRESSolver = Eigen::MINRES<
        SchurOperator<T>,
        Eigen::Lower | Eigen::Upper,
        SchurPreconditioner<T>
    >;
    MINRESSolver minres;

    Vec prev_dy_;
    Vec prev_dx_primal_;
    Vec A_tr_y1_; // cached A^T y1, recomputed once per PMM iteration in update_ssn_system

    // Pre-allocated scratch vectors.
    Vec Ax_ssn_, Bx_ssn_;                     // size M, l (running SpMV products in SSN)
    Vec x_cur_, y2_cur_;                      // size N, l (SSN working iterates)
    Vec u_, v_;                               // size N, l
    Vec new_diag_P_K_, dist_K_u_;             // size N
    Vec new_diag_P_W_, dist_W_v_;             // size l
    Vec y2_active_W_, y2_inactive_W_;         // size l (set to max size)
    Vec dist_W_v_active_, dist_W_v_inactive_; // size l (set to max size)
    Vec dy2_inactive_W_;                      // size l (set to max size)
    Vec r1_, r2_;                             // size N, M+n_active_W
    Vec dxdy_;                                // size N+M+n_active_W
    Vec dy2_;                                 // size l
    Vec Adx_, Bdx_;                           // size M, l
    Vec grad_L_;                              // size N+l
    Vec grad_dist_K_, grad_res_p_;            // size N, M
    Vec grad_dist_W_;                         // size l
    Vec grad_Atr_resp_, grad_Btr_distW_, grad_A_tr_y_, grad_Qx_; // size N
    Vec ls_s_, ls_dv_;                        // size N, l
    Vec ls_v_;                                // size l
    std::vector<Breakpoint> breakpoints_;     // reused across exact_line_search calls
    Vec cg_Hinv_r1_, cg_rhs_;                 // size s = G.rows() (M+n_active_W)
    Vec cg_dx_;                               // size n = N
    Vec ldlt_solve_rhs_;                      // size n+s

    // Stored LDLT factorization for the KKT system [-H, G^T; G, (1/mu)I].
    // ldlt_pattern_dirty_: true when K's dimension changed (n_active_W changed), requires analyzePattern.
    // ldlt_numeric_dirty_: true when K's values changed (H_diag or G rows swapped), requires factorize.
    Eigen::SimplicialLDLT<SpMat> ldlt_;
    bool ldlt_pattern_dirty_ = true;
    bool ldlt_numeric_dirty_ = true;

    // Cached KKT matrix K = [-H, G^T; G, (1/mu)I].
    // When active_W changes (G's sparsity changes): rebuild K from triplets.
    // When only H_diag or mu changes (diagonal-only update): set diagonal entries in-place,
    // skipping the triplet build, setFromTriplets, and makeCompressed calls.
    SpMat K_ldlt_;
    bool K_ldlt_built_ = false;

    SSN(const int Q_info, const Vec& Q_diag, const SpMat& L,
        const SpMat& A, const SpMat& B, const SpMat& A_tr, const SpMat& B_tr,
        const Vec& c, const Vec& b,
        const Vec& D2_ext_inv, const Vec& D1B_diag_inv,
        const Vec& lx, const Vec& ux, const Vec& lw, const Vec& uw,
        int n, int m, int N, int M, int l,
        T ssn_tol, int ssn_max_in_iter, T eps_pinf, T eps_dinf,
        PrintWhen when = PrintWhen::NEVER, PrintWhat what = PrintWhat::NONE)
    : Q_info(Q_info), Q_diag(Q_diag), L(L),
      A(A), B(B), A_tr(A_tr), B_tr(B_tr), c(c), b(b),
      D2_ext_inv(D2_ext_inv), D1B_diag_inv(D1B_diag_inv),
      lx(lx), ux(ux), lw(lw), uw(uw),
      n(n), m(m), N(N), M(M), l(l),
      ssn_tol(ssn_tol), ssn_max_in_iter(ssn_max_in_iter),
      eps_pinf(eps_pinf), eps_dinf(eps_dinf),
      when(when), what(what)
    {
        ones_N = Vec::Ones(N);
        ones_M = Vec::Ones(M);
        ones_l = Vec::Ones(l);
        ssn_iter = 0;
        delta_x = Vec::Zero(N);
        delta_y2 = Vec::Zero(l);

        y2_active_W_.resize(l);
        y2_inactive_W_.resize(l);
        dist_W_v_active_.resize(l);
        dist_W_v_inactive_.resize(l);
        dy2_inactive_W_.resize(l);

        grad_L_.resize(N + l);

        B_rm = B; // Row-major B

        // Cache A's triplets for G.
        G_A_trips_.reserve(A.nonZeros());
        for (int col = 0; col < A.outerSize(); ++col)
            for (typename SpMat::InnerIterator it(A, col); it; ++it)
                G_A_trips_.emplace_back(it.row(), col, it.value());

    }

    void update_ssn_system(const Vec& x, const Vec& y1, const Vec& y2, const Vec& z,
                           const Vec& delta_y1, const Vec& delta_z,
                           T mu, T rho, T alpha, int ssn_iter) {
        this->x = x;
        this->y1 = y1;
        this->y2 = y2;
        this->z = z;
        this->mu = mu;
        this->rho = rho;
        this->alpha = alpha;
        this->ssn_iter = ssn_iter;
        this->delta_y1 = delta_y1;
        this->delta_z = delta_z;

        ldlt_numeric_dirty_ = true;  // mu, rho may have changed.
        A_tr_y1_ = A_tr * y1;        // y1 is fixed for the entire SSN run; cache A^T y1 once.
        linesearch_fail = 0;         // Reset line search failure count for this SSN iteration.
        cg.preconditioner().reset_smw_fail_streak(); // Reset SMW suppression
    }
    static inline T inf_norm(const Vec& v) {
        if (v.size() == 0) return T(0);
        return v.cwiseAbs().maxCoeff();
    }
    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline Vec compute_dist_box(const Vec& v, const Vec& lower, const Vec& upper) {
        return (v - proj(v, lower, upper));
    }
    static void compute_subgrad_and_dist(const Vec& u, const Vec& lower, const Vec& upper,
                                         bool include_bd, Vec& subgrad, Vec& dist) {
        const int sz = static_cast<int>(u.size());
        subgrad.resize(sz);
        dist.resize(sz);
        for (int i = 0; i < sz; ++i) {
            const T ui = u[i], li = lower[i], hi = upper[i];
            const T pi = std::max(li, std::min(hi, ui));
            dist[i]    = ui - pi;
            subgrad[i] = (include_bd ? (ui >= li && ui <= hi) : (ui > li && ui < hi)) ? T(1) : T(0);
        }
    }
    const Vec& compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new, const Vec& Ax_new, const Vec& Bx_new);
    T compute_grad_Lagrangian_unscaled_inf_norm(const Vec& grad_L);
    void split_by_mask(const Vec& u, const BoolArr& mask, int t, Vec& u_sel, Vec& u_unsel);
    void rebuild_G();
    void retrieve_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask, Vec& out);
    bool choose_ldlt(const SpMat& G, const BoolArr& active_K);
    Vec solve_using_cg(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv, const BoolArr& active_K, const Vec& r1, const Vec& r2, T mu, T tol, int max_iter, bool update_prec, bool G_pattern_changed, bool use_ldlt);
    Vec solve_using_ldlt(const SpMat& G, const Vec& H_diag, const Vec& r1, const Vec& r2);
    T exact_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2,
                        const Vec& Ax_curr, const Vec& Bx_curr, const Vec& Adx, const Vec& Bdx,
                        const Vec& dist_K_u, const Vec& dist_W_v);
    void solve_ssn(const T ssn_tol);

};

#include "SSN.tpp"