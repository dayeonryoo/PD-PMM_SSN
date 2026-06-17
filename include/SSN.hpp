#pragma once
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
#include "Printing.hpp"
#include "SchurOperator.hpp"
#include "SchurPreconditioner.hpp"


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
    const Vec& D1A_diag, D1B_diag, D2_diag;
    const Vec& c, b, lx, ux, lw, uw;
    const T obj_const;
    const int n, m, N, M, l;
    Vec x, y1, y2, z;
    Vec delta_x, delta_y1, delta_y2, delta_z;
    int SSN_max_in_iter;
    T mu, rho, gamma, SSN_tol;
    T eps_pinf, eps_dinf;

    // Useful vectors and matrices
    T inf = 1e20;
    T eps_zero = 1e-12; // for checking near-zero values without scaling issues
    Vec ones_N, ones_M, ones_l;
    const SpMat& A_tr, B_tr, L_tr;
    Vec H_diag, H_diag_inv;
    Vec diag_P_K, diag_P_W;
    BoolArr active_W, inactive_W, active_K;
    int n_active_W, n_inactive_W;
    SpMat B_active_W, B_inactive_W, G, G_tr;

    // B_rm: Row-major B for O(nnz_row) active-row access when rebuilding G.
    // G_A_trips_: A's contribution to G (rows 0..M-1), computed once (A is const).
    RowMajorSpMat B_rm;
    std::vector<Triplet> G_A_trips_;

    // Outputs
    int opt, iter, SSN_iter;
    T tol_achieved, obj_val;
    int Krylov_iter = 0, fact = 0, smw_count = 0;
    int linesearch_fail = 0, Krylov_fail = 0;
    bool Krylov_converged = true;

    // Backtracking linesearch parameters
    T beta = 0.4995 / 2;
    T delta = 0.995;

    // Conjugate gradient parameters
    T Krylov_tol = 1e-12;
    int Krylov_max_in_iter = 500;
    bool ldlt_used = false;

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
    Vec A_tr_y1_; // cached A^T y1, recomputed once per PMM iteration in update_SSN_system

    // Pre-allocated scratch vectors for solve_SSN / exact_line_search hot loops.
    Vec Ax_ssn_, Bx_ssn_;                     // size M, l (running SpMV products in SSN)
    Vec x_cur_, y2_cur_;                      // size N, l (SSN working iterates)
    Vec u_, v_;                               // size N, l
    Vec new_diag_P_K_, dist_K_u_;             // size N
    Vec new_diag_P_W_, dist_W_v_;             // size l
    Vec y2_active_W_, y2_inactive_W_;         // size n_active_W, n_inactive_W
    Vec dist_W_v_active_, dist_W_v_inactive_; // size n_active_W, n_inactive_W
    Vec dy2_inactive_W_;                              // size n_inactive_W
    Vec r1_, r2_;                             // size N, M+n_active_W
    Vec dxdy_;                                        // size N+M+n_active_W
    Vec dy2_;                                         // size l
    Vec Adx_, Bdx_;                           // size M, l
    Vec grad_L_;                                      // size N+l
    std::vector<Breakpoint> breakpoints_;             // reused across exact_line_search calls

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

    SSN(const int Q_info_, const Vec& Q_diag_, const SpMat& L_, const SpMat& L_tr_,
        const SpMat& A_, const SpMat& B_, const SpMat& A_tr_, const SpMat& B_tr_,
        const Vec& c_, const Vec& b_, const Vec& D1A_diag_, const Vec& D1B_diag_, const Vec& D2_diag_,
        const Vec& lx_, const Vec& ux_, const Vec& lw_, const Vec& uw_, const T obj_const_,
        int n_, int m_, int N_, int M_, int l_,
        T SSN_tol_, int SSN_max_in_iter_, T eps_pinf_, T eps_dinf_)
    : Q_info(Q_info_), Q_diag(Q_diag_), L(L_), L_tr(L_tr_),
      A(A_), B(B_), A_tr(A_tr_), B_tr(B_tr_),
      c(c_), b(b_), D1A_diag(D1A_diag_), D1B_diag(D1B_diag_), D2_diag(D2_diag_),
      lx(lx_), ux(ux_), lw(lw_), uw(uw_), obj_const(obj_const_),
      n(n_), m(m_), N(N_), M(M_), l(l_),
      SSN_tol(SSN_tol_), SSN_max_in_iter(SSN_max_in_iter_),
      eps_pinf(eps_pinf_), eps_dinf(eps_dinf_)
    {
        ones_N = Vec::Ones(N);
        ones_M = Vec::Ones(M);
        ones_l = Vec::Ones(l);
        SSN_iter = 0;
        delta_x = Vec::Zero(N);
        delta_y2 = Vec::Zero(l);

        // Convert column-major B to row-major once for efficient row-selective access.
        B_rm = B;

        // Cache A's triplets (rows 0..M-1 of G are always A — A is const).
        G_A_trips_.reserve(A.nonZeros());
        for (int col = 0; col < A.outerSize(); ++col)
            for (typename SpMat::InnerIterator it(A, col); it; ++it)
                G_A_trips_.emplace_back(it.row(), col, it.value());

    }

    void update_SSN_system(const Vec& x_, const Vec& y1_, const Vec& y2_, const Vec& z_,
                           const Vec& delta_y1_, const Vec& delta_z_,
                           T mu_, T rho_, T gamma_, int SSN_iter_) {
        x = x_;
        y1 = y1_;
        y2 = y2_;
        z = z_;
        mu = mu_;
        rho = rho_;
        gamma = gamma_;
        SSN_iter = SSN_iter_;
        delta_y1 = delta_y1_;
        delta_z = delta_z_;
        ldlt_numeric_dirty_ = true;    // mu, rho may have changed
        A_tr_y1_ = A_tr * y1; // y1 is fixed for the entire SSN run; cache A^T y1 once
    }
    static inline T inf_norm(const Vec& v) {
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
    T compute_Lagrangian(const Vec& x_new, const Vec& y2_new, const Vec& Ax_new, const Vec& Bx_new);
    Vec compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new, const Vec& Ax_new, const Vec& Bx_new);
    Vec Clarke_subgrad_of_proj(const Vec& u, const Vec& lower, const Vec& upper, const bool include_bd);
    bool is_P_unchanged(const Vec& diag_P, const Vec& new_diag_P);
    void split_by_mask(const Vec& u, const BoolArr& mask, Vec& u_sel, Vec& u_unsel);
    void rebuild_G();
    SpMat scale_columns(const SpMat& M, const Vec& d);
    void retrive_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask, Vec& out);
    SpMat stack_rows(const SpMat& A, const SpMat& B);
    bool form_schur(const SpMat& G);
    Vec solve_using_cg(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv, const BoolArr& active_K, const Vec& r1, const Vec& r2, T mu, T tol, int max_iter, bool update_prec, bool G_pattern_changed);
    Vec solve_using_minres(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv, const BoolArr& active_K, const Vec& r1, const Vec& r2, T mu, T tol, int max_iter, bool update_prec, bool prec_pattern_changed);
    Vec solve_using_schur(const SpMat& G, const SpMat& G_tr, const Vec& H_diag_inv, const Vec& r1, const Vec& r2);
    Vec solve_using_LDLT(const SpMat& G, const Vec& H_diag, const Vec& r1, const Vec& r2);
    T backtracking_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2,
                               const Vec& Ax_curr, const Vec& Bx_curr, const Vec& Adx, const Vec& Bdx);
    T exact_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2,
                        const Vec& Ax_curr, const Vec& Bx_curr, const Vec& Adx, const Vec& Bdx,
                        const Vec& dist_K_u, const Vec& dist_W_v);
    void solve_SSN(const T eps);

};

#include "SSN.tpp"