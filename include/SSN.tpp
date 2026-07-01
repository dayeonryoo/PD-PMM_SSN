#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>
#include <Eigen/SparseCholesky>
#include <cassert>

template <typename T>
T SSN<T>::compute_Lagrangian(const Vec& x_new, const Vec& y2_new, const Vec& Ax_new, const Vec& Bx_new) {
    using Vec = typename SSN<T>::Vec;

    Vec dist_K = compute_dist_box(z / mu + x_new, lx, ux);
    Vec dist_W = compute_dist_box(Bx_new + ((1 - gamma) * y2_new - y2) / mu, lw, uw);

    Vec res_p = Ax_new - b;

    // Compute Lagrangian
    T L;
    if (Q_info == 0) {
        L = c.dot(x_new)
            - y1.dot(res_p) + (mu / 2) * res_p.squaredNorm()
            - z.squaredNorm() / (2 * mu) + (mu / 2) * dist_K.squaredNorm()
            + (mu / (2 * gamma)) * dist_W.squaredNorm() + ((1 - gamma) / (2 * mu)) * y2_new.squaredNorm() - y2.squaredNorm() / (2 * mu)
            + (x_new - x).squaredNorm() / (2 * rho);
    } else {
        L = c.dot(x_new) + 0.5 * Q_diag.cwiseProduct(x_new).dot(x_new)
            - y1.dot(res_p) + (mu / 2) * res_p.squaredNorm()
            - z.squaredNorm() / (2 * mu) + (mu / 2) * dist_K.squaredNorm()
            + (mu / (2 * gamma)) * dist_W.squaredNorm() + ((1 - gamma) / (2 * mu)) * y2_new.squaredNorm() - y2.squaredNorm() / (2 * mu)
            + (x_new - x).squaredNorm() / (2 * rho);
    }
    return L;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new, const Vec& Ax_new, const Vec& Bx_new) {
    using Vec = typename SSN<T>::Vec;

    Vec dist_K = compute_dist_box(z / mu + x_new, lx, ux);
    Vec dist_W = compute_dist_box(Bx_new + ((1 - gamma) * y2_new - y2) / mu, lw, uw);

    Vec res_p = Ax_new - b;
    Vec A_tr_y = - A_tr_y1_ + mu * A_tr * res_p + mu * dist_K + (mu / gamma) * B_tr * dist_W;

    // Compute gradient of Lagrangian
    Vec grad_L_x;
    if (Q_info == 0) {
        grad_L_x = c + A_tr_y + (x_new - x) / rho;
    } else {
        Vec Qx = Q_diag.cwiseProduct(x_new);
        grad_L_x = c + Qx + A_tr_y + (x_new - x) / rho;
    }
    Vec grad_L_y2 = ((1 - gamma) / gamma) * dist_W + ((1 - gamma) / mu) * y2_new;

    Vec grad_L(N + l);
    grad_L << grad_L_x, grad_L_y2;

    return grad_L;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::clarke_subgrad_of_proj(const Vec& u, const Vec& lower, const Vec& upper, const bool include_bd) {
    using Vec = typename SSN<T>::Vec;
    using BoolArr = typename SSN<T>::BoolArr;

    BoolArr mask;
    if (include_bd) mask = (u.array() >= lower.array()) && (u.array() <= upper.array());
    else mask = (u.array() > lower.array()) && (u.array() < upper.array());

    Vec grad_proj = mask.cast<T>().matrix();
    return grad_proj;
}

template <typename T>
bool SSN<T>::is_P_unchanged(const Vec& diag_P, const Vec& new_diag_P) {
    if (diag_P.size() == 0) return false; // At first SSN iteration.
    for (int i = 0; i < diag_P.size(); ++i) {
        if (diag_P[i] != new_diag_P[i]) return false;
    }
    return true;
}

template <typename T>
void SSN<T>::split_by_mask(const Vec& u, const BoolArr& mask, Vec& u_sel, Vec& u_unsel) {
    int t = static_cast<int>(mask.count());
    u_sel.resize(t);
    u_unsel.resize(mask.size() - t);

    int i_sel = 0;
    int i_unsel = 0;
    for (int i = 0; i < mask.size(); ++i) {
        if (mask(i)) {
            u_sel(i_sel++) = u(i);
        } else {
            u_unsel(i_unsel++) = u(i);
        }
    }
}

template <typename T>
void SSN<T>::rebuild_G() {
    using RIt = typename RowMajorSpMat::InnerIterator;

    const int n_act   = n_active_W;
    const int n_inact = l - n_act;

    // Partitioning B into active and inactive
    std::vector<Triplet> B_act_trips, B_inact_trips;
    B_act_trips.reserve(B_rm.nonZeros());
    B_inact_trips.reserve(B_rm.nonZeros());

    // G = [A; B_active_W]
    std::vector<Triplet> G_trips;
    G_trips.reserve(G_A_trips_.size() + B_rm.nonZeros());
    G_trips.insert(G_trips.end(), G_A_trips_.begin(), G_A_trips_.end());

    int i_act = 0, i_inact = 0;
    for (int i = 0; i < l; ++i) {
        if (active_W(i)) {
            for (RIt it(B_rm, i); it; ++it) {
                B_act_trips.emplace_back(i_act, it.col(), it.value());
                G_trips.emplace_back(M + i_act, it.col(), it.value());
            }
            ++i_act;
        } else {
            for (RIt it(B_rm, i); it; ++it)
                B_inact_trips.emplace_back(i_inact, it.col(), it.value());
            ++i_inact;
        }
    }

    B_active_W.resize(n_act, N);
    B_active_W.setFromTriplets(B_act_trips.begin(), B_act_trips.end());
    B_active_W.makeCompressed();

    B_inactive_W.resize(n_inact, N);
    B_inactive_W.setFromTriplets(B_inact_trips.begin(), B_inact_trips.end());
    B_inactive_W.makeCompressed();

    G.resize(M + n_act, N);
    G.setFromTriplets(G_trips.begin(), G_trips.end());
    G.makeCompressed();
    G_tr = G.transpose();
}

template <typename T> // used only in solve_using_schur
typename SSN<T>::SpMat SSN<T>::scale_columns(const SpMat& M, const Vec& d) {
    assert(M.cols() == d.size());

    SpMat M_scaled = M;
    for (int j = 0; j < M_scaled.outerSize(); ++j) {
        T scale = d(j);
        for (typename SpMat::InnerIterator it(M_scaled, j); it; ++it) {
            it.valueRef() *= scale;
        }
    }
    return M_scaled;
}

template <typename T> // used only in solve_using_schur
typename SSN<T>::SpMat SSN<T>::stack_rows(const SpMat& A, const SpMat& B) {
    using SpMat = typename SSN<T>::SpMat;
    using Triplet = typename SSN<T>::Triplet;

    assert(A.cols() == B.cols());

    int A_rows = A.rows();
    int B_rows = B.rows();
    int A_cols = A.cols();

    SpMat stack(A_rows + B_rows, A.cols());
    std::vector<Triplet> trpl;
    trpl.reserve(A.nonZeros() + B.nonZeros());

    for (int col = 0; col < A_cols; ++col) {
        for (typename SpMat::InnerIterator it(A, col); it; ++it) {
            trpl.emplace_back(it.row(), col, it.value());
        }
    }

    for (int col = 0; col < A_cols; ++col) {
        for (typename SpMat::InnerIterator it(B, col); it; ++it) {
            trpl.emplace_back(A_rows + it.row(), col, it.value());
        }
    }

    stack.setFromTriplets(trpl.begin(), trpl.end());
    return stack;
}

template <typename T>
void SSN<T>::retrieve_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask, Vec& out) {
    assert(u_sel.size() == mask.count());
    assert(u_unsel.size() == mask.size() - mask.count());

    out.resize(mask.size());
    int i_sel = 0, i_unsel = 0;
    for (int i = 0; i < mask.size(); ++i)
        out(i) = mask(i) ? u_sel(i_sel++) : u_unsel(i_unsel++);
}

template <typename T>
bool SSN<T>::choose_ldlt(const SpMat& G, const BoolArr& active_K) {
    // Compare the estimated total work required to factorize the KKT matrix K and the Schur complement S.
    //  Ratio = (s / (s+t)) * (|K|^2 / |S|^2), where s = G.rows() and t = n_act_K.
    // |K| = nnz in the KKT matrix [-H_act_K, G_act_K^T; G_act_K, (1/mu)I] (active_K columns only).
    // |S| is overestimated via the densest active_K column and the pigeonhole principle.
    // Returns true (prefer LDLT on K) when ratio < 0.1.
    const int s = G.rows();

    // Pass 1: count active_K columns (t), their total nnz, and the densest one (hat_k, G_hat).
    long long t = 0, G_act_nnz = 0;
    int G_hat = 0, hat_k = -1;
    for (int k = 0; k < N; ++k) {
        if (!active_K(k)) continue;
        ++t;
        int a_k = G.isCompressed()
                ? G.outerIndexPtr()[k + 1] - G.outerIndexPtr()[k] // nnz in column k
                : (int)G.col(k).nonZeros();
        G_act_nnz += a_k;
        if (a_k > G_hat) { G_hat = a_k; hat_k = k; }
    }

    const long long K_nnz = (t + s) + 2LL * G_act_nnz;

    // Overestimate |S|: s (diagonal) + off-diagonal from densest block + pigeonhole corrections.
    long long S_nnz = (long long)s + (long long)G_hat * G_hat - G_hat;
    for (int k = 0; k < N; ++k) {
        if (!active_K(k) || k == hat_k) continue;
        int a_k = G.isCompressed()
                ? G.outerIndexPtr()[k + 1] - G.outerIndexPtr()[k]
                : (int)G.col(k).nonZeros();
        int o_k = std::max(0, G_hat + a_k - s); // Guaranteed overlap by pigeonhole
        S_nnz += (long long)a_k * a_k
                   - a_k
                   - (long long) o_k * o_k
                   + o_k;
    }

    if (S_nnz <= 0) {
        std::cout << "0: ";
        return true;
    }

    const double ratio = ((double)s / (t + s))
                       * ((double)K_nnz / S_nnz)
                       * ((double)K_nnz / S_nnz);

    std::cout << ratio << ": ";
    return ratio < 0.1;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::solve_using_cg(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv,
                                            const BoolArr& active_K, const Vec& r1, const Vec& r2,
                                            T mu, T tol, int max_iter, bool update_prec, bool prec_pattern_changed,
                                            bool use_ldlt) {
    using Vec = typename SSN<T>::Vec;

    const int s = G.rows();
    const int n = G.cols();

    // Schur complement system: S dy = G H_inv G^T dy + (1/mu) dy = G H_inv r1 + r2
    SchurOperator<T> S(G, G_tr, H_diag_inv, mu);
    Vec rhs = G * H_diag_inv.cwiseProduct(r1) + r2;

    cg.setTolerance(tol);
    cg.setMaxIterations(max_iter);

    // Set up preconditioner and call cg.compute(). 
    // force_rebuild=true skips SMW and recomputes G E G^T from scratch.
    auto setup_prec = [&](bool force_rebuild) {
        cg.preconditioner().setData(G, G_tr, H_diag, active_K, active_W, B_rm, mu,
                                    update_prec || force_rebuild, prec_pattern_changed);
        cg.preconditioner().set_use_ldlt(use_ldlt); // Factorization method for a preconditioner
        if (force_rebuild || cg.preconditioner().smw_suppressed())
            cg.preconditioner().force_full_rebuild();
        int prec_fact_before = cg.preconditioner().fact_count();
        cg.compute(S);
        fact      += cg.preconditioner().fact_count() - prec_fact_before;
        smw_count  = cg.preconditioner().smw_count();
    };

    // Run preconditioned CG.
    // Returns false and increments krylov_fail on preconditioner failure or solver non-convergence.
    auto attempt_solve = [&](Vec& dy_out) -> bool {
        if (cg.preconditioner().info() != Eigen::Success) {
            std::cout << "[WARN] Krylov failed due to preconditioner failure.\n";
            krylov_fail++;
            return false;
        }
        Vec dy_;
        bool warm_start = (prev_dy_.size() == s);
        if (warm_start) {
            dy_ = cg.solveWithGuess(rhs, prev_dy_);
        } else {
            dy_ = cg.solve(rhs);
        }
        krylov_iter += cg.iterations();
        if (cg.info() != Eigen::Success) {
            std::cout << "[WARN] Krylov failed to converge.\n";
            krylov_fail++;
            return false;
        }
        dy_out = std::move(dy_);
        return true;
    };

    // Set up and attempt to solve by PCG.
    // bad_alloc can come from build() (P_base_ = G E G^T) or from Eigen's CG iteration internals;
    // fall back to LDLT in either case.
    Vec dy_;
    bool ok = false;
    try {
        setup_prec(false);
        ok = attempt_solve(dy_);
    } catch (const std::bad_alloc&) {
        std::cout << "[WARN] CG failed due to bad_alloc.\n";
        krylov_fail++;
        krylov_converged = false;
        ldlt_used = true;
        return solve_using_ldlt(G, H_diag, r1, r2);
    }

    // If CG succeeded and the preconditioner used SMW, reset the SMW fail streak.
    if (ok && cg.preconditioner().used_smw())
        cg.preconditioner().reset_smw_fail_streak();

    // If CG failed and the preconditioner used SMW, retry after a full rebuild.
    // If it already used a full factorization: fall back to LDLT.
    if (!ok && cg.preconditioner().used_smw()) {
        cg.preconditioner().record_smw_rebuild();
        try {
            setup_prec(true);
            ok = attempt_solve(dy_);
        } catch (const std::bad_alloc&) {
            std::cout << "[WARN] CG failed due to bad_alloc.\n";
            krylov_fail++;
            krylov_converged = false;
            ldlt_used = true;
            return solve_using_ldlt(G, H_diag, r1, r2);
        }
    }

    // If CG failed again, fall back to LDLT.
    if (!ok) {
        std::cout << "[WARN] CG failed, falling back to solving the augmented Lagrangian system via LDLT.\n";
        krylov_converged = false;
        ldlt_used = true;
        return solve_using_ldlt(G, H_diag, r1, r2);
    }

    prev_dy_ = dy_;

    // Recover dx = H_inv (G^T dy_ - r1)
    Vec dx = H_diag_inv.cwiseProduct(G_tr * dy_ - r1);

    Vec dxdy_(n + s);
    dxdy_.head(n) = dx;
    dxdy_.tail(s) = dy_;
    return dxdy_;
}

template <typename T> // not in used
typename SSN<T>::Vec SSN<T>::solve_using_minres(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv,
                                               const BoolArr& active_K, const Vec& r1, const Vec& r2,
                                               T mu, T tol, int max_iter, bool update_prec, bool prec_pattern_changed) {
    using Vec = typename SSN<T>::Vec;

    const int s = G.rows();
    const int n = G.cols();

    // Schur complement system: S dy = G H_inv G^T dy + (1/mu) dy = G H_inv r1 + r2
    SchurOperator<T> S(G, G_tr, H_diag_inv, mu);
    Vec rhs = G * H_diag_inv.cwiseProduct(r1) + r2;

    minres.setTolerance(tol);
    minres.setMaxIterations(max_iter);

    // Set up preconditioner and call minres.compute().
    // force_rebuild=true skips SMW and recomputes G E G^T from scratch.
    auto setup_prec = [&](bool force_rebuild) {
        minres.preconditioner().setData(G, G_tr, H_diag, active_K, active_W, B_rm, mu,
                                        update_prec || force_rebuild, prec_pattern_changed);
        if (force_rebuild || cg.preconditioner().smw_suppressed())
            minres.preconditioner().force_full_rebuild();
        int prec_fact_before = minres.preconditioner().fact_count();
        minres.compute(S);
        fact      += minres.preconditioner().fact_count() - prec_fact_before;
        smw_count  = minres.preconditioner().smw_count();
    };

    // Run preconditioned MINRES.
    // Returns false and increments krylov_fail on preconditioner failure or solver non-convergence.
    auto attempt_solve = [&](Vec& dy_out) -> bool {
        if (minres.preconditioner().info() != Eigen::Success) {
            krylov_fail++;
            return false;
        }
        Vec dy_;
        if (prev_dy_.size() == s)
            dy_ = minres.solveWithGuess(rhs, prev_dy_);
        else
            dy_ = minres.solve(rhs);
        krylov_iter += minres.iterations();
        if (minres.info() != Eigen::Success) {
            krylov_fail++;
            return false;
        }
        dy_out = std::move(dy_);
        return true;
    };

    // Set up and attempt to solve by MINRES.
    // bad_alloc can come from build() (P_base_ = G E G^T) or from Eigen's MINRES iteration internals;
    // fall back to LDLT in either case.
    Vec dy_;
    bool ok = false;
    try {
        setup_prec(false);
        ok = attempt_solve(dy_);
    } catch (const std::bad_alloc&) {
        krylov_fail++;
        krylov_converged = false;
        ldlt_used = true;
        return solve_using_ldlt(G, H_diag, r1, r2);
    }

    // If MINRES succeeded and the preconditioner used SMW, reset the SMW fail streak.
    if (ok && minres.preconditioner().used_smw())
        minres.preconditioner().reset_smw_fail_streak();
    
    // If MINRES failed and the preconditioner used SMW, retry after a full rebuild.
    // If it already used a full factorization: fall back to LDLT.
    if (!ok && minres.preconditioner().used_smw()) {
        minres.preconditioner().record_smw_rebuild();
        try {
            setup_prec(true);
            ok = attempt_solve(dy_);
        } catch (const std::bad_alloc&) {
            krylov_fail++;
            krylov_converged = false;
            ldlt_used = true;
            return solve_using_ldlt(G, H_diag, r1, r2);
        }
    }

    // If MINRES failed again, fall back to LDLT.
    if (!ok) {
        krylov_converged = false;
        ldlt_used = true;
        return solve_using_ldlt(G, H_diag, r1, r2);
    }

    prev_dy_ = dy_;

    // Recover dx = H_inv (G^T dy_ - r1).
    Vec dx = H_diag_inv.cwiseProduct(G_tr * dy_ - r1);

    Vec dxdy_(n + s);
    dxdy_.head(n) = dx;
    dxdy_.tail(s) = dy_;
    return dxdy_;
}

template <typename T> // not in used
typename SSN<T>::Vec SSN<T>::solve_using_schur(const SpMat& G, const SpMat& G_tr, const Vec& H_diag_inv, const Vec& r1, const Vec& r2) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;

    const int s = G.rows();
    const int n = G.cols();

    // Compute the Schur complement of J (self-adjoint and PD)
    // Schur = G H_inv G^T + D, where D = 1/mu I_{m + n_active_W}
    SpMat GH_inv = scale_columns(G, H_diag_inv);
    SpMat Schur = GH_inv * G_tr; 
    for (int i = 0; i < s; ++i) {
        Schur.coeffRef(i, i) += 1 / mu;
    }
    Schur.makeCompressed();

    // Compute the rhs = G * H_inv * r1 + r2.
    Vec rhs = GH_inv * r1 + r2;

    // Solve: Schur * dy_ = rhs, where dy_ = [dλ; dy2_active].
    Eigen::SimplicialLLT<SpMat> chol;
    chol.compute(Schur);
    if (chol.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky factorization failed");
    }
    Vec dy_ = chol.solve(rhs);
    if (chol.info() != Eigen::Success) {
        throw std::runtime_error("Solving linear system via Cholesky failed");
    }

    // Retrieve dx
    Vec dx = H_diag_inv.cwiseProduct(G_tr * dy_ - r1);

    Vec dxdy_(n + s); 
    dxdy_.head(n) = dx;
    dxdy_.tail(s) = dy_;
    return dxdy_;
}

template <typename T> // as a fallback of PCG
typename SSN<T>::Vec SSN<T>::solve_using_ldlt(const SpMat& G, const Vec& H_diag, const Vec& r1, const Vec& r2) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using Triplet = typename SSN<T>::Triplet;

    const int s = G.rows();
    const int n = G.cols();
    const int N_tot = n + s;

    Vec rhs(N_tot);
    rhs << r1, r2;

    // If stored factorization is for a different system size, force re-analyze.
    if (!ldlt_pattern_dirty_ && ldlt_.rows() > 0 && ldlt_.rows() != static_cast<Eigen::Index>(N_tot)) {
        ldlt_pattern_dirty_ = true;
        ldlt_numeric_dirty_ = true;
        K_ldlt_built_       = false;
    }

    if (ldlt_pattern_dirty_ || ldlt_numeric_dirty_) {
        if (ldlt_pattern_dirty_ || !K_ldlt_built_) {
            // Full rebuild: G's sparsity changed (active_W changed) or first call.
            // Assemble K = [-H, G^T; G, (1/mu) I] from triplets and cache it.
            std::vector<Triplet> trip;
            trip.reserve(N_tot + 2 * G.nonZeros());

            for (int i = 0; i < n; ++i)
                trip.emplace_back(i, i, -H_diag(i));
            const T mu_inv = T(1) / mu;
            for (int i = 0; i < s; ++i)
                trip.emplace_back(n + i, n + i, mu_inv);
            for (int col = 0; col < G.outerSize(); ++col)
                for (typename SpMat::InnerIterator it(G, col); it; ++it) {
                    trip.emplace_back(n + it.row(), it.col(), it.value());
                    trip.emplace_back(it.col(), n + it.row(), it.value());
                }

            K_ldlt_.resize(N_tot, N_tot);
            K_ldlt_.setFromTriplets(trip.begin(), trip.end());
            K_ldlt_.makeCompressed();
            K_ldlt_built_ = true;

            if (ldlt_pattern_dirty_) {
                ldlt_.analyzePattern(K_ldlt_);
                ldlt_pattern_dirty_ = false;
            }
        } else {
            // Pattern unchanged (active_W same): only diagonal values changed (H_diag, mu).
            // Update top-left and bottom-right diagonal entries in-place; G blocks stay.
            for (int i = 0; i < n; ++i)
                K_ldlt_.coeffRef(i, i) = -H_diag(i);
            const T mu_inv = T(1) / mu;
            for (int i = 0; i < s; ++i)
                K_ldlt_.coeffRef(n + i, n + i) = mu_inv;
        }

        ldlt_.factorize(K_ldlt_);
        if (ldlt_.info() != Eigen::Success)
            throw std::runtime_error("LDLT factorization of the augmented Lagrangian system failed.");
        ldlt_numeric_dirty_ = false;
        fact++;
    }

    Vec dxdy_ = ldlt_.solve(rhs);
    if (ldlt_.info() != Eigen::Success)
        throw std::runtime_error("Solving the augmented Lagrangian system via LDLT failed.");
    return dxdy_;
}

template <typename T> // not in used
T SSN<T>::backtracking_line_search(const Vec& x_curr, const Vec& y2_curr,
                                    const Vec& dx, const Vec& dy2,
                                    const Vec& Ax_curr, const Vec& Bx_curr,
                                    const Vec& Adx, const Vec& Bdx) {
    using Vec = typename SSN<T>::Vec;

    T alpha = delta;
    int m = 1;

    // Lagrangian and descent slope at current point — no SpMVs, use pre-computed Ax/Bx
    T L = compute_Lagrangian(x_curr, y2_curr, Ax_curr, Bx_curr);
    Vec grad_L = compute_grad_Lagrangian(x_curr, y2_curr, Ax_curr, Bx_curr);
    T grad_desc = grad_L.head(N).dot(dx) + grad_L.tail(l).dot(dy2);

    // Armijo backtracking: reduce alpha until sufficient decrease is achieved
    while (true) {
        Vec Ax_new = Ax_curr + alpha * Adx;
        Vec Bx_new = Bx_curr + alpha * Bdx;
        T L_new = compute_Lagrangian(x_curr + alpha * dx, y2_curr + alpha * dy2, Ax_new, Bx_new);
        if (L_new <= L + beta * alpha * grad_desc) break;
        m += 10;
        alpha = pow(delta, m);
        if (alpha < T(1e-5)) { alpha = T(0); break; }
    }
    return alpha;
}

template <typename T>
T SSN<T>::exact_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2,
                            const Vec& Ax_curr, const Vec& Bx_curr, const Vec& Adx, const Vec& Bdx,
                            const Vec& dist_K_u, const Vec& dist_W_v) {
    /*
    psi(t) = <∇ M(u + t du), du>
           = eta t + beta + mu <dist_K (s + t dx), dx> + <mu / gamma dv, dist_W (v + t dv)>,
    where eta  = <(Q + mu A^T A + I_n/rho) dx, dx> + (1 - gamma) / mu ||dy2||^2,
          beta = <c + Q x_curr - A^T y1 + mu A^T (A x_curr - b) + (x_curr - x)/rho, dx> + (1 - gamma) / mu <y2_curr, dy2>,
          s    = z / mu + x_curr,
          v    = B x_curr + ((1 - gamma) y2_curr - y2) / mu,
          dv   = B dx + (1 - gamma) / mu dy2.
    Compute all breakpoints t and corresponding slope changes of psi, and sort them in increasing order.
    Write psi(t) = p + m (t - t_prev).
    For each breakpoint t, if psi(t) >= 0, return t = t_prev - p / m;
    otherwise, set p = psi(t), t_prev = t and continue.
    */
    using Vec = typename SSN<T>::Vec;

    const Vec& Ax = Ax_curr;
    const Vec& Bx = Bx_curr;

    Vec s = z / mu + x_curr;
    Vec v = Bx + ((1 - gamma) * y2_curr - y2) / mu;
    Vec dv = Bdx + (1 - gamma) / mu * dy2;

    // eta: smooth linear term in psi(t)
    T eta = T(0);
    if (Q_info != 0) {
        eta += dx.dot(Q_diag.cwiseProduct(dx));
    }
    eta += mu * (Adx).squaredNorm();
    eta += (1 / rho) * dx.squaredNorm();
    eta += (1 - gamma) / mu * dy2.squaredNorm();

    // beta: constant term in psi(t)
    T beta = T(0);
    if (Q_info != 0) {
        beta += dx.dot(Q_diag.cwiseProduct(x_curr));
    }
    beta += dx.dot(c - A_tr_y1_ + mu * A_tr * (Ax - b) + (x_curr - x) / rho);
    beta += (1 - gamma) / mu * dy2.dot(y2_curr);

    // Reuse the member breakpoints vector.
    breakpoints_.clear();
    if (breakpoints_.capacity() < static_cast<size_t>(2 * (N + l)))
        breakpoints_.reserve(2 * (N + l));
    auto& breakpoints = breakpoints_;

    // Build K breakpoints and accumulate initial slope m.
    T m = eta;
    for (int i = 0; i < N; ++i) {
        const T s_i = s(i);
        const T dx_i = dx(i);
        const T li = lx(i), ui = ux(i);

        if (s_i < li - eps_zero || s_i > ui + eps_zero)
            m += mu * dx_i * dx_i;

        if (std::abs(dx_i) < eps_direction) continue;

        const T change = mu * dx_i * dx_i;
        const T t_l = (li - s_i) / dx_i;
        const T t_u = (ui - s_i) / dx_i;
        if (t_l > eps_zero) breakpoints.push_back({t_l, dx_i > 0 ? -change : +change});
        if (t_u > eps_zero) breakpoints.push_back({t_u, dx_i > 0 ? +change : -change});
    }

    // Build W breakpoints and accumulate initial slope m.
    for (int i = 0; i < l; ++i) {
        const T v_i = v(i);
        const T dv_i = dv(i);
        const T li = lw(i), ui = uw(i);

        if (v_i < li - eps_zero || v_i > ui + eps_zero)
            m += mu / gamma * dv_i * dv_i;

        if (std::abs(dv_i) < eps_direction) continue;

        const T change = mu / gamma * dv_i * dv_i;
        const T t_l = (li - v_i) / dv_i;
        const T t_u = (ui - v_i) / dv_i;
        if (t_l > eps_zero) breakpoints.push_back({t_l, dv_i > 0 ? -change : +change});
        if (t_u > eps_zero) breakpoints.push_back({t_u, dv_i > 0 ? +change : -change});
    }

    // Trivial case
    if (breakpoints.empty()) return T(1);

    // Sort breakpoints by t in ascending order.
    std::sort(breakpoints.begin(), breakpoints.end(), [](const Breakpoint& a, const Breakpoint& b){ return a.t < b.t; });

    // Merge entries with identical t.
    // Two breakpoints t = (bound - s)/d computed independently agree to within
    // ~4 * eps_machine * |t|, so a relative tolerance of 100 * eps is sufficient
    // to catch true duplicates for both float and double without over-merging
    // genuinely distinct breakpoints.
    const T merge_tol = T(100) * std::numeric_limits<T>::epsilon();
    int n_uniq = 0;
    for (size_t i = 0; i < breakpoints.size(); ) {
        T t = breakpoints[i].t;
        T slope_change_sum = T(0);
        while (i < breakpoints.size() && std::abs(breakpoints[i].t - t) < merge_tol * std::max<T>(1, std::abs(t))) {
            slope_change_sum += breakpoints[i].slope_change;
            ++i;
        }
        breakpoints[n_uniq++] = {t, slope_change_sum};
    }
    breakpoints.resize(n_uniq);

    // Check if psi(0) >= 0; if so, return 0 (no crossing, linesearch failed).
    T p = beta;
    p += mu * dist_K_u.dot(dx);
    p += mu / gamma * dist_W_v.dot(dv);
    if (p >= eps_zero * (T(1) + std::abs(beta))) return T(0);

    // If psi(0) < 0, check at every breakpoint t.
    T t_prev = T(0);
    for (const Breakpoint& bp : breakpoints) {
        T t = bp.t;
        T p_t = p + m * (t - t_prev);
        if (p_t >= 0) return t_prev - p / std::max(m, eps_zero * (T(1) + eta));

        // Cross the breakpoint(s)
        t_prev = t;
        p = p_t;
        m += bp.slope_change;
    }

    // Checking the last breakpoint.
    // m should be >= 0.
    if (m > -eps_zero * (T(1) + eta)) return t_prev - p / std::max(m, eps_zero * (T(1) + eta));
    return T(0); // safeguard: m truly negative
}

template <typename T>
void SSN<T>::solve_ssn(const T eps) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using BoolArr = typename SSN<T>::BoolArr;
    using Triplet = typename SSN<T>::Triplet;

    // Intialize iteration counter and set starting points
    x_cur_ = x;
    y2_cur_ = y2;
    int _iter = 0, _opt = -1;

    // Useful matvecs
    Ax_ssn_.noalias() = A * x_cur_;
    Bx_ssn_.noalias() = B * x_cur_;

    // SSN main loop
    while (_iter < ssn_max_in_iter) {
        // ----------------------------------------------
        // Structure:
        // Let M(u), with u = (x, y_2), be the proximal augmented Lagrangian associated with the subproblem of interest.
        // Until (||∇M(u_{k_j})|| < eps), for some given eps, do:
        //     1) Compute a Clarke subgradient J of ∇M(u_{k_j})
        //        and solve J du = - ∇M(u_{k_j}) for the Newton direction du;
        //     2) Perform exact line search to determine the step size alpha;
        //     3) Update the variables;
        //     j = j + 1;
        // End
        // ----------------------------------------------

        // Timer for SSN iteration
        auto t0_ssn = std::chrono::steady_clock::now();

        // ========== Preporation for Cholesky decomposition ==========
        auto t0_chol_prep = std::chrono::steady_clock::now();

        u_.noalias() = z / mu + x_cur_;
        v_ = Bx_ssn_ + ((1 - gamma) * y2_cur_ - y2) / mu;

        // Single pass each: Clarke subgradient and distance for K and W
        compute_subgrad_and_dist(u_, lx, ux, false, new_diag_P_K_, dist_K_u_);
        compute_subgrad_and_dist(v_, lw, uw, true,  new_diag_P_W_, dist_W_v_);

        // If P_K and P_W are unchanged, reuse the preconditioner and LDLT factorization.
        bool update_prec = false;
        // prec_pattern_changed: true when P = G E G^T + (1/mu)I has a new sparsity pattern.
        // Triggered by active_K changes (which alter E's nonzero pattern) or G structure changes (active_W).
        bool prec_pattern_changed = false;

        // Detect active-set changes; recompute on any single change.
        bool first_ssn_iter = (diag_P_K.size() == 0);
        bool pk_changed = first_ssn_iter || (diag_P_K.array() != new_diag_P_K_.array()).any();
        bool pw_changed = first_ssn_iter || (diag_P_W.array() != new_diag_P_W_.array()).any();

        if (pk_changed) {
            update_prec = true;
            prec_pattern_changed = true;
            ldlt_numeric_dirty_ = true;

            diag_P_K = new_diag_P_K_;
            active_K = (diag_P_K.array() == 1);

            // H = Q + mu(I_N - P_K) + I_N / rho
            if (Q_info == 0) {
                H_diag = mu * (ones_N - diag_P_K) + ones_N / rho;
            } else {
                H_diag = Q_diag + mu * (ones_N - diag_P_K) + ones_N / rho;
            }
            H_diag = H_diag.cwiseMax(eps_zero); // safeguard for non-positive diagonal entries
            H_diag_inv = H_diag.cwiseInverse();
        }

        if (pw_changed) {
            update_prec = true;
            prec_pattern_changed = true;
            ldlt_pattern_dirty_ = true;
            ldlt_numeric_dirty_ = true;

            prev_dy_.resize(0); // No warm-starting for CG

            diag_P_W = new_diag_P_W_;
            active_W = (diag_P_W.array() == 0);
            inactive_W = (diag_P_W.array() == 1);
            n_active_W = active_W.count();
            n_inactive_W = l - n_active_W;

            // Rebuild G = [A; B_active_W], B_active_W, B_inactive_W, G_tr.
            rebuild_G();
        }

        // Compute dy2 in inactive_W:
        //     dy2_inactive_W = - (mu / gamma) * dist_W(v)(inactive_W) - y2(inactive_W)
        split_by_mask(y2_cur_, active_W, y2_active_W_, y2_inactive_W_);
        split_by_mask(dist_W_v_, active_W, dist_W_v_active_, dist_W_v_inactive_);
        dy2_inactive_W_ = -(mu / gamma) * dist_W_v_inactive_ - y2_inactive_W_;

        // Compute the RHS vector.
        if (Q_info == 0) {
            r1_ = c + mu * dist_K_u_
                 - B_tr * y2_cur_ - B_inactive_W.transpose() * dy2_inactive_W_
                 + (x_cur_ - x) / rho;
        } else {
            r1_ = c + Q_diag.cwiseProduct(x_cur_) + mu * dist_K_u_
                 - B_tr * y2_cur_ - B_inactive_W.transpose() * dy2_inactive_W_
                 + (x_cur_ - x) / rho;
        }
        r2_.resize(M + n_active_W);
        r2_.head(M) = y1 / mu - Ax_ssn_ + b;
        r2_.tail(n_active_W) = -dist_W_v_active_ - (gamma / mu) * y2_active_W_;

        auto t1_chol_prep = std::chrono::steady_clock::now();
        double timer_chol_prep = time_diff_ms(t0_chol_prep, t1_chol_prep);
        // std::cout << "  Prep to solve SSN system took " << timer_chol_prep << " ms.\n";
        
        // Solve for dx and dy2_active_W.
        auto t0_solve_lin_sys = std::chrono::steady_clock::now();
        if ((pk_changed || pw_changed) && ldlt_decisions_made_ < 3) {
            auto t0_ratio_comp = std::chrono::steady_clock::now();
            use_ldlt = choose_ldlt(G, active_K);
            ++ldlt_decisions_made_;
            if (use_ldlt) std::cout << "LDLT used.\n";
            else std::cout << "Chol used.\n";
            auto t1_ratio_compt = std::chrono::steady_clock::now();
            // std::cout << "choosing a system took " << time_diff_ms(t0_ratio_comp, t1_ratio_compt) << "ms.\n";
        } // Determines the factorization method for a preconditioner; locked after the first 3 decisions
        dxdy_ = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, r1_, r2_, mu, krylov_tol, krylov_max_in_iter, update_prec, prec_pattern_changed, use_ldlt);

        auto t1_solve_lin_sys = std::chrono::steady_clock::now();
        double timer_solve_line_sys = time_diff_ms(t0_solve_lin_sys, t1_solve_lin_sys);
        // std::cout << "  Solving SSN system took " << timer_solve_line_sys << " ms.\n";

        // Split dxdy_ into dx and dy2_active_W.
        const auto dx = dxdy_.head(N);
        const auto dy2_active_W = dxdy_.tail(n_active_W);

        assert(dy2_active_W.size() == n_active_W);
        assert(dy2_inactive_W_.size() == n_inactive_W);
        assert(active_W.size() == l);

        retrieve_row_order(dy2_active_W, dy2_inactive_W_, active_W, dy2_);

        // ========== Exact linesearch ==========
        auto t0_alpha = std::chrono::steady_clock::now();
        Adx_.noalias() = A * dx;
        Bdx_.noalias() = B * dx;
        T alpha = exact_line_search(x_cur_, y2_cur_, dx, dy2_,
                                    Ax_ssn_, Bx_ssn_, Adx_, Bdx_,
                                    dist_K_u_, dist_W_v_);
        auto t1_alpha = std::chrono::steady_clock::now();
        double timer_alpha = time_diff_ms(t0_alpha, t1_alpha);
        // std::cout << "  Linesearch took " << timer_alpha << " ms.\n";
        // std::cout << "  alpha = " << alpha << "\n";

        // ========== Update x and y2 ==========
        if (alpha <= T(0)) { // Line search failed.
            linesearch_fail++;
            _opt = 3;
            break;
        } else {
            x_cur_  += alpha * dx;
            y2_cur_ += alpha * dy2_;
            if (_iter % 10 == 9) { // Reset incremental drift every 10 iterations.
                Ax_ssn_.noalias() = A * x_cur_;
                Bx_ssn_.noalias() = B * x_cur_;
            } else {
                Ax_ssn_ += alpha * Adx_;
                Bx_ssn_ += alpha * Bdx_;
            }
        }

        // Compute gradient of Lagrangian at current (x, y2).
        grad_L_ = compute_grad_Lagrangian(x_cur_, y2_cur_, Ax_ssn_, Bx_ssn_);
        tol_achieved = inf_norm(grad_L_);
        _iter++;

        auto t1_ssn = std::chrono::steady_clock::now();
        double timer_ssn = time_diff_ms(t0_ssn, t1_ssn);
        // std::cout << "  SSN iteration took " << timer_ssn << " ms.\n";

        // Check termination criterion.
        if (tol_achieved < eps) {
            _opt = 0; // Optimality achieved.
            break;
        }
    }

    if (_opt == -1) {
        _opt = 2; // Maximum number of SSN inner iterations reached without convergence.
    }
    x  = x_cur_;
    y2 = y2_cur_;
    opt = _opt;
    iter = _iter;
}
