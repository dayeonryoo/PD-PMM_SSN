#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <Eigen/SparseCholesky>
#include <cassert>

template <typename T>
T SSN<T>::compute_Lagrangian(const Vec& x_new, const Vec& y2_new) {
    using Vec = typename SSN<T>::Vec;

    // Evalueate dist_K(z/mu + x_new)
    Vec dist_K = compute_dist_box(z / mu + x_new, lx, ux);

    // Evaluate dist_W(B*x_new - (y2 - y2_new/2)/mu)
    Vec dist_W = compute_dist_box(B * x_new + ((1 - gamma) * y2_new - y2) / mu, lw, uw);

    // Evaluate primal residual A x_new - b
    Vec res_p = A * x_new - b;

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

    // Evalueate Dist_K (z/mu + x_new)
    Vec dist_K = compute_dist_box(z / mu + x_new, lx, ux);

    // Evaluate Dist_W (B*x_new + (y2_new/2 - y2)/mu)
    Vec dist_W = compute_dist_box(Bx_new + ((1 - gamma) * y2_new - y2) / mu, lw, uw);

    // Primal residual: A x_new - b
    Vec res_p = Ax_new - b;

    // Compute gradient of Lagrangian
    Vec grad_L_x;
    if (Q_info == 0) {
        grad_L_x = c - A_tr_y1_ + mu * A_tr * res_p
                    + mu * dist_K
                    + (mu / gamma) * B_tr * dist_W
                    + (x_new - x) / rho;
    } else {
        grad_L_x = c + Q_diag.cwiseProduct(x_new) - A_tr_y1_ + mu * A_tr * res_p
                    + mu * dist_K
                    + (mu / gamma) * B_tr * dist_W
                    + (x_new - x) / rho;
    }
    Vec grad_L_y2 = ((1 - gamma) / gamma) * dist_W + ((1 - gamma) / mu) * y2_new;

    // Combine gradients
    Vec grad_L(N + l);
    grad_L << grad_L_x, grad_L_y2;

    return grad_L;

}

template <typename T>
typename SSN<T>::Vec SSN<T>::Clarke_subgrad_of_proj(const Vec& u, const Vec& lower, const Vec& upper, const bool include_bd) {
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
void SSN<T>::build_B_active_inactive(const SpMat& B, const BoolArr& active, SpMat& B_active, SpMat& B_inactive) {
    using Triplet = typename SSN<T>::Triplet;

    const int l = B.rows();
    const int N = B.cols();

    const int n_active = static_cast<int>(active.count());
    const int n_inactive = l - n_active;

    B_active.resize(n_active, N);
    B_inactive.resize(n_inactive, N);

    std::vector<Triplet> trip_act;
    std::vector<Triplet> trip_inact;
    trip_act.reserve(B.nonZeros());
    trip_inact.reserve(B.nonZeros());

    Eigen::VectorXi row_map_act(l);
    Eigen::VectorXi row_map_inact(l);

    int i_act = 0;
    int i_inact = 0;
    for (int i = 0; i < l; ++i) {
        if (active(i)) {
            row_map_act(i) = i_act++;
        } else {
            row_map_inact(i) = i_inact++;
        }
    }

    for (int col = 0; col < N; ++col) {
        for (typename SpMat::InnerIterator it(B, col); it; ++it) {
            const int i = it.row();
            if (active(i)) {
                trip_act.emplace_back(row_map_act(i), col, it.value());
            } else {
                trip_inact.emplace_back(row_map_inact(i), col, it.value());
            }
        }
    }

    B_active.setFromTriplets(trip_act.begin(), trip_act.end());
    B_inactive.setFromTriplets(trip_inact.begin(), trip_inact.end());
    
}

template <typename T>
void SSN<T>::rebuild_G() {
    using RIt = typename RowMajorSpMat::InnerIterator;

    const int n_act   = n_active_W;
    const int n_inact = l - n_act;

    // Single pass over B_rm rows, partitioning into active and inactive.
    // Simultaneously builds:
    //   B_act_trips  → B_active_W  (rows 0..n_act-1)
    //   B_inact_trips → B_inactive_W (rows 0..n_inact-1)
    //   G_trips       → G = [A; B_active_W] (rows 0..M+n_act-1)
    std::vector<Triplet> B_act_trips, B_inact_trips;
    B_act_trips.reserve(B_rm.nonZeros());
    B_inact_trips.reserve(B_rm.nonZeros());

    // G_trips starts from the pre-cached A part (never changes).
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

template <typename T>
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

template <typename T>
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
typename SSN<T>::Vec SSN<T>::retrive_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask) {
    assert(u_sel.size() == mask.count());
    assert(u_unsel.size() == mask.size() - mask.count());

    int i_sel = 0;
    int i_unsel = 0;
    Vec u(mask.size());
    for (int i = 0; i < mask.size(); ++i) {
        if (mask(i)) {
            assert(i_sel < u_sel.size());
            u(i) = u_sel(i_sel++);
        } else {
            assert(i_unsel < u_unsel.size());
            u(i) = u_unsel(i_unsel++);
        }
    }
    return u;
}

template <typename T>
bool SSN<T>::form_schur(const SpMat& G) {
    // |KKT| = N + s + 2|G|
    // |Schur| ~ |G H_inv G^T| approximated by using denstest column in G
    //         ~ s + (r_hat^2 - r_hat) + sum_{i != i_hat}(r_i^2 - r_i - t_i^2 + t_i)
    // If (s / (N + s)) |KKT|^2 / |Schur|^2 >= 2, choose KKT system;
    // otherwise, form the Schur complement.
    using Index = typename SpMat::Index;
    using StorageIndex = typename SpMat::StorageIndex;

    const Index s = G.rows();
    const Index N = G.cols();

    if (s == 0) return true;

    const long long nnzG = static_cast<long long>(G.nonZeros());
    const long long KKT = static_cast<long long>(N) + static_cast<long long>(s) + 2LL * nnzG;

    // r[i] = nnz of column i; densest column found in the same pass
    std::vector<long long> r(N);
    const StorageIndex* outer = G.outerIndexPtr();
    Index i_hat = 0;
    long long r_hat = 0;
    for (Index i = 0; i < N; ++i) {
        r[i] = static_cast<long long>(outer[i+1] - outer[i]);
        if (r[i] > r_hat) { r_hat = r[i]; i_hat = i; }
    }

    // Square function
    auto sq = [](long long x) { return x * x; };

    // Accumulate the first three terms of |Schur|
    // |Schur| ~ s + (r_hat^2 - r_hat) + sum_{i != i_hat}(r_i^2 - r_i - t_i^2 + t_i)
    // with t_i = [r_hat + r_i - s]_+
    long long Schur = static_cast<long long>(s);
    Schur += sq(r_hat) - r_hat;

    // Loop over all columns in G
    for (Index i = 0; i < N; ++i) {
        if (i == i_hat) continue;

        const long long r_i = static_cast<long long>(r[i]);
        const long long t_i = std::max<long long>(0, r_hat + r_i - s);

        // Accumulate r_i^2 - r_i - t_i^2 + t_i
        Schur += sq(r_i) - r_i - sq(t_i) + t_i;
    }

    double ratio = (double(s) / (double(N + s))) * (double(sq(KKT)) / double(sq(Schur)));
    return ratio > 0.05;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::solve_using_cg(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv,
                                            const BoolArr& active_K, const Vec& r1, const Vec& r2,
                                            T mu, T tol, int max_iter, bool update_prec, bool prec_pattern_changed) {
    using Vec = typename SSN<T>::Vec;

    const int s = G.rows();
    const int n = G.cols();

    SchurOperator<T> S(G, G_tr, H_diag_inv, mu);
    Vec rhs = G * H_diag_inv.cwiseProduct(r1) + r2;
    cg.setTolerance(tol);
    cg.setMaxIterations(max_iter);

    // Set up preconditioner and call cg.compute(). force_rebuild=true skips SMW and
    // recomputes G E G^T from scratch, regardless of what changed since the last full build.
    auto setup_prec = [&](bool force_rebuild) {
        cg.preconditioner().setData(G, G_tr, H_diag, active_K, active_W, B_rm, mu,
                                    update_prec || force_rebuild, prec_pattern_changed);
        if (force_rebuild)
            cg.preconditioner().force_full_rebuild();
        int prec_fact_before = cg.preconditioner().fact_count();
        cg.compute(S);
        fact      += cg.preconditioner().fact_count() - prec_fact_before;
        smw_count  = cg.preconditioner().smw_count();
    };

    // Run CG with the current preconditioner. Returns false and increments Krylov_fail
    // on preconditioner failure or solver non-convergence, leaving dy_out unchanged.
    auto attempt_solve = [&](Vec& dy_out) -> bool {
        if (cg.preconditioner().info() != Eigen::Success) {
            Krylov_fail++;
            std::cout << "[CG] Preconditioner factorization failed with info = "
                      << cg.preconditioner().info() << ".\n";
            return false;
        }
        Vec dy_;
        if (prev_dy_.size() == s)
            dy_ = cg.solveWithGuess(rhs, prev_dy_);
        else
            dy_ = cg.solve(rhs);
        Krylov_iter += cg.iterations();
        if (cg.info() != Eigen::Success) {
            Krylov_fail++;
            std::cout << "[CG] Krylov solver failed with info = " << cg.info() << ".\n";
            return false;
        }
        dy_out = std::move(dy_);
        return true;
    };

    // First CG attempt. Wrap setup + solve together: bad_alloc can come from
    // build() (P_base_ = G E G^T) or from Eigen's CG iteration internals.
    Vec dy_;
    bool ok = false;
    try {
        setup_prec(false);
        ok = attempt_solve(dy_);
    } catch (const std::bad_alloc&) {
        Krylov_fail++;
        std::cout << "[CG] std::bad_alloc. Falling back to LDLT.\n";
        Krylov_converged = false;
        ldlt_used = true;
        return solve_using_LDLT(G, H_diag, r1, r2);
    }

    if (ok && cg.preconditioner().used_smw())
        cg.preconditioner().reset_smw_fail_streak();

    // If CG failed and the preconditioner used SMW (possibly stale), retry after a full rebuild.
    // If it already used a full factorization, the preconditioner is as fresh as possible: go to LDLT.
    if (!ok && cg.preconditioner().used_smw()) {
        cg.preconditioner().record_smw_rebuild();
        std::cout << "[CG] SMW preconditioner may be stale; retrying with full rebuild"
                  << (cg.preconditioner().smw_suppressed() ? " (SMW now suppressed)" : "") << ".\n";
        try {
            setup_prec(true);
            ok = attempt_solve(dy_);
        } catch (const std::bad_alloc&) {
            Krylov_fail++;
            std::cout << "[CG] std::bad_alloc on retry. Falling back to LDLT.\n";
            Krylov_converged = false;
            ldlt_used = true;
            return solve_using_LDLT(G, H_diag, r1, r2);
        }
    }

    if (!ok) {
        std::cout << "[CG] Falling back to LDLT.\n";
        Krylov_converged = false;
        ldlt_used = true;
        return solve_using_LDLT(G, H_diag, r1, r2);
    }

    prev_dy_ = dy_;

    // Recover dx = H_inv (G^T dy_ - r1)
    Vec dx = H_diag_inv.cwiseProduct(G_tr * dy_ - r1);

    Vec dxdy_(n + s);
    dxdy_.head(n) = dx;
    dxdy_.tail(s) = dy_;
    return dxdy_;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::solve_using_cg_primal(const SpMat& G, const SpMat& G_tr, const Vec& H_diag,
                                                    const Vec& r1, const Vec& r2,
                                                    T mu, T tol, int max_iter) {
    using Vec = typename SSN<T>::Vec;

    const int s = G.rows();
    const int n = G.cols();

    // Diagonal preconditioner: diag(H + mu G^T G)^{-1}.
    // diag(G^T G)_i = ||column i of G||^2, computed in O(nnz(G)) without forming G^T G.
    Vec prec_inv = H_diag;
    for (int col = 0; col < G.outerSize(); ++col) {
        T sq = T(0);
        for (typename SpMat::InnerIterator it(G, col); it; ++it)
            sq += it.value() * it.value();
        prec_inv(col) += mu * sq;
    }
    prec_inv = prec_inv.cwiseInverse();

    // RHS: mu G^T r2 - r1  (derived by eliminating dy from the Newton system)
    Vec rhs = mu * (G_tr * r2) - r1;

    // Matrix-free preconditioned CG for (H + mu G^T G) dx = rhs.
    // Each mat-vec: (H + mu G^T G) v = H.*v + mu * G^T (G v)  — no explicit G^T G formed.
    auto matvec = [&](const Vec& v) -> Vec {
        return H_diag.cwiseProduct(v) + mu * (G_tr * (G * v));
    };

    Vec x = (prev_dx_primal_.size() == n) ? prev_dx_primal_ : Vec::Zero(n);
    Vec r = rhs - matvec(x);
    Vec z = prec_inv.cwiseProduct(r);
    Vec p = z;
    T rz = r.dot(z);
    const T tol_sq = tol * tol * rhs.squaredNorm();

    for (int iter = 0; iter < max_iter; ++iter) {
        if (r.squaredNorm() <= tol_sq) break;
        Vec Ap = matvec(p);
        T pAp = p.dot(Ap);
        if (std::abs(pAp) < T(1e-30) * std::abs(rz)) break;
        T alpha = rz / pAp;
        x += alpha * p;
        r -= alpha * Ap;
        z = prec_inv.cwiseProduct(r);
        T rz_new = r.dot(z);
        T beta = rz_new / rz;
        p = z + beta * p;
        rz = rz_new;
    }
    prev_dx_primal_ = x;

    // Recover dy = mu (r2 - G dx)
    Vec dy = mu * (r2 - G * x);

    Vec dxdy(n + s);
    dxdy.head(n) = x;
    dxdy.tail(s) = dy;
    return dxdy;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::solve_using_minres(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv,
                                               const BoolArr& active_K, const Vec& r1, const Vec& r2,
                                               T mu, T tol, int max_iter, bool update_prec, bool prec_pattern_changed) {
    using Vec = typename SSN<T>::Vec;

    const int s = G.rows();
    const int n = G.cols();

    SchurOperator<T> S(G, G_tr, H_diag_inv, mu);

    // Precompute H_inv * r1 once: reused for rhs and dx recovery.
    Vec r1_scaled = H_diag_inv.cwiseProduct(r1);
    Vec rhs(s);
    rhs.noalias() = G * r1_scaled;
    rhs += r2;

    minres.setTolerance(tol);
    minres.setMaxIterations(max_iter);
    minres.preconditioner().setData(G, G_tr, H_diag, active_K, active_W, B_rm, mu, update_prec, prec_pattern_changed);
    int prec_fact_before = minres.preconditioner().fact_count();
    minres.compute(S);
    fact      += minres.preconditioner().fact_count() - prec_fact_before;
    smw_count  = minres.preconditioner().smw_count();

    if (minres.preconditioner().info() != Eigen::Success) {
        throw std::runtime_error("Preconditioner setup failed.");
    }

    Vec dy_;
    if (prev_dy_.size() == s) {
        dy_ = minres.solveWithGuess(rhs, prev_dy_);
    } else {
        dy_ = minres.solve(rhs);
    }

    Krylov_iter += minres.iterations();

    if (minres.info() != Eigen::Success) {
        Krylov_fail++;
        std::cout << "[MINRES] Krylov solver failed with info = " << minres.info() << ". Falling back to LDLT.\n";
        Krylov_converged = false;
        return solve_using_LDLT(G, H_diag, r1, r2);
    }

    // Recover dx = H_inv (G^T dy - r1) = H_inv * G^T * dy - r1_scaled.
    // Write dx directly into dxdy_ to avoid a separate size-n allocation and temporaries.
    Vec dxdy_(n + s);
    dxdy_.tail(s) = dy_;
    dxdy_.head(n).noalias() = G_tr * dy_;
    dxdy_.head(n).array() *= H_diag_inv.array();
    dxdy_.head(n) -= r1_scaled;

    prev_dy_ = std::move(dy_);
    return dxdy_;
}

template <typename T>
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

    // Retrive dx
    Vec dx = H_diag_inv.cwiseProduct(G_tr * dy_ - r1);

    Vec dxdy_(n + s); 
    dxdy_.head(n) = dx;
    dxdy_.tail(s) = dy_;
    return dxdy_;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::solve_using_LDLT(const SpMat& G, const Vec& H_diag, const Vec& r1, const Vec& r2) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using Triplet = typename SSN<T>::Triplet;

    const int s = G.rows();
    const int n = G.cols();
    const int N_tot = n + s;

    Vec rhs(N_tot);
    rhs << r1, r2;

    // If stored factorization is for a different system size, force re-analyze.
    if (!ldlt_pattern_dirty_ && ldlt_.rows() > 0 &&
        ldlt_.rows() != static_cast<Eigen::Index>(N_tot)) {
        std::cerr << "[SSN LDLT] stale analysis: ldlt_.rows()=" << ldlt_.rows()
                  << " vs N_tot=" << N_tot << " (n_active_W=" << n_active_W
                  << ", M=" << M << ", N=" << N << "); forcing re-analyze\n";
        ldlt_pattern_dirty_ = true;
        ldlt_numeric_dirty_ = true;
        K_ldlt_built_ = false;
    }

    if (ldlt_pattern_dirty_ || ldlt_numeric_dirty_) {
        if (ldlt_pattern_dirty_ || !K_ldlt_built_) {
            // Full rebuild: G's sparsity changed (active_W changed) or first call.
            // Assemble K = [-H, G^T; G, (1/mu) I] from triplets and cache it.
            // All diagonal entries are always inserted so in-place updates are safe later.
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

template <typename T>
typename SSN<T>::Vec SSN<T>::solve_using_primal_ldlt(const Vec& H_diag, const Vec& r1, const Vec& r2) {
    // Solves the SSN system via the N×N primal normal equations:
    //   (H + mu G^T G) dx = mu G^T r2 - r1
    //   dy = mu (r2 - G dx)
    // where G = [A; B_active_W], G^T G = A^T A + B_active_W^T B_active_W.
    // A^T A is cached; only B_active_W^T B_active_W needs to be recomputed on active_W changes.
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;

    const int s = M + n_active_W;

    // Compute A^T A lazily on first use (A is const, so only once per SSN lifetime).
    if (A_tr_A_.rows() == 0)
        A_tr_A_ = A_tr * A;

    if (primal_pattern_dirty_ || primal_numeric_dirty_) {
        SpMat P = A_tr_A_;
        if (n_active_W > 0) {
            SpMat BaTBa = B_active_W.transpose() * B_active_W;
            P += BaTBa;
        }
        // Add mu-scaled G^T G and diagonal H
        P *= mu;
        for (int i = 0; i < N; ++i)
            P.coeffRef(i, i) += H_diag(i);
        P.makeCompressed();

        if (primal_pattern_dirty_) {
            primal_llt_.analyzePattern(P);
            primal_pattern_dirty_ = false;
        }
        primal_llt_.factorize(P);
        if (primal_llt_.info() != Eigen::Success)
            throw std::runtime_error("Primal LLT factorization failed.");
        primal_numeric_dirty_ = false;
        fact++;
    }

    // rhs = mu G^T r2 - r1 = mu (A^T r2_A + B_active_W^T r2_B) - r1
    Vec r2_A = r2.head(M);
    Vec r2_B = r2.tail(n_active_W);
    Vec rhs = mu * (A_tr * r2_A) - r1;
    if (n_active_W > 0)
        rhs += mu * (B_active_W.transpose() * r2_B);

    Vec dx = primal_llt_.solve(rhs);
    if (primal_llt_.info() != Eigen::Success)
        throw std::runtime_error("Primal LLT solve failed.");

    // Recover dy = mu (r2 - G dx)
    Vec dy(s);
    dy.head(M) = mu * (r2_A - A * dx);
    if (n_active_W > 0)
        dy.tail(n_active_W) = mu * (r2_B - B_active_W * dx);

    Vec dxdy(N + s);
    dxdy.head(N) = dx;
    dxdy.tail(s) = dy;
    return dxdy;
}

template <typename T>
T SSN<T>::backtracking_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2) {
    using Vec = typename SSN<T>::Vec;

    // Increase m until alpha = delta^m breaks the Armijo-Goldstein condition
    T alpha = delta;
    int m = 1;

    // Evaluate Lagrangian and its gradient at current u = [x; y]
    T L = compute_Lagrangian(x_curr, y2_curr);
    Vec Ax_curr = A * x_curr;
    Vec Bx_curr = B * x_curr;
    Vec grad_L = compute_grad_Lagrangian(x_curr, y2_curr, Ax_curr, Bx_curr);

    T grad_desc = grad_L.head(N).dot(dx) + grad_L.tail(l).dot(dy2);

    Vec Adx = A * dx;
    Vec Bdx = B * dx;

    // Iterate until finding the largest step size satisfying the Armijo-Goldstein condition
    while (true) {

        // Evaluate Lagrangian at u_new = u + alpha * du
        Vec x_new = x_curr + alpha * dx;
        Vec y2_new = y2_curr + alpha * dy2;
        T L_new = compute_Lagrangian(x_new, y2_new);

        if (L_new <= L + beta * alpha * grad_desc) break;

        m += 10;
        alpha = pow(delta, m);
        
        if (alpha < 1e-2) { // Lower bound on alpha
            // std::cout << "  SSN: Backtracking linesearch failed.\n";
            alpha = T(0);
            break;
        }
    }
    return alpha;
}

template <typename T>
T SSN<T>::exact_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2,
                             const Vec& Ax_curr, const Vec& Bx_curr, const Vec& Adx, const Vec& Bdx) {
    /*
    psi(t) = <∇ M(u + t du), du>,
            = eta t + beta + mu <dist_K (s + t dx), dx> + <mu/gamma dv, dist_W (v + t dv)>,
    where eta  = <(Q + mu A^T A + I_n/rho) dx, dx> + (1-gamma)/mu ||dy2||^2,
          beta = <c + Q x_curr - A^T y1 + mu A^T (A x_curr - b) + (x_curr - x)/rho, dx> + (1 - gamma)/mu <y2, dy2>,
          s = z / mu + x_curr,
          v = B x_curr + ((1-gamma) y2_curr - y2) / mu,
          dv = B dx + (1-gamma)/mu dy2.
    Compute all breakpoints t and corresponding slope changes of psi, and sort them in increasing order.
    Write psi(t) = p + m(t - t_prev).
    For each breakpoint t, if psi(t) >= 0, return t = t_prev - p / m;
    otherwise, set p = psi(t), t_prev = t and continue.
    */
    using Vec = typename SSN<T>::Vec;

    // Ax_curr, Bx_curr, Adx, Bdx are all passed in to avoid redundant SpMVs
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

    // Breakpoint and slope change of psi when crossing it
    struct Breakpoint {
        T t;
        T slope_change;
    };
    std::vector<Breakpoint> breakpoints;
    breakpoints.reserve(2 * (N + l));

    // Build K breakpoints and accumulate initial slope m in one pass
    T m = eta;
    for (int i = 0; i < N; ++i) {
        const T s_i = s(i);
        const T dx_i = dx(i);
        const T li = lx(i), ui = ux(i);

        if (s_i < li - eps_zero || s_i > ui + eps_zero)
            m += mu * dx_i * dx_i;

        if (std::abs(dx_i) < eps_zero) continue;

        const T change = mu * dx_i * dx_i;
        const T t_l = (li - s_i) / dx_i;
        const T t_u = (ui - s_i) / dx_i;
        if (t_l > eps_zero) breakpoints.push_back({t_l, dx_i > 0 ? -change : +change});
        if (t_u > eps_zero) breakpoints.push_back({t_u, dx_i > 0 ? +change : -change});
    }

    // Build W breakpoints and accumulate initial slope m in one pass
    for (int i = 0; i < l; ++i) {
        const T v_i = v(i);
        const T dv_i = dv(i);
        const T li = lw(i), ui = uw(i);

        if (v_i < li - eps_zero || v_i > ui + eps_zero)
            m += mu / gamma * dv_i * dv_i;

        if (std::abs(dv_i) < eps_zero) continue;

        const T change = mu / gamma * dv_i * dv_i;
        const T t_l = (li - v_i) / dv_i;
        const T t_u = (ui - v_i) / dv_i;
        if (t_l > eps_zero) breakpoints.push_back({t_l, dv_i > 0 ? -change : +change});
        if (t_u > eps_zero) breakpoints.push_back({t_u, dv_i > 0 ? +change : -change});
    }

    // Sort by t
    std::sort(breakpoints.begin(), breakpoints.end(), [](const Breakpoint& a, const Breakpoint& b){ return a.t < b.t; });

    // Deduplicate in-place: merge entries with identical t
    int n_uniq = 0;
    for (size_t i = 0; i < breakpoints.size(); ) {
        T t = breakpoints[i].t;
        T slope_change_sum = T(0);
        while (i < breakpoints.size() && std::abs(breakpoints[i].t - t) < eps_zero * std::max<T>(1, std::abs(t))) {
            slope_change_sum += breakpoints[i].slope_change;
            ++i;
        }
        breakpoints[n_uniq++] = {t, slope_change_sum};
    }
    breakpoints.resize(n_uniq);

    // Check if psi(0) >= 0
    Vec dist_K_s = compute_dist_box(s, lx, ux);
    Vec dist_W_v = compute_dist_box(v, lw, uw);
    T p = beta;
    p += mu * dist_K_s.dot(dx);
    p += mu / gamma * dist_W_v.dot(Bdx);
    p += (1 - gamma) / gamma * dist_W_v.dot(dy2);
    if (p >= eps_zero * (T(1) + std::abs(beta))) return T(0); // No crossing, linesearch failed.

    // If psi(0) < 0, check at every breakpoint t.
    T t_prev = T(0);
    // m was computed during breakpoint generation above

    // Check at each breakpoint t
    for (const Breakpoint& bp : breakpoints) {
        T t = bp.t;
        T p_t = p + m * (t - t_prev);
        if (p_t >= 0) return t_prev - p / m;

        // Cross the breakpoint(s)
        t_prev = t;
        p = p_t;
        m += bp.slope_change;
    }

    // Checking the last breakpoint.
    // m should be >= 0 (sum of squares); slightly negative means floating-point cancellation.
    if (m > -eps_zero * (T(1) + eta)) return t_prev - p / std::max(m, eps_zero * (T(1) + eta));
    return T(0); // safeguard: m truly negative
}

template <typename T>
SSN_result<T> SSN<T>::solve_SSN(const T eps) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using BoolArr = typename SSN<T>::BoolArr;
    using Triplet = typename SSN<T>::Triplet;

    // Intialize iteration counter and set starting points
    SSN_result<T> result;
    result.x = x;
    result.y2 = y2;
    result.iter = 0;
    result.opt = -1;
    result.tol_achieved = T(0);

    // Useful matvecs
    Vec Ax = A * result.x;
    Vec Bx = B * result.x;

    // SSN main loop
    while (result.iter < SSN_max_in_iter) {
        // ----------------------------------------------
        // Structure:
        // Let M(u), with u = (x,y_2), be the proximal augmented Lagrangian
        // associated with the subproblem of interest.
        // Until (|| \nabla M(u_{k_j}) || < eps), for some given eps, do:
        //     1) Compute a Clarke subgradient J of \nabla M(u_{k_j})
        //        and solve J du = - \nabla M(u_{k_j}) for the Newton direction du;
        //     2) Perform a backtracking line search to determine the step size alpha;
        //     3) Update the variables;
        //     j = j + 1;
        // End
        // ----------------------------------------------

        // TIMER FOR SSN ITERATION
        auto t0_ssn = std::chrono::steady_clock::now();

        // ========== Preporation for Cholesky decomposition ==========
        auto t0_chol_prep = std::chrono::steady_clock::now();

        Vec u = z / mu + result.x;
        Vec v = Bx + ((1 - gamma) * result.y2 - y2) / mu;

        // Single pass each: Clarke subgradient and distance for K and W
        Vec new_diag_P_K, dist_K_u;
        Vec new_diag_P_W, dist_W_v;
        compute_subgrad_and_dist(u, lx, ux, false, new_diag_P_K, dist_K_u);
        compute_subgrad_and_dist(v, lw, uw, true,  new_diag_P_W, dist_W_v);

        // If P_K and P_W are unchanged, reuse the preconditioner and LDLT factorization.
        bool update_prec = false;
        // prec_pattern_changed: true when P = G E G^T + (1/mu)I may have a new sparsity pattern.
        // Triggered by active_K changes (which alter E's nonzero pattern) or G structure changes (active_W).
        bool prec_pattern_changed = false;

        // Detect active-set changes; recompute on any single change.
        bool first_ssn_iter = (diag_P_K.size() == 0);
        bool pk_changed = first_ssn_iter || (diag_P_K.array() != new_diag_P_K.array()).any();
        bool pw_changed = first_ssn_iter || (diag_P_W.array() != new_diag_P_W.array()).any();

        if (pk_changed) {
            update_prec = true;
            prec_pattern_changed = true;
            ldlt_numeric_dirty_ = true;
            primal_numeric_dirty_ = true;

            diag_P_K = new_diag_P_K;
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
            primal_pattern_dirty_ = true;
            primal_numeric_dirty_ = true;

            diag_P_W = new_diag_P_W;
            active_W = (diag_P_W.array() == 0);
            inactive_W = (diag_P_W.array() == 1);
            n_active_W = active_W.count();
            n_inactive_W = l - n_active_W;

            // Rebuild G = [A; B_active_W], B_active_W, B_inactive_W, G_tr
            // in a single row-major pass using pre-cached A trips.
            rebuild_G();
        }

        // Compute dy2 in inactive_W:
        // dy2_inactive_W = - (mu / gamma) * dist_W(v)(inactive_W) - y2(inactive_W)
        Vec y2_active_W, y2_inactive_W;
        split_by_mask(result.y2, active_W, y2_active_W, y2_inactive_W);

        Vec dist_W_v_active_W, dist_W_v_inactive_W;
        split_by_mask(dist_W_v, active_W, dist_W_v_active_W, dist_W_v_inactive_W);

        Vec dy2_inactive_W = -(mu / gamma) * dist_W_v_inactive_W - y2_inactive_W;
        
        // Compute the RHS vector
        Vec r1;
        if (Q_info == 0) {
            r1 = c + mu * dist_K_u
                 - B_tr * result.y2 - B_inactive_W.transpose() * dy2_inactive_W
                 + (result.x - x) / rho;
        } else {
            r1 = c + Q_diag.cwiseProduct(result.x) + mu * dist_K_u
                 - B_tr * result.y2 - B_inactive_W.transpose() * dy2_inactive_W
                 + (result.x - x) / rho;
        }
        Vec r2(M + n_active_W);
        r2.head(M) = y1 / mu - Ax + b;
        r2.tail(n_active_W) = -dist_W_v_active_W - (gamma / mu) * y2_active_W;

        auto t1_chol_prep = std::chrono::steady_clock::now();
        double timer_chol_prep = time_diff_ms(t0_chol_prep, t1_chol_prep);
        // std::cout << "  Prep to solve SSN system took " << timer_chol_prep << " ms.\n";

        // Solve for dx and dy2_active_W
        auto t0_solve_lin_sys = std::chrono::steady_clock::now();
        Krylov_tol= std::max(T(1e-16), T(0.5) * Krylov_tol);
        Vec dxdy_;
        if (ldlt_used) {
            dxdy_ = solve_using_LDLT(G, H_diag, r1, r2);
        } else {
            dxdy_ = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, r1, r2, mu, Krylov_tol, Krylov_max_in_iter, update_prec, prec_pattern_changed);
        }


        auto t1_solve_lin_sys = std::chrono::steady_clock::now();
        double timer_solve_line_sys = time_diff_ms(t0_solve_lin_sys, t1_solve_lin_sys);
        // std::cout << "  Solving SSN system took " << timer_solve_line_sys << " ms.\n";

        Vec dx = dxdy_.head(N);
        Vec dy2_active_W = dxdy_.tail(n_active_W);

        assert(dy2_active_W.size() == n_active_W);
        assert(dy2_inactive_W.size() == n_inactive_W);
        assert(active_W.size() == l);

        Vec dy2 = retrive_row_order(dy2_active_W, dy2_inactive_W, active_W);

        // ========== Backtracking/exact linesearch ==========
        auto t0_alpha = std::chrono::steady_clock::now();
        Vec Adx = A * dx;
        Vec Bdx = B * dx;
        T alpha;
        // alpha = backtracking_line_search(result.x, result.y2, dx, dy2);
        alpha = exact_line_search(result.x, result.y2, dx, dy2, Ax, Bx, Adx, Bdx);
        /*
        if (do_exact) {
            // std::cout << "  Exact linesearch applied.\n";
            alpha = exact_line_search(result.x, result.y2, dx, dy2);
        } else {
            // std::cout << "  Backtracking linesearch applied.\n";
            alpha = backtracking_line_search(result.x, result.y2, dx, dy2);
        }
        */
        auto t1_alpha = std::chrono::steady_clock::now();
        double timer_alpha = time_diff_ms(t0_alpha, t1_alpha);
        // std::cout << "  Linesearch took " << timer_alpha << " ms.\n";
        // std::cout << "  alpha = " << alpha << "\n";

        // ========== Update x and y2 ==========
        if (alpha == 0) { // If linesearch fails, use gradient descent.
            linesearch_fail++;
            Vec grad_L_gd = compute_grad_Lagrangian(result.x, result.y2, Ax, Bx);
            // std::cout << "GD applied: ||grad_x|| = " << grad_L_gd.head(N).norm() << ", ||grad_y2|| = " << grad_L_gd.tail(l).norm() << "\n";
            T stepsize = 1e-4;
            result.x -= stepsize * grad_L_gd.head(N);
            result.y2 -= stepsize * grad_L_gd.tail(l);
            Ax = A * result.x;
            Bx = B * result.x;
        } else {
            result.x += alpha * dx;
            result.y2 += alpha * dy2;
            Ax += alpha * Adx;
            Bx += alpha * Bdx;
        }

        // Compute gradient of Lagrangian at current (x, y2)
        Vec grad_L = compute_grad_Lagrangian(result.x, result.y2, Ax, Bx);
        result.tol_achieved = inf_norm(grad_L);

        result.iter++;
        auto t1_ssn = std::chrono::steady_clock::now();
        double timer_ssn = time_diff_ms(t0_ssn, t1_ssn);
        // std::cout << "  SSN iteration took " << timer_ssn << " ms.\n";
        
        // Check termination criterion
        if (result.tol_achieved < eps) {
            result.opt = 0; // Optimality achieved
            break;
        }
    }

    if (result.opt == -1) {
        result.opt = 2; // Maximum number of SSN inner iterations reached without convergence
    }

    return result;
}