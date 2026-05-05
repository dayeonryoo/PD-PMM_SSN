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

    // r[i] = nnz counter of column i
    std::vector<long long> r(N);
    const StorageIndex* outer = G.outerIndexPtr();
    for (Index i = 0; i < N; ++i) {
        r[i] = static_cast<long long>(outer[i+1] - outer[i]);
    }

    // r_hat = nnz of densest column i_hat
    Index i_hat = 0;
    long long r_hat = 0;
    for (Index i = 0; i < N; ++i) {
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

    // Matrix-free Schur operator S = G H_inv G^T + (1/mu) I
    SchurOperator<T> S(G, G_tr, H_diag_inv, mu);

    // rhs = G * H_inv * r1 + r2.
    Vec rhs = G * H_diag_inv.cwiseProduct(r1) + r2;

    cg.setTolerance(tol);
    cg.setMaxIterations(max_iter);
    cg.preconditioner().setData(G, G_tr, H_diag, active_K, mu, update_prec, prec_pattern_changed);
    int prec_fact_before = cg.preconditioner().fact_count();
    cg.compute(S);
    fact += cg.preconditioner().fact_count() - prec_fact_before;

    if (cg.preconditioner().info() != Eigen::Success) {
        throw std::runtime_error("Preconditioner setup failed.");
    }

    Vec dy_;
    if (prev_dy_.size() == s) {
        dy_ = cg.solveWithGuess(rhs, prev_dy_);
    } else {
        dy_ = cg.solve(rhs);
    }

    Krylov_iter += cg.iterations();

    if (cg.info() != Eigen::Success) {
        // ...
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

    Vec rhs = G * H_diag_inv.cwiseProduct(r1) + r2;

    minres.setTolerance(tol);
    minres.setMaxIterations(max_iter);
    minres.preconditioner().setData(G, G_tr, H_diag, active_K, mu, update_prec, prec_pattern_changed);
    int prec_fact_before = minres.preconditioner().fact_count();
    minres.compute(S);
    fact += minres.preconditioner().fact_count() - prec_fact_before;

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
        // ...
    }

    prev_dy_ = dy_;

    Vec dx = H_diag_inv.cwiseProduct(G_tr * dy_ - r1);

    Vec dxdy_(n + s);
    dxdy_.head(n) = dx;
    dxdy_.tail(s) = dy_;
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

    if (ldlt_pattern_dirty_ || ldlt_numeric_dirty_) {
        // Form K = [-H, G^T; G, (1/mu) I]
        std::vector<Triplet> trip;
        trip.reserve(N_tot + 2 * G.nonZeros());

        for (int i = 0; i < n; ++i) {
            const T val = -H_diag(i);
            if (val != T(0)) trip.emplace_back(i, i, val);
        }
        const T mu_inv = T(1) / mu;
        for (int i = 0; i < s; ++i)
            trip.emplace_back(n + i, n + i, mu_inv);
        for (int col = 0; col < G.outerSize(); ++col)
            for (typename SpMat::InnerIterator it(G, col); it; ++it) {
                trip.emplace_back(n + it.row(), it.col(), it.value());
                trip.emplace_back(it.col(), n + it.row(), it.value());
            }

        SpMat K(N_tot, N_tot);
        K.setFromTriplets(trip.begin(), trip.end());
        K.makeCompressed();

        if (ldlt_pattern_dirty_) {
            ldlt_.analyzePattern(K);
            ldlt_pattern_dirty_ = false;
        }
        ldlt_.factorize(K);
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
T SSN<T>::backtracking_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2) {
    using Vec = typename SSN<T>::Vec;

    // Increase m until alpha = delta^m breaks the Armijo-Goldstein condition
    T alpha = delta;
    int m = 1;

    // Evaluate Lagrangian and its gradient at current u = [x; y]
    T L = compute_Lagrangian(x_curr, y2_curr);
    Vec grad_L = compute_grad_Lagrangian(x_curr, y2_curr);

    T grad_desc = grad_L.head(N).dot(dx) + grad_L.tail(l).dot(dy2);

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
T SSN<T>::exact_line_search_w_Lag(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2) {
    using Vec = typename SSN<T>::Vec;

    std::vector<T> breakpoints; // List of breakpoints
    T t; // Breakpoint

    // Breakpoints for K
    for (int i = 0; i < N; ++i) {
        T temp = z(i) / mu + x_curr(i);
        if (dx(i) != 0) {
            t = (ux(i) - temp) / dx(i);
            if (t > 0) breakpoints.push_back(t);

            t = (lx(i) - temp) / dx(i);
            if (t > 0) breakpoints.push_back(t);
        }
    }

    // Breakpoints for W
    Vec Bx = B * x_curr;
    Vec Bdx = B * dx;
    for (int i = 0; i < l; ++i) {
        T ds_i = Bdx(i) + (1 - gamma) * dy2(i) / mu;
        if (ds_i != 0) {
            T s_i = Bx(i) + ((1 - gamma) * y2_curr(i) - y2(i)) / mu;

            t = (uw(i) - s_i) / ds_i;
            if (t > 0) breakpoints.push_back(t);

            t = (lw(i) - s_i) / ds_i;
            if (t > 0) breakpoints.push_back(t);
        }
    }

    // Sort breakpoints in ascending order
    std::sort(breakpoints.begin(), breakpoints.end());

    // Find the smallest breakpoint t which yields grad(u + tdu) >= 0.
    T t_prev = T(0);
    Vec x_prev = x_curr;
    Vec y2_prev = y2_curr;
    Vec grad = compute_grad_Lagrangian(x_curr, y2_curr);
    T phi_prev = grad.head(N).dot(dx) + grad.tail(l).dot(dy2);
    if (phi_prev >= 0) return T(0);

    T t_opt = T(0); // Optimal breakpoint
    Vec x_new, y2_new;
    T phi_new;
    for (T t : breakpoints) {
        x_new = x_curr + t * dx;
        y2_new = y2_curr + t * dy2;
        grad = compute_grad_Lagrangian(x_new, y2_new);
        phi_new = grad.head(N).dot(dx) + grad.tail(l).dot(dy2);
        if (phi_new >= 0) { t_opt = t; break; }
        else { t_prev = t; x_prev = x_new; y2_prev = y2_new; phi_prev = phi_new; }
    }

    // Compute the optimal stepsize in terms of the optimal breakpoint.
    if (t_opt != T(0)) {    
        T tau = t_prev - (phi_prev / (phi_new - phi_prev)) * (t_opt - t_prev);
        return tau;
    } else {
        return T(0); // No crossing
    }
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
    T eps = 1e-10; // Numerical tolerance for checking zero

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

    // Breakpoints and corresponding slope changes for K
    for (int i = 0; i < N; ++i) {
        T s_i = s(i);
        T dx_i = dx(i);
        if (std::abs(dx_i) < eps) continue;

        T change = mu * dx_i * dx_i;
        T t_l = (lx(i) - s_i) / dx_i;
        T t_u = (ux(i) - s_i) / dx_i;

        if (t_l > eps) {
            if (dx_i > 0) breakpoints.push_back({t_l, -change}); // into K
            if (dx_i < 0) breakpoints.push_back({t_l, +change}); // out of K
        } 
        if (t_u > eps) {
            if (dx_i > 0) breakpoints.push_back({t_u, +change}); // out of K
            if (dx_i < 0) breakpoints.push_back({t_u, -change}); // into K
        }
    }

    // Breakpoints and corresponding slope changes for W
    for (int i = 0; i < l; ++i) {
        T v_i = v(i);
        T dv_i = dv(i);
        if (std::abs(dv_i) < eps) continue;
            
        T change = mu / gamma * dv_i * dv_i;
        T t_l = (lw(i) - v_i) / dv_i;
        T t_u = (uw(i) - v_i) / dv_i;

        if (t_l > eps) {
            if (dv_i > 0) breakpoints.push_back({t_l, -change}); // into W
            if (dv_i < 0) breakpoints.push_back({t_l, +change}); // out of W
        }
        if (t_u > eps) {
            if (dv_i > 0) breakpoints.push_back({t_u, +change}); // out of W
            if (dv_i < 0) breakpoints.push_back({t_u, -change}); // into W
        }
    }

    // Sort by t
    std::sort(breakpoints.begin(), breakpoints.end(), [](Breakpoint& a, Breakpoint& b){ return a.t < b.t; });

    // Group breakpoints with the same t by summing up their slope changes
    std::vector<Breakpoint> unique_breakpoints;
    for (size_t i = 0; i < breakpoints.size();) {
        T t = breakpoints[i].t;
        T slope_change_sum = T(0);
        while (i < breakpoints.size() && std::abs(breakpoints[i].t - t) < eps * std::max<T>(1, std::abs(t))) {
            slope_change_sum += breakpoints[i].slope_change;
            ++i;
        }
        unique_breakpoints.push_back({t, slope_change_sum});
    }

    // Check if psi(0) >= 0
    Vec dist_K_s = compute_dist_box(s, lx, ux);
    Vec dist_W_v = compute_dist_box(v, lw, uw);
    T p = beta;
    p += mu * dist_K_s.dot(dx);
    p += mu / gamma * dist_W_v.dot(Bdx);
    p += (1 - gamma) / gamma * dist_W_v.dot(dy2);
    if (p >= eps * (T(1) + std::abs(beta))) return T(0); // No crossing, linesearch failed.

    // If psi(0) < 0, check at every breakpoint t.
    T t_prev = T(0);
    T m = eta; // initial slope at t=0
    for (int i = 0; i < N; ++i) {
        if (s(i) < lx(i) - eps || s(i) > ux(i) + eps) {
            m += mu * dx(i) * dx(i);
        }
    }
    for (int i = 0; i < l; ++i) {
        if (v(i) < lw(i) - eps || v(i) > uw(i) + eps) {
            m += mu / gamma * dv(i) * dv(i);
        }
    }

    // Check at each breakpoint t
    size_t k = 0;
    for (Breakpoint& bp : unique_breakpoints) {
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
    if (m > -eps * (T(1) + eta)) return t_prev - p / std::max(m, eps * (T(1) + eta));
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

        // Compute Clarke subgradient of Proj_K(z/mu + x_new)
        Vec u = z / mu + result.x;
        Vec new_diag_P_K = Clarke_subgrad_of_proj(u, lx, ux, false);

        // Compute Clarke subgradient of Proj_W(B*x_new + ((1 - gamma)*y2_new - y2)/mu)
        Vec v = Bx + ((1 - gamma) * result.y2 - y2) / mu;
        Vec new_diag_P_W = Clarke_subgrad_of_proj(v, lw, uw, true);

        // Compute dist_K(u) and dist_W(v)
        Vec dist_K_u = compute_dist_box(u, lx, ux);
        Vec dist_W_v = compute_dist_box(v, lw, uw);

        // If P_K and P_W are unchanged, reuse the preconditioner and LDLT factorization.
        bool update_prec = false;
        // prec_pattern_changed: true when P = G E G^T + (1/mu) I may have a new sparsity pattern.
        // Triggered by active_K changes (which alter E's diagonal) or G structure changes (active_W).
        bool prec_pattern_changed = false;

        // Compare the new P_K to the previous P_K
        if (!is_P_unchanged(diag_P_K, new_diag_P_K)) {
            update_prec = true;
            prec_pattern_changed = true; // active_K changes E's nonzero pattern, which changes P's pattern
            ldlt_numeric_dirty_ = true;  // H_diag changes, K values change

            diag_P_K = new_diag_P_K;
            active_K = (diag_P_K.array() == 1);

            // H = Q + mu(I_N - P_K) + I_N / rho
            if (Q_info == 0) {
                H_diag = mu * (ones_N - diag_P_K) + ones_N / rho;
            } else {
                H_diag = Q_diag + mu * (ones_N - diag_P_K) + ones_N / rho;
            }
            H_diag_inv = H_diag.cwiseInverse();
        }

        // Compare the new P_W to the previous P_W
        if (!is_P_unchanged(diag_P_W, new_diag_P_W)) {
            update_prec = true;
            prec_pattern_changed = true; // G = [A; B_active_W] changes structure
            ldlt_pattern_dirty_ = true;  // G changes, KKT system pattern changes
            ldlt_numeric_dirty_ = true;

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
        Vec dxdy_;
        if (more_rows_than_cols) {
            dxdy_ = solve_using_LDLT(G, H_diag, r1, r2);
        } else {
            dxdy_ = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, r1, r2, mu, Krylov_tol, Krylov_max_in_iter, update_prec, prec_pattern_changed);
        }
        // dxdy_ = solve_using_minres(G, G_tr, H_diag, H_diag_inv, active_K, r1, r2, mu, Krylov_tol, Krylov_max_in_iter, update_prec, prec_pattern_changed);
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
        if (alpha == 0) { // If linesearch fails,
            // Option 1. Use gradient descent to update x and y2.
            // std::cout << "GD applied: ||grad_x|| = " << grad_L.head(N).norm() << "||grad_y2|| = " << grad_L.tail(l).norm() << "\n";
            // T stepsize = 1e-7;
            // result.x -= stepsize * grad_L.head(N);
            // result.y2 -= stepsize * grad_L.tail(l);

            // Option 2. Terminate and discard; change from backtracking to exact and come back with smaller mu and rho.
            result.opt = 3; // linesearch failed
            do_exact = true;
            break;

            // Option 3. Just carry on with alpha = 1e-7
            // result.x += 1e-7 * dx;
            // result.y2 += 1e-7 * dy2;
        } else {
            // Newton's steps
            delta_x = alpha * dx;
            delta_y2 = alpha * dy2;

            // Update x and y2
            result.x += delta_x;
            result.y2 += delta_y2;

            // Go back to backtracking
            // do_exact = false;

        }

        // Compute gradient of Lagrangian at current (x, y2)
        Ax += alpha * Adx; // Update Ax for gradient computation
        Bx += alpha * Bdx; // Update Bx for gradient computation
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