#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <Eigen/SparseCholesky>

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
typename SSN<T>::Vec SSN<T>::compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new) {
    using Vec = typename SSN<T>::Vec;

    // Evalueate Dist_K (z/mu + x_new)
    Vec dist_K = compute_dist_box(z / mu + x_new, lx, ux);

    // Evaluate Dist_W (B*x_new + (y2_new/2 - y2)/mu)
    Vec dist_W = compute_dist_box(B * x_new + ((1 - gamma) * y2_new - y2) / mu, lw, uw);

    // Primal residual: A x_new - b
    Vec res_p = A * x_new - b;

    // Compute gradient of Lagrangian
    Vec grad_L_x;
    if (Q_info == 0) {
        grad_L_x = c - A_tr * y1 + mu * A_tr * res_p
                    + mu * dist_K
                    + (mu / gamma) * B_tr * dist_W
                    + (x_new - x) / rho;
    } else {
        grad_L_x = c + Q_diag.cwiseProduct(x_new) - A_tr * y1 + mu * A_tr * res_p
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
    int i_sel = 0;
    int i_unsel = 0;
    Vec u(mask.size());
    for (int i = 0; i < mask.size(); ++i) {
        if (mask(i)) {
            u(i) = u_sel(i_sel++);
        } else {
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
                                            T mu, T tol, int max_iter, bool update_prec) {
    using Vec = typename SSN<T>::Vec;

    const int s = G.rows();
    const int n = G.cols();

    // Matrix-free Schur operator S = G H_inv G^T + (1/mu) I
    SchurOperator<T> S(G, G_tr, H_diag_inv, mu);

    // rhs = G * H_inv * r1 + r2.
    Vec rhs = G * H_diag_inv.cwiseProduct(r1) + r2;

    cg.setTolerance(tol);
    cg.setMaxIterations(max_iter);
    
    cg.preconditioner().setData(G, G_tr, H_diag, active_K, mu, update_prec);
    cg.compute(S);

    if (cg.preconditioner().info() != Eigen::Success) {
        throw std::runtime_error("Preconditioner setup failed.");
    }

    Vec dy_;
    if (prev_dy_.size() == s) {
        dy_ = cg.solveWithGuess(rhs, prev_dy_);
    } else {
        dy_ = cg.solve(rhs);
    }

    // std::cout << "PCG took " << cg.iterations() << " iterations.\n";

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
typename SSN<T>::Vec SSN<T>::solve_using_schur(const SpMat& G, const SpMat& G_tr, const Vec& H_diag_inv, const Vec& r1, const Vec& r2) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;

    const int s = G.rows();
    const int n = G.cols();

    // Compute the Schur complement of J (self-adjoint and PD)
    // Schur = G H_inv G^T + D, where D = 1/mu I_{m + n_active_W}
    SpMat GH_inv = scale_columns(G, H_diag_inv);
    SpMat Schur = GH_inv * G_tr; 
    for (int i = 0; i < M + n_active_W; ++i) {
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

    // Form K = [-H, G^T; G, 1/mu]
    std::vector<Triplet> trip;
    trip.reserve(N_tot + 2 * G.nonZeros());

    // Top-left block: -H
    for (int i = 0; i < n; ++i) {
        const T val = -H_diag(i);
        if (val != T(0)) trip.emplace_back(i, i, val);
    }
    // Bottom-right block: 1 / mu
    const T mu_inv = T(1) / mu;
    for (int i = 0; i < s; ++i) {
        trip.emplace_back(n + i, n + i, mu_inv);
    }
    // Off-diagonal blocks: G and G^T
    for (int col = 0; col < G.outerSize(); ++col) {
        for (typename SpMat::InnerIterator it(G, col); it; ++it) {
            const int i = it.row();
            const int j = it.col();
            const T val = it.value();

            trip.emplace_back(n + i, j, val);
            trip.emplace_back(j, n + i, val);
        }
    }
    SpMat K(N_tot, N_tot);
    K.setFromTriplets(trip.begin(), trip.end());
    K.makeCompressed();

    // From RHS vector
    Vec rhs(N_tot);
    rhs << r1, r2;

    // Solve using LDLT
    Eigen::SimplicialLDLT<SpMat> ldlt;
    ldlt.compute(K);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("LDLT factorization of the augmented Lagrangian system failed.");
    }
    Vec dxdy_ = ldlt.solve(rhs);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("Solving the augmented Lagrangian system via LDLT failed.");
    }

    return dxdy_; // [dx; dy_]
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
    if (t_opt = ! T(0)) {    
        T tau = t_prev - (phi_prev / (phi_new - phi_prev)) * (t_opt - t_prev);
        return tau;
    } else {
        return T(0); // No crossing
    }
}

template <typename T>
T SSN<T>::exact_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2) {
    /* 
    psi(t) = <∇ M(u + t du), du>,
            = eta t + beta + mu <proj_K (s + t dx), dx> + <dc, proj_W (v + t dv)>,
    where eta  = <(Q + mu A^T A + I_n/rho) dx, dx> + (1-gamma)/mu ||dy2||^2,
          beta = <c + Q x_curr - A^T y1 + mu A^T (A x_curr - b) + (x_curr - x)/rho, dx> + (1 - gamma)/mu <y2, dy2>,
          s = z / mu + x_curr,
          v = B x_curr + ((1-gamma) y2_curr - y2) / mu,
          dv = B dx + (1-gamma)/mu dy2,
          dc = mu/gamma B dx + (1-gamma)/gamma dy2. 
    Compute all breakpoints t and corresponding slope changes of psi, and sort them in increasing order.
    Write psi(t) = p + m(t - t_prev).
    For each breakpoint t, if psi(t) >= 0, return t = t_prev - p / m;
    otherwise, set p = psi(t), t_prev = t and continue.
    */
    using Vec = typename SSN<T>::Vec;

    // Useful vectors
    Vec Ax = A * x_curr;
    Vec Adx = A * dx;
    Vec Bx = B * x_curr;
    Vec Bdx = B * dx;

    Vec s = z / mu + x_curr;
    Vec v = Bx + ((1 - gamma) * y2_curr - y2) / mu;
    Vec dv = Bdx + (1 - gamma) / mu * dy2;
    Vec dc = mu / gamma * Bdx + (1 - gamma) / gamma * dy2;

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
    beta += dx.dot(c - A_tr * y1 + mu * A_tr * (Ax - b) + (x_curr - x) / rho);
    beta += (1 - gamma) / mu * dy2.dot(y2_curr);

    // Breakpoint and slope change of psi when crossing it
    struct Breakpoint {
        T t;
        T slope_change;
    };
    std::vector<Breakpoint> breakpoints;

    // Breakpoints and corresponding slope changes for K
    for (int i = 0; i < N; ++i) {
        T s_i = s(i);
        T dx_i = dx(i);
        if (dx_i == 0) continue;

        T change = mu * dx_i * dx_i;
        T t_l = (lx(i) - s_i) / dx_i;
        T t_u = (ux(i) - s_i) / dx_i;

        if (dx_i > 0) {
            if (t_l > 0) breakpoints.push_back({t_l, -change}); // into K
            if (t_u > 0) breakpoints.push_back({t_u, +change}); // out of K
        } else { // dx_i < 0
            if (t_l > 0) breakpoints.push_back({t_l, +change}); // out of K
            if (t_u > 0) breakpoints.push_back({t_u, -change}); // into K
        }
    }

    // Breakpoints and corresponding slope changes for W
    for (int i = 0; i < l; ++i) {
        T v_i = v(i);
        T dv_i = dv(i);
        if (dv_i == 0) continue;
            
        T change = dc(i) * dv_i;
        T t_l = (lw(i) - v_i) / dv_i;
        T t_u = (uw(i) - v_i) / dv_i;

        if (dv_i > 0) {
            if (t_l > 0) breakpoints.push_back({t_l, -change}); // into W
            if (t_u > 0) breakpoints.push_back({t_u, +change}); // out of W
        } else { // dv_i < 0
            if (t_l > 0) breakpoints.push_back({t_l, +change}); // out of W
            if (t_u > 0) breakpoints.push_back({t_u, -change}); // into W
        }
    }

    // Sort by t
    std::sort(breakpoints.begin(), breakpoints.end(), [](Breakpoint& a, Breakpoint& b){ return a.t < b.t; });

    // Check if psi(0) >= 0
    Vec dist_K_s = compute_dist_box(s, lx, ux);
    Vec dist_W_v = compute_dist_box(v, lw, uw);
    T p = beta;
    p += mu * dist_K_s.dot(dx);
    p += mu / gamma * dist_W_v.dot(Bdx);
    p += (1 - gamma) / gamma * dist_W_v.dot(dy2);
    if (p >= T(0)) return T(0);

    // If psi(0) < 0, check at every breakpoint t.
    T t_prev = T(0);
    T m = eta; // initial slope at t=0
    for (int i = 0; i < N; ++i) {
        if (s(i) < lx(i) || s(i) > ux(i)) {
            m += mu * dx(i) * dx(i);
        }
    }
    for (int i = 0; i < l; ++i) {
        if (v(i) < lw(i) || v(i) > uw(i)) {
            m += dc(i) * dc(i);
        }
    }

    // Check at each breakpoint t
    size_t k = 0;
    while (k < breakpoints.size()) {
        T t = breakpoints[k].t;
        T p_t = p + m * (t - t_prev);
        if (p_t >= 0) return t_prev - p / m;

        // Taking care of idential breakpoints at the same time
        T change_sum = T(0);
        while (k < breakpoints.size() && std::abs(breakpoints[k].t - t) < 1e-6) {
            change_sum += breakpoints[k].slope_change;
            ++k;
        }

        // Cross the breakpoint(s)
        t_prev = t;
        p = p_t;
        m += change_sum;
    }

    // Checking the last breakpoint.
    if (m > T(0)) return t_prev - p / m;
    return T(0); // safeguard

}

template <typename T>
bool SSN<T>::primal_infeas(const Vec& delta_y1, const Vec& delta_y2, const Vec& delta_z, T eps_pinf) {
    /*
    The QP is determined to be primal infeasible if all 2 conditions hold for nonzero [delta_y1, delta_y2, delta_z]:
    1. ||A^T delta_y1 + B^T delta_y2 + delta_z||_inf <= eps_pinf * max{||delta_y1||_inf, ||delta_y2||_inf, ||delta_z||_inf};
    2. b^T delta_y1 + sum_i [uw_i * max(y2_i, 0)] + sum_i [lw_i * min(y2_i, 0)]
                    + sum_i [ux_i * max(z_i, 0)] + sum_i [lx_i * min(z_i, 0)]
       <= eps_pinf * max{||delta_y1||_inf, ||delta_y2||_inf, ||delta_z||_inf}
       for finite lx_i, ux_i, lw_i, uw_i.
    */
    using Vec = typename SSN<T>::Vec;

    T delta_y1_inf = T(0);
    if (M != 0) delta_y1_inf = inf_norm(delta_y1);
    T delta_y2_inf = T(0);
    if (l != 0) delta_y2_inf = inf_norm(delta_y2);
    T delta_z_inf = inf_norm(delta_z);
    T delta_inf = std::max({delta_y1_inf, delta_y2_inf, delta_z_inf});
    if (delta_inf == T(0)) return false;

    Vec lhs1 = delta_z;
    if (M != 0) lhs1 += A_tr * delta_y1;
    if (l != 0) lhs1 += B_tr * delta_y2;
    bool cond1 = inf_norm(lhs1) <= eps_pinf * delta_inf;

    if (!cond1) return false;

    T lhs2 = T(0);
    if (M != 0) lhs2 += b.dot(delta_y1);
    for (int i = 0; i < l; ++i) {
        if (uw(i) < inf) lhs2 += uw(i) * std::max(delta_y2(i), T(0));
        if (lw(i) > -inf) lhs2 += lw(i) * std::min(delta_y2(i), T(0));
    }
    for (int i = 0; i < N; ++i) {
        if (ux(i) < inf) lhs2 += ux(i) * std::max(delta_z(i), T(0));
        if (lx(i) > -inf) lhs2 += lx(i) * std::min(delta_z(i), T(0));
    }
    bool cond2 = lhs2 <= eps_pinf * delta_inf;

    if (!cond2) return false;

    return true;
}

template <typename T>
bool SSN<T>::dual_infeas(const Vec& delta_x, T eps_dinf) {
    /*
    The QP is determined to be dual infeasible if all 5 conditions hold for nonzero delta_x:
    1. ||Q delta_x||_inf <= eps_dinf * ||delta_x||_inf;
    2. c^T delta_x <= -eps_dinf * ||delta_x||_inf;
    3. A delta_x ∈ [-eps_dinf, eps_dinf] * ||delta_x||_inf;
    4. delta_x_i ∈ [-eps_dinf, eps_dinf] * ||delta_x||_inf  for finite bounds on x_i,
       delta_x_i >= -eps_dinf * ||delta_x||_inf  for finite lower bounds on x_i,
       delta_x_i <= eps_dinf  * ||delta_x||_inf  for finite upper bounds on x_i;
    5. (B delta_x)_i ∈ [-eps_dinf, eps_dinf]  * ||delta_x||_inf for finite bounds on (Bx)_i,
       (B delta_x)_i >= -eps_dinf * ||delta_x||_inf for finite lower bounds on (Bx)_i,
       (B delta_x)_i <= eps_dinf  * ||delta_x||_inf for finite upper bounds on (Bx)_i.
    */
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;

    const T delta_x_inf = inf_norm(delta_x);
    if (delta_x_inf == T(0)) return false;
    const T rhs = eps_dinf * delta_x_inf;
    
    bool cond1 = true;
    if (Q_info != 0) { // Nontrivial Q
        cond1 = inf_norm(Q_diag.cwiseProduct(delta_x)) <= rhs;
    }
    if (!cond1) return false;

    bool cond2 = c.dot(delta_x) <= -rhs;
    if (!cond2) return false;

    bool cond3 = true;
    if (M != 0) {
        cond3 = inf_norm(A * delta_x) <= rhs;
    }
    if (!cond3) return false;

    for (int i = 0; i < N; ++i) {
        if (std::abs(delta_x(i)) > rhs) return false;
        if (lx(i) > -inf && delta_x(i) < -rhs) return false;
        if (ux(i) < inf && delta_x(i) > rhs) return false;
    }

    for (int i = 0; i < l; ++i) {
        T B_delta_x_i = T(0);
        for (typename SpMat::InnerIterator it(B_tr, i); it; ++it) {
            B_delta_x_i += it.value() * delta_x[it.row()];
        }
        if (std::abs(B_delta_x_i) > rhs) return false;
        if (lw(i) > -inf && B_delta_x_i < -rhs) return false;
        if (uw(i) < inf && B_delta_x_i > rhs) return false;
    }

    return true;
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
        Vec v = B * result.x + ((1 - gamma) * result.y2 - y2) / mu;
        Vec new_diag_P_W = Clarke_subgrad_of_proj(v, lw, uw, true);

        // Compute dist_K(u) and dist_W(v)
        Vec dist_K_u = compute_dist_box(u, lx, ux);
        Vec dist_W_v = compute_dist_box(v, lw, uw);

        // If P_K and P_W are unchanged, reuse the preconditioner.
        bool update_prec = false;

        // Compare the new P_K to the previous P_K
        if (!is_P_unchanged(diag_P_K, new_diag_P_K)) {
            // If P_K is unchanged, reconstruct H_diag, H_diag_inv and the preconditioner for CG.
            // Otherwise, reuse them from the previous iteration.
            update_prec = true;

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
            // If P_W is changed, reconstruct:
            // active_W, inactive_W, H_diag, H_diag_inv, B_active_W, B_inactive_W, G, G_tr.
            // Otherwise, reuse them from the previous iteration.
            update_prec = true;

            diag_P_W = new_diag_P_W;
            active_W = (diag_P_W.array() == 0);
            inactive_W = (diag_P_W.array() == 1);
            n_active_W = active_W.count();
            n_inactive_W = l - n_active_W;

            // Active and inactive parts of B w.r.t. W = [lw, uw]
            build_B_active_inactive(B, active_W, B_active_W, B_inactive_W);

            // G = [A ; B_active_W]
            G = stack_rows(A, B_active_W);
            G_tr = G.transpose();
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
        r2.head(M) = y1 / mu - A * result.x + b;
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
            dxdy_ = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, r1, r2, mu, Krylov_tol, Krylov_max_in_iter, update_prec);
        }
        auto t1_solve_lin_sys = std::chrono::steady_clock::now();
        double timer_solve_line_sys = time_diff_ms(t0_solve_lin_sys, t1_solve_lin_sys);
        // std::cout << "  Solving SSN system took " << timer_solve_line_sys << " ms.\n";

        Vec dx = dxdy_.head(N);
        Vec dy2_active_W = dxdy_.tail(n_active_W);
        Vec dy2 = retrive_row_order(dy2_active_W, dy2_inactive_W, active_W);

        // ========== Backtracking/exact linesearch ==========
        auto t0_alpha = std::chrono::steady_clock::now();
        T alpha;
        alpha = exact_line_search(result.x, result.y2, dx, dy2);
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

            // Check infeasibility
            bool dinf = dual_infeas(delta_x, eps_dinf);
            bool pinf = primal_infeas(delta_y1, delta_y2, delta_z, eps_pinf);
            if (dinf) { result.opt = -3; std::cout << "Dual infeasible.\n"; break; } // Dual infeasible
            if (pinf) { result.opt = -2; std::cout << "Primal infeasible.\n"; break; } // Primal infeasible
        }

        // Compute gradient of Lagrangian at current (x, y2)
        Vec grad_L = compute_grad_Lagrangian(result.x, result.y2);
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