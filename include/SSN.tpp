#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <Eigen/SparseCholesky>

template <typename T>
T SSN<T>::get_obj_val(const Vec& x) {
    T obj_val = obj_const + c.dot(x);
    if (Q_info != 0) {
        obj_val += T(0.5) * Q_diag.cwiseProduct(x).dot(x);
    }
    return obj_val;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::printable_x(const Vec& x) {
    using Vec = typename SSN<T>::Vec;
    Vec x_sol;
    if (Q_info == 2) {
        x_sol = x.head(n).array() * D2_diag.array();
    } else {
        x_sol = x.cwiseProduct(D2_diag);
    }
    return x_sol;
}

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
typename SSN<T>::Vec SSN<T>::solve_using_cg(const SpMat& G, const SpMat& G_tr, const Vec& H_diag_inv, const Vec& r1, const Vec& r2,
                                            const T mu, const T tol, const int max_iter) {
    using Vec = typename SSN<T>::Vec;

    const int s = G.rows();
    const int n = G.cols();

    // Build Schur operator
    // SchurOperator<T> S(G, G_tr, H_diag_inv, mu);

    // Compute the rhs = G * H_inv * r1 + r2.
    SpMat G_H_inv = G;
    scale_columns(G_H_inv, H_diag_inv);
    Vec rhs = G_H_inv * r1 + r2;

    SpMat S = G_H_inv * G_tr; 
    for (int i = 0; i < s; ++i) {
        S.coeffRef(i, i) += 1 / mu;
    }
    S.makeCompressed();

    // Solve S dxdy_ = rhs using conjugate gradient
    Eigen::ConjugateGradient<
        // SchurOperator<T>,
        SpMat,
        Eigen::Lower | Eigen::Upper,
        Eigen::IncompleteCholesky<T>> cg;

    cg.setTolerance(tol);
    cg.setMaxIterations(max_iter);

    cg.compute(S);
    Vec dy_ = cg.solve(rhs);

    std::cout << "  CG took " << cg.iterations() << " iterations.\n";

    if (cg.info() != Eigen::Success) {
        throw std::runtime_error("CG failed to converge.");
    }

    // Retrive dx
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

    // Form M = [-H, G^T; G, 1/mu]
    std::vector<Triplet> trip;
    trip.reserve(N_tot + 2 * G.nonZeros());

    // Top-left block: -H_inv
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
        
        if (alpha < 1e-7) { // Lower bound on alpha
            std::cout << "  SSN: Backtracking linesearch failed.\n";
            alpha = T(0);
            break;
        }
    }
    return alpha;
}

template <typename T>
T SSN<T>::exact_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2) {
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

    T t_opt; // Optimal breakpoint
    Vec x_new, y2_new;
    T phi_new;
    for (T t : breakpoints) {
        x_new = x_curr + t * dx;
        y2_new = y2_curr + t * dy2;
        grad = compute_grad_Lagrangian(x_new, y2_new);
        phi_new = grad.head(N).dot(dx) + grad.tail(l).dot(dy2);
        if (phi_new >= 0) { t_opt = t; break; }
        else { t_prev = t; x_prev = x_new; y2_prev = y2_new; phi_prev = phi_prev; }
    }

    // Compute the optimal stepsize in terms of the optimal breakpoint.
    T tau = t_prev - (phi_prev / (phi_new - phi_prev)) * (t_opt - t_prev);
    return tau;
}

template <typename T>
SSN_result<T> SSN<T>::solve_SSN(const T eps) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using BoolArr = typename SSN<T>::BoolArr;
    using Triplet =typename SSN<T>::Triplet;

    // Intialize iteration counter and set starting points
    SSN_result<T> result;
    result.SSN_in_iter = 0;
    result.x = x;
    result.y2 = y2;
    result.SSN_opt = -1;

    // Initialize printing
    auto printer = make_print_function<T, Vec>(SSN_print_label, SSN_print_when, SSN_print_what, SSN_max_in_iter);

    // SSN main loop
    while (result.SSN_in_iter < SSN_max_in_iter) {
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

        // Compute gradient of Lagrangian at current (x, y2)
        Vec grad_L = compute_grad_Lagrangian(result.x, result.y2);
        result.SSN_tol_achieved = grad_L.norm();

        // Ruiz-descale x and shrink x to the original dimension (n, m, l) for printing
        x_sol = printable_x(result.x);

        // Compute objective value for printing
        result.obj_val = get_obj_val(result.x);

        // Check termination criterion
        if (result.SSN_tol_achieved < eps) {
            result.SSN_opt = 0; // Optimality achieved
            printer(result.SSN_in_iter, result.SSN_opt, result.obj_val, x_sol, y1_sol, result.y2, z_sol, result.SSN_tol_achieved);
            break;
        }

        // Print current iteration info
        printer(result.SSN_in_iter, result.SSN_opt, result.obj_val, x_sol, y1_sol, result.y2, z_sol, result.SSN_tol_achieved);

        result.SSN_in_iter++;

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

        // Compare the new P_K to the previous P_K
        if (!is_P_unchanged(diag_P_K, new_diag_P_K)) {
            // If P_K is unchanged, reconstruct H_diag, H_diag_inv.
            // Otherwise, reuse them from the previous iteration.
            diag_P_K = new_diag_P_K;

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
            diag_P_W = new_diag_P_W;

            // Active and inactive sets for (P_W)(v)
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
        Vec dxdy_ = solve_using_cg(G, G_tr, H_diag_inv, r1, r2, mu, Krylov_tol, Krylov_max_in_iter);
        // Vec dxdy_ = solve_using_schur(G, G_tr, H_diag_inv, r1, r2);
        // if (more_rows_than_cols) {
        //     dxdy_ = solve_using_LDLT(G, H_diag, r1, r2);
        // } else {
        //     dxdy_ = solve_using_schur(G, G_tr, H_diag_inv, r1, r2);
        // }
        auto t1_solve_lin_sys = std::chrono::steady_clock::now();
        double timer_solve_line_sys = time_diff_ms(t0_solve_lin_sys, t1_solve_lin_sys);
        // std::cout << "  Solving SSN system took " << timer_solve_line_sys << " ms.\n";

        Vec dx = dxdy_.head(N);
        Vec dy2_active_W = dxdy_.tail(n_active_W);
        Vec dy2 = retrive_row_order(dy2_active_W, dy2_inactive_W, active_W);

        
        // ========== Backtracking linesearch ==========
        auto t0_alpha = std::chrono::steady_clock::now();

        // Backtracking linesearch to find a Newton step size alpha
        T alpha = backtracking_line_search(result.x, result.y2, dx, dy2);
        // if (result.SSN_in_iter == 1) alpha = 0.995;
        // else alpha = backtracking_line_search(result.x, result.y2, dx, dy2);

        auto t1_alpha = std::chrono::steady_clock::now();
        double timer_alpha = time_diff_ms(t0_alpha, t1_alpha);
        // std::cout << "  Backtracking linesearch took " << timer_alpha << " ms.\n";
        

        // ========== Exact linesearch ==========
        // auto t0_alpha = std::chrono::steady_clock::now();
        // T alpha = exact_line_search(result.x, result.y2, dx, dy2);
        // auto t1_alpha = std::chrono::steady_clock::now();
        // double timer_alpha = time_diff_ms(t0_alpha, t1_alpha);
        // std::cout << "  Exact linesearch took " << timer_alpha << " ms.\n";
        // std::cout << "  alpha = " << alpha << "\n";

        // ========== Update x and y2 ==========
        if (alpha == 0) { // If linesearch fails,
            // Option 1. Use gradient descent to update x and y2.
            // std::cout << "GD applied: ||grad_x|| = " << grad_L.head(N).norm() << "||grad_y2|| = " << grad_L.tail(l).norm() << "\n";
            // T stepsize = 1e-7;
            // result.x -= stepsize * grad_L.head(N);
            // result.y2 -= stepsize * grad_L.tail(l);

            // Option 2. Terminate and discard; come back with smaller mu and rho.
            result.SSN_opt = 3; // means linesearch failure
            break;

            // Option 3. Just carry on with alpha = 1e-7
            // result.x += 1e-7 * dx;
            // result.y2 += 1e-7 * dy2;
        } else {
            result.x += alpha * dx;
            result.y2 += alpha * dy2;
        }

        auto t1_ssn = std::chrono::steady_clock::now();
        double timer_ssn = time_diff_ms(t0_ssn, t1_ssn);
        // std::cout << "  SSN iteration took " << timer_ssn << " ms.\n";

    }

    if (result.SSN_opt == -1) {
        result.SSN_opt = 2; // Maximum number of SSN inner iterations reached without convergence
        // Modify x for printing. (This modification is not saved as SSN result.)
        x_sol = printable_x(result.x);
        result.obj_val = get_obj_val(result.x);
        printer(result.SSN_in_iter, result.SSN_opt, result.obj_val, x_sol, y1_sol, result.y2, z_sol, result.SSN_tol_achieved);
    } else if (result.SSN_opt == 3) {
        // Backtracking linesearch failed.
        printer(result.SSN_in_iter, result.SSN_opt, result.obj_val, x_sol, y1_sol, result.y2, z_sol, result.SSN_tol_achieved);
    }

    return result;
}