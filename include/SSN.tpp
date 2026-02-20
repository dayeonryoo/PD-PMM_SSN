#pragma once
#include <iostream>
#include <vector>
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
void SSN<T>::scale_columns(SpMat& M, const Vec& d) {
    assert(M.cols() == d.size());

    for (int j = 0; j < M.outerSize(); ++j) {
        T scale = d(j);
        for (typename SpMat::InnerIterator it(M, j); it; ++it) {
            it.valueRef() *= scale;
        }
    }
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
typename SSN<T>::Vec SSN<T>::solve_via_chol(const SpMat& M, const Vec& r) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;

    assert(M.rows() == M.cols());
    assert(M.rows() == r.size());
    
    Eigen::SimplicialLLT<SpMat> chol;
    chol.compute(M);
    if (chol.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky factorization failed");
    }

    Vec sol = chol.solve(r);
    if (chol.info() != Eigen::Success) {
        throw std::runtime_error("Solving linear system via Cholesky failed");
    }

    return sol;
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

        m += 50;
        alpha = pow(delta, m);
        if (alpha < 1e-7) break; // Lower bound on alpha 
    }

    return alpha;
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

        // Compute the Schur complement of J (self-adjoint and PD)
        // Schur = G H_inv G^T + D, where D = 1/mu I_{m + n_active_W}
        SpMat GH_inv = G; 
        scale_columns(GH_inv, H_diag_inv);
        SpMat Schur = GH_inv * G_tr; 
        for (int i = 0; i < M + n_active_W; ++i) {
            Schur.coeffRef(i, i) += 1 / mu;
        }
        Schur.makeCompressed();

        // Solve: Schur * dy_ = G * H_inv * r1 + r2, where dy_ = [dy1; dy2_active].
        Vec H_inv_r1 = H_diag_inv.cwiseProduct(r1);
        Vec rhs = G * H_inv_r1 + r2;
        auto t1_chol_prep = std::chrono::steady_clock::now();
        double timer_chol_prep = time_diff_ms(t0_chol_prep, t1_chol_prep);
        // std::cout << "  Prep for Cholesky decomposition took " << timer_chol_prep << " ms.\n";

        // ========== Perform Cholesky decomposition ==========
        auto t0_chol = std::chrono::steady_clock::now(); // TIMER FOR CHOL DECOMP
        Vec dy_ = solve_via_chol(Schur, rhs);
        auto t1_chol = std::chrono::steady_clock::now();
        double timer_chol = time_diff_ms(t0_chol, t1_chol);
        // std::cout << "  Cholesky decomposition took " << timer_chol << " ms.\n";

        // Retrive dx and dy2
        Vec dx = H_diag_inv.cwiseProduct(G_tr * dy_ - r1);
        Vec dy2_active_W = dy_.tail(n_active_W);
        Vec dy2 = retrive_row_order(dy2_active_W, dy2_inactive_W, active_W);

        // ========== Backtracking linesearch ==========
        auto t0_alpha = std::chrono::steady_clock::now();

        // Backtracking linesearch to find a Newton step size alpha
        T alpha;
        alpha = backtracking_line_search(result.x, result.y2, dx, dy2);

        auto t1_alpha = std::chrono::steady_clock::now();
        double timer_alpha = time_diff_ms(t0_alpha, t1_alpha);
        // std::cout << "  Backtracking linesearch took " << timer_alpha << " ms.\n";

        // ========== Update x and y2 ==========
        result.x += alpha * dx;
        result.y2 += alpha * dy2;

        auto t1_ssn = std::chrono::steady_clock::now();
        double timer_ssn = time_diff_ms(t0_ssn, t1_ssn);
        // std::cout << "  SSN iteration took " << timer_ssn << " ms.\n";

    }

    if (result.SSN_opt != 0) {
        result.SSN_opt = 2; // Maximum number of SSN inner iterations reached without convergence
        // Modify x for printing. (This modification is not saved as SSN result.)
        x_sol = printable_x(result.x);
        result.obj_val = get_obj_val(result.x);
        printer(result.SSN_in_iter, result.SSN_opt, result.obj_val, x_sol, y1_sol, result.y2, z_sol, result.SSN_tol_achieved);
    }

    return result;
}