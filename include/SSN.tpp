#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

template <typename T>
typename SSN<T>::Vec SSN<T>::proj(const Vec& u, const Vec& lower, const Vec& upper) {
    return u.cwiseMax(lower).cwiseMin(upper);
}

template <typename T>
typename SSN<T>::Vec SSN<T>::compute_dist_box(const Vec& v, const Vec& lower, const Vec& upper) {
    using Vec = typename SSN<T>::Vec;

    Vec dist = v - proj(v, lower, upper);
    return dist;
}

template <typename T>
T SSN<T>::compute_Lagrangian(const Vec& x_new, const Vec& y2_new) {
    using Vec = typename SSN<T>::Vec;

    // Evalueate dist_K(z/mu + x_new)
    Vec dist_K = compute_dist_box(z / mu + x_new, lx, ux);

    // Evaluate dist_W(B*x_new - (y2 - y2_new/2)/mu)
    Vec dist_W = compute_dist_box(B * x_new + (0.5 * y2_new - y2) / mu, lw, uw);

    // Evaluate primal residual A x_new - b
    Vec pr_res = A * x_new - b;

    // Compute Lagrangian
    T L = c.dot(x_new) + 0.5 * x_new.dot(Q * x_new)
          - y1.dot(pr_res) + (mu / 2) * pr_res.squaredNorm()
          - z.squaredNorm() / (2 * mu) + (mu / 2) * dist_K.squaredNorm()
          + mu * dist_W.squaredNorm() + y2_new.squaredNorm() / (4 * mu) - y2.squaredNorm() / (2 * mu)
          + (x_new - x).squaredNorm() / (2 * rho);

    return L;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new) {
    using Vec = typename SSN<T>::Vec;

    // Evalueate Dist_K (z/mu + x_new)
    Vec dist_K = compute_dist_box(z / mu + x_new, lx, ux);

    // Evaluate Dist_W (B*x_new - (y2 - y2_new/2)/mu)
    Vec dist_W = compute_dist_box(B * x_new + (0.5 * y2_new - y2) / mu, lw, uw);

    // Compute gradient of Lagrangian
    Vec grad_L_x = c + Q * x_new - A_tr * y1 + mu * A_tr * (A * x_new - b)
                   + mu * dist_K
                   + 2 * mu * B_tr * dist_W
                   + (x_new - x) / rho;
    Vec grad_L_y2 = dist_W + y2_new / (2 * mu);

    // Combine gradients
    Vec grad_L(n + l);
    grad_L << grad_L_x, grad_L_y2;

    return grad_L;

}

template <typename T>
typename SSN<T>::Vec SSN<T>::Clarke_subgrad_of_proj(const Vec& u, const Vec& lower, const Vec& upper) {
    using Vec = typename SSN<T>::Vec;
    using BoolArr = typename SSN<T>::BoolArr;

    BoolArr mask = (u.array() > lower.array()) && (u.array() < upper.array());
    Vec grad_proj = mask.cast<T>().matrix();

    return grad_proj;
}

template <typename T>
typename SSN<T>::SpMat SSN<T>::build_diag_matrix(const Vec& diag) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using Triplet = typename SSN<T>::Triplet;

    int t = diag.size();
    SpMat M(t, t);
    std::vector<Triplet> trpl;
    trpl.reserve(t);
    for (int i = 0; i < t; ++i) {
        trpl.emplace_back(i, i, diag(i));
    }
    M.setFromTriplets(trpl.begin(), trpl.end());
    
    return M;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::separate_rows(const Vec& u, const BoolArr& mask) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using Triplet = typename SSN<T>::Triplet;
    using BoolArr = typename SSN<T>::BoolArr;

    int t = mask.count();
    Vec u_separated(u.size());
    int selected_row = 0;
    int unselected_row = 0;
    for (int i = 0; i < u.size(); ++i) {
        if (mask(i)) {
            u_separated(selected_row++) = u(i);
        } else {
            u_separated(t + unselected_row++) = u(i);
        }
    }
    
    return u_separated;
}

template <typename T>
typename SSN<T>::SpMat SSN<T>::separate_rows(const SpMat& M, const BoolArr& mask) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using Triplet = typename SSN<T>::Triplet;
    using BoolArr = typename SSN<T>::BoolArr;

    int t = mask.count();
    Eigen::VectorXi row_map(M.rows());
    int selected_row = 0;
    int unselected_row = 0;
    for (int i = 0; i < M.rows(); ++i) {
        if (mask(i)) {
            row_map(i) = selected_row++;
        } else {
            row_map(i) = t + unselected_row++;
        }
    }
    SpMat M_separated(M.rows(), M.cols());
    std::vector<Triplet> trpl;
    trpl.reserve(M.nonZeros());
    for (int col = 0; col < M.cols(); ++col) {
        for (typename SpMat::InnerIterator it(M, col); it; ++it) {
            trpl.emplace_back(row_map(it.row()), col, it.value());
        }
    }
    M_separated.setFromTriplets(trpl.begin(), trpl.end());
    
    return M_separated;
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

    T grad_desc = grad_L.head(n).dot(dx) + grad_L.tail(l).dot(dy2);

    // Iterate until finding the largest step size satisfying the Armijo-Goldstein condition
    while (true) {

        // Evaluate Lagrangian at u_new = u + alpha * du
        Vec x_new = x_curr + alpha * dx;
        Vec y2_new = y2_curr + alpha * dy2;
        T L_new = compute_Lagrangian(x_new, y2_new);

        if (L_new <= L + beta * alpha * grad_desc) break;

        m += 200;
        alpha = pow(delta, m);

        if (alpha < 1e-3) break; // Lower bound on alpha
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

        // Compute gradient of Lagrangian at current (x, y2)
        Vec grad_L = compute_grad_Lagrangian(result.x, result.y2);
        T grad_L_norm = grad_L.norm();

        // Print current iteration info
        printer(result.SSN_in_iter, 0, 0, result.x, y1, result.y2, z, grad_L_norm);

        // Check termination criterion
        if (grad_L_norm < eps) break;

        // Compute Clarke subgradient of Proj_K(z/mu + x_new)
        Vec u = z / mu + result.x;
        Vec diag_P_K = Clarke_subgrad_of_proj(u, lx, ux);

        // Compute Clarke subgradient of Proj_W(B*x_new + (y2_new/2 - y2)/mu)
        Vec v = B * result.x + (0.5 * result.y2 - y2) / mu;
        Vec diag_P_W = Clarke_subgrad_of_proj(v, lw, uw);

        // Compute dist_K(u) and dist_W(v)
        Vec dist_K_u = compute_dist_box(u, lx, ux);
        Vec dist_W_v = compute_dist_box(v, lw, uw);

        // Compute active and inactive sets for (I - P_W)(v)
        BoolArr active_W = (diag_P_W.array() == 0);
        BoolArr inactive_W = (diag_P_W.array() == 1);
        int n_active_W = active_W.count();
        int n_inactive_W = l - n_active_W;

        // Build of Clarke subgradient matrix J_tilde = [-H_tilde G^T; -G D]:

        // H_tilde = diag(Q) + mu(I_n - P_K) + I_n / rho
        Vec H_tilde_diag = Q_diag + mu * (ones_n - diag_P_K) + ones_n / rho;
        Vec H_tilde_diag_inv = H_tilde_diag.cwiseInverse();
        SpMat H_tilde_inv = build_diag_matrix(H_tilde_diag_inv);

        // Active and inactive parts of B w.r.t. W = [lw, uw]
        SpMat B_sep = separate_rows(B, active_W);
        SpMat B_active_W = B_sep.topRows(n_active_W);
        SpMat B_inactive_W = B_sep.bottomRows(n_inactive_W);

        // G = [A ; B_active_W]
        SpMat G = stack_rows(A, B_active_W);
        SpMat G_tr = G.transpose();

        // D = [I_m / mu, 0 ; 0, (I_m - P_W/2)_active_W / mu]
        SpMat D = build_diag_matrix(Vec::Ones(m + n_active_W) / mu);

        // Compute dy2 in inactive_W:
        // dy2_inactive_W = ((I - P_W/2)^{-1} - mu * dist_W(v) - y2/2)(inactive_W)
        //                = 2 - mu * dist_W(v)(inactive_W) - y2(inactive_W)/2 
        Vec y2_sep = separate_rows(result.y2, active_W);
        Vec y2_active_W = y2_sep.head(n_active_W);
        Vec y2_inactive_W = y2_sep.tail(n_inactive_W);

        Vec dist_W_v_sep = separate_rows(dist_W_v, active_W);
        Vec dist_W_v_active_W = dist_W_v_sep.head(n_active_W);
        Vec dist_W_v_inactive_W = dist_W_v_sep.tail(n_inactive_W);

        Vec dy2_inactive_W = -2 * (mu * dist_W_v_inactive_W + y2_inactive_W / 2);

        // Compute the RHS vector
        Vec r1 = c + Q * result.x + mu * dist_K_u
                 - B_tr * result.y2 - B_inactive_W.transpose() * dy2_inactive_W
                 + (result.x - x) / rho;
        Vec r2(m + n_active_W);
        r2.head(m) = y1 / mu - A * result.x + b;
        r2.tail(n_active_W) = -dist_W_v_active_W - y2_active_W / (2 * mu);

        // Compute the Schur complement of J_tilde
        SpMat Schur_tilde = G * H_tilde_inv * G_tr + D; // Self-adjoint and PD

        // Perform Cholesky factorization on the approximated Schur complement
        // to solve Schur_tilde * dy_ = G * H_tilde_inv * r1 + r2, where dy_ = [dy1; dy2_active].
        Vec rhs = G * H_tilde_inv * r1 + r2;
        Vec dy_ = solve_via_chol(Schur_tilde, rhs);

        // Retrive dx and dy2
        Vec dx = H_tilde_inv * (G_tr * dy_ - r1);
        Vec dy2_active_W = dy_.tail(n_active_W);
        Vec dy2 = retrive_row_order(dy2_active_W, dy2_inactive_W, active_W);

        // Backtracking line search to find a Newton step size alpha
        T alpha = backtracking_line_search(result.x, result.y2, dx, dy2);

        // Udpate x and y2
        result.x += alpha * dx;
        result.y2 += alpha * dy2;
        
        x = result.x;
        y2 = result.y2;
        result.SSN_in_iter++;

    }

    // Final tolerance and printing
    result.SSN_tol_achieved = compute_grad_Lagrangian(result.x, result.y2).norm();
    printer(result.SSN_in_iter, 0, 0, result.x, y1, result.y2, z, result.SSN_tol_achieved);

    return result;
}