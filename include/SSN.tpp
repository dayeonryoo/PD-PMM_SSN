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
typename SSN<T>::Vec SSN<T>::compute_box_proj(const Vec& v, const Vec& lower, const Vec& upper) {
    using Vec = typename SSN<T>::Vec;

    Vec proj = v.cwiseMax(lower).cwiseMin(upper);
    return proj;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::compute_dist_box(const Vec& v, const Vec& lower, const Vec& upper) {
    using Vec = typename SSN<T>::Vec;

    Vec proj = compute_box_proj(v, lower, upper);
    Vec dist = v - proj;
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
    T L = c.dot(x_new) + 0.5 * x_new.transpose() * Q * x_new
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
    SpMat A_tr = A.transpose();
    SpMat B_tr = B.transpose();
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
T SSN<T>::backtracking_line_search(const Vec& x_curr, const Vec& y2_curr, const Vec& dx, const Vec& dy2) {
    using Vec = typename SSN<T>::Vec;

    // Increase m until alpha = delta^m breaks the Armijo-Goldstein condition
    T alpha = delta;
    int m = 1;

    // Evaluate Lagrangian and its gradient at current u = [x; y]
    T L = compute_Lagrangian(x_curr, y2_curr);
    Vec grad_L = compute_grad_Lagrangian(x_curr, y2_curr);

    // Evaluate Lagrangian at u_new = u + alpha * du
    Vec x_new = x_curr + alpha * dx;
    Vec y2_new = y2_curr + alpha * dy2;
    T L_new = compute_Lagrangian(x_new, y2_new);

    // Iterate until finding the largest step size satisfying the Armijo-Goldstein condition
    T grad_desc = grad_L.segment(0, n).dot(dx) + grad_L.segment(n, l).dot(dy2);
    while (L_new > L + beta * alpha * grad_desc) {
        m += 200;
        alpha = pow(delta, m);
        if (alpha < 1e-3) break; // Lower bound on alpha

        // Evaluate Lagrangian at u_new for next iteration
        x_new += alpha * dx;
        y2_new += alpha * dy2;
        L_new = compute_Lagrangian(x_new, y2_new);
    }

    return alpha;
}


template <typename T>
SSN_result<T> SSN<T>::solve_SSN() {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<T>;

    // Intialize iteration counter and set starting points
    SSN_result<T> result;
    result.SSN_in_iter = 0;
    result.x = x;
    result.y2 = y2;

    // SSN main loop
    while (result.SSN_in_iter < SSN_max_in_iter) {
        // ----------------------------------------------
        // Structure:
        // Let M(u), with u = (x,y_2), be the proximal augmented Lagrangian
        // associated with the subproblem of interest.
        // Until (|| \nabla M(u_{k_j}) || < SSN_tol), do:
        //     1) Compute a Clarke subgradient J of \nabla M(u_{k_j})
        //        and solve J du = - \nabla M(u_{k_j}) for the Newton direction du;
        //     2) Perform a backtracking line search to determine the step size alpha;
        //     3) Update the variables;
        //     j = j + 1;
        // End
        // ----------------------------------------------

        // Compute gradient and check termination criteria
        Vec grad_L = compute_grad_Lagrangian(result.x, result.y2);
        T grad_L_norm = grad_L.norm();
        if (grad_L_norm < SSN_tol) break;

        // Compute Clarke subgradient of Proj_K(z/mu + x_new)
        Vec u = z / mu + result.x;
        BoolArr K_mask = (u.array() > lx.array()) && (u.array() < ux.array());
        Vec diag_P_K = K_mask.cast<T>().matrix();

        // Compute Clarke subgradient of Proj_W(B*x_new + (y2_new/2 - y2)/mu)
        Vec v = B * result.x + (0.5 * result.y2 - y2) / mu;
        BoolArr W_mask = (v.array() > lw.array()) && (v.array() < uw.array());
        Vec diag_P_W = W_mask.cast<T>().matrix();

        // Compute dist_K(u) and dist_W(v)
        Vec dist_K_u = compute_dist_box(u, lx, ux);
        Vec dist_W_v = compute_dist_box(v, lw, uw);

        // Compute active and inactive sets for (I - P_W)(v)
        BoolArr active_W = (diag_P_W.array() == 0);
        BoolArr inactive_W = (diag_P_W.array() == 1);
        int n_active_W = active_W.count();
        int n_inactive_W = l - n_active_W;

        // Useful vectors and matrices
        Vec ones_n = Vec::Ones(n);
        Vec ones_l = Vec::Ones(l);
        Vec ones_m = Vec::Ones(m);
        Vec Q_diag = Q.diagonal();
        SpMat A_tr = A.transpose();
        SpMat B_tr = B.transpose();

        // Build of Clarke subgradient matrix J_tilde = [-H_tilde G^T; -G D]:

        // H_tilde = diag(Q) + mu(I_n - P_K) + I_n / rho
        Vec H_tilde_diag = Q_diag + mu * (ones_n - diag_P_K) + ones_n / rho;
        Vec H_tilde_diag_inv = H_tilde_diag.cwiseInverse();
        SpMat H_tilde_inv(n, n);
        std::vector<Triplet> H_tilde_inv_tripl;
        H_tilde_inv_tripl.reserve(n);
        for (int i = 0; i < n; ++i) {
            H_tilde_inv_tripl.emplace_back(i, i, H_tilde_diag_inv(i));
        }
        H_tilde_inv.setFromTriplets(H_tilde_inv_tripl.begin(), H_tilde_inv_tripl.end());

        // Active and inactive parts of B w.r.t. W = [lw, uw]
        Eigen::VectorXi row_map_act(l); // Indices for active rows
        Eigen::VectorXi row_map_inact(l); // Indicices for inactive rows
        int new_row_act = 0;
        int new_row_inact = 0;
        for (int i = 0; i < l; ++i) {
            if (active_W(i)) {
                row_map_act(i) = new_row_act;
                new_row_act++;
            } else {
                row_map_inact(i) = new_row_inact;
                new_row_inact++;
            }
        }
        SpMat B_active_W(n_active_W, n);
        SpMat B_inactive_W(n_inactive_W, n);
        std::vector<Triplet> B_act_trpl;
        std::vector<Triplet> B_inact_trpl;
        B_act_trpl.reserve(B.nonZeros());
        B_inact_trpl.reserve(B.nonZeros());
        for (int i = 0; i < n; ++i) { // Iterate columns
            for (typename SpMat::InnerIterator it(B, i); it; ++it) { // Iterate nnz in col i
                int old_row = it.row();
                if (active_W(old_row)) {
                    int new_row = row_map_act(old_row);
                    B_act_trpl.emplace_back(new_row, i, it.value());
                } else {
                    int new_row = row_map_inact(old_row);
                    B_inact_trpl.emplace_back(new_row, i, it.value());
                }
            }
        }
        B_active_W.setFromTriplets(B_act_trpl.begin(), B_act_trpl.end());
        B_inactive_W.setFromTriplets(B_inact_trpl.begin(), B_inact_trpl.end());

        // G = [A ; B_active_W]
        SpMat G(m + n_active_W, n);
        std::vector<Triplet> G_trpl;
        G_trpl.reserve(A.nonZeros() + B_active_W.nonZeros());
        for (int col = 0; col < n; ++col) {
            for (typename SpMat::InnerIterator it(A, col); it; ++it) {
                G_trpl.emplace_back(it.row(), col, it.value());
            }
        }
        for (int col = 0; col < n; ++col) {
            for (typename SpMat::InnerIterator it(B_active_W, col); it; ++it) {
                G_trpl.emplace_back(m + it.row(), col, it.value());
            }
        }
        G.setFromTriplets(G_trpl.begin(), G_trpl.end());
        SpMat G_tr = G.transpose();

        // D = [I_m / mu, 0 ; 0, (I_m - P_W/2)_active_W / mu]
        SpMat D(m + n_active_W, m + n_active_W);
        std::vector<Triplet> D_trpl;
        D_trpl.reserve(m + n_active_W);
        for (int i = 0; i < m + n_active_W; ++i) {
            D_trpl.emplace_back(i, i, 1 / mu);
        }
        D.setFromTriplets(D_trpl.begin(), D_trpl.end());

        
        // Compute dy2 in inactive_W:
        // dy2_inactive_W = ((I - P_W/2)^{-1} - mu * dist_W(v) - y2/2)(inactive_W)
        //                = 2 - mu * dist_W(v)(inactive_W) - y2(inactive_W)/2 
        Vec y2_active_W(n_active_W);
        Vec y2_inactive_W(n_inactive_W);
        Vec dist_W_v_active_W(n_active_W);
        Vec dist_W_v_inactive_W(n_inactive_W);
        int i_act = 0;
        int i_inact = 0;
        for (int i = 0; i < l; ++i) {
            if (active_W(i)) {
                dist_W_v_active_W(i_act) = dist_W_v(i);
                y2_active_W(i_act) = result.y2(i);
                i_act++;
            } else {
                dist_W_v_inactive_W(i_inact) = dist_W_v(i);
                y2_inactive_W(i_inact) = result.y2(i);
                i_inact++;
            }
        }
        Vec dy2_inactive_W = 2 * Vec::Ones(n_inactive_W) - mu * dist_W_v_inactive_W - y2_inactive_W / 2;

        // Compute the RHS vector
        Vec r1 = c + Q * result.x + mu * dist_K_u
                 - B_tr * result.y2 - B_inactive_W.transpose() * dy2_inactive_W
                 + (result.x - x) / rho;
        Vec r2(m + n_active_W);
        r2.segment(0, m) = y1 / mu - A * result.x + b;
        r2.segment(m, n_active_W) = -dist_W_v_active_W - y2_active_W / (2 * mu);

        // Compute the Schur complement of J_tilde
        SpMat Schur_tilde = G * H_tilde_inv * G_tr + D; // Self-adjoint and PD

        // Perform Cholesky factorization on the approximated Schur complement
        // to solve Schur_tilde * dy_ = G * r1 + r2, where dy_ = [dy1; dy2_active].
        Vec rhs = G * r1 + r2;
        Eigen::SimplicialLLT<SpMat> chol;
        chol.compute(Schur_tilde);
        if (chol.info() != Eigen::Success) {
            std::cerr << "Cholesky factorization faild\n";
            break;
        }
        Vec dy_ = chol.solve(rhs);
        if (chol.info() != Eigen::Success) {
            std::cerr << "Solving via Cholesky factorization faild\n";
            break;
        }

        // Retrive dx and dy2
        Vec dx = H_tilde_inv * (G_tr * dy_ - r1);
        Vec dy2_active_W = dy_.segment(m, n_active_W);
        Vec dy2(l);
        i_act = 0;
        i_inact = 0;
        for (int i = 0; i < l; ++i) {
            if (active_W(i)) {
                dy2(i) = dy2_active_W(i_act);
                i_act++;
            } else {
                dy2(i) = dy2_inactive_W(i_inact);
                i_inact++;
            }
        }

        // Backtracking line search to find a Newton step size alpha
        T alpha = backtracking_line_search(result.x, result.y2, dx, dy2);

        // Udpate x and y2
        result.x += alpha * dx;
        result.y2 += alpha * dy2;
        
        result.SSN_in_iter++;

        std::cout << "  SSN iter " << result.SSN_in_iter << ": x = (" << result.x.transpose() << ")\n";
    }

    result.SSN_tol_achieved = compute_grad_Lagrangian(result.x, result.y2).norm();
    std::cout << "  SSN tol achieved = " << result.SSN_tol_achieved << std::endl;

    return result;
}