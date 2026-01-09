#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;
using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

// =============================================================
// Helper functions
// --------------------------------------------------------------
Vec compute_residual_norms(const SpMat& Q, const SpMat& A, const SpMat& B,
                           const Vec& c, const Vec& b,
                           const Vec& x, const Vec& y1, const Vec& y2, const Vec& z,
                           const Vec& lx, const Vec& ux, const Vec& lw, const Vec& uw) {
    T res_p = (A * x - b).norm() / (1 + b.norm());
    T res_d = (c + Q * x - A.transpose() * y1 - B.transpose() * y2 + z).norm() / (1 + c.norm());
    Vec proj_K = (x + z).cwiseMax(lx).cwiseMin(ux);
    T compl_x = (x - proj_K).norm();
    Vec proj_W = (B * x - y2).cwiseMax(lw).cwiseMin(uw);
    T compl_w = (B * x - proj_W).norm();

    Vec res_norms(4);
    res_norms << res_p, res_d, compl_x, compl_w;

    return res_norms;

}

Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
    return u.cwiseMax(lower).cwiseMin(upper);
}

T compute_Lagrangian(const SpMat& Q, const SpMat& A, const SpMat& B,
                     const Vec& c, const Vec& b,
                     const Vec& x, const Vec& y1, const Vec& y2, const Vec& z,
                     const Vec& lx, const Vec& ux, const Vec& lw, const Vec& uw,
                     const Vec& x_new, const Vec& y2_new, const T mu, const T rho) {
    Vec v1 = z / mu + x_new;
    Vec dist_K = v1 - proj(v1, lx, ux);
    Vec v2 = B * x_new + (0.5 * y2_new - y2) / mu;
    Vec dist_W = v2 - proj(v2, lw, uw);
    Vec res_p = A * x_new - b;

    T L = c.dot(x_new) + 0.5 * x_new.dot(Q * x_new)
          - y1.dot(res_p) + (mu / 2) * res_p.squaredNorm()
          - z.squaredNorm() / (2 * mu) + (mu / 2) * dist_K.squaredNorm()
          + mu * dist_W.squaredNorm() + y2_new.squaredNorm() / (4 * mu) - y2.squaredNorm() / (2 * mu)
          + (x_new - x).squaredNorm() / (2 * rho); 

    return L;

}

Vec compute_grad_Lagrangian(const SpMat& Q, const SpMat& A, const SpMat& B,
                            const Vec& c, const Vec& b,
                            const Vec& x, const Vec& y1, const Vec& y2, const Vec& z,
                            const Vec& lx, const Vec& ux, const Vec& lw, const Vec& uw,
                            const Vec& x_new, const Vec& y2_new, const T mu, const T rho) {
    Vec v1 = z / mu + x_new;
    Vec dist_K = v1 - proj(v1, lx, ux);

    Vec v2 = B * x_new + (0.5 * y2_new - y2) / mu;
    Vec dist_W = v2 - proj(v2, lw, uw);

    SpMat A_tr = A.transpose();
    SpMat B_tr = B.transpose();

    Vec grad_x = c + Q * x_new - A_tr * y1 + mu * A_tr * (A * x_new - b)
                 + mu * dist_K
                 + 2 * mu * B_tr * dist_W
                 + (x_new - x) / rho;
    Vec grad_y2 = dist_W + y2_new / (2 * mu);

    Vec grad_L(x.size() + y2.size());
    grad_L << grad_x, grad_y2;

    return grad_L;
    
}

T backtracking_line_search(const SpMat& Q, const SpMat& A, const SpMat& B,
                           const Vec& c, const Vec& b,
                           const Vec& x, const Vec& y1, const Vec& y2, const Vec& z,
                           const Vec& lx, const Vec& ux, const Vec& lw, const Vec& uw,
                           const Vec& x_curr, const Vec& y2_curr, const T mu, const T rho,
                           const Vec& dx, const Vec& dy2, const T beta, const T delta) {
    // Initialize the Newton step and its power.
    T alpha = delta;
    int m = 1;

    // Lagrangian and its gradient at current x and y2
    T L = compute_Lagrangian(Q, A, B, c, b,
                             x, y1, y2, z,
                             lx, ux, lw, uw,
                             x_curr, y2_curr, mu, rho);
    Vec grad_L = compute_grad_Lagrangian(Q, A, B, c, b,
                                         x, y1, y2, z,
                                         lx, ux, lw, uw,
                                         x_curr, y2_curr, mu, rho);

    // Take a Newton step with alpha
    Vec x_new = x_curr + alpha * dx;
    Vec y2_new = y2_curr + alpha * dy2;

    // Compute Lagrangian at new x and y2
    T L_new = compute_Lagrangian(Q, A, B, c, b,
                                 x, y1, y2, z,
                                 lx, ux, lw, uw,
                                 x_new, y2_new, mu, rho);

    // Iterate until finding the largest step size alpha satisfying the Armijo-Goldstein condition
    T grad_desc = grad_L.segment(0, x.size()).dot(dx) + grad_L.segment(x.size(), y2.size()).dot(dy2);
    while (L_new > L + beta * alpha * grad_desc) {
        m += 200;
        alpha = pow(delta, m);
        if (alpha < 1e-3) break; // Lower bound on alpha

        // Update x_new and y2_new and evaluate the new Lagrangian for next iteration
        x_new += alpha * dx;
        y2_new += alpha * dy2;
        L_new = compute_Lagrangian(Q, A, B, c, b,
                                   x, y1, y2, z,
                                   lx, ux, lw, uw,
                                   x_new, y2_new, mu, rho);
    }

    return alpha;
}

void update_PMM_params(const T res_p, const T res_d,
                       const T new_res_p, const T new_res_d,
                       T& mu, T& rho, const T reg_limit) {
    bool cond_p = 0.95 * res_p > new_res_p;
    bool cond_d = 0.905 * res_d > new_res_d;

    if (cond_p || cond_d){
        mu = std::min(reg_limit, 1.2 * mu);
        rho = std::min(1e2 * reg_limit, 1.4 * rho);
    } else {
        mu = std::min(reg_limit, 1.1 * mu);
        rho = std::min(1e2 * reg_limit, 1.1 * rho);
    };
}

// =============================================================

int main() {

// =============================================================
//      min  c^T x + (1/2) x^T Q x,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw
// --------------------------------------------------------------
//      c = [1; 2], Q = 0, A = [1; 1], b = 0, B = I_2,
//      lx = [0; 0], ux = [1; 1], lw = [0; 0], uw = [1; 1]
// --------------------------------------------------------------
//      Expected solution: x = [0; 0]
//      Expected objective value: f(x) = 0
// =============================================================

    // Define problem data
    const int n = 2;
    const int m = 1;
    const int l = 2;

    // Q = 0
    SpMat Q(n, n);
    Q.setZero();

    // A = [1, 1]
    SpMat A(m, n);
    std::vector<Triplet> A_trpl;
    A_trpl.emplace_back(0, 0, 1.0);
    A_trpl.emplace_back(0, 1, 1.0);
    A.setFromTriplets(A_trpl.begin(), A_trpl.end());
    
    // B = I_2
    SpMat B(l, n);
    std::vector<Triplet> B_trpl;
    B_trpl.emplace_back(0, 0, 1.0);
    B_trpl.emplace_back(1, 1, 1.0);
    B.setFromTriplets(B_trpl.begin(), B_trpl.end());

    // c = [1; 2], b = 0
    Vec c(n);
    c << 1.0, 2.0;
    Vec b(m);
    b << 0.0;

    // 0 <= x, w <= 1
    Vec lx(n), ux(n);
    lx.setZero();
    ux.setOnes();
    Vec lw(l), uw(l);
    lw.setZero();
    uw.setOnes();

    T tol = 0.1;
    int max_iter = 5;

    T SSN_tol = tol;
    int SSN_max_iter = 100; // max total SSN iter
    int SSN_max_in_iter = 3; // max SSN iter per PMM iter

    // Initialize variables for SSN_PMM
    int PMM_iter = 0;
    int opt = -1;
    Vec x = Vec::Zero(n);
    Vec y1 = Vec::Zero(m);
    Vec y2 = Vec::Zero(l);
    Vec z = Vec::Zero(n);
    T mu = 5e1;
    T rho = 1e2;

    // Useful vectors and matrices
    Vec ones_n = Vec::Ones(n);
    Vec ones_l = Vec::Ones(l);
    Vec ones_m = Vec::Ones(m);
    Vec Q_diag = Q.diagonal();
    SpMat A_tr = A.transpose();
    SpMat B_tr = B.transpose();

    // PMM main loop
    while (PMM_iter < max_iter) {

        // Compute residuals.
        Vec res_norms = compute_residual_norms(Q, A, B, c, b,
                                               x, y1, y2, z,
                                               lx, ux, lw, uw);
        T res_p = res_norms(0);
        T res_d = res_norms(1); // Needed to update PMM params

        // Check termination criteria.
        T max_res_norm = res_norms.maxCoeff();
        if (max_res_norm < tol) {
            opt = 0;
            break;
        }

        PMM_iter++;
        std::cout << "PMM iter " << PMM_iter << ": res norms = (" << res_norms.transpose() << ")\n";

        // Initialize SSN variables.
        int SSN_in_iter = 0;
        Vec SSN_x = x;
        Vec SSN_y2 = y2;
        T beta = 0.4995 / 2;
        T delta = 0.995;
        T eta = 0.1 * SSN_tol;
        T gamma = 0.1;
        T reg_limit = 1e6;

        // SSN main loop
        while (SSN_in_iter < SSN_max_in_iter) {

            // Compute gradient and its norm
            Vec grad_L = compute_grad_Lagrangian(Q, A, B, c, b,
                                                 x, y1, y2, z,
                                                 lx, ux, lw, ux,
                                                 SSN_x, SSN_y2, mu, rho);
            T grad_L_norm = grad_L.norm();
            std::cout << "  SSN iter " << SSN_in_iter << ": grad_norm = " << grad_L_norm << "\n";

            // Check termination criteria
            if (grad_L_norm < SSN_tol) break;

            // Compute Clarke subgradient of Proj_K(z/mu + x_new)
            Vec v1 = z / mu + SSN_x;
            BoolArr K_mask = (v1.array() > lx.array()) && (v1.array() < ux.array());
            Vec diag_P_K = K_mask.cast<T>().matrix();

            // Compute Clarke subgradient of Proj_W(B*x_new + (y2_new/2 - y2)/mu)
            Vec v2 = B * SSN_x + (0.5 * SSN_y2 - y2) / mu;
            BoolArr W_mask = (v2.array() > lw.array()) && (v2.array() < uw.array());
            Vec diag_P_W = W_mask.cast<T>().matrix();

            std::cout << "              diag(P_K) = (" << diag_P_K.transpose() << ")\n";
            std::cout << "              diag(P_W) = (" << diag_P_W.transpose() << ")\n";
            
            // Compute active and inactive sets for P_W
            BoolArr active_W = (diag_P_W.array() == 0);
            BoolArr inactive_W = (diag_P_W.array() == 1);
            int n_active_W = active_W.count();
            int n_inactive_W = inactive_W.count();
            std::cout << "              # active indices = " << n_active_W << "\n";

            // Compute H_tilde = diag(Q) + mu(I_n - P_K) + I_n/rho
            Vec H_tilde_diag = Q_diag + mu * (ones_n - diag_P_K) + ones_n / rho;
            Vec H_tilde_diag_inv = H_tilde_diag.cwiseInverse();
            SpMat H_tilde_inv(n, n);
            std::vector<Triplet> H_tilde_inv_tripl;
            H_tilde_inv_tripl.reserve(n);
            for (int i = 0; i < n; ++i) {
                H_tilde_inv_tripl.emplace_back(i, i, H_tilde_diag_inv(i));
            }
            H_tilde_inv.setFromTriplets(H_tilde_inv_tripl.begin(), H_tilde_inv_tripl.end());

            std::cout << "              diag(H_tilde) = (" << H_tilde_diag.transpose() << ")\n";
            std::cout << "              diag(H_tilde_inv) = (" << H_tilde_diag_inv.transpose() << ")\n";

            // Construct B_active_W and B_inactive_W
            Eigen::VectorXi row_map_act(l); // Indices for active rows
            Eigen::VectorXi row_map_inact(l); // Indicices for inactive rows
            row_map_act.setConstant(-1); // Inactive rows stay at -1
            row_map_inact.setConstant(-1); // Active rows stay at -1
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
                for (SpMat::InnerIterator it(B, i); it; ++it) { // Iterate nnz in col i
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

            std::cout << "              diag(B_active_W) = (" << B_active_W.diagonal().transpose() << ")\n";
            std::cout << "              diag(B_inactive_W) = (" << B_inactive_W.diagonal().transpose() << ")\n";

            // Construct G = [A; B_active_W]
            SpMat G(m + n_active_W, n);
            std::vector<Triplet> G_trpl;
            G_trpl.reserve(A.nonZeros() + B_active_W.nonZeros());
            for (int i = 0; i < n; ++i) { // Fill elements in A
                for (SpMat::InnerIterator it(A, i); it; ++it) {
                    G_trpl.emplace_back(it.row(), i, it.value());
                }
            }
            for (int i = 0; i < n; ++i) { // Fill elements in B_active_W
                for (SpMat::InnerIterator it(B_active_W, i); it; ++it) {
                    G_trpl.emplace_back(m + it.row(), i, it.value());
                }
            }
            G.setFromTriplets(G_trpl.begin(), G_trpl.end());
            SpMat G_tr = G.transpose();

            // std::cout << "              G = " << Eigen::MatrixXd(G) << "\n";

            // Construct D = [I_m/mu, 0; 0, (I_m - P_K/2)_active_W / mu]
            //             = I_{m + n_active_W} / mu
            SpMat D(m + n_active_W, m + n_active_W);
            std::vector<Triplet> D_trpl;
            D_trpl.reserve(m + n_active_W);
            for (int i = 0; i < m + n_active_W; ++i) {
                D_trpl.emplace_back(i, i, 1 / mu);
            }
            D.setFromTriplets(D_trpl.begin(), D_trpl.end());

            std::cout << "              diag(D) = (" << D.diagonal().transpose() << ")\n";

            // Separate y2, dist_W(B*x_new + (y2_new/2 - y2)/mu) w.r.t. active_W and inactive_W
            Vec y2_active_W(n_active_W);
            Vec y2_inactive_W(n_inactive_W);
            Vec dist_W = v2 - proj(v2, lw, uw);
            Vec dist_W_active_W(n_active_W);
            Vec dist_W_inactive_W(n_inactive_W);
            int i_act = 0;
            int i_inact = 0;
            for (int i = 0; i < l; ++i) {
                if (active_W(i)) {
                    dist_W_active_W(i_act) = dist_W(i);
                    y2_active_W(i_act) = SSN_y2(i);
                    i_act++;
                } else {
                    dist_W_inactive_W(i_inact) = dist_W(i);
                    y2_inactive_W(i_inact) = SSN_y2(i);
                    i_inact++;
                }
            }

            // Compute dy2_inactive_W = (I - P_W/2)^{-1} - mu * dist_W - y2/2, all in inactive_W
            Vec dy2_inactive_W = 2 * Vec::Ones(n_inactive_W) - mu * dist_W_inactive_W - y2_inactive_W / 2;
            
            // Compute the RHS vector
            Vec dist_K = v1 - proj(v1, lx, ux);
            Vec r1 = c + Q * SSN_x + mu * dist_K - B_tr * SSN_y2 - B_inactive_W.transpose() * dy2_inactive_W + (SSN_x - x) / rho;
            Vec r2(m + n_active_W);
            r2.segment(0, m) = y1 / mu - A * SSN_x + b;
            r2.segment(m, n_active_W) = -dist_W_active_W - y2_active_W / (2 * mu);

            std::cout << "              r1 = (" << r1.transpose() << ")\n";
            std::cout << "              r2 = (" << r2.transpose() << ")\n";

            // Compute the Schur complement; Schur_tilde = G * H_tilde_inv * G_tr + D
            SpMat Schur_tilde = G * H_tilde_inv * G_tr + D; // Self-adjoint and PD

            // Perform Cholesky factorization on the approximated Schur complement
            // to solve Schur_tilde * dy_ = G * r1 + r2.
            Vec rhs = G * r1 + r2;
            Eigen::SimplicialLLT<SpMat> chol;
            chol.compute(Schur_tilde); // Cholesky factorization
            Vec dy_ = chol.solve(rhs); // Solve with Cholesky factorization
            // std::cout << "              dy_ = " << dy_.transpose() << "\n";

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
            std::cout << "              dx = (" << dx.transpose() << ")\n";
            std::cout << "              dy2 = (" << dy2.transpose() << ")\n";

            // Backtracking line search to find a Newton step size alpha
            T alpha = backtracking_line_search(Q, A, B, c, b,
                                               x, y1, y2, z,
                                               lx, ux, lw, uw,
                                               SSN_x, SSN_y2, mu, rho,
                                               dx, dy2, beta, delta);
            std::cout << "              alpha = " << alpha << "\n";

            // Update x and y2;
            SSN_x += alpha * dx;
            SSN_y2 += alpha * dy2;

            SSN_in_iter++;
        }

        // Update x and y2.
        x = SSN_x;
        y2 = SSN_y2;

        // Update multipliers y1 and z.
        y1 -= mu * (A * x - b);
        z += mu * (x - proj(z / mu + x, lx, ux));

        std::cout << "  x = (" << x.transpose() << ")\n";
        std::cout << "  y1 = (" << y1.transpose() << ")\n";
        std::cout << "  y2 = (" << y2.transpose() << ")\n";
        std::cout << "  z = (" << z.transpose() << ")\n";
        
        // Compute the new residual norms
        Vec new_res_norms = compute_residual_norms(Q, A, B, c, b,
                                                   x, y1, y2, z,
                                                   lx, ux, lw, uw);
        T new_res_p = new_res_norms(0);
        T new_res_d = new_res_norms(1);

        std::cout << "res norms = (" << new_res_norms.transpose() << ")\n";

        // Update penalty parameters
        update_PMM_params(res_p, res_d, new_res_p, new_res_d, mu, rho, reg_limit);
        std::cout << "mu = " << mu << ", rho = " << rho << "\n";
    }

    return 0;
}