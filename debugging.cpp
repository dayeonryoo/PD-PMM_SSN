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
    int SSN_max_in_iter = 5; // max SSN iter per PMM iter

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
            SpMat H_tilde(n, n);
            std::vector<Triplet> H_tilde_tripl;
            H_tilde_tripl.reserve(n);
            for (int i = 0; i < n; ++i) {
                H_tilde_tripl.emplace_back(i, i, H_tilde_diag(i));
            }
            H_tilde.setFromTriplets(H_tilde_tripl.begin(), H_tilde_tripl.end());

            std::cout << "              diag(H_tilde) = " << H_tilde_diag.transpose() << "\n";

            // Construct B_active_W and B_inactive_W
            SpMat B_active_W(n_active_W, n);
            SpMat B_inactive_W(n_inactive_W, n);
            // should i explicitly do .setZero()? i don't think so
            std::vector<Triplet> B_act_trpl;
            std::vector<Triplet> B_inact_trpl;
            int new_col_act = 0;
            int new_col_inact = 0;
            for (int i = 0; i < l; ++i) {
                if (active_W(i)) {
                    for (SpMat::InnerIterator it(B, i); it; ++it) {
                        B_act_trpl.emplace_back(it.row(), new_col_act, it.value());
                    }
                    new_col_act++;
                } else {
                    for (SpMat::InnerIterator it(B, i); it; ++it) {
                        B_inact_trpl.emplace_back(it.row(), new_col_inact, it.value());
                    }
                    new_col_inact++;
                }   
            }
            B_active_W.setFromTriplets(B_act_trpl.begin(), B_act_trpl.end());
            B_inactive_W.setFromTriplets(B_inact_trpl.begin(), B_inact_trpl.end());

            std::cout << "              diag(B_active_W) = " << B_active_W.diagonal().transpose() << "\n";
            std::cout << "              diag(B_inactive_W) = " << B_inactive_W.diagonal().transpose() << "\n";

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

            std::cout << "              diag(D) = " << D.diagonal().transpose() << "\n";

            // Separate y2, dist_W(B*x_new + (y2_new/2 - y2)/mu) w.r.t. active_W and inactive_W
            Vec y2_active_W(n_active_W);
            Vec y2_inactive_W(n_inactive_W);
            Vec dist_W = v2 - proj(v2, lw, uw);
            Vec dist_W_active_W(n_active_W);
            Vec dist_W_inactive_W(n_inactive_W);
            

            SSN_in_iter++;
        }

        // Update x and y2.
        x = SSN_x;
        y2 = SSN_y2;

        // Update y1 and z.
        y1 -= mu * (A * x - b);
        z += mu * (x - proj(z / mu + x, lx, ux));

        std::cout << "  x = (" << x.transpose() << ")\n";
        std::cout << "  y1 = (" << y1.transpose() << ")\n";
        std::cout << "  y2 = (" << y2.transpose() << ")\n";
        std::cout << "  z = (" << z.transpose() << ")\n";
        
    }


    return 0;
}