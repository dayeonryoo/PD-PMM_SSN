#pragma once
#include <iostream>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <chrono>
#include "SSN.hpp"

template <typename T>
void SSN_PMM<T>::get_Q_info(const SpMat& Q) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    if (Q.nonZeros() == 0) {
        Q_info = 0;
        // std::cout << "QInfo: Zero matrix.\n";
        return;
    }

    if (Q.rows() != Q.cols()) {
        throw std::invalid_argument("Given Q is not a square matrix (n x n).");
    }
    // Q is given as a lower triangular matrix.

    // Is Q = 0?
    if (Q.rows() == 0) {
        Q_info = 0;
    } else { // Is Q diagonal or not?
        Q_info = 1;
        for (int k = 0; k < Q.outerSize(); ++k) {
            for (typename SpMat::InnerIterator it(Q, k); it; ++it) {
                if (it.row() != it.col()) {
                    Q_info = 2;
                }
            }
        }
    }

    if (Q_info == 1) {
        problem_Q_diag = Q.diagonal();
        // std::cout << "Q_info = diagonal\n";
    }
    if (Q_info == 2) {
        // std::cout << "Q_info = general\n";
    }
}

template <typename T>
void SSN_PMM<T>::determine_dimensions(const Problem<T>& problem) {
    // Determine n
    if (Q_info == 0) {
        if (problem.c.size() != 0) {
            n = problem.c.size();
        } else if (problem.A.cols() != 0) {
            n = problem.A.cols();
        } else if (problem.B.cols() != 0) {
            n = problem.B.cols();
        } else if (problem.lx.size() != 0) {
            n = problem.lx.size();
        } else if (problem.ux.size() != 0) {
            n = problem.ux.size();
        } else {
            throw std::invalid_argument("Problem dimension n cannot be determined from the provided data.");
        }
    } else if (Q_info == 1) {
        n = problem_Q_diag.size();
    } else { // Q_info == 2
        n = Q.rows();
    }

    // Determine m
    if (problem.A.rows() != 0) {
        m = problem.A.rows();
    } else if (problem.b.size() != 0) {
        m = problem.b.size();
    } else {
        m = 0;
    }

    // Determine l
    if (problem.B.rows() != 0) {
        l = problem.B.rows();
    } else if (problem.lw.size() != 0) {
        l = problem.lw.size();
    } else if (problem.uw.size() != 0) {
        l = problem.uw.size();
    } else {
        l = 0;
    }
}

template <typename T>
void SSN_PMM<T>::check_dimensions(const Problem<T>& problem) {

    // Check dimensions consistency
    if (problem.c.size() != 0 && problem.c.size() != n) {
        std::cout << "n = " << n << ", but c.size() = " << problem.c.size() << "\n";
        throw std::invalid_argument("Dimension mismatch: c should be a vector of size n.");
    }
    if ((problem.A.rows() != 0 && problem.A.rows() != m) || (problem.A.cols() != 0 && problem.A.cols() != n)) {
        throw std::invalid_argument("Dimension mismatch: A should be m x n.");
    }
    if (problem.b.size() != m) {
        throw std::invalid_argument("Dimension mismatch: b should be a vector of size m.");
    }
    if ((problem.B.rows() != 0 && problem.B.rows() != l) || (problem.B.cols() != 0 && problem.B.cols() != n)) {
        throw std::invalid_argument("Dimension mismatch: B should be l x n.");
    }
    if (problem.lx.size() != 0 && problem.lx.size() != n) {
        throw std::invalid_argument("Dimension mismatch: lx should be a vector of size n.");
    }
    if (problem.ux.size() != 0 && problem.ux.size() != n) {
        throw std::invalid_argument("Dimension mismatch: ux should be a vector of size n.");
    }
    if (problem.lw.size() != 0 && problem.lw.size() != l) {
        throw std::invalid_argument("Dimension mismatch: lw should be a vector of size l.");
    }
    if (problem.uw.size() != 0 && problem.uw.size() != l) {
        throw std::invalid_argument("Dimension mismatch: uw should be a vector of size l.");
    }
}

template <typename T>
void SSN_PMM<T>::ruiz_scaling(const SpMat& Q, const Vec& Q_diag, const SpMat& A, const SpMat& B, const Vec& c, const Vec& b, const Vec& lx, const Vec& ux) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    const int max_ruiz_iter = 10;
    const T ruiz_tol = 1e-3;
    const T eps = std::numeric_limits<T>::epsilon();

    A_ruiz = A;
    B_ruiz = B;
    c_ruiz = c;
    b_ruiz = b;
    lx_ruiz = lx;
    ux_ruiz = ux;
    if (Q_info == 2) {
        Q_ruiz = Q;
    } else {
        Q_diag_ruiz = Q_diag;
    }

    D1_diag = Vec::Ones(m); // Cumulative row scaling factors
    D2_diag = Vec::Ones(n); // Cumulative column scaling factors

    if (A_ruiz.rows() == 0 || A_ruiz.cols() == 0) return;

    for (int k = 0; k < max_ruiz_iter; ++k) {

        // Compute row-/column-wise infinity norms of current A_ruiz
        Vec row_max = Vec::Zero(m); // row_max(i) = max_j |A(i,j)|
        Vec col_max = Vec::Zero(n); // col_max(j) = max_i |A(i,j)|
        for (int col = 0; col < n; ++col) {
            for (typename SpMat::InnerIterator it(A_ruiz, col); it; ++it) {
                const int i = it.row();
                const int j = it.col();
                const T val = std::abs(it.value());
                if (val > row_max(i)) row_max(i) = val;
                if (val > col_max(j)) col_max(j) = val;
            }
        }

        // Check convergence
        const T max_row = row_max.maxCoeff();
        const T max_col = col_max.maxCoeff();
        if (std::abs(T(1) - max_row) < ruiz_tol && std::abs(T(1) - max_col) < ruiz_tol) break;

        // Scaling factors: dr, dc = sqrt(max)
        Vec dr = Vec::Ones(m);
        Vec dc = Vec::Ones(n);

        for (int i = 0; i < m; ++i) if (row_max(i) > eps) dr(i) = std::sqrt(row_max(i));
        for (int j = 0; j < n; ++j) if (col_max(j) > eps) dc(j) = std::sqrt(col_max(j));

        Vec dr_inv = dr.cwiseInverse();
        Vec dc_inv = dc.cwiseInverse();

        // Scale A_ruiz: A <- diag(dr)^{-1} A diag(dc)^{-1}
        for (int j = 0; j < n; ++j) {
            const T col_fac = dc_inv(j);
            for (typename SpMat::InnerIterator it(A_ruiz, j); it; ++it) {
                const T row_fac = dr_inv(it.row());
                it.valueRef() *= row_fac * col_fac;
            }
        }
        // Scale B_ruiz: B <- B D2 with D2 <- D2 * diag(dc)^{-1}
        for (int j = 0; j < n; ++j) {
            const T col_fac = dc_inv(j);
            for (typename SpMat::InnerIterator it(B_ruiz, j); it; ++it) {
                it.valueRef() *= col_fac;
            }
        }

        // Scale Q_ruiz if Q is nonzero: Q <- D2 Q D2
        if (Q_info == 2) {
            for (int j = 0; j < n; ++j) {
                const T col_fac = dc_inv(j);
                for (typename SpMat::InnerIterator it(Q_ruiz, j); it; ++it) {
                    const int i = it.row();
                    it.valueRef() *= dc_inv(i) * col_fac; // Q_ij *= d_i * d_j
                }
            }
        } else if (Q_info == 1) {
            Q_diag_ruiz.array() *= dc_inv.array().square(); // Q_ii *= d_i * d_i
        }

        // Scale c_ruiz: c <- D2 c
        if (c_ruiz.size() != 0) c_ruiz.array() *= dc_inv.array();

        // Scale b_ruiz: b <- D1 b with D1 <- D1 * diag(dr)^{-1}
        if (b_ruiz.size() != 0) b_ruiz.array() *= dr_inv.array();

        // Scale lx_ruiz, ux_ruiz: D2^{-1} lx <= D2^{-1} x <= D2^{-1} ux
        if (lx_ruiz.size() != 0) lx_ruiz.array() *= dc.array();
        if (ux_ruiz.size() != 0) ux_ruiz.array() *= dc.array();

        // Accumulate scaling factors (D <- D * diag(d)^{-1})
        D1_diag.array() *= dr_inv.array();
        D2_diag.array() *= dc_inv.array();
    }

}

template <typename T>
void SSN_PMM<T>::set_L_from_LLT(const SpMat& Q) {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;

    SpMat Q_full = Q.template selfadjointView<Eigen::Lower>();
    for (int i = 0; i < Q_full.rows(); ++i) {
        Q_full.coeffRef(i, i) += T(1e-10); // regularization for numerical stability
    }
    Q_full.makeCompressed();

    Eigen::SimplicialLDLT<SpMat> ldlt;
    ldlt.compute(Q_full);
    auto P = ldlt.permutationP();

    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("LDLT factorization on Q failed. Q is possibly singular.");
    }

    const Vec D = ldlt.vectorD();
    for (int i = 0; i < D.size(); ++i) {
        if (D(i) < -1e-2) {
            std::cout << "D(" << i << "): " << D(i) << "\n";
            throw std::invalid_argument("Q is not PSD.");
        }
    }

    const int n = Q.rows();
    std::vector<Triplet> trip;
    trip.reserve(n);
    for (int i = 0; i < n; ++i) {
        T val;
        if (D(i) > T(0)) {
            val = std::sqrt(D(i));
        } else {
            val = T(0);
        }
        if (val != T(0)) {
            trip.emplace_back(i, i, val);
        }
    }
    SpMat D_sqrt(n, n);
    D_sqrt.setFromTriplets(trip.begin(), trip.end());

    SpMat L_D = ldlt.matrixL(); // lower triangular from LDL^T
    L = P.transpose() * L_D * D_sqrt; // lower triangular from LL^T
    L_tr = L.transpose();
    
}

template <typename T>
void SSN_PMM<T>::set_default(const Problem<T>& problem) {
    using SpMat   = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;

    // T inf = std::numeric_limits<T>::infinity();
    T inf = 1e20;

    if (Q_info == 2) {
        N = 2 * n; M = m + n;

        // c', lx', ux'
        if (problem.c.size() == 0) {
            c = Vec::Zero(N);
        } else {
            c.resize(N);
            c << c_ruiz, Vec::Zero(n);
        }
        if (problem.b.size() == 0) {
            b = Vec::Zero(M);
        } else {
            b.resize(M);
            b << b_ruiz, Vec::Zero(n);
        }
        if (problem.lx.size() == 0) {
            lx = Vec::Constant(N, -inf);
        } else {
            lx.resize(N);
            lx << lx_ruiz, Vec::Constant(n, -inf);
        }
        if (problem.ux.size() == 0) {
            ux = Vec::Constant(N, inf);
        } else {
            ux.resize(N);
            ux << ux_ruiz, Vec::Constant(n, inf);
        }

        Q_diag.resize(N);
        Q_diag << Vec::Zero(n), Vec::Ones(n);

        // L s.t. Q = LL^T
        set_L_from_LLT(Q_ruiz);

        // A' = [A_ruiz 0; L^T -I]
        A.resize(M, N);
        {
            std::vector<Triplet> trip;
            trip.reserve(A_ruiz.nonZeros() + L.nonZeros() + n);

            // Top-left block: A (ruiz scaled)
            if (A_ruiz.rows() != 0 && A_ruiz.cols() != 0) {
                for (int k = 0; k < n; ++k) {
                    for (typename SpMat::InnerIterator it(A_ruiz, k); it; ++it) {
                        trip.emplace_back(it.row(), it.col(), it.value());
                    }
                }
            }
            // Bottom-left block: L^T
            for (int k = 0; k < n; ++k) {
                for (typename SpMat::InnerIterator it(L, k); it; ++it) {
                    trip.emplace_back(m + it.col(), it.row(), it.value());
                }
            }
            // Bottom-right block: -I_n
            for (int i = 0; i < n; ++i) {
                trip.emplace_back(m + i, n + i, T(-1));
            }
            A.setFromTriplets(trip.begin(), trip.end());
        }

        // B' = [B_ruiz 0]
        B.resize(l, N);
        {
            std::vector<Triplet> trip;
            trip.reserve(B_ruiz.nonZeros());

            if (B_ruiz.rows() != 0 && B_ruiz.cols() != 0) {
                for (int k = 0; k < n; ++k) {
                    for (typename SpMat::InnerIterator it(B_ruiz, k); it; ++it) {
                        trip.emplace_back(it.row(), it.col(), it.value());
                    }
                }
            }
            B.setFromTriplets(trip.begin(), trip.end());
        }

    } else {
        N = n; M = m;

        if (problem.c.size() == 0) {
            c = Vec::Zero(N);
        } else {
            c = c_ruiz;
        }
        if (problem.A.rows() == 0 || problem.A.cols() == 0) {
            A = SpMat(M, N);
        } else {
            A = A_ruiz;
        }
        if (problem.b.size() != 0) {
            b = b_ruiz;
        }
        if (problem.B.rows() == 0 || problem.B.cols() == 0) {
            B = SpMat(l, N);
        } else {
            B = B_ruiz;
        }
        if (problem.lx.size() == 0) {
            lx = Vec::Constant(N, -inf);
        } else {
            lx = lx_ruiz;
        }
        if (problem.ux.size() == 0) {
            ux = Vec::Constant(N, inf);
        } else {
            ux = ux_ruiz;
        }
        Q_diag = Q_diag_ruiz;
    }

    // lw, uw
    if (problem.lw.size() == 0) {
        lw = Vec::Constant(l, -inf);
    } else {
        lw = problem.lw;
    }
    if (problem.uw.size() == 0) {
        uw = Vec::Constant(l, inf);
    } else {
        uw = problem.uw;
    }

}

template <typename T>
void SSN_PMM<T>::initialize_sols() {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using Triplet = typename SSN<T>::Triplet;

    x = Vec::Zero(N);
    y1 = Vec::Zero(M);
    y2 = Vec::Zero(l);
    z = Vec::Zero(N);

    /*
    y2 = Vec::Zero(l);
    z = Vec::Zero(N);

    // Form K = [-(Q + I_N/rho), A^T; A, I_M/mu]
    std::vector<Triplet> trip;
    trip.reserve(N + M + 2 * A.nonZeros());

    // Top-left block: -(Q + I_N/rho)
    for (int i = 0; i < N; ++i) {
        const T val = Q_diag(i) + 1/rho;
        if (val != T(0)) trip.emplace_back(i, i, -val);
    }
    // Bottom-right block: I_M / mu
    const T mu_inv = T(1) / mu;
    for (int i = 0; i < M; ++i) {
        trip.emplace_back(N + i, N + i, mu_inv);
    }
    // Off-diagonal blocks: A and A^T
    for (int col = 0; col < N; ++col) {
        for (typename SpMat::InnerIterator it(A, col); it; ++it) {
            const int i = it.row();
            const int j = it.col();
            const T val = it.value();

            trip.emplace_back(N + i, j, val);
            trip.emplace_back(j, N + i, val);
        }
    }
    SpMat K(N + M, N + M);
    K.setFromTriplets(trip.begin(), trip.end());
    K.makeCompressed();

    // From RHS vector
    Vec rhs(N + M);
    rhs << c, b;

    // Solve using LDLT
    Eigen::SimplicialLDLT<SpMat> ldlt;
    ldlt.compute(K);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("Warm-starting via LDLT failed.");
    }
    Vec xy1 = ldlt.solve(rhs);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("Warm-starting via LDLT failed.");
    }
    x = xy1.head(N);
    y1 = xy1.tail(M);
    */
}

template <typename T>
void SSN_PMM<T>::check_bounds() {
    // Check lower and upper bounds
    for (int i = 0; i < N; ++i) {
        if (lx(i) > ux(i)) {
            throw std::invalid_argument("Problem is infeasible: lx should be <= ux.");
        }
    }
    for (int i = 0; i < l; ++i) {
        if (lw(i) > uw(i)) {
            throw std::invalid_argument("Problem is infeasible: lw should be <= uw.");
        }
    }
}

template <typename T>
typename SSN_PMM<T>::Vec SSN_PMM<T>::compute_residual_norms() {
    // Primal residual norm
    T res_p;
    if (M > 0) res_p = (A * x - b).norm() / (1 + b.norm());
    else res_p = T(0);

    // Dual residual norm
    T res_d;
    if (Q_info == 0) {
        res_d = (c - A_tr * y1 - B_tr * y2 + z).norm() / (1 + c.norm());
    } else {
        res_d = (c + Q_diag.cwiseProduct(x) - A_tr * y1 - B_tr * y2 + z).norm() / (1 + c.norm());
    }

    // Complementarity residual norm for box constraints
    T compl_x = (x - proj(x + z, lx, ux)).norm();

    // Complementarity residual norm for Bx constraints
    Vec Bx = B * x;
    T compl_w;
    if (l > 0) compl_w = (Bx - proj(Bx - y2, lw, uw)).norm();
    else compl_w = T(0);

    // Collect residual norms
    Vec res_norms(4);
    res_norms << res_p, res_d, compl_x, compl_w;

    return res_norms;
}

template <typename T>
typename SSN_PMM<T>::Vec SSN_PMM<T>::compute_residual_norms_inf() {
    // Primal residual norm
    T res_p;
    if (M == 0) res_p = T(0);
    else res_p = inf_norm(A * x - b) / (1 + inf_norm(b)); 

    // Dual residual norm
    T res_d;
    if (Q_info == 0) {
        res_d = inf_norm(c - A_tr * y1 - B_tr * y2 + z) / (1 + inf_norm(c));
    } else {
        res_d = inf_norm(c + Q_diag.cwiseProduct(x) - A_tr * y1 - B_tr * y2 + z) / (1 + inf_norm(c));
    }

    // Complementarity residual norm for box constraints
    T compl_x = inf_norm(x - proj(x + z, lx, ux));

    // Complementarity residual norm for Bx constraints
    T compl_w;
    if (l == 0) compl_w = T(0);
    else {
        Vec Bx = B * x;
        compl_w = inf_norm(Bx - proj(Bx - y2, lw, uw));
    }

    // Collect residual norms
    Vec res_norms(4);
    res_norms << res_p, res_d, compl_x, compl_w;

    return res_norms;
}

template <typename T>
T SSN_PMM<T>::objective_value(const Vec& x) {
    T obj_val = obj_const + c.dot(x);
    if (Q_info != 0) {
        obj_val += T(0.5) * Q_diag.cwiseProduct(x).dot(x);
    }
    return obj_val;
}

template <typename T>
void SSN_PMM<T>::printable_sol(const Vec& x, const Vec& y1, const Vec& z) {
    // Ruiz descale, and shrink to original dimension if needed
    if (Q_info == 2) {
        x_sol = x.head(n).array() * D2_diag.array();
        y1_sol = y1.head(m).array() * D1_diag.array();
        z_sol = z.head(n).array() / D2_diag.array();
    } else {
        x_sol = x.cwiseProduct(D2_diag);
        y1_sol = y1.cwiseProduct(D1_diag);
        z_sol = z.cwiseQuotient(D2_diag);
    }
}

template <typename T>
void SSN_PMM<T>::update_PMM_parameters(const T res_p, const T res_d, const T new_res_p, const T new_res_d) {
    // If the overall primal and dual residual error is decreased,
    // we increase the penalty parameters aggressively.
    // If not, we continue increasing the parameters slowly
    // up to the regularization threshold.

    bool cond_p = 0.95 * res_p > new_res_p;
    bool cond_d = 0.95 * res_d > new_res_d;

    if (cond_p || cond_d){
        mu = std::min(reg_limit, 1.2*mu);
        rho = std::min(1e2*reg_limit, 1.4*rho);
        // std::cout << "Aggressive update of PMM parameters.\n";
    } else {
        mu = std::min(reg_limit, 1.05*mu);
        rho = std::min(1e2*reg_limit, 1.05*rho);
        // std::cout << "Mild update of PMM parameters.\n";
    }

}

template <typename T>
T SSN_PMM<T>::compute_p(const Vec& x) {
    using Vec = typename SSN_PMM<T>::Vec;
    if (l == 0) return T(0);
    Vec Bx = B * x;
    T p = inf_norm(Bx - proj(Bx, lw, uw));
    return p;
}

template <typename T>
void SSN_PMM<T>::update_with_bcl(const T p, const Vec& y2_hat, const T res) {
    if (p <= eps_bcl) {
        // Accept y2 from SSN; keep mu and rho unchanged; fast decrease eps
        // std::cout << "  SSN result ACCEPTED.\n";
        y2 = y2_hat;
        mu = std::min(reg_limit, 1.05 * mu);
        rho = std::min(reg_limit, 1.05 * rho);
        eps_bcl = std::max(eps_limit, eps_bcl / std::pow(mu, 0.9));
        SSN_tol = std::max(eps_limit, SSN_tol / mu);
    } else {
        // Reject y2 from SSN; increase mu and rho; reset eps
        // std::cout << "  SSN result rejected.\n";
        mu = std::min(reg_limit, 1.1 * mu);
        rho = std::min(reg_limit, 1.2 * rho);
        eps_bcl = std::max(eps_limit, mu0 / std::pow(mu, 0.09));
        SSN_tol = std::max(eps_limit, mu0 / mu);
    }
}

template <typename T>
bool SSN_PMM<T>::primal_infeas_y1(const Vec& delta_y1, T eps_pinf) {
    if (M == 0) return false;

    T delta_y1_inf = inf_norm(delta_y1);
    if (delta_y1_inf == T(0)) return false;

    T rhs_y1 = eps_pinf * delta_y1_inf;
    
    bool cond1 = inf_norm(A_tr * delta_y1) <= rhs_y1;
    if (!cond1) return false;

    bool cond2 = b.dot(delta_y1) <= -rhs_y1;
    if (!cond2) return false;

    return true;
}

template <typename T>
bool SSN_PMM<T>::primal_infeas_z(const Vec& delta_z, T eps_pinf) {

    T delta_z_inf = inf_norm(delta_z);
    if (delta_z_inf == T(0)) return false;

    T rhs_z = eps_pinf * delta_z_inf;

    T lhs = T(0);
    for (int i = 0; i < N; ++i) {
        if (lx(i) > -1e20 && delta_z(i) < T(0)) {
            lhs += lx(i) * delta_z(i);
        }
        if (ux(i) < 1e20 && delta_z(i) > T(0)) {
            lhs += ux(i) * delta_z(i);
        }
    }

    return lhs <= -rhs_z;
}

template <typename T>
Solution<T> SSN_PMM<T>::solve() {

    // Initialize variables
    opt = -1;
    PMM_iter = 0;
    SSN_iter = 0;
    Vec y2_hat = y2;
    Vec res_norms;
    SSN_tol_achieved = T(0);
    bool pinf_y1z = false;

    // Build the Newton system
    SSN<T> NS(Q_info, Q_diag, L, L_tr,
            A, B, A_tr, B_tr, c, b, D1_diag, D2_diag,
            lx, ux, lw, uw, obj_const, n, m, N, M, l,
            SSN_tol, SSN_max_in_iter, more_rows_than_cols,
            eps_pinf, eps_dinf);

    // Print header
    print_header(when, what);

    // SSN-PMM main loop
    while (PMM_iter < max_iter) {
        // ----------------------------------------------
        // Structure:
        // Until (primal infeasibility, dual infeasibility, complementarity) < tol, do:
        //     1) Call Semismooth Newton method to approximately minimize the augmented Lagrangian w.r.t. x;
        //     2) Update multipliers y1, y2, z;
        //     3) Update penalty parameters mu, rho;
        //     k = k + 1;
        // End
        // ----------------------------------------------

        // TIMER FOR PMM ITERATION
        auto t0_pmm = std::chrono::steady_clock::now();

        // Call semismooth Newton method
        NS.update_SSN_system(x, y1, y2, z, mu, rho, gamma, SSN_iter, pinf_y1z);
        SSN_result<T> NS_solution = NS.solve_SSN(SSN_tol);

        SSN_iter += NS_solution.iter;
        SSN_tol_achieved = NS_solution.tol_achieved;

        // If SSN was not "successful", discard the recent SSN and revert to solution from previous PMM iter.
        if (SSN_iter >= SSN_max_iter) { // Max total SSN iteration is reached
            opt = 2;
            break;
        } else if (NS_solution.opt == 3) { // Linesearch failed
            std::cout << std::setw(8) << PMM_iter << std::setw(8) << SSN_iter << ": Linesearch failed. Retrying with smaller mu and rho.\n";
            if (mu == mu0 && rho == mu0) { opt = 3; break; }
            else {
                mu = std::max(mu0, 0.5 * mu);
                rho = std::max(mu0, 0.5 * rho);
                PMM_iter++;
                continue;
            }
        } else if (NS_solution.opt == -2) { // Primal infeasible
            opt = -2;
            break;
        } else if (NS_solution.opt == -3) { // Dual infeasible
            opt = -3;
            break;
        }

        // Update x and store candidate y2.
        x = NS_solution.x;
        y2_hat = NS_solution.y2;

        // Update y2 and penalty parameters based on BCL
        T p = compute_p(x); // L_inf primal feasibility violation of lw <= Bx <= uw
        update_with_bcl(p, y2_hat, PMM_tol_achieved);

        // Update y1 and z
        Vec delta_y1 = mu * (A * x - b);
        y1 -= delta_y1;

        Vec delta_z = mu * (x - proj(z / mu + x, lx, ux));
        z += delta_z;

        // Compute residuals
        res_norms = compute_residual_norms_inf();
        PMM_tol_achieved = res_norms.maxCoeff();

        // Check infeasibility with y1 and z.
        // Inside SSN iteration together with x and y2, infeasibility will be detected.
        bool pinf_y1 = primal_infeas_y1(delta_y1, eps_pinf);
        bool pinf_z = primal_infeas_z(delta_z, eps_pinf);
        pinf_y1z = pinf_y1 || pinf_z;

        // Ruiz-descale and shrink to the original dimension (n, m, l) for printing
        printable_sol(x, y1, z); // (Modifies x_sol, y1_sol, z_sol.)
        obj_val = objective_value(x);

        // Check termination criterion
        if (PMM_tol_achieved < tol) {
            opt = 0; // Optimal solution found
            if (when != PrintWhen::NEVER || when != PrintWhen::ALWAYS) {
                print(PrintWhen::ALWAYS, what, PMM_iter, SSN_iter, obj_val, res_norms, SSN_tol_achieved, mu, rho, eps_bcl, SSN_tol);
            }
            break;
        }

        // Print current iteration info
        print(when, what, PMM_iter, SSN_iter, obj_val, res_norms, SSN_tol_achieved, mu, rho, eps_bcl, SSN_tol);
        PMM_iter++;

        // TIMER FOR PMM ITERATION
        auto t1_pmm = std::chrono::steady_clock::now();
        double timer_pmm = time_diff_ms(t0_pmm, t1_pmm);
        // std::cout << "PMM iteration took " << timer_pmm << " ms.\n";
        // std::cout << "=====================================================\n";

    }

    // Check if max number of PMM iterations is reached
    if (opt == -1) { opt = 1; }

    // Check if infeasiblity is detected
    if (opt == -2 || opt == -3) {
        obj_val = 1e20;
        res_norms = 1e20 * Vec::Ones(4);
        PMM_tol_achieved = 1e20;
        SSN_tol_achieved = 1e20;
        x_sol = Vec::Zero(n);
        y1_sol = Vec::Zero(m);
        y2 = Vec::Zero(l);
        z_sol = Vec::Zero(n);
    }

    print(when, what, PMM_iter, SSN_iter, obj_val, res_norms, SSN_tol_achieved, mu, rho, eps_bcl, SSN_tol);
    return Solution<T>(opt, x_sol, y1_sol, y2, z_sol, obj_val, PMM_iter, SSN_iter, PMM_tol_achieved, SSN_tol_achieved);
}
