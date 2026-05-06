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
void SSN_PMM<T>::ruiz_scaling(const Problem<T>& problem, const Vec& problem_Q_diag) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    const int max_ruiz_iter = 10;
    const T ruiz_tol = 1e-3;
    const T eps = std::numeric_limits<T>::epsilon();

    A_ruiz = problem.A;
    B_ruiz = problem.B;
    c_ruiz = problem.c;
    b_ruiz = problem.b;
    lx_ruiz = problem.lx;
    ux_ruiz = problem.ux;
    lw_ruiz = problem.lw;
    uw_ruiz = problem.uw;

    if (Q_info == 2) {
        Q_ruiz = problem.Q;
    } else {
        Q_diag_ruiz = problem_Q_diag;
    }

    D2_diag = Vec::Ones(n);  // Column scalings for x
    D1A_diag = Vec::Ones(m); // Row scalings for equality constraint matrix A
    D1B_diag = Vec::Ones(l); // Row scalings for box constraint matrix B

    if (n == 0) return;

    for (int k = 0; k < max_ruiz_iter; ++k) {

        // Compute row-/column-wise infinity norms of constraint matrices
        Vec row_max_A = Vec::Zero(m); // row_max(i) = max_j |A(i,j)|
        Vec row_max_B = Vec::Zero(l); // row_max(i) = max_j |B(i,j)|
        Vec col_max = Vec::Ones(n); // col_max(j) = max_i |[A; B; I](i,j)| (starts at 1 due to I)

        // Contribution from A
        for (int col = 0; col < n; ++col) {
            for (typename SpMat::InnerIterator it(A_ruiz, col); it; ++it) {
                const int i = it.row();
                const T val = std::abs(it.value());
                if (val > row_max_A(i)) row_max_A(i) = val;
                if (val > col_max(col)) col_max(col) = val;
            }
        }

        // Contribution from B
        for (int col = 0; col < n; ++col) {
            for (typename SpMat::InnerIterator it(B_ruiz, col); it; ++it) {
                const int i = it.row();
                const T val = std::abs(it.value());
                if (val > row_max_B(i)) row_max_B(i) = val;
                if (val > col_max(col)) col_max(col) = val;
            }
        }

        // Check convergence on [A; B; I]
        T row_dev = T(0);
        if (m > 0) row_dev = std::max(row_dev, (row_max_A.array() - T(1)).abs().maxCoeff());
        if (l > 0) row_dev = std::max(row_dev, (row_max_B.array() - T(1)).abs().maxCoeff());

        T col_dev = (col_max.array() - T(1)).abs().maxCoeff();
        if (row_dev < ruiz_tol && col_dev < ruiz_tol) break;

        // Scaling factors: dr, dc = sqrt(max_norms)
        Vec drA = Vec::Ones(m);
        Vec drB = Vec::Ones(l);
        Vec dc  = Vec::Ones(n);

        for (int i = 0; i < m; ++i) if (row_max_A(i) > eps) drA(i) = std::sqrt(row_max_A(i));
        for (int i = 0; i < l; ++i) if (row_max_B(i) > eps) drB(i) = std::sqrt(row_max_B(i));
        for (int j = 0; j < n; ++j) if (col_max(j) > eps)   dc(j) = std::sqrt(col_max(j));

        Vec drA_inv = drA.cwiseInverse();
        Vec drB_inv = drB.cwiseInverse();
        Vec dc_inv = dc.cwiseInverse();

        // Scale A: A <-  D1A^{-1} A D2^{-1}
        for (int j = 0; j < n; ++j) {
            const T col_fac = dc_inv(j);
            for (typename SpMat::InnerIterator it(A_ruiz, j); it; ++it) {
                const T row_fac = drA_inv(it.row());
                it.valueRef() *= row_fac * col_fac;
            }
        }
        // Scale B: B <- D1B^{-1} B D2^{-1}
        for (int j = 0; j < n; ++j) {
            const T col_fac = dc_inv(j);
            for (typename SpMat::InnerIterator it(B_ruiz, j); it; ++it) {
                const T row_fac = drB_inv(it.row());
                it.valueRef() *= row_fac * col_fac;
            }
        }

        // Scale I: I <- D1I^{-1} I D2^{-1}
        // This is represented through the variable substitution and scaling of lx, ux below.

        // Scale Q if Q is nonzero: Q <- D2^{-1} Q D2^{-1}
        if (Q_info == 2) {
            for (int j = 0; j < n; ++j) {
                const T col_fac = dc_inv(j);
                for (typename SpMat::InnerIterator it(Q_ruiz, j); it; ++it) {
                    const T row_fac = dc_inv(it.row());
                    it.valueRef() *= row_fac * col_fac; // Q_ij *= 1/d_i * 1/d_j
                }
            }
        } else if (Q_info == 1) {
            Q_diag_ruiz.array() *= dc_inv.array().square(); // Q_ii *= 1/d_i * 1/d_i
        }

        // Scale c: c <- D2^{-1} c
        if (c_ruiz.size() == n) c_ruiz.array() *= dc_inv.array();

        // Scale b: b <- D1A^{-1} b
        if (b_ruiz.size() == m) b_ruiz.array() *= drA_inv.array();

        // Scale lw, uw:  lw, uw <- D1B^{-1} lw, uw
        if (lw_ruiz.size() == l) {
            for (int i = 0; i < l; ++i) {
                if (lw_ruiz(i) > -inf) lw_ruiz(i) *= drB_inv(i);
            }
        }
        if (uw_ruiz.size() == l) {
            for (int i = 0; i < l; ++i) {
                if (uw_ruiz(i) < inf) uw_ruiz(i) *= drB_inv(i);
            }
        }

        // Scale lx, ux:  lx, ux <- D2 lx, ux
        if (lx_ruiz.size() == n) {
            for (int i = 0; i < n; ++i) {
                if (lx_ruiz(i) > -inf) lx_ruiz(i) *= dc(i);
            }
        }
        if (ux_ruiz.size() == n) {
            for (int i = 0; i < n; ++i) {
                if (ux_ruiz(i) < inf) ux_ruiz(i) *= dc(i);
            }
        }

        // Accumulate scaling factors (D <- D * diag(d))
        if (m > 0) D1A_diag.array() *= drA.array();
        if (l > 0) D1B_diag.array() *= drB.array();
        D2_diag.array() *= dc.array();
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
        lw = lw_ruiz;
    }
    if (problem.uw.size() == 0) {
        uw = Vec::Constant(l, inf);
    } else {
        uw = uw_ruiz;
    }

    // Decide whether to solve KKT or Schur
    more_rows_than_cols = N < M + l;
    // more_rows_than_cols = true; // always solve KKT
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
typename SSN_PMM<T>::Vec SSN_PMM<T>::compute_residual_norms_inf(const Vec& Ax, const Vec& Bx, const Vec& Qx) {
    // Primal residual norm
    T res_p;
    if (M == 0) res_p = T(0);
    else {
        // T denom = 1 + inf_norm(b); // original denominator based only on a constant term b
        T denom = 1 + std::max(inf_norm(Ax), inf_norm(b));
        res_p = inf_norm(Ax - b) / denom;
    }

    // Dual residual norm
    Vec num = c + z;
    // T denom = 1 + inf_norm(c); // original denominator based only on a constant term c
    T denom = std::max(inf_norm(c), inf_norm(z));
    if (Q_info != 0) {
        num += Qx;
        denom = std::max(denom, inf_norm(Qx));
    }
    if (M != 0) {
        Vec A_tr_y1 = A_tr * y1;
        num -= A_tr_y1;
        denom = std::max(denom, inf_norm(A_tr_y1));
    }
    if (l != 0) {
        Vec B_tr_y2 = B_tr * y2;
        num -= B_tr_y2;
        denom = std::max(denom, inf_norm(B_tr_y2));
    }
    denom += T(1);
    T res_d = inf_norm(num) / denom;

    // Complementarity residual norm for box constraints on x
    Vec proj_K = proj(x + z, lx, ux);
    T compl_x = inf_norm(x - proj_K);
    compl_x /= (T(1) + std::max(inf_norm(x), inf_norm(proj_K)));

    // Complementarity residual norm for Bx constraints
    T compl_w;
    if (l == 0) compl_w = T(0);
    else {
        Vec proj_W = proj(Bx - y2, lw, uw);
        compl_w = inf_norm(Bx - proj_W);
        compl_w /= (T(1) + std::max(inf_norm(Bx), inf_norm(proj_W)));
    }

    Vec res_norms(4);
    res_norms << res_p, res_d, compl_x, compl_w;
    return res_norms;
}

template <typename T>
T SSN_PMM<T>::objective_value(const Vec& x, const Vec& Qx) {
    T obj_val = obj_const + c.dot(x);
    if (Q_info != 0) {
        obj_val += T(0.5) * Qx.dot(x);
    }
    return obj_val;
}

template <typename T>
void SSN_PMM<T>::printable_sol(const Vec& x, const Vec& y1, const Vec& y2, const Vec& z) {
    // Ruiz descale, and shrink to original dimension if needed
    if (Q_info == 2) {
        x_sol = x.head(n).array() / D2_diag.array();
        y1_sol = y1.head(m).array() / D1A_diag.array();
        y2_sol = y2.head(l).array() / D1B_diag.array();
        z_sol = z.head(n).array() * D2_diag.array();
    } else {
        x_sol = x.cwiseQuotient(D2_diag);
        y1_sol = y1.cwiseQuotient(D1A_diag);
        y2_sol = y2.cwiseQuotient(D1B_diag);
        z_sol = z.cwiseProduct(D2_diag);
    }
}

template <typename T>
void SSN_PMM<T>::update_PMM_parameters(const Vec& res_norms, const Vec& new_res_norms, int SSN_opt, T SSN_tol_achieved) {
    using Vec = typename SSN_PMM<T>::Vec;
    
    // Looking at the residual's reduction
    T worst_res = new_res_norms.maxCoeff();
    T worst_ratio = (new_res_norms.array() / (res_norms.array().abs() + 1e-12)).maxCoeff();

    bool ssn_success = SSN_opt == 0;
    bool ssn_good = SSN_tol_achieved < T(0.1) * SSN_tol;
    bool ssn_bad = SSN_tol_achieved > T(10) * SSN_tol;
    bool stagnating = worst_ratio > T(0.99);

    // Update stagnation counter
    if (!ssn_success || stagnating) stagnation++;
    else stagnation = 0;

    if (ssn_success) {
        if (ssn_good || stagnating) {
            // Reliable SSN solve -> aggessive increase to speed up convergence
            // Stagnating -> aggressive increase to escape possible local difficulties
            mu = std::min(mu_limit, T(1.20) * mu);
            rho = std::min(rho_limit, T(1.15) * rho);
            // std::cout << "Reliable SSN solve\n";
        } else if (worst_ratio < T(0.95)) {
            // Good progress -> mild increase
            mu = std::min(mu_limit, T(1.10) * mu);
            rho = std::min(rho_limit, T(1.05) * rho);
            // std::cout << "Good progress\n";
        }
        SSN_tol = std::max(eps_limit, std::min({worst_res, T(0.90) * SSN_tol, std::pow(worst_res, T(1.2))}));

    } else {
        // Unsuccessful SSN
        if (ssn_bad) {
            mu = std::max(T(5), T(0.99) * mu);
            rho = std::max(T(5), T(0.99) * rho);
            SSN_tol = std::min(worst_res, T(1.05) * SSN_tol);
            // std::cout << "Bad SSN solve\n";
        } else if (stagnating) {
            // Mild increase to escape possible local difficulties
            mu = std::min(mu_limit, T(1.05) * mu);
            rho = std::min(rho_limit, T(1.05) * rho);
            // std::cout << "Mild increase\n";
        }
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
void SSN_PMM<T>::update_with_bcl(const Vec& y2_hat, T compl_W, T new_compl_W, int PMM_iter) {
    if (new_compl_W / compl_W < 1.0) {
        // std::cout << "SSN y2 accepted.\n";
        y2 = y2_hat;
    }
}

template <typename T>
bool SSN_PMM<T>::primal_infeas(const Vec& cert_y1, const Vec& cert_y2, const Vec& cert_z) {
    /*
    Primal infeasibility certificate at the PMM level:
    cert_y1 = delta_y1 = y1_new - y1_old
    cert_y2 = y2_new - y2_old (after SSN iterations)
    cert_z  = delta_z = z_new - z_old

    The QP is determined to be primal infeasible if all 2 conditions hold for nonzero [delta_y1, delta_y2, delta_z]:
    1. ||A^T cert_y1 + B^T cert_y2 - cert_z||_inf <= eps_pinf * max{||cert_y1||_inf, ||cert_y2||_inf, ||cert_z||_inf};
    2. -b^T cert_y1 + sum_i [uw_i * max(-cert_y2_i, 0)] + sum_i [lw_i * min(-cert_y2_i, 0)]
                     + sum_i [ux_i * max(cert_z_i, 0)]   + sum_i [lx_i * min(cert_z_i, 0)]
       <= -eps_pinf * max{||cert_y1||_inf, ||cert_y2||_inf, ||cert_z||_inf}
       for finite lx_i, ux_i, lw_i, uw_i.
    */
    T cert_inf = std::max({M > 0 ? inf_norm(cert_y1) : T(0),
                           l  > 0 ? inf_norm(cert_y2) : T(0),
                           inf_norm(cert_z)});
    if (cert_inf < T(1e-12)) return false;

    // Condition 2
    T lhs2 = T(0);
    if (M > 0) lhs2 -= b.dot(cert_y1);
    for (int i = 0; i < l; ++i) {
        if (uw(i) < inf) lhs2 += uw(i) * std::max(-cert_y2(i), T(0));
        if (lw(i) > -inf) lhs2 += lw(i) * std::min(-cert_y2(i), T(0));
    }
    for (int i = 0; i < N; ++i) {
        if (ux(i) < inf) lhs2 += ux(i) * std::max(cert_z(i), T(0));
        if (lx(i) > -inf) lhs2 += lx(i) * std::min(cert_z(i), T(0));
    }
    if (lhs2 > -eps_pinf * cert_inf) return false;

    // Condition 1
    Vec lhs1 = -cert_z;
    if (M > 0) lhs1 += A_tr * cert_y1;
    if (l  > 0) lhs1 += B_tr * cert_y2;
    return inf_norm(lhs1) <= eps_pinf * cert_inf;
}

template <typename T>
bool SSN_PMM<T>::dual_infeas(const Vec& delta_x, const Vec& Adx, const Vec& Bdx) {
    /*
    Dual infeasibility certificate at the PMM level:
    delta_x = x_new - x_old (PMM-level primal change used as certificate direction).
    Adx = A * delta_x, Bdx = B * delta_x — pre-computed by caller.

    The QP is determined to be dual infeasible if all 5 conditions hold for nonzero delta_x:
    1. ||Q delta_x||_inf <= eps_dinf * ||delta_x||_inf;
    2. c^T delta_x <= -eps_dinf * ||delta_x||_inf;
    3. ||A delta_x||_inf <= eps_dinf * ||delta_x||_inf;
    4. (delta_x)_i ∈ [-eps_dinf, eps_dinf] * ||delta_x||_inf  for finite bounds on x_i,
       (delta_x)_i >= -eps_dinf * ||delta_x||_inf  for finite lower bounds on x_i,
       (delta_x)_i <= eps_dinf  * ||delta_x||_inf  for finite upper bounds on x_i;
    5. (B delta_x)_i ∈ [-eps_dinf, eps_dinf]  * ||delta_x||_inf for finite bounds on (Bx)_i,
       (B delta_x)_i >= -eps_dinf * ||delta_x||_inf for finite lower bounds on (Bx)_i,
       (B delta_x)_i <= eps_dinf  * ||delta_x||_inf for finite upper bounds on (Bx)_i.
    */
    T eps_zero = T(1e-12);
    const T delta_x_inf = inf_norm(delta_x);
    if (delta_x_inf < eps_zero) return false;
    const T rhs = eps_dinf * delta_x_inf;

    // Conditions 1, 2, 3
    if (Q_info != 0 && inf_norm(Q_diag.cwiseProduct(delta_x)) > rhs) return false;
    if (c.dot(delta_x) > -rhs) return false;
    if (M > 0 && inf_norm(Adx) > rhs) return false;

    // Condition 4
    for (int i = 0; i < N; ++i) {
        const bool has_lx = lx(i) > -inf;
        const bool has_ux = ux(i) < inf;
        if (has_lx && has_ux) { if (std::abs(delta_x(i)) > rhs) return false; }
        else if (has_lx)      { if (delta_x(i) < -rhs)          return false; }
        else if (has_ux)      { if (delta_x(i) >  rhs)          return false; }
    }

    // Condition 5
    for (int i = 0; i < l; ++i) {
        const bool has_lw = lw(i) > -inf;
        const bool has_uw = uw(i) < inf;
        if (has_lw && has_uw) { if (std::abs(Bdx(i)) > rhs) return false; }
        else if (has_lw)      { if (Bdx(i) < -rhs)          return false; }
        else if (has_uw)      { if (Bdx(i) >  rhs)          return false; }
    }
    return true;
}

template <typename T>
Solution<T> SSN_PMM<T>::solve() {
    
    // Initialize variables
    opt = -1;
    PMM_iter = 0;
    SSN_iter = 0;
    Vec y2_hat = y2;
    SSN_tol_achieved = T(0);
    Vec delta_y1 = Vec::Zero(M);
    Vec delta_z = Vec::Zero(N);
    Vec Ax_old = M > 0 ? Vec(A * x) : Vec::Zero(M);
    Vec Bx_old = l  > 0 ? Vec(B * x) : Vec::Zero(l);
    Vec Qx_init = Q_info != 0 ? Vec(Q_diag.cwiseProduct(x)) : Vec::Zero(N);
    Vec res_norms = compute_residual_norms_inf(Ax_old, Bx_old, Qx_init);
    auto solving_start = std::chrono::steady_clock::now();

    // Build the Newton system
    SSN<T> NS(Q_info, Q_diag, L, L_tr,
            A, B, A_tr, B_tr, c, b, D1A_diag, D1B_diag, D2_diag,
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
        Vec x_old = x;
        Vec y2_old = y2;
        NS.update_SSN_system(x, y1, y2, z, delta_y1, delta_z, mu, rho, gamma, SSN_iter);
        SSN_result<T> NS_solution = NS.solve_SSN(SSN_tol);

        SSN_iter += NS_solution.iter;
        SSN_tol_achieved = NS_solution.tol_achieved;

        // If SSN was not "successful", discard the recent SSN and revert to solution from previous PMM iter.
        if (SSN_iter >= SSN_max_iter) { // Max total SSN iteration is reached
            opt = 2;
            break;
        } else if (NS_solution.opt == 3) { // Linesearch failed
            linesearch_fail++;
            // std::cout << std::setw(8) << PMM_iter << std::setw(8) << SSN_iter << ": Linesearch failed with mu = " << mu << ", rho = " << rho << ".\n";
            if (mu == mu0 && rho == mu0) { opt = 3; break; }
            else {
                mu = std::max(mu0, 0.9 * mu);
                rho = std::max(mu0, 0.9 * rho);
                // SSN_tol = std::max(eps_limit, 0.9 * SSN_tol);
                // PMM_iter++;
                continue;
            }
        }

        // Update x and store candidate y2.
        x = NS_solution.x;
        y2_hat = NS_solution.y2;

        // Compute Ax, Bx, Qx for the new x.
        Vec Ax = M > 0 ? Vec(A * x) : Vec::Zero(M);
        Vec Bx = l  > 0 ? Vec(B * x) : Vec::Zero(l);
        Vec Qx = Q_info != 0 ? Vec(Q_diag.cwiseProduct(x)) : Vec::Zero(N);
        Vec Adx = Ax - Ax_old;
        Vec Bdx = Bx - Bx_old;

        // Update y2 and penalty parameters.
        // T p = compute_p(x); // L_inf primal feasibility violation of lw <= Bx <= uw
        // update_with_bcl(y2_hat, res_norms(3), p, PMM_iter);
        y2 = y2_hat;

        // Update y1 and z.
        delta_y1 = -mu * (Ax - b);
        delta_z = mu * (x - proj(z / mu + x, lx, ux));
        y1 += delta_y1;
        z += delta_z;

        // Primal/dual infeasibility checks.
        if (primal_infeas(delta_y1, y2 - y2_old, delta_z)) {
            opt = -2; std::cout << "Primal infeasible.\n"; break;
        }
        if (dual_infeas(x - x_old, Adx, Bdx)) {
            opt = -3; std::cout << "Dual infeasible.\n"; break;
        }

        // Update PMM parameters based on the progress of residual norms and SSN solve quality.
        Vec new_res_norms = compute_residual_norms_inf(Ax, Bx, Qx);
        PMM_tol_achieved = new_res_norms.maxCoeff();
        update_PMM_parameters(res_norms, new_res_norms, NS_solution.opt, SSN_tol_achieved);
        res_norms = new_res_norms; // for next iteration
        PMM_tol_achieved = res_norms.maxCoeff();

        // Ruiz-descale and shrink to the original dimension (n, m, l) for printing.
        printable_sol(x, y1, y2, z); // (Modifies x_sol, y1_sol, y2_sol, z_sol.)
        obj_val = objective_value(x, Qx);

        // Check termination criterion.
        if (PMM_tol_achieved < tol) {
            opt = 0; // Optimal solution found
            if (when != PrintWhen::NEVER || when != PrintWhen::ALWAYS) {
                print(PrintWhen::ALWAYS, what, PMM_iter, SSN_iter, obj_val, res_norms, SSN_tol_achieved, mu, rho, eps_bcl, SSN_tol, linesearch_fail);
            }
            break;
        }

        // Print current iteration info.
        print(when, what, PMM_iter, SSN_iter, obj_val, res_norms, SSN_tol_achieved, mu, rho, eps_bcl, SSN_tol, linesearch_fail);
        PMM_iter++;

        // Carry Ax, Bx forward.
        Ax_old = std::move(Ax);
        Bx_old = std::move(Bx);

        // TIMER FOR PMM ITERATION
        auto t1_pmm = std::chrono::steady_clock::now();
        double timer_pmm = time_diff_ms(t0_pmm, t1_pmm);
        // std::cout << "PMM iteration took " << timer_pmm << " ms.\n";
        // std::cout << "=====================================================\n";

        auto solving_current = std::chrono::steady_clock::now();
        double solving_current_time = time_diff_ms(solving_start, solving_current) * 1e-3; // in seconds
        if (solving_current_time > time_limit) {
            opt = 4; // Time limit reached
            break;
        }

    }

    // Check if max number of PMM iterations is reached.
    if (opt == -1) { opt = 1; }

    // Check if infeasiblity is detected.
    if (opt == -2 || opt == -3) {
        obj_val = 1e20;
        res_norms = 1e20 * Vec::Ones(4);
        PMM_tol_achieved = 1e20;
        SSN_tol_achieved = 1e20;
        x_sol = Vec::Zero(n);
        y1_sol = Vec::Zero(m);
        y2_sol = Vec::Zero(l);
        z_sol = Vec::Zero(n);
    }

    auto solving_end = std::chrono::steady_clock::now();
    double solving_time = time_diff_ms(solving_start, solving_end) * 1e-3; // in seconds

    Krylov_iter = NS.Krylov_iter;
    fact = NS.fact;

    print(when, what, PMM_iter, SSN_iter, obj_val, res_norms, SSN_tol_achieved, mu, rho, eps_bcl, SSN_tol, linesearch_fail);
    return Solution<T>(opt, x_sol, y1_sol, y2_sol, z_sol, obj_val, PMM_iter, SSN_iter, Krylov_iter, fact, PMM_tol_achieved, SSN_tol_achieved, solving_time, linesearch_fail);
}
