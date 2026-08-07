#pragma once
#include <iostream>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <chrono>
#include <limits>
#include "SSN.hpp"

template <typename T>
void SSN_PMM<T>::get_Q_info(const SpMat& Q) {
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    if (Q.rows() != Q.cols()) {
        throw std::invalid_argument("Given Q is not a square matrix (n x n).");
    }

    if (Q.nonZeros() == 0 || Q.rows() == 0) {
        Q_info = 0; // Q is zero.
    } else {
        Q_info = 1; // Q is diagonal until proven otherwise.
        for (int k = 0; k < Q.outerSize() && Q_info < 2; ++k) {
            for (typename SpMat::InnerIterator it(Q, k); it; ++it) {
                if (it.row() != it.col() && it.value() != T(0)) {
                    Q_info = 2;
                    break;
                }
            }
        }
    }

    if (Q_info == 1) {
        problem_Q_diag = Q.diagonal();
    }
}

template <typename T>
void SSN_PMM<T>::determine_dimensions(const Problem<T>& problem) {
    // Determine n.
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

    // Determine m.
    if (problem.A.rows() != 0) {
        m = problem.A.rows();
    } else if (problem.b.size() != 0) {
        m = problem.b.size();
    } else {
        m = 0;
    }

    // Determine l.
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

    // Check dimensions consistency.
    if (Q_info == 1 && problem_Q_diag.size() != n) {
        throw std::invalid_argument("Dimension mismatch: Q diagonal should be a vector of size n.");
    }
    if (Q_info == 2 && (problem.Q.rows() != n || problem.Q.cols() != n)) {
        throw std::invalid_argument("Dimension mismatch: Q should be an n x n matrix.");
    }
    if (problem.c.size() != 0 && problem.c.size() != n) {
        std::cout << "n = " << n << ", but c.size() = " << problem.c.size() << "\n";
        throw std::invalid_argument("Dimension mismatch: c should be a vector of size n.");
    }
    if ((problem.A.rows() != 0 || problem.A.cols() != 0) && (problem.A.rows() != m || problem.A.cols() != n)) {
        throw std::invalid_argument("Dimension mismatch: A should be m x n.");
    }
    if (problem.b.size() != 0 && problem.b.size() != m) {
        throw std::invalid_argument("Dimension mismatch: b should be a vector of size m.");
    }
    if ((problem.B.rows() != 0 || problem.B.cols() != 0) && (problem.B.rows() != l || problem.B.cols() != n)) {
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
    /*
    drA = Diag(sqrt(||A_{i.}||_inf))_i; drB = Diag(sqrt(||B_{i.}||_inf))_i; dc = Diag(sqrt(||[A; B; I]_{.j}||_inf))_j;
    D1A <- D1A / drA; D1B <- D1B / drB; D2 <- D2 / dc;
    A <- D1A A D2; B <- D1B B D2; c <- D2 c; b <- D1A b; (lx ux) <- D2^{-1} (lx, ux); (lw, uw) <- D1B (lw, uw).

    Given scaled (x, y1, y2, z), unscale them as follows:
    x_unscaled = D2 x; y1_unscaled = D1A y1; y2_unscaled = D1B y2; z_unscaled = D2^{-1} z.
    */
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    const int max_ruiz_iter = 10;
    const T ruiz_tol = 1e-3;
    const T eps = std::sqrt(std::numeric_limits<T>::epsilon()); // for sqrt of near-zero row/column max.

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

        // Compute row-/column-wise infinity norms of constraint matrices.
        Vec row_max_A = Vec::Zero(m); // row_max(i) = max_j |A(i,j)|
        Vec row_max_B = Vec::Zero(l); // row_max(i) = max_j |B(i,j)|
        Vec col_max = Vec::Ones(n);   // col_max(j) = max_i |[A; B; I](i,j)| (starts at 1 due to I)

        // Contribution from A.
        for (int col = 0; col < n; ++col) {
            for (typename SpMat::InnerIterator it(A_ruiz, col); it; ++it) {
                const int i = it.row();
                const T val = std::abs(it.value());
                if (val > row_max_A(i)) row_max_A(i) = val;
                if (val > col_max(col)) col_max(col) = val;
            }
        }

        // Contribution from B.
        for (int col = 0; col < n; ++col) {
            for (typename SpMat::InnerIterator it(B_ruiz, col); it; ++it) {
                const int i = it.row();
                const T val = std::abs(it.value());
                if (val > row_max_B(i)) row_max_B(i) = val;
                if (val > col_max(col)) col_max(col) = val;
            }
        }

        // Check convergence on [A; B; I].
        T row_dev = T(0);
        if (m > 0) row_dev = std::max(row_dev, (row_max_A.array() - T(1)).abs().maxCoeff());
        if (l > 0) row_dev = std::max(row_dev, (row_max_B.array() - T(1)).abs().maxCoeff());
        T col_dev = (col_max.array() - T(1)).abs().maxCoeff();
        if (row_dev < ruiz_tol && col_dev < ruiz_tol) break;

        // Scaling factors: dr, dc = sqrt(max_norms).
        Vec drA(m), drA_inv(m);
        Vec drB(l), drB_inv(l);
        Vec dc(n),  dc_inv(n);
        for (int i = 0; i < m; ++i) {
            drA(i)     = (row_max_A(i) > eps) ? std::sqrt(row_max_A(i)) : T(1);
            drA_inv(i) = T(1) / drA(i);
        }
        for (int i = 0; i < l; ++i) {
            drB(i)     = (row_max_B(i) > eps) ? std::sqrt(row_max_B(i)) : T(1);
            drB_inv(i) = T(1) / drB(i);
        }
        for (int j = 0; j < n; ++j) {
            dc(j)     = (col_max(j) > eps) ? std::sqrt(col_max(j)) : T(1);
            dc_inv(j) = T(1) / dc(j);
        }

        // Scale A: A <- D1A A D2.
        for (int j = 0; j < n; ++j) {
            const T col_fac = dc_inv(j);
            for (typename SpMat::InnerIterator it(A_ruiz, j); it; ++it) {
                const T row_fac = drA_inv(it.row());
                it.valueRef() *= row_fac * col_fac;
            }
        }
        // Scale B: B <- D1B B D2.
        for (int j = 0; j < n; ++j) {
            const T col_fac = dc_inv(j);
            for (typename SpMat::InnerIterator it(B_ruiz, j); it; ++it) {
                const T row_fac = drB_inv(it.row());
                it.valueRef() *= row_fac * col_fac;
            }
        }

        // Scale I: I <- D1 I D2;
        // this is represented through the variable substitution and scaling of lx, ux below.

        // Scale Q if Q is nonzero: Q <- D2 Q D2.
        if (Q_info == 2) {
            for (int j = 0; j < n; ++j) {
                const T col_fac = dc_inv(j);
                for (typename SpMat::InnerIterator it(Q_ruiz, j); it; ++it) {
                    const T row_fac = dc_inv(it.row());
                    it.valueRef() *= row_fac * col_fac; // Q_ij *= 1/dc_i * 1/dc_j
                }
            }
        } else if (Q_info == 1) {
            Q_diag_ruiz.array() *= dc_inv.array().square(); // Q_ii *= 1/dc_i * 1/dc_i
        }

        // Scale c: c <- D2 c.
        if (c_ruiz.size() == n) c_ruiz.array() *= dc_inv.array();

        // Scale b: b <- D1A b.
        if (b_ruiz.size() == m) b_ruiz.array() *= drA_inv.array();

        // Scale lw, uw: (lw, uw) <- D1B (lw, uw).
        const bool has_lw = (lw_ruiz.size() == l);
        const bool has_uw = (uw_ruiz.size() == l);
        if (has_lw || has_uw) {
            for (int i = 0; i < l; ++i) {
                const T di = drB_inv(i);
                if (has_lw && lw_ruiz(i) > -inf) lw_ruiz(i) *= di;
                if (has_uw && uw_ruiz(i) <  inf) uw_ruiz(i) *= di;
            }
        }

        // Scale lx, ux: (lx, ux) <- D2^{-1} (lx, ux).
        const bool has_lx = (lx_ruiz.size() == n);
        const bool has_ux = (ux_ruiz.size() == n);
        if (has_lx || has_ux) {
            for (int i = 0; i < n; ++i) {
                const T di = dc(i);
                if (has_lx && lx_ruiz(i) > -inf) lx_ruiz(i) *= di;
                if (has_ux && ux_ruiz(i) <  inf) ux_ruiz(i) *= di;
            }
        }

        // Accumulate scaling factors (D <- D / diag(d)).
        if (m > 0) D1A_diag.array() *= drA_inv.array();
        if (l > 0) D1B_diag.array() *= drB_inv.array();
        D2_diag.array() *= dc_inv.array();

    }
}

template <typename T>
void SSN_PMM<T>::set_L_from_LLT(const SpMat& Q) {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;

    const int n = Q.rows();
    const T delta = std::sqrt(std::numeric_limits<T>::epsilon());

    // Regularization for numerical stability.
    SpMat I(n, n);
    I.setIdentity();
    SpMat Q_reg = Q + delta * I;
    Q_reg.makeCompressed();

    Eigen::SimplicialLDLT<SpMat> ldlt;
    ldlt.compute(Q_reg);

    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("LDLT factorization on Q failed. Q is possibly singular.");
    }

    // Clamp negative D values to 0.
    Vec D = ldlt.vectorD();
    T min_D = D.minCoeff();
    if (min_D < -delta) {
        std::cerr << "[warn] Q has negative LDLT diagonal entry " << min_D
                  << "; treating as PSD (clamping to 0).\n";
    }
    Vec D_sqrt = D.cwiseMax(T(0)).cwiseSqrt();

    auto P = ldlt.permutationP();
    SpMat L_D = ldlt.matrixL(); // lower triangular from LDL^T

    // Diagonal scaling.
    L = (P.transpose() * L_D) * D_sqrt.asDiagonal();
}

template <typename T>
void SSN_PMM<T>::build_reformulated_vecs(int n, int m, int N, int M, T inf,
                                         const Vec& c_in, const Vec& b_in, const Vec& lx_in, const Vec& ux_in,
                                         Vec& c_out, Vec& b_out, Vec& lx_out, Vec& ux_out) {
     if (N == n) { // Q diagonal or zero: no reformulation

        if (c_in.size() == 0) c_out = Vec::Zero(N);
        else c_out = c_in;

        if (b_in.size() != 0) b_out = b_in;

        if (lx_in.size() == 0) lx_out = Vec::Constant(N, -inf);
        else lx_out = lx_in;

        if (ux_in.size() == 0) ux_out = Vec::Constant(N, inf);
        else ux_out = ux_in;
        
    } else { // general Q: reformulate to N = 2n, M = m + n

        if (c_in.size() == 0) {
            c_out = Vec::Zero(N);
        } else {
            c_out.resize(N);
            c_out << c_in, Vec::Zero(n);
        }
        if (b_in.size() == 0) {
            b_out = Vec::Zero(M);
        } else {
            b_out.resize(M);
            b_out << b_in, Vec::Zero(n);
        }
        if (lx_in.size() == 0) {
            lx_out = Vec::Constant(N, -inf);
        } else {
            lx_out.resize(N);
            lx_out << lx_in, Vec::Constant(n, -inf);
        }
        if (ux_in.size() == 0) {
            ux_out = Vec::Constant(N, inf);
        } else {
            ux_out.resize(N);
            ux_out << ux_in, Vec::Constant(n, inf);
        }
    }
}

template <typename T>
void SSN_PMM<T>::set_default(const Problem<T>& problem) {
    using SpMat   = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;

    // For general real, symmetric positive semidefinite Q, we use the reformulation:
    if (Q_info == 2) {
        N = 2 * n; M = m + n;

        // L s.t. Q = LL^T
        set_L_from_LLT(Q_ruiz);

        // Q = [0 0; 0 I_n]
        Q_diag.resize(N);
        Q_diag << Vec::Zero(n), Vec::Ones(n);

        // A = [A_ruiz, 0; L^T, -I]
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

        // B = [B_ruiz, 0]
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

    } else { // Q is diagonal or zero; no reformulation needed.
        N = n; M = m;

        Q_diag = Q_diag_ruiz;

        if (problem.A.rows() == 0 || problem.A.cols() == 0) {
            A = SpMat(M, N);
        } else {
            A = A_ruiz;
        }

        if (problem.B.rows() == 0 || problem.B.cols() == 0) {
            B = SpMat(l, N);
        } else {
            B = B_ruiz;
        }
    }

    // Set up scaled and unscaled data, reformulated in case of general non-diagonal Q.
    build_reformulated_vecs(n, m, N, M, inf, c_ruiz, b_ruiz, lx_ruiz, ux_ruiz, c, b, lx, ux);
    // c_orig is set in the constructor.
    Vec place_holder;
    build_reformulated_vecs(n, m, N, M, inf, problem.c, problem.b, problem.lx, problem.ux, place_holder, b_orig, lx_orig, ux_orig);

    // lw, uw remain the same for any Q
    if (problem.lw.size() == 0) {
        lw = Vec::Constant(l, -inf);
        lw_orig = Vec::Constant(l, -inf);
    } else {
        lw = lw_ruiz;
        lw_orig = problem.lw;
    }
    if (problem.uw.size() == 0) {
        uw = Vec::Constant(l, inf);
        uw_orig = Vec::Constant(l, inf);
    } else {
        uw = uw_ruiz;
        uw_orig = problem.uw;
    }

    // Extended scaling vectors for general Q reformulation.
    D1A_ext = Vec::Ones(M); D1A_ext.head(m) = D1A_diag;
    D2_ext = Vec::Ones(N); D2_ext.head(n) = D2_diag;
    D1A_ext_inv = D1A_ext.cwiseInverse();
    D2_ext_inv = D2_ext.cwiseInverse();
    D1B_diag_inv = D1B_diag.cwiseInverse();

}

template <typename T>
void SSN_PMM<T>::initialize_sols() { // using 0 vectors
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;
    using Triplet = typename SSN<T>::Triplet;

    x = Vec::Zero(N);
    y1 = Vec::Zero(M);
    y2 = Vec::Zero(l);
    z = Vec::Zero(N);

    Ax_scratch_     = Vec::Zero(M);
    Bx_scratch_     = Vec::Zero(l);
    Qx_scratch_     = Vec::Zero(N);
    Ax_old_scratch_ = Vec::Zero(M);
    Bx_old_scratch_ = Vec::Zero(l);
    Adx_scratch_    = Vec::Zero(M);
    Bdx_scratch_    = Vec::Zero(l);
    x_old_scratch_  = Vec::Zero(N);
    y2_old_scratch_ = Vec::Zero(l);

    A_tr_y1_scratch_        = Vec::Zero(N);
    B_tr_y2_scratch_        = Vec::Zero(N);
    num_scratch_            = Vec::Zero(N);
    proj_K_scratch_         = Vec::Zero(N);
    proj_K_unscaled_scratch_= Vec::Zero(N);
    proj_W_scratch_         = Vec::Zero(l);
    proj_W_unscaled_scratch_= Vec::Zero(l);
    Ax_unscaled_scratch_    = Vec::Zero(M);
    z_unscaled_scratch_     = Vec::Zero(N);
    num_unscaled_scratch_   = Vec::Zero(N);
    x_unscaled_scratch_     = Vec::Zero(N);
    Bx_unscaled_scratch_    = Vec::Zero(l);
    y2_unscaled_scratch_    = Vec::Zero(l);
}

template <typename T>
void SSN_PMM<T>::check_bounds() {
    // Check lower and upper bounds.
    for (int i = 0; i < n; ++i) {
        if (lx_orig(i) > ux_orig(i)) {
            throw std::invalid_argument("Problem is infeasible: lx should be <= ux.");
        }
    }
    for (int i = 0; i < l; ++i) {
        if (lw_orig(i) > uw_orig(i)) {
            throw std::invalid_argument("Problem is infeasible: lw should be <= uw.");
        }
    }
}

template <typename T>
typename SSN_PMM<T>::ResVec SSN_PMM<T>::compute_residual_unscaled_inf_norms(const Vec& Ax, const Vec& Bx, const Vec& Qx, ResVec& res_norms_scaled) {
    // Return the unscaled residual norms (primal, dual, complementarity for x, complementarity for Bx),
    // and update the scaled norms in-place.
    Vec& A_tr_y1 = A_tr_y1_scratch_;
    Vec& B_tr_y2 = B_tr_y2_scratch_;

    // ===== Scaled residual norms =====
    T res_p; // Primal residual norm
    if (M == 0) res_p = T(0);
    else {
        T denom = 1 + std::max(inf_norm(Ax), inf_norm(b));
        res_p = inf_norm(Ax - b) / denom;
    }

    // Dual residual norm
    Vec& num = num_scratch_;
    num.noalias() = c + z;
    T denom = std::max(inf_norm(c), inf_norm(z));
    if (Q_info != 0) {
        num += Qx;
        denom = std::max(denom, inf_norm(Qx));
    }
    if (M != 0) {
        A_tr_y1.noalias() = A_tr * y1;
        num -= A_tr_y1;
        denom = std::max(denom, inf_norm(A_tr_y1));
    }
    if (l != 0) {
        B_tr_y2.noalias() = B_tr * y2;
        num -= B_tr_y2;
        denom = std::max(denom, inf_norm(B_tr_y2));
    }
    denom += T(1); // denom = 1 + max(||c||_inf, ||z||_inf, ||Qx||_inf, ||A^T y1||_inf, ||B^T y2||_inf)
    T res_d = inf_norm(num) / denom;

    // Complementarity residual norm for box constraints on x
    Vec& proj_K = proj_K_scratch_;
    proj_K.noalias() = proj(x + z, lx, ux);
    T compl_x = inf_norm(x - proj_K);
    compl_x /= (T(1) + std::max(inf_norm(z), inf_norm(proj_K)));

    // Complementarity residual norm for Bx constraints
    T compl_w;
    if (l == 0) compl_w = T(0);
    else {
        Vec& proj_W = proj_W_scratch_;
        proj_W.noalias() = proj(Bx - y2, lw, uw);
        compl_w = inf_norm(Bx - proj_W);
        compl_w /= (T(1) + std::max(inf_norm(y2), inf_norm(proj_W)));
    }

    res_norms_scaled << res_p, res_d, compl_x, compl_w; // Update scaled residual norms in-place.

    // ===== Unscaled residual norms =====
    T res_p_unscaled; // Primal residual norm
    if (M == 0) res_p_unscaled = T(0);
    else {
        Vec& Ax_unscaled = Ax_unscaled_scratch_;
        Ax_unscaled.noalias() = Ax.cwiseProduct(D1A_ext_inv);
        T denom_unscaled = T(1) + std::max(inf_norm(Ax_unscaled), inf_norm(b_orig));
        res_p_unscaled = inf_norm(Ax_unscaled - b_orig) / denom_unscaled;
    }

    // Dual residual norm
    Vec& z_unscaled = z_unscaled_scratch_;
    Vec& num_unscaled = num_unscaled_scratch_;
    z_unscaled.noalias() = z.cwiseProduct(D2_ext_inv);
    num_unscaled.noalias() = num.cwiseProduct(D2_ext_inv);
    T denom_unscaled = std::max(inf_norm(c_orig), inf_norm(z_unscaled));
    if (Q_info != 0) denom_unscaled = std::max(denom_unscaled, inf_norm(Qx.cwiseProduct(D2_ext_inv)));
    if (M != 0)      denom_unscaled = std::max(denom_unscaled, inf_norm(A_tr_y1.cwiseProduct(D2_ext_inv)));
    if (l != 0)      denom_unscaled = std::max(denom_unscaled, inf_norm(B_tr_y2.cwiseProduct(D2_ext_inv)));
    denom_unscaled += T(1);
    T res_d_unscaled = inf_norm(num_unscaled) / denom_unscaled;

    // Complementarity residual norm for box constraints on x
    Vec& x_unscaled = x_unscaled_scratch_;
    x_unscaled.noalias() = x.cwiseProduct(D2_ext);
    Vec& proj_K_unscaled = proj_K_unscaled_scratch_;
    proj_K_unscaled.noalias() = proj(x_unscaled + z_unscaled, lx_orig, ux_orig);
    T compl_x_unscaled = inf_norm(x_unscaled - proj_K_unscaled) / (T(1) + std::max(inf_norm(z_unscaled), inf_norm(proj_K_unscaled)));

    // Complementarity residual norm for box constraints on Bx
    T compl_w_unscaled;
    if (l == 0) compl_w_unscaled = T(0);
    else {
        Vec& Bx_unscaled = Bx_unscaled_scratch_;
        Vec& y2_unscaled = y2_unscaled_scratch_;
        Bx_unscaled.noalias() = Bx.cwiseProduct(D1B_diag_inv);
        y2_unscaled.noalias() = y2.cwiseProduct(D1B_diag);
        Vec& proj_W_unscaled = proj_W_unscaled_scratch_;
        proj_W_unscaled.noalias() = proj(Bx_unscaled - y2_unscaled, lw_orig, uw_orig);
        compl_w_unscaled = inf_norm(Bx_unscaled - proj_W_unscaled) / (T(1) + std::max(inf_norm(y2_unscaled), inf_norm(proj_W_unscaled)));
    }

    ResVec res_norms_unscaled; // Return unscaled residual norms.
    res_norms_unscaled << res_p_unscaled, res_d_unscaled, compl_x_unscaled, compl_w_unscaled;
    return res_norms_unscaled;
}

template <typename T>
T SSN_PMM<T>::objective_value(const Vec& x_orig) {
    T obj_val = obj_const + c_orig.dot(x_orig);
    if (Q_info != 0) {
        Vec Qx = Q.template selfadjointView<Eigen::Lower>() * x_orig;
        obj_val += T(0.5) * x_orig.dot(Qx);
    }
    return obj_val;
}

template <typename T>
void SSN_PMM<T>::printable_sol(const Vec& x, const Vec& y1, const Vec& y2, const Vec& z) {
    // Ruiz descale, and shrink to original dimension if needed
    if (Q_info == 2) {
        x_sol = x.head(n).cwiseProduct(D2_diag);
        y1_sol = y1.head(m).cwiseProduct(D1A_diag);
        y2_sol = y2.head(l).cwiseProduct(D1B_diag);
        z_sol = z.head(n).cwiseQuotient(D2_diag);
    } else {
        x_sol = x.cwiseProduct(D2_diag);
        y1_sol = y1.cwiseProduct(D1A_diag);
        y2_sol = y2.cwiseProduct(D1B_diag);
        z_sol = z.cwiseQuotient(D2_diag);
    }
}

template <typename T>
void SSN_PMM<T>::update_PMM_parameters(const ResVec& res_norms, const ResVec& new_res_norms, int ssn_opt, T ssn_res, int ssn_inner_iters) {

    T worst_res = new_res_norms.maxCoeff();
    
    if (ssn_opt != 3) {
        mu = std::min(mu_limit, T(1.2) * mu);
        rho = std::min(rho_limit, T(10) * rho);
        if (ssn_opt == 0) {
            ssn_tol = std::max(eps_limit, T(0.1) * ssn_tol);
        }

    } else {
        // Linesearch failed.
        mu = std::max(mu0, T(0.5) * mu);
        rho = std::max(rho0, T(0.5) * rho);
        ssn_tol = std::min({worst_res, T(1.1) * ssn_tol, T(1e-2)});

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

    Infeasibility is determined in unscaled scope.
    */
    T cert_inf = std::max({M > 0 ? inf_norm(cert_y1.cwiseProduct(D1A_ext)) : T(0),
                           l > 0 ? inf_norm(cert_y2.cwiseProduct(D1B_diag)) : T(0),
                           inf_norm(cert_z.cwiseQuotient(D2_ext))}); // Unscaled certificate norms
    if (cert_inf < T(100) * std::numeric_limits<T>::epsilon()) return false;

    // Condition 2
    T lhs2 = T(0); // scale-invariant
    if (M > 0) lhs2 -= b.dot(cert_y1);
    for (int i = 0; i < l; ++i) {
        const T cy2i = -cert_y2(i);
        if (uw(i) < inf)  lhs2 += uw(i) * std::max(cy2i, T(0));
        if (lw(i) > -inf) lhs2 += lw(i) * std::min(cy2i, T(0));
    }
    for (int i = 0; i < N; ++i) {
        const T czi = cert_z(i);
        if (ux(i) < inf)  lhs2 += ux(i) * std::max(czi, T(0));
        if (lx(i) > -inf) lhs2 += lx(i) * std::min(czi, T(0));
    }
    if (lhs2 > -eps_pinf * cert_inf) return false;

    // Condition 1
    Vec lhs1 = -cert_z;
    if (M > 0) lhs1 += A_tr * cert_y1;
    if (l  > 0) lhs1 += B_tr * cert_y2;
    return inf_norm(lhs1.cwiseQuotient(D2_ext)) <= eps_pinf * cert_inf;
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

    Infeasibility is determined in unscaled scope.
    */
    const T delta_x_inf = inf_norm(delta_x.cwiseProduct(D2_ext));
    if (delta_x_inf < eps_zero) return false;
    const T rhs = eps_dinf * delta_x_inf;

    // Conditions 1, 2, 3
    if (Q_info != 0 && inf_norm(Q_diag.cwiseProduct(delta_x).cwiseQuotient(D2_ext)) > rhs) return false;
    if (c.dot(delta_x) > -rhs) return false;
    if (M > 0 && inf_norm(Adx.cwiseQuotient(D1A_ext)) > rhs) return false;

    // Condition 4
    for (int i = 0; i < N; ++i) {
        const bool has_lx = lx(i) > -inf;
        const bool has_ux = ux(i) < inf;
        const T dx_i_unscaled = delta_x(i) * D2_ext(i);
        if (has_lx && has_ux) { if (std::abs(dx_i_unscaled) > rhs) return false; }
        else if (has_lx)      { if (dx_i_unscaled < -rhs)          return false; }
        else if (has_ux)      { if (dx_i_unscaled >  rhs)          return false; }
    }

    // Condition 5
    for (int i = 0; i < l; ++i) {
        const bool has_lw = lw(i) > -inf;
        const bool has_uw = uw(i) < inf;
        const T Bdx_i_unscaled = Bdx(i) / D1B_diag(i);
        if (has_lw && has_uw) { if (std::abs(Bdx_i_unscaled) > rhs) return false; }
        else if (has_lw)      { if (Bdx_i_unscaled < -rhs)          return false; }
        else if (has_uw)      { if (Bdx_i_unscaled >  rhs)          return false; }
    }
    return true;
}

template <typename T>
Solution<T> SSN_PMM<T>::solve() {
    auto solving_start = std::chrono::steady_clock::now();

    // If an error occurred during setup, exit immediately with opt = -1.
    if (setup_failed) {
        auto solving_end = std::chrono::steady_clock::now();
        double solve_time = time_diff_s(solving_start, solving_end); // in seconds
        return Solution<T>(opt, Vec::Zero(n), Vec::Zero(m), Vec::Zero(l), Vec::Zero(n),
                            T(1e20), 0, 0, 0, 0, 0, T(1e20), T(1e20),
                            setup_time, solve_time, 0, 0);
    }

    // Initialize variables.
    opt = -1;
    pmm_iter = 0;
    ssn_iter = 0;
    ssn_tol_achieved = T(0);
    bool solve_error = false;

    Vec delta_y1 = Vec::Zero(M);
    Vec delta_z = Vec::Zero(N);

    if (M > 0) Ax_old_scratch_.noalias() = A * x; else Ax_old_scratch_.setZero();
    if (l  > 0) Bx_old_scratch_.noalias() = B * x; else Bx_old_scratch_.setZero();
    if (Q_info != 0) Qx_scratch_.noalias() = Q_diag.cwiseProduct(x); else Qx_scratch_.setZero();

    ResVec res_norms = compute_residual_unscaled_inf_norms(Ax_old_scratch_, Bx_old_scratch_, Qx_scratch_, res_norms_scaled);
    pmm_tol_achieved = res_norms.maxCoeff();

    // Build the Newton system.
    SSN<T> NS(Q_info, Q_diag, L,
            A, B, A_tr, B_tr, c, b,
            D2_ext_inv, D1B_diag_inv,
            lx, ux, lw, uw,
            n, m, N, M, l,
            ssn_tol, ssn_max_in_iter,
            eps_pinf, eps_dinf,
            when, what);

    // Print header
    print_header(when, what);

    try {
    // SSN-PMM main loop
    while (pmm_iter < max_iter) {
        // ----------------------------------------------
        // Structure:
        // Until (primal infeasibility, dual infeasibility, complementarity) < tol, do:
        //     1) Call Semismooth Newton method to approximately minimize the augmented Lagrangian w.r.t. x and y2;
        //     2) Update multipliers y1, z if SSN solve is accurate enough;
        //     3) Update penalty parameters mu, rho;
        //     k = k + 1;
        // End
        // ----------------------------------------------

        // Call semismooth Newton method.
        x_old_scratch_  = x;
        y2_old_scratch_ = y2;
        NS.update_ssn_system(x, y1, y2, z, delta_y1, delta_z, mu, rho, alpha, ssn_iter);
        NS.solve_ssn(ssn_tol);

        ssn_iter += NS.iter;
        ssn_tol_achieved = NS.tol_achieved;
        linesearch_fail += NS.linesearch_fail;

        // If SSN reached max total iteratioins, terminate.
        if (ssn_iter >= ssn_max_iter) { 
            opt = 2;
            break;
        }

        // Accept x and y2.
        x = NS.x;
        y2 = NS.y2;

        // Compute Ax, Bx, Qx for the new x.
        if (M > 0) Ax_scratch_.noalias() = A * x; else Ax_scratch_.setZero();
        if (l > 0) Bx_scratch_.noalias() = B * x; else Bx_scratch_.setZero();
        if (Q_info != 0) Qx_scratch_.noalias() = Q_diag.cwiseProduct(x); else Qx_scratch_.setZero();
        Adx_scratch_.noalias() = Ax_scratch_ - Ax_old_scratch_;
        Bdx_scratch_.noalias() = Bx_scratch_ - Bx_old_scratch_;

        // Infeasibility checks
        if (primal_infeas(delta_y1, y2 - y2_old_scratch_, delta_z)) {
            opt = -2; std::cout << "[Infeasibility] Primal infeasible.\n";
            break;
        }
        if (dual_infeas(x - x_old_scratch_, Adx_scratch_, Bdx_scratch_)) {
            opt = -3; std::cout << "[Infeasibility] Dual infeasible.\n";
            break;
        }

        // Update multipliers y1, z if SSN solve is accurate enough.
        if (NS.opt == 0 || ssn_tol_achieved <= T(100) * pmm_tol_achieved) {
            delta_y1 = -mu * (Ax_scratch_ - b);
            y1 += delta_y1;
            delta_z = mu * (x - proj(z / mu + x, lx, ux));
            z += delta_z;
        } 
        // else, keep y1, z from previous PMM iteration

        // Compute new residual norms.
        ResVec new_res_norms = compute_residual_unscaled_inf_norms(Ax_scratch_, Bx_scratch_, Qx_scratch_, res_norms_scaled);
        pmm_tol_achieved = new_res_norms.maxCoeff();

        // Ruiz-descale and shrink to the original dimension (n, m, l) for printing.
        printable_sol(x, y1, y2, z); // (Modifies x_sol, y1_sol, y2_sol, z_sol.)
        obj_val = objective_value(x_sol);

        pmm_iter++;
        
        // Print current iteration info.
        print(when, what, pmm_iter, ssn_iter, NS.krylov_iter, NS.fact, obj_val, new_res_norms, ssn_tol_achieved, mu, rho, ssn_tol, linesearch_fail, NS.krylov_fail);
        
        // Check termination criterion.
        if (pmm_tol_achieved < tol) {
            opt = 0; // Optimal solution found.
            break;
        }

        // Update PMM parameters based on SSN solve quality.
        update_PMM_parameters(res_norms, new_res_norms, NS.opt, ssn_tol_achieved, NS.iter);
        res_norms = new_res_norms; // for next iteration

        // Carry Ax, Bx forward.
        Ax_old_scratch_.swap(Ax_scratch_);
        Bx_old_scratch_.swap(Bx_scratch_);

        auto solving_current = std::chrono::steady_clock::now();
        double solving_current_time = time_diff_s(solving_start, solving_current); // in seconds
        if (solving_current_time > time_limit) {
            opt = 4; // Time limit reached.
            break;
        }

    }
    } catch (const std::exception& e) {
        std::cerr << "[SSN_PMM] Solve error: " << e.what() << "\n";
        opt = -1;
        solve_error = true;
    }

    // Check if max number of PMM iterations is reached.
    if (opt == -1 && !solve_error) { opt = 1; }

    // Check if infeasiblity or a numerical error is detected.
    if (opt == -2 || opt == -3 || solve_error) {
        obj_val = 1e20;
        res_norms = ResVec::Constant(1e20);
        pmm_tol_achieved = 1e20;
        ssn_tol_achieved = 1e20;
        x_sol = Vec::Zero(n);
        y1_sol = Vec::Zero(m);
        y2_sol = Vec::Zero(l);
        z_sol = Vec::Zero(n);
    }

    auto solving_end = std::chrono::steady_clock::now();
    double solve_time = time_diff_s(solving_start, solving_end); // in seconds

    krylov_iter = NS.krylov_iter;
    fact = NS.fact;
    krylov_fail = NS.krylov_fail;
    ldlt_used = NS.ldlt_used;

    return Solution<T>(opt, x_sol, y1_sol, y2_sol, z_sol, obj_val, pmm_iter, ssn_iter, krylov_iter, fact, NS.smw_count, pmm_tol_achieved, ssn_tol_achieved, setup_time, solve_time, linesearch_fail, krylov_fail);
}
