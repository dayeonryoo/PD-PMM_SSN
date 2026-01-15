#pragma once

#include <iostream>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "SSN.hpp"

template <typename T>
void SSN_PMM<T>::get_Q_info(const SpMat& Q) {
    using SpMat = Eigen::SparseMatrix<T>;

    if (Q.rows() != Q.cols()) {
        throw std::invalid_argument("Given Q is not a square matrix (n x n).");
    }
    if (!Q.isApprox(Q.transpose(), 1e-8)) {
        throw std::invalid_argument("Given Q is not symmetric.");
    }

    // Is Q = 0?
    if (Q.rows() == 0 || Q.nonZeros() == 0) {
        Q_info = QInfo::Zero;
    } else { // Is Q diagonal or not?
        Q_info = QInfo::Diagonal;
        for (int k = 0; k < Q.outerSize(); ++k) {
            for (typename SpMat::InnerIterator it(Q, k); it; ++it) {
                if (it.row() != it.col()) {
                    Q_info = QInfo::General;
                }
            }
        }
    }
    if (Q_info == QInfo::Zero) {
        std::cout << "Given Q is zero.\n";
    } else if (Q_info == QInfo::Diagonal) {
        std::cout << "Given Q is diagonal.\n";
    } else {
        std::cout << "Given Q is general.\n";
    }

}

template <typename T>
void SSN_PMM<T>::determine_dimensions(const Problem<T>& problem) {

    // Determine n
    if (problem.c.size() != 0) {
        n = problem.c.size();
    } else if (problem.Q.rows() != 0) {
        n = problem.Q.rows();
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
void SSN_PMM<T>::set_default(const Problem<T>& problem) {
    using SpMat   = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;

    // PMM parameters
    mu = 5e1;
    rho = 1e2;
    if (problem.tol == 0.0) tol = 1e-6;
    if (problem.max_iter == 0) max_iter = 1e3;

    // SSN parameters
    SSN_max_iter = 4000;
    SSN_max_in_iter = 40;
    SSN_tol = tol;
    reg_limit = 1e6;

    T inf = std::numeric_limits<T>::infinity();

    if (Q_info == QInfo::General) {
        N = 2 * n;
        M = m + n;

        // c', b', lx', ux'
        if (problem.c.size() == 0) {
            c = Vec::Zero(N);
        } else {
            c.resize(N);
            c << problem.c, Vec::Zero(n);
        }
        if (problem.b.size() == 0) {
            b = Vec::Zero(M);
        } else {
            b.resize(M);
            b << problem.b, Vec::Zero(n);
        }
        if (problem.lx.size() == 0) {
            lx = Vec::Constant(N, -inf);
        } else {
            lx.resize(N);
            lx << problem.lx, Vec::Constant(n, -inf);
        }
        if (problem.ux.size() == 0) {
            ux = Vec::Constant(N, inf);
        } else {
            ux.resize(N);
            ux << problem.ux, Vec::Constant(n, inf);
        }

        // Q' = [0_n 0_n; 0_n I_n]
        Q_diag.resize(N);
        Q_diag << Vec::Zero(n), Vec::Ones(n);

        // L s.t. Q = LL^T
        SpMat Qc = problem.Q;
        Qc.makeCompressed();
        Eigen::SimplicialLLT<SpMat> llt;
        llt.compute(Qc);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("Cholesky factorization of Q failed.");
        }
        L = llt.matrixL();

        // A' = [A 0; L^T -I]
        {
            std::vector<Triplet> trip;
            trip.reserve(problem.A.nonZeros() + L.nonZeros() + n);

            // Top-left block: A
            if (problem.A.rows() != 0 && problem.A.cols() != 0) {
                for (int k = 0; k < problem.A.outerSize(); ++k) {
                    for (typename SpMat::InnerIterator it(problem.A, k); it; ++it) {
                        trip.emplace_back(it.row(), it.col(), it.value());
                    }
                }
            }
            // Bottom-left block: L^T
            for (int k = 0; k < L.outerSize(); ++k) {
                for (typename SpMat::InnerIterator it(L, k); it; ++it) {
                    trip.emplace_back(m + it.col(), it.row(), it.value());
                }
            }
            // Bottom-right block: -I_n
            for (int i = 0; i < n; ++i) {
                trip.emplace_back(m + i, n + i, T(-1));
            }
            A.resize(M, N);
            A.setFromTriplets(trip.begin(), trip.end());
        }

        // B' = [B 0]
        {
            std::vector<Triplet> trip;
            trip.reserve(problem.B.nonZeros());

            if (problem.B.rows() != 0 && problem.B.cols() != 0) {
                for (int k = 0; k < problem.B.outerSize(); ++k) {
                    for (typename SpMat::InnerIterator it(problem.B, k); it; ++it) {
                        trip.emplace_back(it.row(), it.col(), it.value());
                    }
                }
            }
            B.resize(l, N);
            B.setFromTriplets(trip.begin(), trip.end());
        }

    } else {
        N = n;
        M = m;

        if (problem.c.size() == 0) {
            c = Vec::Zero(N);
        } else {
            c = problem.c;
        }
        if (problem.A.rows() == 0 || problem.A.cols() == 0) {
            A = SpMat(M, N);
        } else {
            A = problem.A;
        }
        if (problem.b.size() == 0) {
            b = Vec::Zero(M);
        } else {
            b = problem.b;
        }
        if (problem.B.rows() == 0 || problem.B.cols() == 0) {
            B = SpMat(l, N);
        } else {
            B = problem.B;
        }
        if (problem.lx.size() == 0) {
            lx = Vec::Constant(N, -inf);
        } else {
            lx = problem.lx;
        }
        if (problem.ux.size() == 0) {
            ux = Vec::Constant(N, inf);
        } else {
            ux = problem.ux;
        }
        if (Q_info == QInfo::Diagonal) {
            Q_diag = problem.Q.diagonal();
        }
    }

    // Initial solution
    x = Vec::Zero(N);
    y1 = Vec::Zero(M);
    y2 = Vec::Zero(l);
    z = Vec::Zero(N);

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
void SSN_PMM<T>::check_dimensions() {
    if (c.size() != N) {
        throw std::invalid_argument("Dimension mismatch: c should be a vector of size n.");
    }
    if (A.rows() != M || A.cols() != N) {
        throw std::invalid_argument("Dimension mismatch: A should be m x n.");
    }
    if (b.size() != M) {
        throw std::invalid_argument("Dimension mismatch: b should be a vector of size m.");
    }
    if (B.rows() != l || B.cols() != N) {
        throw std::invalid_argument("Dimension mismatch: B should be l x n.");
    }
    if (lx.size() != N || ux.size() != N) {
        throw std::invalid_argument("Dimension mismatch: lx and ux should be a vector of size n.");
    }
    if (lw.size() != l || uw.size() != l) {
        throw std::invalid_argument("Dimension mismatch: lw and uw should be a vector of size l.");
    }
}

template <typename T>
void SSN_PMM<T>::check_infeasibility() {
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
typename SSN_PMM<T>::Vec SSN_PMM<T>::proj(const Vec& u, const Vec& lower, const Vec& upper) {
    return u.cwiseMax(lower).cwiseMin(upper);
}

template <typename T>
typename SSN_PMM<T>::Vec SSN_PMM<T>::compute_residual_norms() {
    // Primal residual norm
    T res_p = (A * x - b).norm() / (1 + b.norm());

    // Dual residual norm
    T res_d;
    if (Q_info == QInfo::Zero) {
        res_d = (c - A_tr * y1 - B_tr * y2 + z).norm() / (1 + c.norm());
    } else {
        res_d = (c + Q_diag.cwiseProduct(x) - A_tr * y1 - B_tr * y2 + z).norm() / (1 + c.norm());
    }

    // Complementarity residual norm for box constraints
    T compl_x = (x - proj(x + z, lx, ux)).norm();

    // Complementarity residual norm for Bx constraints
    T compl_w = (B * x - proj(B * x - y2, lw, uw)).norm();

    // Collect residual norms
    Vec res_norms(4);
    res_norms << res_p, res_d, compl_x, compl_w;

    return res_norms;
}

template <typename T>
void SSN_PMM<T>::update_PMM_parameters(const T res_p, const T res_d,
                                       const T new_res_p, const T new_res_d) {
    // If the overall primal and dual residual error is decreased,
    // we increase the penalty parameters aggressively.
    // If not, we continue increasing the parameters slowly
    // up to the regularization threshold.

    bool cond_p = 0.95 * res_p > new_res_p;
    bool cond_d = 0.95 * res_d > new_res_d;

    if (cond_p || cond_d){
        mu = std::min(reg_limit, 1.2*mu);
        rho = std::min(1e2*reg_limit, 1.4*rho);
    } else {
        mu = std::min(reg_limit, 1.1*mu);
        rho = std::min(1e2*reg_limit, 1.1*rho);
    };

}

template <typename T>
Solution<T> SSN_PMM<T>::solve() {
    
    // Initialize variables
    opt = -1;
    PMM_iter = 0;
    SSN_iter = 0;

    // Initialize printing functions
    auto printer = make_print_function<T, Vec>(PMM_print_label, PMM_print_when, PMM_print_what, max_iter);

    // Build the Newton system.
    SSN<T> NS(Q_info, Q_diag, L, A, B,
        A_tr, B_tr, c, b,
        lx, ux, lw, uw,
        x, y1, y2, z,
        mu, rho, N, M, l,
        SSN_tol, SSN_max_in_iter,
        SSN_print_when, SSN_print_what);

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

        // Compute residuals
        Vec res_norms = compute_residual_norms();
        PMM_tol_achieved = res_norms.maxCoeff();

        // Primal and dual residuals (needed to update PMM params)
        T res_p = res_norms(0);
        T res_d = res_norms(1);

        // Compute objective value
        if (Q_info == QInfo::Zero) {
            obj_val = c.dot(x);
        } else {
            obj_val = c.dot(x) + 0.5 * Q_diag.cwiseProduct(x).dot(x);
        }

        // Check termination criterion
        if (PMM_tol_achieved < tol) {
            opt = 0; // Optimal solution found
            printer(PMM_iter, opt, obj_val, x, y1, y2, z, PMM_tol_achieved);
            break;
        }
        PMM_iter++;

        // Print current iteration info
        printer(PMM_iter, opt, obj_val, x, y1, y2, z, PMM_tol_achieved);

        // PRINTING ALL RESIDUALS
        std::cout << "  res_p = " << res_p << "\n  res_d = " << res_d
                  << "\n  compl_x = " << res_norms(2) << "\n  compl_w = " << res_norms(3) << "\n";

        // Update the Newton system
        NS.x = x;
        NS.y1 = y1;
        NS.y2 = y2;
        NS.z = z;
        NS.mu = mu;
        NS.rho = rho;

        // Calculate adaptive SSN tolerance eps_k
        Vec res_vec(3);
        res_vec << 0.1 * res_p, 0.1 * res_d, T(1);
        T min_res_vec = res_vec.minCoeff();
        T max_res_vec = res_vec.maxCoeff();
        SSN_tol_achieved = 2 * max_res_vec;
        
        T eps1 = std::max(0.1*max_res_vec, min_res_vec);
        T eps2 = std::max(min_res_vec, SSN_tol);
        std::cout << "  eps_k = " << eps1 << " (outer), " << eps2 << " (inner)\n";

        // Call semismooth Newton method to update x and y2
        while (SSN_tol_achieved > eps1) {
            SSN_result<T> NS_solution = NS.solve_SSN(eps2);
            x = NS_solution.x;
            y2 = NS_solution.y2;

            // Update the Newton system in case of a restart
            NS.x = x;
            NS.y2 = y2;

            SSN_tol_achieved = NS_solution.SSN_tol_achieved;
            SSN_iter += NS_solution.SSN_in_iter;
            if (SSN_iter >= SSN_max_iter) break;
        }
        std::cout << "SSN iter: " << SSN_iter << "\n    tol = " << SSN_tol_achieved << "\n";

        // Update multipliers
        y1 -= mu * (A * x - b);
        z += mu * (x - proj(z / mu + x, lx, ux));

        // Compute the new residual norms
        Vec new_res_norms = compute_residual_norms();
        T new_res_p = new_res_norms(0);
        T new_res_d = new_res_norms(1);

        // Update penalty parameters
        update_PMM_parameters(res_p, res_d, new_res_p, new_res_d);

        // TIMER FOR PMM ITERATION
        auto t1_pmm = std::chrono::steady_clock::now();
        double timer_pmm = time_diff_ms(t0_pmm, t1_pmm);
        std::cout << "PMM iteration took " << timer_pmm << " ms.\n";
        std::cout << "=====================================================\n";

    }
    if (opt != 0) {
        opt = 1; // Maximum number of PMM iterations reached
        printer(PMM_iter, opt, obj_val, x, y1, y2, z, PMM_tol_achieved);
    }
    return Solution<T>(opt, x, y1, y2, z, obj_val, PMM_iter, SSN_iter, PMM_tol_achieved, SSN_tol_achieved);
}
