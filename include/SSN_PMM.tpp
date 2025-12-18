#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include "SSN.hpp"

template <typename T>
void SSN_PMM<T>::determine_dimensions() {

    // Determine n
    if (c.size() != 0) {
        n = c.size();
    } else if (Q.rows() != 0) {
        n = Q.rows();
    } else if (A.cols() != 0) {
        n = A.cols();
    } else if (B.cols() != 0) {
        n = B.cols();
    } else if (lx.size() != 0) {
        n = lx.size();
    } else if (ux.size() != 0) {
        n = ux.size();
    } else {
        throw std::invalid_argument("Problem dimension n cannot be determined from the provided data.");
    }

    // Determine m
    m = A.rows();

    // Determine l
    l = B.rows();

}

template <typename T>
void SSN_PMM<T>::set_default() {
    // PMM parameters
    mu = 5e1;
    rho = 1e2;
    if (tol == 0.0) tol = 1e-6;
    if (max_iter == 0) max_iter = 1e3;

    // SSN parameters
    SSN_max_iter = 4000;
    SSN_max_in_iter = 40;
    SSN_tol = tol;
    reg_limit = 1e6;

    // Printing
    PMM_print_label = PrintLabel::PMM;

    // Initial solution
    x = Vec::Zero(n);
    y1 = Vec::Zero(m);
    y2 = Vec::Zero(l);
    z = Vec::Zero(n);

    T inf = std::numeric_limits<T>::infinity();

    // Matrices and vectors
    if (c.size() == 0) {
        c = Vec::Zero(n);
    }
    if (Q.rows() == 0 || Q.cols() == 0) {
        Q = SpMat(n, n);
    }
    if (A.rows() == 0 || A.cols() == 0) {
        A = SpMat(m, n);
    }
    if (b.size() == 0) {
        b = Vec::Zero(m);
    }
    if (B.rows() == 0 || B.cols() == 0) {
        B = SpMat(l, n);
    }
    if (lx.size() == 0) {
        lx = Vec::Constant(n, -inf);
    }
    if (ux.size() == 0) {
        ux = Vec::Constant(n, inf);
    }
    if (lw.size() == 0) {
        lw = Vec::Constant(l, -inf);
    }
    if (uw.size() == 0) {
        uw = Vec::Constant(l, inf);
    }
}

template <typename T>
void SSN_PMM<T>::check_dimensionality() {
    if (Q.rows() != n || Q.cols() != n) {
        throw std::invalid_argument("Dimension mismatch: Q should be n x n.");
    }
    if (c.size() != n) {
        throw std::invalid_argument("Dimension mismatch: c should be a vector of size n.");
    }
    if (A.rows() != m || A.cols() != n) {
        throw std::invalid_argument("Dimension mismatch: A should be m x n.");
    }
    if (b.size() != m) {
        throw std::invalid_argument("Dimension mismatch: b should be a vector of size m.");
    }
    if (B.rows() != l || B.cols() != n) {
        throw std::invalid_argument("Dimension mismatch: B should be l x n.");
    }
    if (lx.size() != n || ux.size() != n) {
        throw std::invalid_argument("Dimension mismatch: lx and ux should be a vector of size n.");
    }
    if (lw.size() != l || uw.size() != l) {
        throw std::invalid_argument("Dimension mismatch: lw and uw should be a vector of size l.");
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
    T res_d = (c + Q * x - A.transpose() * y1 - B.transpose() * y2 + z).norm() / (1 + c.norm());

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

    // Initialize printing function
    auto printer = make_print_function<T, Vec>(PMM_print_label, PMM_print_when, PMM_print_what, max_iter);

    // Build the Newton system.
    SSN<T> NS(Q, A, B, c, b,
        lx, ux, lw, uw,
        x, y1, y2, z,
        mu, rho, n, m, l,
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

        // Compute residuals
        Vec res_norms = compute_residual_norms();
        T res_p = res_norms(0); // Needed to update PMM params
        T res_d = res_norms(1);
        PMM_tol_achieved = res_norms.maxCoeff();

        // Compute objective value
        obj_val = c.dot(x) + 0.5 * x.dot(Q * x);

        // Check termination criterion
        if (PMM_tol_achieved < tol) {
            opt = 0; // Optimal solution found
            break;
        }
        PMM_iter++;

        // Print current iteration info
        printer(PMM_iter, opt, obj_val, x, y1, y2, z, PMM_tol_achieved);

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
        T eps_k = std::max(min_res_vec, SSN_tol);
        SSN_tol_achieved = 2 * max_res_vec;

        // Call semismooth Newton method to update x and y2
        while (SSN_tol_achieved > std::max(0.1*max_res_vec, min_res_vec)) {
            SSN_result<T> NS_solution = NS.solve_SSN(eps_k);
            x = NS_solution.x;
            y2 = NS_solution.y2;

            // Update the Newton system in case of a restart
            NS.x = x;
            NS.y2 = y2;

            SSN_tol_achieved = NS_solution.SSN_tol_achieved;
            SSN_iter += NS_solution.SSN_in_iter;
            if (SSN_iter >= SSN_max_iter) break;
        }

        // Update multipliers
        y1 -= mu * (A * x - b);
        z += mu * (x - proj(z / mu + x, lx, ux));

        // Compute the new residual norms
        Vec new_res_norms = compute_residual_norms();
        T new_res_p = new_res_norms(0);
        T new_res_d = new_res_norms(1);

        // Update penalty parameters
        update_PMM_parameters(res_p, res_d, new_res_p, new_res_d);

    }
    if (opt != 0) opt = 1; // Maximum number of PMM iterations reached
    printer(PMM_iter, opt, obj_val, x, y1, y2, z, PMM_tol_achieved);

    return Solution<T>(opt, x, y1, y2, z, obj_val, PMM_iter, SSN_iter, PMM_tol_achieved, SSN_tol_achieved);
}
