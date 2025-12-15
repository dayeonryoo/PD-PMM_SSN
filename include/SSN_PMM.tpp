#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include "SSN.hpp"

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
    bool cond_d = 0.905 * res_d > new_res_d;

    if (cond_p || cond_d){
        mu = std::min(reg_limit, 1.2 * mu);
        rho = std::min(1e2 * reg_limit, 1.4 * rho);
    } else {
        mu = std::min(reg_limit, 1.1 * mu);
        rho = std::min(1e2 * reg_limit, 1.1 * rho);
    };

}

template <typename T>
Solution<T> SSN_PMM<T>::solve() {
    
    // Initialize variables
    opt = -1;
    PMM_iter = 0;
    SSN_iter = 0;
    x = Vec::Zero(n);
    y1 = Vec::Zero(m);
    y2 = Vec::Zero(l);
    z = Vec::Zero(n);

    // Build the Newton system.
    SSN<T> NS(Q, A, B, c, b,
        lx, ux, lw, uw,
        x, y1, y2, z,
        mu, rho, n, m, l,
        SSN_tol, SSN_max_in_iter);

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

        // Compute residuals and check termination criteria.
        Vec res_norms = compute_residual_norms();
        T res_p = res_norms(0); // Needed to update PMM params
        T res_d = res_norms(1);
        T max_res_norm = res_norms.maxCoeff();
        if (max_res_norm < tol) {
            opt = 0; // Optimal solution found
            break;
        }

        PMM_iter++;
        std::cout << "PMM iter " << PMM_iter << ": x = (" << x.transpose() << "), res norms = (" << res_norms.transpose() << ")\n";

        // Update the Newton system
        NS.x = x;
        NS.y1 = y1;
        NS.y2 = y2;
        NS.z = z;
        NS.mu = mu;
        NS.rho = rho;

        // Call semismooth Newton method to update x and y2
        Vec res_vec(3);
        res_vec << 0.1 * res_p, 0.1 * res_d, T(1);
        SSN_tol = std::max(res_vec.minCoeff(), SSN_tol);
        SSN_tol_achieved = 2 * res_vec.maxCoeff();
        while (SSN_tol_achieved > std::max(0.1*res_vec.maxCoeff(), res_vec.minCoeff())) {
            SSN_result<T> NS_solution = NS.solve_SSN();
            x = NS_solution.x;
            y2 = NS_solution.y2;
            NS.x = x;
            NS.y2 = y2;
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

    PMM_tol_achieved = compute_residual_norms().maxCoeff();

    return Solution<T>(opt, x, y1, y2, z, obj_val, PMM_iter, SSN_iter, PMM_tol_achieved);
}