#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "SSN.hpp"


template <typename T>
T SSN_PMM<T>::max_residual_norm() {
    // Dual residual norm
    T res_d = (c + Q * x - A.transpose() * y1 - B.transpose() * y2 + z).norm() / (1 + c.norm());
    
    // Primal residual norm
    T res_p = (A * x - b).norm() / (1 + b.norm());

    // Complementarity residual norm for box constraints
    Vec temp_compl_x = x + z;
    temp_compl_x = temp_compl_x.cwiseMax(lx).cwiseMin(ux);
    T compl_x = (x - temp_compl_x).norm();

    // Complementarity residual norm for Bx constraints
    Vec w = B * x;
    Vec temp_compl_B = w - y2;
    temp_compl_B = temp_compl_B.cwiseMax(lw).cwiseMin(uw);
    T compl_B = (w - temp_compl_B).norm();

    // Collect residual norms
    Vec res_norms(4);
    res_norms << res_d, res_p, compl_x, compl_B;

    return res_norms.maxCoeff();
}

template <typename T>
Solution<T> SSN_PMM<T>::solve() {
    
    // Initialize parameters
    T mu = 5e1;
    T rho = 1e2;
    int SSN_max_iters = 4000; // Maximum number of total SSN iterations
    int SSN_max_in_iter = 40; // Maximum number of SSN iterations per PMM iteration
    T SSN_tol = tol; // Tolerance for SSN termination
    T reg_limit = 1e6; // Maximum value for the penalty parameters

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
    while (PMM_iter < max_it) {
    // ----------------------------------------------
    // Structure:
    // Until (primal infeasibility, dual infeasibility, complementarity) < tol, do:
    //     1) Call Semismooth Newton method to approximately minimize the augmented Lagrangian w.r.t. x;
    //     2) Update multipliers y1, y2, z;
    //     3) Update penalty parameters mu, rho;
    //     k = k + 1;
    // End
    // ----------------------------------------------

        // Check termination criteria.
        if (max_residual_norm() < tol) {
            opt = 0; // Optimal solution found
            break;
        }

        // Update the Newton system
        NS.x = x;
        NS.y1 = y1;
        NS.y2 = y2;
        NS.z = z;
        NS.mu = mu;
        NS.rho = rho;

        // Call semismooth Newton method.
        SSN_result<T> NS_solution = NS.solve_SSN();
        x = NS_solution.x;
        y2 = NS_solution.y2;
        // Update multipliers
        y1 += mu * (A * x - b);
        z += mu * x - mu * ((z / mu) + x).cwiseMax(lx).cwiseMin(ux);

        ++PMM_iter;
        SSN_iter += NS_solution.SSN_in_iter;
    }
    return Solution<T>(opt, x, y1, y2, z, obj_val, PMM_iter, SSN_iter);
}