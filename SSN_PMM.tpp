#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "SSN_PMM.hpp"
#include "Solution.hpp"
using namespace std;
using namespace Eigen;


template <typename T>
typename SSN_PMM<T>::Vec SSN_PMM<T>::compute_residuals() {
    // Dual residual
    T res_d = (c + Q * x - A.transpose() * y1 - B.transpose() * y2 + z).norm() / (1 + c.norm());
    
    // Primal residual
    T res_p = (A * x - b).norm() / (1 + b.norm());

    // Complementarity residual for box constraints
    Vec temp_compl_x = x + z;
    temp_compl_x = temp_compl_x.cwiseMax(lx).cwiseMin(ux);
    T compl_x = (x - temp_compl_x).norm();

    // Complementarity residual for Bx constraints
    Vec w = B * x;
    Vec temp_compl_B = w - y2;
    temp_compl_B = temp_compl_B.cwiseMax(lw).cwiseMin(uw);
    T compl_B = (w - temp_compl_B).norm();

    // Collect residuals
    Vec residuals(4);
    residuals << res_d, res_p, compl_x, compl_B;
    return residuals;
}

template <typename T>
Solution<T> SSN_PMM<T>::solve() {
    
    // Initialize parameters
    T mu = 5e1;
    T rho = 1e2;
    int max_SSN_iters = 4000; // Maximum number of total SSN iterations
    int SSN_maxit = 40; // Maximum number of SSN iterations per PMM iteration
    T SSN_tol = tol; // Tolerance for SSN termination
    int PMM_iter = 0;
    int SSN_iter = 0;
    T reg_limit = 1e6; // Maximum value for the penalty parameters

    // SSN-PMM Main loop
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
        Vec residuals = compute_residuals();
        T res_norm = residuals.maxCoeff();
        if (res_norm < tol) {
            opt = 0; // Optimal solution found
            break;
        }

        // Build or update the Newton system.
        if (PMM_iter == 0) {
            // Initial construction of the Newton system
            // (Implementation of system construction goes here)
        } else {
            // Update the Newton system based on new mu, rho
            // (Implementation of system update goes here)
        }

        // Call semismooth Newton method to find the x-update.

        ++PMM_iter;
    }
    return Solution<T>(opt, x, y1, y2, z, obj_val, PMM_iter, SSN_iter);
}