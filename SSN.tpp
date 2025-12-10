#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

template <typename T>
SSN_result<T> SSN<T>::solve_SSN() {

    // Set the semismooth Newton parameters
    T beta = 0.4995 / 2;
    T delta = 0.995;
    T eta = SSN_tol / 10;
    T gamma = 0.1;

    // Intialize iteration counter and set starting points
    SSN_result<T> result;
    result.SSN_in_iter = 0;
    result.x = x;
    result.y2 = y2;

    // SSN main loop
    while (result.SSN_in_iter < SSN_max_in_iter) {
        // ----------------------------------------------
        // Structure:
        // Let M(u), with u = (x,y_2), be the proximal augmented Lagrangian
        // associated with the subproblem of interest.
        // Until (|| \nabla M(u_{k_j}) || < SSN_tol), do:
        //     1) Compute a Clarke subgradient J of \nabla M(u_{k_j})
        //        and solve J du = - \nabla M(u_{k_j}) for the Newton direction du;
        //     2) Perform a backtracking line search to determine the step size alpha;
        //     3) Update the variables;
        //     j = j + 1;
        // End
        // ----------------------------------------------

        result.SSN_in_iter++;
    }

    return result;
}