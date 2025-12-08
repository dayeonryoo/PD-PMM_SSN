#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace std;
using namespace Eigen;

// =============================================================
//      min  c^T x + (1/2) x^T Q x,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw
// =============================================================
// INPUT: Problem
// --------------------------------------------------------------
// A class containing the data of the problem to be solved:
//    .Q       -> n x n quadratic coefficient matrix
//                *** If empty, provide as Mat::Zero(n,n) ***
//    .A       -> m x n linear equality constraint matrix
//                *** If empty, provide as Mat::Zero(0,n) ***
//    .B       -> l x n coefficient matrix
//                *** If empty, provide as Mat::Zero(0,n) ***
//    .b       -> m-dim right-hand side vector for linear equality constraints
//    .c       -> n-dim coefficient vector
//    .lx      -> n-dim lower bound vector for box constraints on x
//    .ux      -> n-dim upper bound vector for box constraints on x
//    .lw      -> l-dim lower bound vector for box constraints on Bx
//    .uw      -> l-dim upper bound vector for box constraints on Bx
//    .tol     -> tolerance for termination
//    .max_it  -> maximum allowed number of PMM iterations
// =============================================================
// OUTPUT: Solution
// --------------------------------------------------------------
// A class containing the solution of the PMM_SSN solver:
//    .opt     -> Integer indicating the termination status:
//                  0: optimal solution found
//                  1: maximum number of iterations reached
//                  2: termination due to numerical errors
//    .x       -> Optimal primal solution vector
//    .y1      -> Lagrangian multipliers corresponding to Ax = b
//    .y2      -> Lagrangian multipliers corresponding to Bx = w
//    .z       -> Lagrangian multipliers corresponding to box constraints on x
//    .obj_val -> Optimal objective value
//    .PMM_it  -> number of PMM iterations performed to terminate
//    .SSN_it  -> number of SSN iterations performed to terminate
// --------------------------------------------------------------

template <typename T>
class PMM_SSN {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat = Eigen::SparseMatrix<T>;


};