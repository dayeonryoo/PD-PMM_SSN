#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
using namespace std;
using namespace Eigen;

// =============================================================
//      min  c^T x + (1/2) x^T Q x,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw
// =============================================================

template <typename T>
class Solution {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    int opt;        // Termination status
                    //   0: optimal solution found
                    //   1: maximum number of iterations reached
                    //   2: termination due to numerical errors
    Vec x;          // Optimal primal solution vector
    Vec y1;         // Lagrangian multipliers for Ax = b
    Vec y2;         // Lagrangian multipliers for Bx = w
    Vec z;          // Lagrangian multipliers for box constraints on x
    T obj_val;      // Optimal objective value
    int PMM_it;     // Number of PMM iterations performed
    int SSN_it;     // Number of SSN iterations performed

    Solution(const int opt_, const Vec& x_, const Vec& y1_, const Vec& y2_,
             const Vec& z_, const T obj_val_, const int PMM_it_, const int SSN_it_)
    : opt(opt_), x(x_), y1(y1_), y2(y2_), z(z_), obj_val(obj_val_), PMM_it(PMM_it_), SSN_it(SSN_it_)
    {}

};