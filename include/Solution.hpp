#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>

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
    int PMM_iter;   // Number of PMM iterations performed
    int SSN_iter;   // Number of SSN iterations performed per PMM iteration
    T PMM_tol_achieved; // Tolerance achieved by PMM
    T SSN_tol_achieved; // Tolerance achieved by SSN

    Solution(const int opt_, const Vec& x_, const Vec& y1_, const Vec& y2_,
             const Vec& z_, const T obj_val_, const int PMM_iter_, const int SSN_iter_,
             const T PMM_tol_achieved_ = 0, const T SSN_tol_achieved_ = 0)
    : opt(opt_), x(x_), y1(y1_), y2(y2_), z(z_), obj_val(obj_val_),
      PMM_iter(PMM_iter_), SSN_iter(SSN_iter_),
      PMM_tol_achieved(PMM_tol_achieved_), SSN_tol_achieved(SSN_tol_achieved_)
    {}

    void print_summary() const {
        std::cout << "\n";
        std::cout << "Solution Summary:" << std::endl;
        std::cout << "Termination status (opt): " << opt << std::endl;
        std::cout << "Optimal objective value (obj_val): " << obj_val << std::endl;
        // std::cout << "Optimal solution (x): (" << x.transpose() << ")\n";
        std::cout << "Number of PMM iterations (PMM_iter): " << PMM_iter << std::endl;
        std::cout << "Number of SSN iterations (SSN_iter): " << SSN_iter << std::endl;
        std::cout << "PMM tolerance achieved (PMM_tol_achieved): " << PMM_tol_achieved << std::endl;
        std::cout << "SSN tolerance achieved (SSN_tol_achieved): " << SSN_tol_achieved << std::endl;
    }

};