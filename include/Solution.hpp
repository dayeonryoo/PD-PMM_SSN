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

    int opt;         // Termination status
                     //  -3: termination due to dual infeasibility
                     //  -2: termination due to primal infeasibility
                     //  -1: termination due to numerical errors
                     //   0: optimal solution found
                     //   1: maximum number of PMM iterations reached
                     //   2: maximum number of SSN iterations reached
                     //   3: linesearch failed
                     //   4: time limit exceeded

    Vec x;           // Optimal primal solution vector
    Vec y1;          // Lagrangian multipliers for Ax = b
    Vec y2;          // Lagrangian multipliers for Bx = w
    Vec z;           // Lagrangian multipliers for box constraints on x
    T obj_val;       // Optimal objective value

    int PMM_iter;    // Number of PMM iterations performed
    int SSN_iter;    // Number of SSN iterations performed
    int Krylov_iter; // Number of Krylov iterations performed
    int fact;        // Number of factorizations performed

    T PMM_tol_achieved; // Tolerance achieved by PMM
    T SSN_tol_achieved; // Tolerance achieved by SSN
    
    double solving_time; // Total time in seconds taken to solve the problem
    int linesearch_fail; // Number of linesearch failures

    Solution(int opt_, const Vec& x_, const Vec& y1_, const Vec& y2_, const Vec& z_,
             T obj_val_, int PMM_iter_, int SSN_iter_, int Krylov_iter_, int fact_,
             T PMM_tol_achieved_, T SSN_tol_achieved_,
             double solving_time_, int linesearch_fail_)
    : opt(opt_), x(x_), y1(y1_), y2(y2_), z(z_), obj_val(obj_val_),
      PMM_iter(PMM_iter_), SSN_iter(SSN_iter_), Krylov_iter(Krylov_iter_), fact(fact_),
      PMM_tol_achieved(PMM_tol_achieved_), SSN_tol_achieved(SSN_tol_achieved_),
      solving_time(solving_time_), linesearch_fail(linesearch_fail_)
    {}

    void print_summary() const {
        std::cout << "\n";
        std::cout << "Solution Summary:" << std::endl;
        std::cout << "Termination status (opt): " << opt << std::endl;
        std::cout << "Problem dimensions: n = " << x.size() << ", m = " << y1.size() << ", l = " << y2.size() << std::endl;
        if (opt == -2) {
            std::cout << "Problem is primal infeasible.\n";
        } else if (opt == -3) {
            std::cout << "Problem is dual infeasible.\n";
        } else {
            std::cout << "Optimal objective value (obj_val): " << obj_val << std::endl;
            std::cout << "Number of PMM iterations (PMM_iter): " << PMM_iter << std::endl;
            std::cout << "Number of SSN iterations (SSN_iter): " << SSN_iter << std::endl;
            std::cout << "Number of Krylov iterations (Krylov_iter): " << Krylov_iter << std::endl;
            std::cout << "Number of factorizations (fact): " << fact << std::endl;
            std::cout << "PMM tolerance achieved (PMM_tol_achieved): " << PMM_tol_achieved << std::endl;
            std::cout << "SSN tolerance achieved (SSN_tol_achieved): " << SSN_tol_achieved << std::endl;
        }
        std::cout << "Number of linesearch failures (linesearch_fail): " << linesearch_fail << std::endl;
        std::cout << "Total solving time (solving_time): " << solving_time << " seconds\n";
    }

};