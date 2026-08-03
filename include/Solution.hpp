#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>

// =============================================================
//      min  c^T x + 0.5 x^T Q x + obj_const,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw.
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

    int pmm_iter;    // Number of PMM iterations performed
    int ssn_iter;    // Number of SSN iterations performed
    int krylov_iter; // Number of Krylov iterations performed
    int fact;        // Number of factorizations performed
    int smw_count;   // Number of SMW preconditioner applications

    T pmm_tol_achieved; // Tolerance achieved by PMM
    T ssn_tol_achieved; // Tolerance achieved by SSN

    double setup_time;   // Wall-clock time in seconds spent in the SSN_PMM constructor
    double solve_time;   // Wall-clock time in seconds spent in solve()
    double run_time;     // setup_time + solve_time
    int linesearch_fail; // Number of linesearch failures
    int krylov_fail;     // Number of Krylov failures

    Solution(int opt, const Vec& x, const Vec& y1, const Vec& y2, const Vec& z,
             T obj_val, int pmm_iter, int ssn_iter, int krylov_iter, int fact, int smw_count,
             T pmm_tol_achieved, T ssn_tol_achieved,
             double setup_time, double solve_time, int linesearch_fail, int krylov_fail)
    : opt(opt), x(x), y1(y1), y2(y2), z(z), obj_val(obj_val),
      pmm_iter(pmm_iter), ssn_iter(ssn_iter), krylov_iter(krylov_iter), fact(fact), smw_count(smw_count),
      pmm_tol_achieved(pmm_tol_achieved), ssn_tol_achieved(ssn_tol_achieved),
      setup_time(setup_time), solve_time(solve_time), run_time(setup_time + solve_time),
      linesearch_fail(linesearch_fail), krylov_fail(krylov_fail)
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
            std::cout << "Number of PMM iterations (pmm_iter): " << pmm_iter << std::endl;
            std::cout << "Number of SSN iterations (ssn_iter): " << ssn_iter << std::endl;
            std::cout << "Number of Krylov iterations (krylov_iter): " << krylov_iter << std::endl;
            std::cout << "Number of factorizations (fact): " << fact << std::endl;
            std::cout << "PMM tolerance achieved (pmm_tol_achieved): " << pmm_tol_achieved << std::endl;
            std::cout << "SSN tolerance achieved (ssn_tol_achieved): " << ssn_tol_achieved << std::endl;
        }
        std::cout << "Number of linesearch failures (linesearch_fail): " << linesearch_fail << std::endl;
        std::cout << "Number of Krylov failures (krylov_fail): " << krylov_fail << std::endl;
        std::cout << "Run time (run_time): " << run_time << " seconds\n";
    }

};
