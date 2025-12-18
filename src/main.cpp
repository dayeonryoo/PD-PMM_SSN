#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include "SSN_PMM.hpp"
#include "Problem.hpp"
#include "Printing.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

int main() {

// ==============================================================
//     min  c^T x + (1/2) x^T Q x,
//     s.t. A x = b,
//          B x = w,
//          lx <= x <= ux,
//          lw <= w <= uw
// ---------------------------------------------------------------
//     n = 6, m = 2, l = 3,
//     c = [1, 0, -2, 0, 1, 0]^T, Q = 0,
//     A = [1, 0, 1, 0, 0, 0
//          0, 0, 0, 2, 0, 1],
//     b = [3, 5]^T,
//     B = [1, -1, 0, 0, 0, 0
//          0, 0, 2, 0, -1, 0,
//          0, 0, 0, 1, 0, 1],
//     lx = [0, -inf, 0, -inf, 0, -inf]^T,
//     ux = [5, inf, 4, inf, 3, inf]^T,
//     lw = [-2, -1, 0]^T,
//     uw = [2, 3, 4]^T
// ---------------------------------------------------------------
//     Expected solution: x1 = 0, x3 = 3, x5 = 3,
//                        -2 <= x2 <= 2, 1 <= x4 <= 5, x6 = 5 - 2x4
// ==============================================================

    // Define problem data
    const int n = 6;
    const int m = 2;
    const int l = 3;

    SpMat Q0(n, n);

    SpMat A(m, n);
    std::vector<Triplet> A_trpl;
    A_trpl.emplace_back(0, 0, 1.0);
    A_trpl.emplace_back(0, 2, 1.0);
    A_trpl.emplace_back(1, 3, 2.0);
    A_trpl.emplace_back(1, 5, 1.0);
    A.setFromTriplets(A_trpl.begin(), A_trpl.end());
    
    SpMat B(l, n);
    std::vector<Triplet> B_trpl;
    B_trpl.emplace_back(0, 0, 1.0);
    B_trpl.emplace_back(0, 1, -1.0);
    B_trpl.emplace_back(1, 2, 2.0);
    B_trpl.emplace_back(1, 4, -1.0);
    B_trpl.emplace_back(2, 3, 1.0);
    B_trpl.emplace_back(2, 5, 1.0);
    B.setFromTriplets(B_trpl.begin(), B_trpl.end());

    Vec c(n);
    c << 1.0, 0.0, -2.0, 0.0, 1.0, 0.0;
    Vec b(m);
    b << 3.0, 5.0;

    const T inf = std::numeric_limits<T>::infinity();

    Vec lx(n), ux(n);
    lx << 0.0, -inf, 0.0, -inf, 0.0, -inf;
    ux << 5.0, inf, 4.0, inf, 3.0, inf;
    Vec lw(l), uw(l);
    lw << -2.0, -1.0, 0.0;
    uw << 2.0, 3.0, 4.0;

    T tol = 1e-6;
    int max_iter = 1e3;
    PrintWhen PMM_print_when = PrintWhen::ALWAYS;
    PrintWhat PMM_print_what = PrintWhat::FULL;
    PrintWhen SSN_print_when = PrintWhen::END_ONLY;
    PrintWhat SSN_print_what = PrintWhat::SUMMARY;

    std::cout << "Q = 0, i.e. LP problem.\n";

    // Create Problem instance
    Problem<T> LP(Q0, A, B, c, b, lx, ux, lw, uw, tol, max_iter,
                  PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);

    // Solve the problem using SSN_PMM
    SSN_PMM<T> LP_solver(LP); 
    Solution<T> LP_sol = LP_solver.solve();
    LP_sol.print_summary();
    
    std::cout << "===================================================================\n";
    

//  Q = diag(2, 0, 1, 0, 1, 0)


    SpMat diag_Q(n, n);
    std::vector<Triplet> diag_Q_trpl;
    diag_Q_trpl.emplace_back(0, 0, 2.0);
    diag_Q_trpl.emplace_back(2, 2, 1.0);
    diag_Q_trpl.emplace_back(4, 4, 1.0);
    diag_Q.setFromTriplets(diag_Q_trpl.begin(), diag_Q_trpl.end());

    std::cout << "Q is diagonal.\n";

    // Create Problem instance with quadratic term
    Problem<T> diag_QP(diag_Q, A, B, c, b, lx, ux, lw, uw, tol, max_iter,
                    PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);

    // Solve the problem using SSN_PMM
    SSN_PMM<T> diag_QP_solver(diag_QP); 
    Solution<T> diag_QP_sol = diag_QP_solver.solve();
    diag_QP_sol.print_summary();

    std::cout << "===================================================================\n";
    

    // Q = [ 2, 0, -1, 0,  0, 0
    //       0, 0,  0, 0,  0, 0
    //      -1, 0,  2, 0, -1, 0
    //       0, 0,  0, 0,  0, 0
    //       0, 0, -1, 0,  2, 0
    //       0, 0,  0, 0,  0, 0]


    SpMat SPSD_Q(n, n);
    std::vector<Triplet> SPSD_Q_trpl;
    SPSD_Q_trpl.emplace_back(0, 0, 2.0);
    SPSD_Q_trpl.emplace_back(0, 2, -1.0);
    SPSD_Q_trpl.emplace_back(2, 0, -1.0);
    SPSD_Q_trpl.emplace_back(2, 2, 2.0);
    SPSD_Q_trpl.emplace_back(2, 4, -1.0);
    SPSD_Q_trpl.emplace_back(4, 2, -1.0);
    SPSD_Q_trpl.emplace_back(4, 4, 2.0);
    SPSD_Q.setFromTriplets(SPSD_Q_trpl.begin(), SPSD_Q_trpl.end());

    std::cout << "Q is symmetric positive semidefinite.\n";

    // Create Problem instance with quadratic term
    Problem<T> SPSD_QP(SPSD_Q, A, B, c, b, lx, ux, lw, uw, tol, max_iter,
                    PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);
    
    // Solve the problem using SSN_PMM
    SSN_PMM<T> SPSD_QP_solver(SPSD_QP);
    Solution<T> SPSD_QP_sol = SPSD_QP_solver.solve();
    SPSD_QP_sol.print_summary();


    

    return 0;
}