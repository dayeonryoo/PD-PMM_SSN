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
  
    std::cout << "==================== Q = 0 ====================\n";
    // min x1 + 2 x2 + 0.5 x3 s.t. x1 + x2 + x3 = 1, x >= 0
    // Expected solution (0, 0, 1), with optimal value 0.5

    Problem<T> QP1;
    QP1.PMM_print_when = PrintWhen::ALWAYS;
    QP1.SSN_print_when = PrintWhen::ALWAYS;

    Vec c1(3); c1 << 1, 2, 0.5;
    QP1.c = c1;

    SpMat A1(1, 3);
    A1.insert(0, 0) = 1;
    A1.insert(0, 1) = 1;
    A1.insert(0, 2) = 1;
    QP1.A = A1;

    QP1.b = Vec::Ones(1);
    QP1.lx = Vec::Zero(3);

    SSN_PMM<T> QP1_solver(QP1);
    Solution<T> QP1_sol = QP1_solver.solve();
    QP1_sol.print_summary();

    std::cout << "\nExpected solution x = (0, 0, 1), f(x) = 0.5\n\n";

    std::cout << "==================== Q = diagonal ====================\n";
    // min -4 x1 - 8 x2 + 0.5 (2 x1^2 + 4 x2^2) (unconstrained)
    // Expected solution (2, 2), with optimal value -12

    Problem<T> QP2;
    QP2.PMM_print_when = PrintWhen::ALWAYS;
    QP2.SSN_print_when = PrintWhen::ALWAYS;

    Vec c2(2); c2 << -4.0, -8.0;
    QP2.c = c2;

    SpMat Q2(2, 2);
    Q2.insert(0, 0) = 2.0;
    Q2.insert(1, 1) = 4.0;
    QP2.Q = Q2;

    SSN_PMM<T> QP2_solver(QP2);
    Solution<T> QP2_sol = QP2_solver.solve();
    QP2_sol.print_summary();

    std::cout << "\nExpected solution x = (2, 2), f(x) = -12\n\n";

    std::cout << "==================== Q = PSD ====================\n";
    // c = [-1; -1], Q = [2, -1; -1, 2]
    // Expected solution (1, 1), with optimal value -1

    Problem<T> QP3;
    QP3.PMM_print_when = PrintWhen::NEVER;
    QP3.SSN_print_when = PrintWhen::NEVER;

    Vec c3(2); c3 << -1.0, -1.0;
    QP3.c = c3;

    SpMat Q3(2, 2);
    Q3.insert(0, 0) = 2.0;
    Q3.insert(0, 1) = -1.0;
    Q3.insert(1, 0) = -1.0;
    Q3.insert(1, 1) = 2.0;
    QP3.Q = Q3;

    SSN_PMM<T> QP3_solver(QP3);
    Solution<T> QP3_sol = QP3_solver.solve();
    QP3_sol.print_summary();

    std::cout << "\nExpected solution x = (1, 1), f(x) = -1\n\n";

    std::cout << "==================== Q = diagonal ====================\n";
    // c = [-30; 2; -0.2], Q = diag(100, 1, 0.01)
    // lx = [0; -1; -5], ux = [1; 0.5; 5]
    // Expected solution (0.3, -1, 5), with optimal value -6.875

    Problem<T> QP4;
    QP4.PMM_print_when = PrintWhen::NEVER;
    QP4.SSN_print_when = PrintWhen::NEVER;

    Vec c4(3); c4 << -30.0, 2.0, -0.2;
    QP4.c = c4;

    SpMat Q4(3, 3);
    Q4.insert(0, 0) = 100.0;
    Q4.insert(1, 1) = 1.0;
    Q4.insert(2, 2) = 0.01;
    QP4.Q = Q4;

    Vec lx4(3); lx4 << 0.0, -1.0, -5.0;
    Vec ux4(3); ux4 << 1.0, 0.5, 5.0;
    QP4.lx = lx4;
    QP4.ux = ux4;

    SSN_PMM<T> QP4_solver(QP4);
    Solution<T> QP4_sol = QP4_solver.solve();
    QP4_sol.print_summary();

    std::cout << "\nExpected solution x = (0.3, -1, 5), f(x) = -6.875\n\n";

    return 0;
}