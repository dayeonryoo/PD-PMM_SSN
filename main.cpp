#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include "SSN_PMM.hpp"
#include "Problem.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

int main() {

// =============================================================
//      min  c^T x + (1/2) x^T Q x,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw
// --------------------------------------------------------------
//      c = [1; 2], Q = 0, A = [1; 1], b = 0, B = I_2,
//      lx = [0; 0], ux = [1; 1], lw = [0; 0], uw = [1; 1]
// --------------------------------------------------------------
//      Expected solution: x = [0; 0]
//      Expected objective value: f(x) = 0
// =============================================================

    // Define problem data
    const int n = 2;
    const int m = 1;
    const int l = 2;

    // Q = 0
    SpMat Q(n, n);
    Q.setZero();

    // A = [1; 1]
    SpMat A(m, n);
    std::vector<Triplet> A_trpl;
    A_trpl.emplace_back(0, 0, 1.0);
    A_trpl.emplace_back(0, 1, 1.0);
    A.setFromTriplets(A_trpl.begin(), A_trpl.end());
    
    // B = I_2
    SpMat B(l, n);
    std::vector<Triplet> B_trpl;
    B_trpl.emplace_back(0, 0, 1.0);
    B_trpl.emplace_back(1, 1, 1.0);
    B.setFromTriplets(B_trpl.begin(), B_trpl.end());

    // c = [1; 2], b = 0
    Vec c(n);
    c << 1.0, 2.0;
    Vec b(m);
    b << 0.0;

    // 0 <= x, w <= 1
    Vec lx(n), ux(n);
    lx.setZero();
    ux.setOnes();
    Vec lw(l), uw(l);
    lw.setZero();
    uw.setOnes();

    T tol = 0.1;
    int max_iter = 5;

    // Create Problem instance
    Problem<T> problem(Q, A, B, c, b, lx, ux, lw, uw, tol, max_iter);

    // Solve the problem using SSN_PMM
    SSN_PMM<T> solver(problem); 
    Solution<T> solution = solver.solve();

    std::cout << "Final result:\n";

    if (solution.opt == 0) {
        std::cout << "Optimal solution found at iteration " << solution.PMM_iter << std::endl;
    } else {
        std::cout << "opt = " << solution.opt << std::endl;
    }
    std::cout << "x = " << solution.x.transpose() << std::endl;
    // std::cout << "y1 = " << solution.y1.transpose() << std::endl;
    // std::cout << "y2 = " << solution.y2.transpose() << std::endl;
    // std::cout << "z = " << solution.z.transpose() << std::endl;
    std::cout << "f(x) = " << solution.obj_val << std::endl;
    std::cout << "PMM tol achieved = " << solution.PMM_tol_achieved << std::endl;

    return 0;
}