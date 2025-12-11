#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include "SSN_PMM.hpp"
#include "Problem.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;

int main() {

    // Define problem data
    SpMat Q(6,6);

    std::vector<Eigen::Triplet<T>> Atr;
    Atr.emplace_back(0, 0, 1.0);
    Atr.emplace_back(0, 2, 1.0);
    Atr.emplace_back(1, 3, 2.0);
    Atr.emplace_back(1, 5, 1.0);
    SpMat A(2,6);
    A.setFromTriplets(Atr.begin(), Atr.end());

    std::vector<Eigen::Triplet<T>> Btr;
    Btr.emplace_back(0, 0, 1.0);
    Btr.emplace_back(0, 1, -1.0);
    Btr.emplace_back(1, 2, 2.0);
    Btr.emplace_back(1, 4, -1.0);
    Btr.emplace_back(2, 3, 1.0);
    Btr.emplace_back(2, 5, 1.0);
    SpMat B(3,6);
    B.setFromTriplets(Btr.begin(), Btr.end());

    Vec c = Vec::Zero(6);
    c(0) = 1.0;
    c(2) = -2.0;
    c(4) = 1.0;
    Vec b(2);
    b << 3, 5;

    T inf = std::numeric_limits<T>::infinity();
    Vec lx(6), ux(6);
    lx << 0, -inf, 0, -inf, 0, -inf;
    ux << 5, inf, 4, inf, 3, inf;
    Vec lw(3), uw(3);
    lw << -2, -1, 0;
    uw << 2, 3, 4;

    // Create Problem instance
    Problem<T> problem(Q, A, B, c, b, lx, ux, lw, uw);

    // Solve the problem using SSN_PMM
    SSN_PMM<T> solver(problem); 
    Solution<T> solution = solver.solve();

    if (solution.opt == 0) {
        std::cout << "Optimal solution found.\n";
    } else {
        std::cout << "opt = " << solution.opt << std::endl;
    }
    std::cout << "x = " << solution.x << std::endl;
    std::cout << "y1 = " << solution.y1 << std::endl;
    std::cout << "y2 = " << solution.y2 << std::endl;
    std::cout << "z = " << solution.z << std::endl;
    std::cout << "f(x) = " << solution.obj_val << std::endl;

    return 0;
}