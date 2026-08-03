#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "SSN_PMM.hpp"
#include "Problem.hpp"
#include "Printing.hpp"
#include "PDEgenerator.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

int main() {

    double tol = 1e-6;
    int max_iter = 1000;
    double time_limit = 1000.0; // in seconds
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::SSN;
    double inf =  std::numeric_limits<double>::infinity();

    std::time_t curr_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::ctime(&curr_time);

    // std::cout << "========== Solving L1/L2 Poisson ==========\n";
    // PDPMMdata<T> data = pdegen::make_poisson_l1l2_control<T>(6, 1e-2, 1e-2);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();

    // std::cout << "========== Solving L1/L2 ConvDiff ==========\n";
    // PDPMMdata<T> data = pdegen::make_convdiff_l1l2_control<T>(6, 1e-2, 1e-2);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();


    // std::cout << "========== Solving L2 control-constrained Poisson ==========\n";
    // PDPMMdata<T> data = pdegen::make_poisson_l2_control<T>(7, 1e-6, -inf, inf, 0, 300);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();

    // std::cout << "========== Solving L2 state-constrained Poisson ==========\n";
    // PDPMMdata<T> data = pdegen::make_poisson_l2_state_control<T>(6, 1e0, -0.1, 0.002, -inf, inf);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();

    std::cout << "========== Solving L2 ConvDiff ==========\n";
    PDPMMdata<T> data = pdegen::make_convdiff_l2_control<T>(6, 1e-1, 0.0, 0.2, -0.75, 0.75);
    Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    SSN_PMM<T> solver(pb);
    Solution<T> sol = solver.solve();
    sol.print_summary();

}