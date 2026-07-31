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

void print_feasibility(const PDPMMdata<T>& pd, const Vec x, const T tol) {
    std::cout << "\nChecking feasibility of solution x from PMM_SSN solver:\n";
    std::cout << "  ||Ax - b||_2 = ";
    if (pd.A.rows() == 0) std::cout << "0 (m = 0)\n";
    else std::cout << (pd.A * x - pd.b).norm() << "\n";
    std::cout << "  ||Ax - b||_inf = ";
    if (pd.A.rows() == 0) std::cout << "0 (m = 0)\n";
    else std::cout << (pd.A * x - pd.b).cwiseAbs().maxCoeff() << "\n";

    std::cout << "\n  Elements of x outside bounds:\n";
    for (int i = 0; i < pd.c.size(); ++i) {
        if (x[i] < pd.lx[i] - tol || x[i] > pd.ux[i] + tol) {
            std::cout << "      Variable " << i << " out of bounds: x = " << x[i]
                      << ", [" << pd.lx[i] << ", " << pd.ux[i] << "]\n";
        }
    }
    std::cout << "\n  Elements of Bx outside bounds:\n";
    Vec Bx = pd.B * x;
    for (int i = 0; i < pd.lw.size(); ++i) {
        T Bx_i = Bx[i];
        if (Bx_i < pd.lw[i] - tol || Bx_i > pd.uw[i] + tol) {
            std::cout << "      Variable " << i << " out of bounds: Bx = " << Bx_i
                      << ", [" << pd.lw[i] << ", " << pd.uw[i] << "]\n";
        }
    }
}

int main() {

    double tol = 1e-9;
    int max_iter = 1000;
    double time_limit = 1000.0; // in seconds
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::TUNING;
    double inf =  std::numeric_limits<double>::infinity();

    std::time_t curr_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::ctime(&curr_time);

    // ============ L1/L2 ============
    // ----- Poisson -----
    std::cout << "========== Solving L1/L2 Poisson ==========\n";
    // tol = 1e-6;
    // PDPMMdata<T> data = pdegen::make_poisson_L1L2_control<T>(6, 1e-2, 1e-2);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();

    // ----- Convection-diffusion -----
    std::cout << "========== Solving L1/L2 ConvDiff ==========\n";
    tol = 1e-10;
    PDPMMdata<T> data = pdegen::make_convdiff_L1L2_control<T>(6, 1e-2, 1e-2);
    Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    SSN_PMM<T> solver(pb);
    Solution<T> sol = solver.solve();
    sol.print_summary();
    std::cout << "obj_val - obj_const = " << sol.obj_val - solver.obj_const << "\n";


    // ============ L2 ============
    // ----- Poisson control -----
    // std::cout << "========== Solving L2 control-constrained Poisson ==========\n";
    // PDPMMdata<T> data = pdegen::make_poisson_control<T>(9, 1e-6, -inf, inf, 0, 300);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();

    // ----- Poisson state -----
    // std::cout << "========== Solving L2 state-constrained Poisson ==========\n";
    // tol = 1e-8;
    // PDPMMdata<T> data = pdegen::make_poisson_state_control<T>(9, 1e0, -0.1, 0.002, -inf, inf);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();

    // ----- Convdiff -----
    // std::cout << "========== Solving L2 ConvDiff ==========\n";
    // tol = 1e-10;
    // PDPMMdata<T> data = pdegen::make_convdiff_control<T>(7, 0.1, 0.0, 0.2, -0.75, 0.75);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();

    // ----- 3D Poisson -----
    // std::cout << "========== Solving L2 3D Poisson ==========\n";
    // tol = 1e-10;
    // PDPMMdata<T> data = pdegen::make_poisson_control_3d<T>(6, 1.0, -inf, inf, 0.0, 0.01);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();

    // ----- Helmholtz -----
    // std::cout << "========== Solving L2 Helmholtz equation ==========\n";
    // PDPMMdata<T> data = pdegen::make_helmholtz_control<T>(10, 1e-2, 20, -0.0005, 0.0005, -inf, inf);
    // Problem<T> pb(data, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver(pb);
    // Solution<T> sol = solver.solve();
    // sol.print_summary();
}