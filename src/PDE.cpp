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

    double tol = 1e-6;
    int max_iter = 1000;
    double time_limit = 180.0; // in seconds
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::TUNING;

    // ====== Poisson ======
    // PDPMMdata<T> data1 = pdegen::make_poisson_L1L2_control<T>(8, 1e-2, 1e-6);
    // Problem<T> pb1(data1, tol, max_iter, time_limit, when, what);
    // SSN_PMM<T> solver1(pb1);

    // Solution<T> sol1 = solver1.solve();
    // sol1.print_summary();
    // print_feasibility(data1, sol1.x, tol);

    // ====== Convection-diffusion ======
    PDPMMdata<T> data2 = pdegen::make_convdiff_L1L2_control<T>(10, 1e-4, 1e-2);
    Problem<T> pb2(data2, tol, max_iter, time_limit, when, what);
    SSN_PMM<T> solver2(pb2);

    Solution<T> sol2 = solver2.solve();
    sol2.print_summary();
    // print_feasibility(data2, sol2.x, tol);

}