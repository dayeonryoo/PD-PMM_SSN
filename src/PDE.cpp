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

    double tol = 1e-4;
    int max_iter = 30;
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::MINIMAL;

    // ====== Poisson ======
    PDPMMdata<T> data1 = pdegen::make_poisson_L1L2_control_default<T>();
    Problem<T> pb1(data1, tol, max_iter, when, what);
    SSN_PMM<T> solver1(pb1);

    auto start1 = std::chrono::high_resolution_clock::now();
    Solution<T> sol1 = solver1.solve();
    sol1.print_summary();
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<T> elapsed1 = end1 - start1;
    std::cout << "\nPMM solver took " << elapsed1.count() << " s.\n";

    print_feasibility(data1, sol1.x, tol);

    // ====== Convection-diffusion ======
    PDPMMdata<T> data2 = pdegen::make_convdiff_L1L2_control_default<T>();
    Problem<T> pb2(data2, tol, max_iter, when, what);
    SSN_PMM<T> solver2(pb2);

    auto start2 = std::chrono::high_resolution_clock::now();
    Solution<T> sol2 = solver2.solve();
    sol2.print_summary();
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<T> elapsed2 = end2 - start2;
    std::cout << "\nPMM solver took " << elapsed2.count() << " s.\n";

    print_feasibility(data2, sol2.x, tol);

}