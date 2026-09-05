#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <chrono>
#include <algorithm>
#include <cctype>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ksp_qp.hpp"
#include "problem.hpp"
#include "printing.hpp"
#include "pde_generator.hpp"
#include "cli_args.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

static const std::vector<std::string> kAllProblems = {
    "l1l2_poisson", "l1l2_convdiff", "l2_poisson_control", "l2_poisson_state", "l2_convdiff"
};

// Parses "fem"/"fd" (case-insensitive) into pdegen::Discretization.
static pdegen::Discretization parse_discretization(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                    [](unsigned char c) { return std::tolower(c); });
    if (lower == "fem") return pdegen::Discretization::FEM;
    if (lower == "fd")  return pdegen::Discretization::FD;
    throw std::invalid_argument("--discretization must be 'fem' or 'fd', got: " + s);
}

// Builds the named PDE-constrained QP (see include/pde_generator.hpp for the generators).
// nc is the grid exponent passed straight through to the generator (grid size ~ 2^nc).
KSPQPdata<T> build_problem(const std::string& problem_name, int nc, bool lump_mass,
                            pdegen::Discretization disc) {
    double inf = std::numeric_limits<double>::infinity();

    if (problem_name == "l1l2_poisson") {
        return pdegen::make_poisson_l1l2_control<T>(nc, 1e-2, 1e-2,
            T(-2), T(1.5), -inf, inf, lump_mass, disc);
    } else if (problem_name == "l1l2_convdiff") {
        return pdegen::make_convdiff_l1l2_control<T>(nc, 1e-2, 1e-2,
            T(-2), T(1.5), T(0.02), -inf, inf, lump_mass, disc);
    } else if (problem_name == "l2_poisson_control") {
        return pdegen::make_poisson_l2_control<T>(nc, 1e-2, -inf, inf, 0, 1, lump_mass, disc);
    } else if (problem_name == "l2_poisson_state") {
        return pdegen::make_poisson_l2_state_control<T>(nc, 1, -0.1, 0.002, -inf, inf, lump_mass, disc);
    } else if (problem_name == "l2_convdiff") {
        return pdegen::make_convdiff_l2_control<T>(nc, 1e-1, 0.0, 0.2, -0.75, 0.75, T(0.01), lump_mass, disc);
    }

    throw std::invalid_argument("Unknown problem name: " + problem_name);
}

int main(int argc, char** argv) {
    if (cli::has_flag(argc, argv, "--help") || cli::has_flag(argc, argv, "-h")) {
        std::cout <<
            "Usage: ksp_qp_pde [--name PROBLEM|all] [--nc N] [--tol T] [--max-iter N] [--time-limit S]\n"
            "  Builds and solves a PDE-constrained QP (see include/pde_generator.hpp).\n"
            "  --name PROBLEM   one of: l1l2_poisson, l1l2_convdiff, l2_poisson_control,\n"
            "                   l2_poisson_state, l2_convdiff, or \"all\" to solve every one\n"
            "                   in sequence (default: l2_convdiff)\n"
            "  --nc N           grid exponent passed to the PDE generator (default: 6)\n"
            "  --tol T          primal-dual tolerance (default: 1e-6)\n"
            "  --max-iter N     max PMM iterations (default: 3000)\n"
            "  --time-limit S   time limit in seconds (default: 600)\n"
            "  --lump-mass      use a lumped (diagonal) mass matrix instead of the\n"
            "                   consistent FEM mass matrix (default: off; ignored when\n"
            "                   --discretization fd is used, FD is always lumped)\n"
            "  --discretization fem|fd   spatial discretization for the PDE operator\n"
            "                   (default: fem)\n";
        return 0;
    }

    std::string name = cli::get_str(argc, argv, "--name", "l2_convdiff");
    int nc = cli::get_int(argc, argv, "--nc", 6);
    double tol = cli::get_double(argc, argv, "--tol", 1e-6);
    int max_iter = cli::get_int(argc, argv, "--max-iter", 3000);
    double time_limit = cli::get_double(argc, argv, "--time-limit", 600.0); // in seconds
    bool lump_mass = cli::has_flag(argc, argv, "--lump-mass");
    pdegen::Discretization disc = parse_discretization(cli::get_str(argc, argv, "--discretization", "fem"));
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::SSN;

    std::time_t curr_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::ctime(&curr_time);

    std::vector<std::string> problems = (name == "all") ? kAllProblems : std::vector<std::string>{name};

    for (const std::string& problem_name : problems) {
        std::cout << "========== Solving " << problem_name << " ==========\n";
        KSPQPdata<T> data = build_problem(problem_name, nc, lump_mass, disc);
        Problem<T> pb(data, tol, max_iter, time_limit, when, what);
        KSP_QP<T> solver(pb);
        Solution<T> sol = solver.solve();
        sol.print_summary();
    }

    return 0;
}