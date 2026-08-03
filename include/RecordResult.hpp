#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Problem.hpp"

template <typename T>
struct TestResult {
    std::string system;
    bool agree;
    int opt_status;
    bool diverged;

    std::string name;
    T abs_err;
    T rel_err;

    T obj_val;
    int pmm_iter;
    int ssn_iter;
    int krylov_iter;
    int fact;
    int smw_count;
    T pmm_tol_achieved;
    T ssn_tol_achieved;

    double run_time; // in seconds
    int linesearch_fail;
    int krylov_fail;
};

template <typename T>
void print_feasibility(const PDPMMdata<T>& pd,
                       const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                       const T tol) {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    std::cout << "\nChecking feasibility of solution x from PMM_SSN solver:\n";

    std::cout << "  ||Ax - b||_2 = ";
    if (pd.A.rows() == 0) {
        std::cout << "0 (m = 0)\n";
    } else {
        std::cout << (pd.A * x - pd.b).norm() << "\n";
    }

    std::cout << "  ||Ax - b||_inf = ";
    if (pd.A.rows() == 0) {
        std::cout << "0 (m = 0)\n";
    } else {
        std::cout << (pd.A * x - pd.b).cwiseAbs().maxCoeff() << "\n";
    }

    std::cout << "\n  Elements of x outside bounds:\n";
    for (int i = 0; i < pd.c.size(); ++i) {
        if (x[i] < pd.lx[i] - tol || x[i] > pd.ux[i] + tol) {
            std::cout << "      Variable " << i
                      << " out of bounds: x = " << x[i]
                      << ", [" << pd.lx[i] << ", " << pd.ux[i] << "]\n";
        }
    }

    std::cout << "\n  Elements of Bx outside bounds:\n";
    Vec Bx = pd.B * x;
    for (int i = 0; i < pd.lw.size(); ++i) {
        T Bx_i = Bx[i];
        if (Bx_i < pd.lw[i] - tol || Bx_i > pd.uw[i] + tol) {
            std::cout << "      Variable " << i
                      << " out of bounds: Bx = " << Bx_i
                      << ", [" << pd.lw[i] << ", " << pd.uw[i] << "]\n";
        }
    }
}

inline void write_csv_header(const std::string& path) {
    namespace fs = std::filesystem;

    if (!fs::exists(fs::path(path)) || fs::is_empty(fs::path(path))) {
        std::ofstream csv(path);
        csv << "System,agree,opt_status,diverged,name,abs_err,rel_err,obj_val,"
            << "pmm_iter,ssn_iter,krylov_iter,fact,smw_count,pmm_tol_achieved,ssn_tol_achieved,"
            << "run_time,linesearch_fail,krylov_fail\n";
    } else if (!fs::is_empty(fs::path(path))) {
        std::ofstream csv(path, std::ios::out | std::ios::app);
        csv << "\n";
    }
}

template <typename T>
void append_csv_result(const std::string& path, const TestResult<T>& r) {
    std::ofstream csv(path, std::ios::out | std::ios::app);
    csv << r.system << "," << r.agree << "," << r.opt_status << "," << r.diverged << ","
        << r.name << "," << r.abs_err << "," << r.rel_err << ","
        << r.obj_val << "," << r.pmm_iter << "," << r.ssn_iter << ","
        << r.krylov_iter << "," << r.fact << "," << r.smw_count << ","
        << r.pmm_tol_achieved << "," << r.ssn_tol_achieved << ","
        << r.run_time << "," << r.linesearch_fail << "," << r.krylov_fail << "\n";
}
