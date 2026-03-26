#pragma once
#include <iostream>
#include <iomanip>
#include <functional>
#include <chrono>

inline double time_diff_ms(const std::chrono::steady_clock::time_point& start,
                           const std::chrono::steady_clock::time_point& end) {
    using namespace std::chrono;
    return duration<double, std::milli>(end - start).count();
}

enum class PrintWhen {
    NEVER,
    EVERY10,
    ALWAYS
};

enum class PrintWhat {
    NONE,
    MINIMAL, // iter, tol
    TUNING, // iter, tol, params
    FULL, // iter, obj_val, tol, params
};

enum class PrintLabel {
    SSN,
    PMM
};

void print_header(PrintWhen when, PrintWhat what) {
    if (when == PrintWhen::NEVER || what == PrintWhat::NONE) return;

    const int w_iter = 8;
    const int w_val  = 14;

    std::cout << std::setw(w_iter) << "PMM" << std::setw(w_iter) << "SSN";
    if (what == PrintWhat::FULL) {
        std::cout << std::setw(w_val)  << "Objective";
    }
    std::cout << std::setw(w_val)  << "PrimalRes"
                << std::setw(w_val)  << "DualRes"
                << std::setw(w_val)  << "Comp_x"
                << std::setw(w_val)  << "Comp_Bx"
                << std::setw(w_val)  << "SSN_res";
    if (what == PrintWhat::TUNING || what == PrintWhat::FULL) {
        std::cout << std::setw(w_val) << "mu" << std::setw(w_val)  << "rho"  << std::setw(w_val) << "eps";
    }
    std::cout << "\n";

    std::cout << std::string(w_iter*2 + w_val*5, '-');
    if (what == PrintWhat::TUNING) {
        std::cout << std::string(w_val*3, '-');
    } else if (what == PrintWhat::FULL) {
        std::cout << std::string(w_val*4, '-');
    }
    std::cout << "\n";
}

template <typename T, typename Vec>
void print(PrintWhen when, PrintWhat what, int PMM_iter, int SSN_iter, T obj_val, const Vec& res_norms, T SSN_res, T mu, T rho, T eps_bcl, T eps) {
    if (when == PrintWhen::NEVER || what == PrintWhat::NONE) return;
    if (when == PrintWhen::EVERY10 && PMM_iter % 10 != 0) return;

    const int w_iter = 8;
    const int w_val  = 14;

    std::cout << std::setw(w_iter) << PMM_iter << std::setw(w_iter) << SSN_iter;
    if (what == PrintWhat::FULL) {
        std::cout << std::setw(w_val)  << std::scientific << obj_val;
    }
    T max_res = res_norms.maxCoeff();
    for (T res : res_norms) {
        if (res == max_res) std::cout << "\033[1m" << std::setw(w_val)  << res << "\033[0m";
        else std::cout << std::setw(w_val)  << res;
    }
    std::cout << std::setw(w_val)  << SSN_res;
    if (what == PrintWhat::TUNING || what == PrintWhat::FULL) {
        std::cout << std::setw(w_val)  << mu << std::setw(w_val)  << rho << std::setw(w_val) << eps;
    }
    std::cout << "\n";
}
