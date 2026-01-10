#pragma once
#include <iostream>
#include <functional>

enum class PrintWhen {
    NEVER,
    ALWAYS,
    END_ONLY
};

enum class PrintWhat {
    NONE,
    MINIMAL, // iter, tol
    SUMMARY, // iter, x, y2, tol, 
    FULL // iter, obj_val, x, y1, y2, z, tol
};

enum class PrintLabel {
    SSN,
    PMM
};

template <typename T, typename Vec>
std::function<void(int, int, T, const Vec&, const Vec&, const Vec&, const Vec&, T)>
make_print_function(PrintLabel label, PrintWhen when, PrintWhat what, int max_iter) {
    return [label, when, what, max_iter](int iter, int opt, T obj_val, const Vec& x, const Vec& y1,
                                         const Vec& y2, const Vec& z, T tol) {
        if (when == PrintWhen::NEVER) return;
        if (when == PrintWhen::END_ONLY && opt == -1) return;

        if (what == PrintWhat::MINIMAL) {
            switch (label) {
                case PrintLabel::SSN:
                    std::cout << "SSN ";
                    break;
                case PrintLabel::PMM:
                    std::cout << "PMM ";
                    break;
            }
            std::cout << "iter " << iter << ":\n";
            std::cout << "  tol = " << tol << "\n";
            switch (label) {
                case PrintLabel::PMM:
                    if (opt == 0) {
                        std::cout << "Optimal solution found at PMM iteration " << iter << ".\n";
                    } else if (opt == 1) {
                        std::cout << "Optimal solution not found within the maximum number of PMM iterations.\n";
                    }
                    break;
                case PrintLabel::SSN:
                    if (opt == 0) {
                        std::cout << "Optimal solution found at SSN iteration " << iter << ".\n";
                    } else if (opt == 1) {
                        std::cout << "Optimal solution not found within the maximum number of SSN iterations.\n";
                    }
                    break;
            }
        }
        if (what == PrintWhat::SUMMARY) {
            switch (label) {
                case PrintLabel::SSN:
                    std::cout << "SSN ";
                    break;
                case PrintLabel::PMM:
                    std::cout << "PMM ";
                    break;
            }
            std::cout << "iter " << iter << ":\n";
            std::cout << "  x = (" << x.transpose() << "),\n";
            std::cout << "  y2 = (" << y2.transpose() << "),\n";
            std::cout << "  tol = " << tol << "\n";
            switch (label) {
                case PrintLabel::PMM:
                    if (opt == 0) {
                        std::cout << "Optimal solution found at PMM iteration " << iter << ".\n";
                    } else if (opt == 1) {
                        std::cout << "Optimal solution not found within the maximum number of PMM iterations.\n";
                    }
                    break;
                case PrintLabel::SSN:
                    if (opt == 0) {
                        std::cout << "Optimal solution found at SSN iteration " << iter << ".\n";
                    } else if (opt == 1) {
                        std::cout << "Optimal solution not found within the maximum number of SSN iterations.\n";
                    }
                    break;
            }
        } else if (what == PrintWhat::FULL) {
            switch (label) {
                case PrintLabel::SSN:
                    std::cout << "SSN ";
                    break;
                case PrintLabel::PMM:
                    std::cout << "PMM ";
                    break;
            }
            std::cout << "iter " << iter << ":\n";
            if (label == PrintLabel::PMM) {
                std::cout << "  obj_val = " << obj_val << ",\n";
            }
            std::cout << "  x = (" << x.transpose() << "),\n";
            std::cout << "  y1 = (" << y1.transpose() << "),\n";
            std::cout << "  y2 = (" << y2.transpose() << "),\n";
            std::cout << "  z = (" << z.transpose() << "),\n";
            std::cout << "  tol = " << tol << "\n";
            switch (label) {
                case PrintLabel::PMM:
                    if (opt == 0) {
                        std::cout << "Optimal solution found at PMM iteration " << iter << ".\n";
                    } else if (opt == 1) {
                        std::cout << "Optimal solution not found within the maximum number of PMM iterations.\n";
                    }
                    break;
                case PrintLabel::SSN:
                    if (opt == 0) {
                        std::cout << "Optimal solution found at SSN iteration " << iter << ".\n";
                    } else if (opt == 1) {
                        std::cout << "Optimal solution not found within the maximum number of SSN iterations.\n";
                    }
                    break;
            }
        }
    };
}