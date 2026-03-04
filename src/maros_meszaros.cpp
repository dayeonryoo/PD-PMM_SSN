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
#include "MpsParser.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

struct MarosMeszarosTestResult {
    // Comparison result for a Netlib test problem
    bool agree;
    std::string name;
    bool abs_agree;
    T abs_err;
    bool rel_agree;
    T rel_err;

    // Result summary from PD-PMM_SSN solver
    int opt_status;
    T obj_val;
    int PMM_iter;
    int SSN_iter;
    T PMM_tol_achieved;
    T SSN_tol_achieved;
    double solving_time_sec;
};

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
    std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/maros_meszaros/";
    
    // std::string name = "DUALC1";
    // T obj_val = 6.1552508e+03;

    std::string name = "AUG2DQP";
    T obj_val = 6.2370121e+06;

    // std::string name = "QSIERRA";
    // T obj_val =  2.3750458e+07;


    std::string filename = root + name + ".SIF";

    std::cout << "==================== Solving " + name << " ====================\n";
    
    MpsParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    PDPMMdata<T> pd = parser.to_pdpmm(model);

    T tol = 1e-4;
    int max_iter = 100;
    PrintWhen PMM_print_when = PrintWhen::ALWAYS;
    PrintWhat PMM_print_what = PrintWhat::MINIMAL;
    PrintWhen SSN_print_when = PrintWhen::END_ONLY;
    PrintWhat SSN_print_what = PrintWhat::MINIMAL;

    Problem<T> prob(pd, tol, max_iter, PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);
    SSN_PMM<T> solver(prob);

    std::cout << "n = " << prob.n << ", m = " << prob.m << ", l = " << prob.l << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    Solution<T> sol = solver.solve();
    sol.print_summary();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nPMM solver took " << elapsed.count() << " s.\n";

    // Check feasibility
    // print_feasibility(pd, sol.x, tol);

    // Compare with reference objective value
    T abs_error = std::abs(sol.obj_val - obj_val);
    T rel_error = abs_error / std::abs(obj_val);
    T err_tol = 1e-2;
    if (abs_error < err_tol || rel_error < err_tol) {
        std::cout << "Correct! Absolute error = " << abs_error << ", relative error: " << rel_error << "\n";
    } else {
        std::cout << "Incorrect. Absolute error = " << abs_error << ", relative error: " << rel_error << "\n";
    }

    return 0;
}


void write_csv_header(const std::string& path) {
    namespace fs = std::filesystem;
    if (!fs::exists(fs::path(path)) || fs::is_empty(fs::path(path))) {
        std::ofstream csv(path);
        csv << "agree,name,abs_agree,abs_err,rel_agree,rel_err,opt_status,obj_val,"
            << "PMM_iter,SSN_iter,PMM_tol_achieved,SSN_tol_achieved,solving_time_sec\n";
    }
}

void append_csv_result(const std::string& path, const MarosMeszarosTestResult& r) {
    std::ofstream csv(path, std::ios::out | std::ios::app);
    csv << r.agree << "," << r.name << ","
        << (r.abs_agree ? "1" : "0") << "," << r.abs_err << ","
        << (r.rel_agree ? "1" : "0") << "," << r.rel_err << ","
        << r.opt_status << "," << r.obj_val << ","
        << r.PMM_iter << "," << r.SSN_iter << ","
        << r.PMM_tol_achieved << "," << r.SSN_tol_achieved << ","
        << r.solving_time_sec << "\n";
    csv.close();
}

/*
int main() {
    
    // Filenames and objective values of Maros/Meszaros QPs
    static const std::map<std::string, double> QPs = {
        {"AUG2D",      1.6874118e+06},
        {"AUG2DC",     1.8183681e+06},
        {"AUG2DCQP",   6.4981348e+06},
        {"AUG2DQP",    6.2370121e+06},
        {"AUG3D",      5.5406773e+02},
        {"AUG3DC",     7.7126244e+02},
        {"AUG3DCQP",   9.9336215e+02},
        {"AUG3DQP",    6.7523767e+02},
        {"BOYD1",     -6.1735220e+07},
        {"BOYD2",      2.1256767e+01},
        {"CONT-050",  -4.5638509e+00},
        {"CONT-100",  -4.6443979e+00},
        {"CONT-101",   1.9552733e-01},
        {"CONT-200",  -4.6848759e+00},
        {"CONT-201",   1.9248337e-01},
        // {"CONT-300",   1.9151232e-01}, // dimension too large
        // {"CVXQP1L",    1.0870480e+08}, // too many QNZ
        {"CVXQP1M",    1.0875116e+06},
        {"CVXQP1S",    1.1590718e+04},
        // // {"CVXQP2L",    8.1842458e+07}, // too many QNZ
        {"CVXQP2M",    8.2015543e+05},
        {"CVXQP2S",    8.1209405e+03},
        // {"CVXQP3L",    1.1571110e+08}, // too many QNZ
        {"CVXQP3M",    1.3628287e+06},
        {"CVXQP3S",    1.1943432e+04},
        {"DPKLO1",     3.7009622e-01},
        {"DTOC3",      2.3526248e+02},
        {"DUAL1",      3.5012966e-02},
        {"DUAL2",      3.3733676e-02},
        {"DUAL3",      1.3575584e-01},
        {"DUAL4",      7.4609084e-01},
        {"DUALC1",     6.1552508e+03},
        {"DUALC2",     3.5513077e+03},
        {"DUALC5",     4.2723233e+02},
        {"DUALC8",     1.8309359e+04},
        // {"EXDATA",    -1.4184343e+02}, // too many QNZ
        {"GENHS28",    9.2717369e-01},
        {"GOULDQP2",   1.8427534e-04},
        {"GOULDQP3",   2.0627840e+00},
        {"HS118",      6.6482045e+02},
        {"HS21",      -9.9960000e+01},
        {"HS268",      5.7310705e-07},
        {"HS35",       1.1111111e-01},
        {"HS35MOD",    2.5000000e-01},
        {"HS51",       8.8817842e-16},
        {"HS52",       5.3266476e+00},
        {"HS53",       4.0930233e+00},
        {"HS76",      -4.6818182e+00},
        {"HUES-MOD",   3.4824690e+07},
        {"HUESTIS",    3.4824690e+11},
        {"KSIP",       5.7579794e-01},
        {"LASER",      2.4096014e+06},
        {"LISWET1",    3.6122402e+01},
        {"LISWET10",   4.9485785e+01},
        {"LISWET11",   4.9523957e+01},
        {"LISWET12",   1.7369274e+03},
        {"LISWET2",    2.4998076e+01},
        {"LISWET3",    2.5001220e+01},
        {"LISWET4",    2.5000112e+01},
        {"LISWET5",    2.5034253e+01},
        {"LISWET6",    2.4995748e+01},
        {"LISWET7",    4.9884089e+02},
        {"LISWET8",    7.1447006e+03},
        {"LISWET9",    1.9632513e+03},
        {"LOTSCHD",    2.3984159e+03},
        {"MOSARQP1",  -9.5287544e+02},
        {"MOSARQP2",  -1.5974821e+03},
        {"POWELL20",   5.2089583e+10},
        {"PRIMAL1",   -3.5012965e-02},
        {"PRIMAL2",   -3.3733676e-02},
        {"PRIMAL3",   -1.3575584e-01},
        {"PRIMAL4",   -7.4609083e-01},
        {"PRIMALC1",  -6.1552508e+03},
        {"PRIMALC2",  -3.5513077e+03},
        {"PRIMALC5",  -4.2723233e+02},
        {"PRIMALC8",  -1.8309430e+04},
        // {"Q25FV47",    1.3744448e+07}, // too many QNZ
        {"QADLITTL",   4.8031886e+05},
        {"QAFIRO",    -1.5907818e+00},
        {"QBANDM",     1.6352342e+04},
        {"QBEACONF",   1.6471206e+05},
        {"QBORE3D",    3.1002008e+03},
        {"QBRANDY",    2.8375115e+04},
        {"QCAPRI",     6.6793293e+07},
        {"QE226",      2.1265343e+02},
        {"QETAMACR",   8.6760370e+04},
        {"QFFFFF80",   8.7314747e+05},
        {"QFORPLAN",   7.4566315e+09},
        {"QGFRDXPN",   1.0079059e+11},
        {"QGROW15",   -1.0169364e+08},
        {"QGROW22",   -1.4962895e+08},
        {"QGROW7",    -4.2798714e+07},
        {"QISRAEL",    2.5347838e+07},
        {"QPCBLEND",  -7.8425409e-03},
        {"QPCBOEI1",   1.1503914e+07},
        {"QPCBOEI2",   8.1719623e+06},
        {"QPCSTAIR",   6.2043875e+06},
        {"QPILOTNO",   4.7285869e+06},
        {"QPTEST",     4.3718750e+00},
        {"QRECIPE",   -2.6661600e+02},
        {"QSC205",    -5.8139518e-03},
        {"QSCAGR25",   2.0173794e+08},
        {"QSCAGR7",    2.6865949e+07},
        {"QSCFXM1",    1.6882692e+07},
        {"QSCFXM2",    2.7776162e+07},
        {"QSCFXM3",    3.0816355e+07},
        {"QSCORPIO",   1.8805096e+03},
        {"QSCRS8",     9.0456001e+02},
        {"QSCSD1",     8.6666667e+00}, 
        {"QSCSD6",     5.0808214e+01},
        {"QSCSD8",     9.4076357e+02},
        {"QSCTAP1",    1.4158611e+03},
        {"QSCTAP2",    1.7350265e+03},
        {"QSCTAP3",    1.4387547e+03},
        {"QSEBA",      8.1481801e+07},
        {"QSHARE1B",   7.2007832e+05},
        {"QSHARE2B",   1.1703692e+04},
        // {"QSHELL",     1.5726368e+12}, // too many QNZ
        {"QSHIP04L",   2.4200155e+06},
        {"QSHIP04S",   2.4249937e+06},
        // {"QSHIP08L",   2.3760406e+06}, // too many QNZ
        // {"QSHIP08S",   2.3857289e+06}, // too many QNZ
        // {"QSHIP12L",   3.0188766e+06}, // too many QNZ
        // {"QSHIP12S",   3.0569623e+06}, // too many QNZ
        {"QSIERRA",    2.3750458e+07},
        {"QSTAIR",     7.9854528e+06},
        {"QSTANDAT",   6.4118384e+03},
        {"S268",       5.7310705e-07},
        {"STADAT1",   -2.8526864e+07},
        {"STADAT2",   -3.2626665e+01},
        {"STADAT3",   -3.5779453e+01},
        // {"STCQP1",     1.5514356e+05}, // too many QNZ
        // {"STCQP2",     2.2327313e+04}, // too many QNZ
        {"TAME",       0.0000000e+00},
        {"UBH1",       1.1160008e+00},
        {"VALUES",    -1.3966211e+00},
        {"YAO",        1.9770426e+02},
        {"ZECEVIC2",  -4.1250000e+00},
    };

    std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/";

    // Parameters
    T tol = 1e-4;
    int max_iter = 100;
    PrintWhen PMM_print_when = PrintWhen::EVERY10;
    PrintWhat PMM_print_what = PrintWhat::MINIMAL;
    PrintWhen SSN_print_when = PrintWhen::NEVER;
    PrintWhat SSN_print_what = PrintWhat::MINIMAL;

    // Solver result
    std::string csv_path = root + "results/maros_meszaros_test5.csv";
    write_csv_header(csv_path);

    for (const auto& [name, ref_obj_val] : QPs) {
        // Build full path and check if file exists
        std::string filename = root + "data/maros_meszaros/" + name + ".SIF";
        if (!std::filesystem::exists(filename)) {
            std::cerr << "SKIP: File not found: " << name << "\n";
            continue;
        }

        std::cout << "\n==========Solving " << name << "==========\n";

        try {
            // Read problem data from the file
            MpsParser<T> parser;
            ParsedModel<T> model = parser.parse(filename);
            PDPMMdata<T> pd = parser.to_pdpmm(model);

            // Construct the problem and solver
            Problem<T> prob(pd, tol, max_iter, PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);
            SSN_PMM<T> solver(prob);

            // Solve the QP
            auto t0 = std::chrono::steady_clock::now();
            Solution<T> sol = solver.solve();
            sol.print_summary();
            auto t1 = std::chrono::steady_clock::now();
            double solving_time_sec = time_diff_ms(t0, t1) * 1e-3;
            std::cout << "\nPMM solver took " << solving_time_sec << " s.\n";

            // Compare
            T abs_err = std::abs(sol.obj_val - ref_obj_val);
            T rel_err = abs_err / std::abs(sol.obj_val);
            T err_tol = 1e-2;
            bool abs_agree = abs_err < err_tol;
            bool rel_agree = rel_err < err_tol;
            bool agree = abs_agree || rel_agree;
            if (agree) std::cout << "\nCORRECT! Absolute error = " << abs_err << ", Relative error = " << rel_err << "\n";
            else std::cout << "\nIncorrect Absolute error = " << abs_err << ", Relative error = " << rel_err << "\n";

            // Record result
            MarosMeszarosTestResult result = {
                agree, name, abs_agree, abs_err, rel_agree, rel_err,
                sol.opt, sol.obj_val, sol.PMM_iter, sol.SSN_iter,
                sol.PMM_tol_achieved, sol.SSN_tol_achieved, solving_time_sec
            };
            append_csv_result(csv_path, result);

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n";
            MarosMeszarosTestResult result = {
                false, name, false, -1.0, false, -1.0,
                -1, -1.0, -1, -1,
                -1.0, -1.0, -1.0
            };
            append_csv_result(csv_path, result);
        }
    }

    return 0;
}
*/