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

#include "Highs.h"
#include "lp_to_pdpmm.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

struct NetlibTestResult {
    // Comparison result for a Netlib test problem
    std::string name;
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

void write_csv_header(const std::string& path) {
    namespace fs = std::filesystem;
    if (!fs::exists(fs::path(path)) || fs::is_empty(fs::path(path))) {
        std::ofstream csv(path);
        csv << "name,rel_agree,rel_err,opt_status,obj_val,"
            << "PMM_iter,SSN_iter,PMM_tol_achieved,SSN_tol_achieved,solving_time_sec\n";
    }
}

void append_csv_result(const std::string& path, const NetlibTestResult& r) {
    std::ofstream csv(path, std::ios::out | std::ios::app);
    csv << r.name << "," << (r.rel_agree ? "1" : "0") << "," << r.rel_err << ","
        << r.opt_status << "," << r.obj_val << ","
        << r.PMM_iter << "," << r.SSN_iter << ","
        << r.PMM_tol_achieved << "," << r.SSN_tol_achieved << ","
        << r.solving_time_sec << "\n";
    csv.close();
}

/*
int main() {

    std::string filename = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib/80BAU3B.mps";
    
    // Solving via HiGHS
    Highs h;
    h.setOptionValue("output_flag", true);
    h.readModel(filename);
    h.run();
    double ref_obj_val = h.getObjectiveValue();

    std::cout << "===============================================\n";

    // Extract problem data from the mps file
    const HighsLp& lp = h.getLp();
    PDPMMdata<T> pd = lp_to_pdpmm<T>(lp);

    // Construct the problem and solver
    T tol = 1e-4;
    int max_iter = 100;
    PrintWhen PMM_print_when = PrintWhen::ALWAYS;
    PrintWhat PMM_print_what = PrintWhat::MINIMAL;
    PrintWhen SSN_print_when = PrintWhen::END_ONLY;
    PrintWhat SSN_print_what = PrintWhat::MINIMAL;

    Problem<T> prob(pd.Q, pd.A, pd.B, pd.c, pd.b, pd.lx, pd.ux, pd.lw, pd.uw,
                    tol, max_iter, PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);
    SSN_PMM<T> solver(prob);
    
    // Solve the LP using PD-PMM_SSN solver
    auto t0 = std::chrono::steady_clock::now();
    Solution<T> sol = solver.solve();
    auto t1 = std::chrono::steady_clock::now();
    double solving_time_sec = time_diff_ms(t0, t1) * 1e-3;
    sol.print_summary();
    std::cout << "\nPD-PMM solver took " << solving_time_sec << " s.\n";
    T obj_val = sol.obj_val;
   
    // Compare
    std::cout << std::setprecision(5) << std::scientific;
    T abs_err = std::abs(obj_val - ref_obj_val);
    T err = abs_err / std::abs(ref_obj_val);
    bool abs_agree = abs_err <= 1e-4;
    bool agree = err <= 1e-4;
    if (agree) std::cout << "\nCORRECT! Relative error = " << err << "\n";
    else std::cout << "\nIncorrect. Relative error = " << err << "\n";

    // std::cout << "\nChecking feasibility of solution x from PMM_SSN solver:\n";
    // std::cout << "  ||Ax - b||_inf = ";
    // if (prob.A.rows() == 0) std::cout << "0 (m = 0)\n";
    // else std::cout << (prob.A * sol.x - prob.b).cwiseAbs().maxCoeff() << "\n";

    // std::cout << "\n  Elements of x outside bounds:\n";
    // for (int i = 0; i < prob.c.size(); ++i) {
    //     if (sol.x[i] < prob.lx[i] - tol || sol.x[i] > prob.ux[i] + tol) {
    //         std::cout << "      Variable " << i << " out of bounds: x = " << sol.x[i]
    //                   << ", [" << prob.lx[i] << ", " << prob.ux[i] << "]\n";
    //     }
    // }
    // std::cout << "\n  Elements of Bx outside bounds:\n";
    // for (int i = 0; i < prob.lw.size(); ++i) {
    //     T Bx_i = (prob.B * sol.x)[i];
    //     if (Bx_i < prob.lw[i] - tol || Bx_i > prob.uw[i] + tol) {
    //         std::cout << "      Variable " << i << " out of bounds: Bx = " << Bx_i
    //                   << ", [" << prob.lw[i] << ", " << prob.uw[i] << "]\n";
    //     }
    // }

    return 0;
}
*/

int main() {

    // Filenames of Netlib test problems (without .mps extension)
    std::vector<std::string> netlib_names = {"25FV47","80BAU3B","ADLITTLE","AFIRO","AGG","AGG2","AGG3","BANDM","BEACONFD","BLEND","BNL1","BNL2","BOEING1","BOEING2","BORE3D","BRANDY","CAPRI","CYCLE","CZPROB","D2Q06C","D6CUBE","DEGEN2","DEGEN3","DFL001",
                                            "E226","ETAMACRO","FFFFF800","FINNIS","FIT1D","FIT1P","FIT2D","FIT2P","FORPLAN","GANGES","GFRD-PNC","GREENBEA","GREENBEB","GROW15","GROW22","GROW7","ISRAEL","KB2","LOTFI","MAROS","MAROS-R7","MODSZK1","NESM",
                                            "PEROLD","PILOT","PILOT.JA","PILOT.WE","PILOT4","PILOT87","PILOTNOV","QAP8","QAP12","QAP15","RECIPE","SC105","SC205","SC50A","SC50B","SCAGR25","SCAGR7","SCFXM1","SCFXM2","SCFXM3","SCORPION","SCRS8","SCSD1","SCSD6","SCSD8","SCTAP1","SCTAP2","SCTAP3",
                                            "SEBA","SHARE1B","SHARE2B","SHELL","SHIP04L","SHIP04S","SHIP08L","SHIP08S","SHIP12L","SHIP12S","SIERRA","STAIR","STANDATA","STANDGUB","STANDMPS","STOCFOR1","STOCFOR2","STOCFOR3","TRUSS","TUFF","VTP.BASE","WOOD1P","WOODW"};
    // std::vector<std::string> kennington_names = {"CRE-A","CRE-B","CRE-C","CRE-D","KEN-07","KEN-11","KEN-13","KEN-18","OSA-07","OSA-14","OSA-30","OSA-60","PDS-02","PDS-06","PDS-10","PDS-20"};

    // Root
    std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/";

    // Parameters in common
    T tol = 1e-4;
    int max_iter = 1e2;
    PrintWhen PMM_print_when = PrintWhen::EVERY10;
    PrintWhat PMM_print_what = PrintWhat::MINIMAL;
    PrintWhen SSN_print_when = PrintWhen::NEVER;
    PrintWhat SSN_print_what = PrintWhat::NONE;

    // Solver result
    std::string csv_path = root + "results/netlib_test_w_bcl_ruiz_gamma=07.csv";
    write_csv_header(csv_path);

    for (const auto& name : netlib_names) {

        // Build full path and check if file exists
        std::string filename = root + "data/netlib/" + name + ".mps";
        if (!std::filesystem::exists(filename)) {
            std::cerr << "SKIP: File not found: " << name << "\n";
            continue;
        }

        std::cout << "\n==========Solving " << name << "==========\n";
        
        try {
            // 1. Solve using HiGHS
            Highs h;
            h.setOptionValue("output_flag", false);
            h.readModel(filename);
            h.run();
            double ref_obj_val = h.getObjectiveValue();

            // 2. Extract problem data from the mps file
            const HighsLp& lp = h.getLp();
            PDPMMdata<T> pd = lp_to_pdpmm<T>(lp);

            // 3. Construct the problem and solver
            Problem<T> prob(pd.Q, pd.A, pd.B, pd.c, pd.b, pd.lx, pd.ux, pd.lw, pd.uw,
                            tol, max_iter, PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);
            SSN_PMM<T> solver(prob);

            // 4. Solve the LP
            auto t0 = std::chrono::steady_clock::now();
            Solution<T> sol = solver.solve();
            auto t1 = std::chrono::steady_clock::now();
            double solving_time_sec = time_diff_ms(t0, t1) * 1e-3;
            std::cout << "\nPD-PMM solver took " << solving_time_sec << " s.\n";

            int opt_status = sol.opt;
            T obj_val = sol.obj_val;
            int PMM_iter = sol.PMM_iter;
            int SSN_iter = sol.SSN_iter;
            T PMM_tol_achieved = sol.PMM_tol_achieved;
            T SSN_tol_achieved = sol.SSN_tol_achieved;

            // 5. Compare
            T abs_err = std::abs(obj_val - ref_obj_val);
            T rel_err = abs_err / std::abs(ref_obj_val);
            bool rel_agree = rel_err <= 1e-4;
            if (rel_agree) std::cout << "\nCORRECT! Relative error = " << rel_err << "\n";
            else std::cout << "\nIncorrect. Relative error = " << rel_err << "\n";

            // 6. Store result
            NetlibTestResult result = {
                name, rel_agree, rel_err,
                opt_status, obj_val, PMM_iter, SSN_iter,
                PMM_tol_achieved, SSN_tol_achieved,
                solving_time_sec
            };
            append_csv_result(csv_path, result);

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n"; 
            NetlibTestResult result = {
                name, false, -1.0,
                -1, -1.0, -1, -1,
                -1.0, -1.0,
                -1.0
            };
            append_csv_result(csv_path, result);
        }
    }

    return 0;
}

