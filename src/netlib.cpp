#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <chrono>
#include <ctime>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "SSN_PMM.hpp"
#include "Problem.hpp"
#include "Printing.hpp"
#include "MpsParser.hpp"
#include "RecordResult.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

// ==================== Individual Netlib LP ====================
/*
int main() {

    std::string filename = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib/ADLITTLE.mps";
    T ref_obj_val = 2.2549496316e+05;

    std::cout << "===============================================\n";

    // Parameters for PD-PMM_SSN solver
    T tol = 1e-4;
    int max_iter = 500;
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::MINIMAL;

    // Extract problem data from the mps file using our MpsParser and construct solver
    MpsParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    PDPMMdata<T> pd = parser.to_pdpmm(model);

    Problem<T> prob(pd, tol, max_iter, when, what);
    SSN_PMM<T> solver(prob);
    
    // Solve the LP using PD-PMM_SSN solver
    auto t0 = std::chrono::steady_clock::now();
    Solution<T> sol = solver.solve();
    auto t1 = std::chrono::steady_clock::now();
    double solving_time_sec = time_diff_ms(t0, t1) * 1e-3;
    sol.print_summary();
    std::cout << "\nPD-PMM solver took " << solving_time_sec << " s.\n";
    T obj_val = sol.obj_val;
   
    // Check feasibility
    // print_feasibility(pd, sol.x, tol);

    // Check convergence
    if (sol.opt == 0) {
        std::cout << "Solver converged!\n";
    } else if (sol.opt == 3) {
        std::cout << "Lineserach failed. Solver terminated.\n";
    } else if (sol.opt < 0) {
        std::cout << "Solver detected infeasibility.\n";
    } else {
        std::cout << "Solver hit the max iteration before converging.\n";
    }
    if (sol.opt <= 0 && sol.PMM_tol_achieved > 1e0) {
        std::cout << "Solver possibly diverged.\n"; 
    }

    // Compare with reference objective value
    std::cout << std::setprecision(5) << std::scientific;
    T abs_err = std::abs(obj_val - ref_obj_val);
    T err = abs_err / std::abs(ref_obj_val);
    bool agree = err <= 1e-4;
    if (agree) std::cout << "\nCORRECT! Relative error = " << err << "\n";
    else std::cout << "\nIncorrect. Relative error = " << err << "\n";

    return 0;
}
*/

// ==================== Run a set of Netlib LPs ====================
/*
int main() {
    std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/";

    // Netlib LPs
    std::map<std::string,double> LPs = {
        {"25FV47", 5.5018458883E+03},
        {"80BAU3B", 9.8723216072E+05},
        {"ADLITTLE", 2.2549496316E+05},
        {"AFIRO", -4.6475314286E+02},
        {"AGG", -3.5991767287E+07},
        {"AGG2", -2.0239252356E+07},
        {"AGG3", 1.0312115935E+07},
        {"BANDM", -1.5862801845E+02},
        {"BEACONFD", 3.3592485807E+04},
        {"BLEND", -3.0812149846E+01},
        {"BNL1", 1.9776292856E+03},
        {"BNL2", 1.8112365404E+03},
        {"BOEING1", -3.3521356751E+02},
        {"BOEING2", -3.1501872802E+02},
        {"BORE3D", 1.3730803942E+03},
        {"BRANDY", 1.5185098965E+03},
        {"CAPRI", 2.6900129138E+03},
        {"CYCLE", -5.2263930249E+00},
        {"CZPROB", 2.1851966989E+06},
        {"D2Q06C", 1.2278423615E+05},
        {"D6CUBE", 3.1549166667E+02},
        {"DEGEN2", -1.4351780000E+03},
        {"DEGEN3", -9.8729400000E+02},
        {"DFL001", 1.12664E+07},
        {"E226", -1.8751929066E+01},
        {"ETAMACRO", -7.5571521774E+02},
        {"FFFFF800", 5.5567961165E+05},
        {"FINNIS", 1.7279096547E+05},
        {"FIT1D", -9.1463780924E+03},
        {"FIT1P", 9.1463780924E+03},
        {"FIT2D", -6.8464293294E+04},
        {"FIT2P", 6.8464293232E+04},
        {"FORPLAN", -6.6421873953E+02},
        {"GANGES", -1.0958636356E+05},
        {"GFRD-PNC", 6.9022359995E+06},
        {"GREENBEA", -7.2462405908E+07},
        {"GREENBEB", -4.3021476065E+06},
        {"GROW15", -1.0687094129E+08},
        {"GROW22", -1.6083433648E+08},
        {"GROW7", -4.7787811815E+07},
        {"ISRAEL", -8.9664482186E+05},
        {"KB2", -1.7499001299E+03},
        {"LOTFI", -2.5264706062E+01},
        {"MAROS", -5.8063743701E+04},
        {"MAROS-R7", 1.4971851665E+06},
        {"MODSZK1", 3.2061972906E+02},
        {"NESM", 1.4076073035E+07},
        {"PEROLD", -9.3807580773E+03},
        {"PILOT", -5.5740430007E+02},
        {"PILOT.JA", -6.1131344111E+03},
        {"PILOT.WE", -2.7201027439E+06},
        {"PILOT4", -2.5811392641E+03},
        {"PILOT87", 3.0171072827E+02},
        {"PILOTNOV", -4.4972761882E+03},
        {"QAP8", 2.0350000000E+02},
        {"QAP12", 5.2289435056E+02},
        {"QAP15", 1.0409940410E+03},
        {"RECIPE", -2.6661600000E+02},
        {"SC105", -5.2202061212E+01},
        {"SC205", -5.2202061212E+01},
        {"SC50A", -6.4575077059E+01},
        {"SC50B", -7.0000000000E+01},
        {"SCAGR25", -1.4753433061E+07},
        {"SCAGR7", -2.3313892548E+06},
        {"SCFXM1", 1.8416759028E+04},
        {"SCFXM2", 3.6660261565E+04},
        {"SCFXM3", 5.4901254550E+04},
        {"SCORPION", 1.8781248227E+03},
        {"SCRS8", 9.0429998619E+02},
        {"SCSD1", 8.6666666743E+00},
        {"SCSD6", 5.0500000078E+01},
        {"SCSD8", 9.0499999993E+02},
        {"SCTAP1", 1.4122500000E+03},
        {"SCTAP2", 1.7248071429E+03},
        {"SCTAP3", 1.4240000000E+03},
        {"SEBA", 1.5711600000E+04},
        {"SHARE1B", -7.6589318579E+04},
        {"SHARE2B", -4.1573224074E+02},
        {"SHELL", 1.2088253460E+09},
        {"SHIP04L", 1.7933245380E+06},
        {"SHIP04S", 1.7987147004E+06},
        {"SHIP08L", 1.9090552114E+06},
        {"SHIP08S", 1.9200982105E+06},
        {"SHIP12L", 1.4701879193E+06},
        {"SHIP12S", 1.4892361344E+06},
        {"SIERRA", 1.5394362184E+07},
        {"STAIR", -2.5126695119E+02},
        {"STANDATA", 1.2576995000E+03},
        {"STANDMPS", 1.4060175000E+03},
        {"STOCFOR1", -4.1131976219E+04},
        {"STOCFOR2", -3.9024408538E+04},
        {"STOCFOR3", -3.9976661576E+04},
        {"TRUSS", 4.5881584719E+05},
        {"TUFF", 2.9214776509E-01},
        {"VTP.BASE", 1.2983146246E+05},
        {"WOOD1P", 1.4429024116E+00},
        {"WOODW", 1.3044763331E+00}
    };
    
    // Parameters in common
    T tol = 1e-4;
    int max_iter = 500;
    PrintWhen when = PrintWhen::EVERY10;
    PrintWhat what = PrintWhat::TUNING;

    // Solver result
    std::string csv_path = root + "results/netlib_lp4.csv";
    write_csv_header(csv_path);

    for (const auto& [name, ref_obj_val] : LPs) {

        // Build full path and check if file exists
        std::string filename = root + "data/netlib/" + name + ".mps";
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

            Problem<T> prob(pd, tol, max_iter, when, what);
            SSN_PMM<T> solver(prob);

            std::time_t curr_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "Compuation started at " << std::ctime(&curr_time);

            // Solve the LP
            auto t0 = std::chrono::steady_clock::now();
            Solution<T> sol = solver.solve();
            auto t1 = std::chrono::steady_clock::now();
            double solving_time_sec = time_diff_ms(t0, t1) * 1e-3;
            std::cout << "\nPD-PMM solver took " << solving_time_sec << " s.\n";

            // Check convergence
            bool diverged = false;
            if (sol.opt == 0) {
                std::cout << "Solver converged!\n";
            } else if (sol.opt == 3) {
                std::cout << "Lineserach failed. Solver terminated.\n";
            } else {
                std::cout << "Solver hit the max iteration before converging.\n";
            }
            if (sol.PMM_tol_achieved > 1e0) {
                diverged = true;
                std::cout << "Solver possibly diverged.\n"; 
            }

            // Compare
            T abs_err = std::abs(sol.obj_val - ref_obj_val);
            T rel_err = abs_err / std::abs(ref_obj_val);
            T err_tol = 1e-2;
            bool abs_agree = abs_err < err_tol;
            bool rel_agree = rel_err < err_tol;
            bool agree = abs_agree || rel_agree;
            if (agree) std::cout << "\nCORRECT! Absolute error = " << abs_err << ", Relative error = " << rel_err << "\n";
            else std::cout << "\nIncorrect Absolute error = " << abs_err << ", Relative error = " << rel_err << "\n";

            // Store result
            std::string system = "S";
            TestResult<T> result = {
                system, agree, sol.opt, diverged,
                name, abs_err, rel_err,
                sol.obj_val, sol.PMM_iter, sol.SSN_iter,
                sol.PMM_tol_achieved, sol.SSN_tol_achieved,
                solving_time_sec
            };
            append_csv_result(csv_path, result);

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n"; 
            TestResult<T> result = {
                e.what(), false, -5, false,
                name, -1.0, -1.0,
                -1.0, -1, -1,
                -1.0, -1.0, -1.0
            };
            append_csv_result(csv_path, result);
        }
    }

    return 0;
}
*/

// ==================== Netlib infeasible problems ====================

int main() {
    // std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/";
    std::string root = "/Users/dianaryoo/Desktop/KCL/PD-PMM_SSN/";

    std::vector<std::string> LPs = {
        "BGDBG1",
        "BGETAM",
        "BGINDY",
        "BGPRTR",
        "BOX1",
        "CERIA3D",
        "CHEMCOM",
        "CPLEX1",
        "CPLEX2",
        "EX72A",
        "EX73A",
        "FOREST6",
        "GALENET",
        "GOSH",
        "GRAN",
        "GREENBEA",
        "ITEST2",
        "ITEST6",
        "KLEIN1",
        "KLEIN2",
        "KLEIN3",
        "MONDOU2",
        "PANG",
        "PILOT4I",
        "QUAL",
        "REACTOR",
        "REFINERY",
        "VOL1",
        "WOODINFE"
    };
    
    // Parameters in common
    T tol = 1e-6;
    int max_iter = 1000;
    double time_limit = 600.0; // in seconds

    PrintWhen when = PrintWhen::EVERY10;
    PrintWhat what = PrintWhat::TUNING;

    // Solver result
    std::string csv_path = root + "results/0504netlib_infeas.csv";

    // Write header
    if (!std::filesystem::exists(std::filesystem::path(csv_path))
        || std::filesystem::is_empty(std::filesystem::path(csv_path))) {
        std::ofstream csv(csv_path);
        csv << "System,infeas_detected,opt_status,name,PMM_iter,SSN_iter,Krylov_iter,fact,PMM_res,SSN_res,solving_time_sec,linesearch_fails\n";
    }

    // Loop over all LPs
    for (const std::string name : LPs) {
        // Build full path and check if file exists
        std::string filename = root + "data/netlib_infeas/" + name + ".mps";
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

            Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
            SSN_PMM<T> solver(prob);

            std::time_t curr_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "Compuation started at " << std::ctime(&curr_time);

            // Chosen system:
            std::string system = "S";

            // Solve the LP
            auto t0 = std::chrono::steady_clock::now();
            Solution<T> sol = solver.solve();
            auto t1 = std::chrono::steady_clock::now();
            double solving_time_sec = time_diff_ms(t0, t1) * 1e-3;
            std::cout << "\nPD-PMM solver took " << solving_time_sec << " s.\n";

            // Store the result
            bool infeas_detected = (sol.opt == -2 || sol.opt == -3);
            std::ofstream csv(csv_path, std::ios::out | std::ios::app);
            csv << system << "," << infeas_detected << "," << sol.opt << ","
                << name << "," << sol.PMM_iter << "," << sol.SSN_iter << ","
                << sol.Krylov_iter << "," << sol.fact << ","
                << sol.PMM_tol_achieved << "," << sol.SSN_tol_achieved << ","
                << solving_time_sec << "," << sol.linesearch_fail << "\n";
            csv.close();

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n"; 
            std::ofstream csv(csv_path, std::ios::out | std::ios::app);
            csv << e.what() << "," << 0 << "," << -1 << ","
                << name << "," << -1 << "," << -1 << ","
                << -1 << "," << -1 << ","
                << -1.0 << "," << -1.0 << "," << -1.0 << "," << 0 << "\n";
            csv.close();
        }
    }
}


/*
int main() {

    std::string root = "/Users/dianaryoo/Desktop/KCL/PD-PMM_SSN/data/netlib_infeas/";
    std::string name = "GOSH";
    std::string filename = root + name + ".mps";

    // Parameters for PD-PMM_SSN solver
    T tol = 1e-6;
    int max_iter = 1000;
    PrintWhen when = PrintWhen::EVERY10;
    PrintWhat what = PrintWhat::TUNING;

    // Extract problem data from the mps file using our MpsParser and construct solver
    MpsParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    PDPMMdata<T> pd = parser.to_pdpmm(model);

    Problem<T> prob(pd, tol, max_iter, when, what);
    SSN_PMM<T> solver(prob);

    std::cout << "================================================ Solving " << name << " =================================================\n";

    // Chosen system
    std::cout << "n = " << solver.n << ", m = " << solver.m << ", l = " << solver.l << "\n";
    std::cout << "N = " << solver.N << ", M = " << solver.M << "\n";
    std::cout << "Solving Schur (CG with LDLT fallback).\n";

    // Solve the LP
    auto t0 = std::chrono::steady_clock::now();
    Solution<T> sol = solver.solve();
    auto t1 = std::chrono::steady_clock::now();
    double solving_time_sec = time_diff_ms(t0, t1) * 1e-3;
    sol.print_summary();
    std::cout << "\nPD-PMM solver took " << solving_time_sec << " s.\n";
    T obj_val = sol.obj_val;
   
    // Check feasibility
    // print_feasibility(pd, sol.x, tol);

    return 0;
}
*/