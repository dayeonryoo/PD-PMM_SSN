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

#include "ksp_qp.hpp"
#include "problem.hpp"
#include "printing.hpp"
#include "mps_format_parser.hpp"
#include "record_result.hpp"
#include "cli_args.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

// ==================== Solving Netlib LPs ====================
/*
int main(int argc, char** argv) {
    if (cli::has_flag(argc, argv, "--help") || cli::has_flag(argc, argv, "-h")) {
        std::cout <<
            "Usage: ksp_qp_netlib [--root DIR] [--in DIR] [--name PROBLEM|all] [--tol T] [--max-iter N] [--time-limit S] [--out FILE]\n"
            "  Solves one Netlib LP from ROOT/IN/PROBLEM.mps, or sweeps the whole feasible set with --name all.\n"
            "  --root DIR       working directory; --in and --out are resolved relative to this (default: ./)\n"
            "  --in DIR         directory containing the .mps files, relative to --root (default: data/netlib/)\n"
            "  --name PROBLEM   problem name, without extension, or \"all\" to sweep every LP with a\n"
            "                   known reference objective (default: AFIRO)\n"
            "  --tol T          primal-dual tolerance (default: 1e-6)\n"
            "  --max-iter N     max PMM iterations (default: 3000)\n"
            "  --time-limit S   time limit in seconds (default: 60)\n"
            "  --out FILE       (--name all only) output CSV path, relative to --root (default: results/netlib_all.csv)\n";
        return 0;
    }

    std::string root = cli::get_str(argc, argv, "--root", "./");
    if (!root.empty() && root.back() != '/') root += '/';
    std::string in_dir = cli::get_str(argc, argv, "--in", "data/netlib/");
    if (!in_dir.empty() && in_dir.back() != '/') in_dir += '/';
    std::string data_dir = root + in_dir;
    std::string name = cli::get_str(argc, argv, "--name", "AFIRO");

    T tol = cli::get_double(argc, argv, "--tol", 1e-6);
    int max_iter = cli::get_int(argc, argv, "--max-iter", 3000);
    double time_limit = cli::get_double(argc, argv, "--time-limit", 60.0); // in seconds

    if (name == "all") {
        // Netlib LPs with known reference objective values
        std::map<std::string, double> LPs = {
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

        PrintWhen when = PrintWhen::NEVER;
        PrintWhat what = PrintWhat::TUNING;

        std::string csv_path = root + cli::get_str(argc, argv, "--out", "results/netlib_all.csv");
        write_csv_header(csv_path);

        for (const auto& [lp_name, ref_obj_val] : LPs) {

            // Build full path and check if file exists
            std::string filename = data_dir + lp_name + ".mps";
            if (!std::filesystem::exists(filename)) {
                std::cerr << "SKIP: File not found: " << lp_name << "\n";
                continue;
            }

            std::cout << "\n==========Solving " << lp_name << "==========\n";

            try {
                // Read problem data from the file
                MpsFormatParser<T> parser;
                ParsedModel<T> model = parser.parse(filename);
                KSPQPdata<T> pd = parser.to_kspqp(model);

                Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
                KSP_QP<T> solver(prob);

                // Solve the LP
                Solution<T> sol = solver.solve();

                // Compare
                T abs_err = std::abs(sol.obj_val - ref_obj_val);
                T rel_err = abs_err / std::abs(ref_obj_val);
                T err_tol = 1e-2;
                bool agree = (abs_err < err_tol) || (rel_err < err_tol);
                bool diverged = sol.pmm_tol_achieved > 1e0;

                std::string system = solver.kkt_ldlt_used ? "L" : "S";

                // Store result
                TestResult<T> result = {
                    system,
                    agree, static_cast<int>(sol.opt), diverged, lp_name,
                    abs_err, rel_err, sol.obj_val,
                    sol.pmm_iter, sol.ssn_iter, sol.krylov_iter, sol.fact, sol.smw_count,
                    sol.pmm_tol_achieved, sol.ssn_tol_achieved,
                    sol.run_time, sol.linesearch_fail, sol.krylov_fail
                };
                append_csv_result(csv_path, result);

            } catch (const std::exception& e) {
                std::cerr << "ERROR solving " << lp_name << ": " << e.what() << "\n";
                TestResult<T> result = {
                    e.what(),
                    false, -1, false, lp_name,
                    -1.0, -1.0, -1.0,
                    -1, -1, -1, -1, -1,
                    -1.0, -1.0,
                    -1.0, -1, -1
                };
                append_csv_result(csv_path, result);
            }
        }

        return 0;
    }

    // ---- Single-problem solve ----
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::TUNING;

    std::string filename = data_dir + name + ".mps";
    std::cout << "==================== Solving " + name << " ====================\n";

    MpsFormatParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    KSPQPdata<T> pd = parser.to_kspqp(model);

    Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
    KSP_QP<T> solver(prob);

    // Solve:
    Solution<T> sol = solver.solve();
    sol.print_summary();

    return 0;
}
*/
// ==================== Netlib infeasible problems ====================

int main(int argc, char** argv) {
    if (cli::has_flag(argc, argv, "--help") || cli::has_flag(argc, argv, "-h")) {
        std::cout <<
            "Usage: ksp_qp_netlib [--root DIR] [--in DIR] [--name PROBLEM|all] [--tol T] [--max-iter N] [--time-limit S] [--out FILE]\n"
            "  Solves one Netlib LP from ROOT/IN/PROBLEM.mps, or sweeps the whole feasible set with --name all.\n"
            "  --root DIR       working directory; --in and --out are resolved relative to this (default: ./)\n"
            "  --in DIR         directory containing the .mps files, relative to --root (default: data/netlib/)\n"
            "  --name PROBLEM   problem name, without extension, or \"all\" to sweep every LP with a\n"
            "                   known reference objective (default: AFIRO)\n"
            "  --tol T          primal-dual tolerance (default: 1e-6)\n"
            "  --max-iter N     max PMM iterations (default: 3000)\n"
            "  --time-limit S   time limit in seconds (default: 60)\n"
            "  --out FILE       (--name all only) output CSV path, relative to --root (default: results/netlib_all.csv)\n";
        return 0;
    }

    std::string root = cli::get_str(argc, argv, "--root", "./");
    if (!root.empty() && root.back() != '/') root += '/';
    std::string in_dir = cli::get_str(argc, argv, "--in", "data/netlib_infeas/");
    if (!in_dir.empty() && in_dir.back() != '/') in_dir += '/';
    std::string data_dir = root + in_dir;
    std::string name = cli::get_str(argc, argv, "--name", "BGDBG1");

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
    T tol = cli::get_double(argc, argv, "--tol", 1e-6);
    int max_iter = cli::get_int(argc, argv, "--max-iter", 3000);
    double time_limit = cli::get_double(argc, argv, "--time-limit", 60.0); // in seconds

    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::TUNING;

    // Solver result
    std::string csv_path = root + cli::get_str(argc, argv, "--out", "results/netlib_infeas_all.csv");

    // Write header
    if (!std::filesystem::exists(std::filesystem::path(csv_path))
        || std::filesystem::is_empty(std::filesystem::path(csv_path))) {
        std::ofstream csv(csv_path);
        csv << "System,infeas_detected,opt_status,name,pmm_iter,ssn_iter,krylov_iter,fact,pmm_res,ssn_res,solving_time_sec,linesearch_fails,krylov_fails\n";
    }

    // Loop over all LPs
    for (const std::string name : LPs) {
        // Build full path and check if file exists
        std::string filename = data_dir + name + ".mps";
        if (!std::filesystem::exists(filename)) {
            std::cerr << "SKIP: File not found: " << name << "\n";
            continue;
        }

        std::cout << "\n==========Solving " << name << "==========\n";

        try {
            // Read problem data from the file
            MpsFormatParser<T> parser;
            ParsedModel<T> model = parser.parse(filename);
            KSPQPdata<T> pd = parser.to_kspqp(model);

            Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
            KSP_QP<T> solver(prob);

            // Solve the LP
            Solution<T> sol = solver.solve();

            // Chosen system:
            std::string system = solver.kkt_ldlt_used ? "L" : "S";

            // Store the result
            int opt = static_cast<int>(sol.opt);
            bool infeas_detected = (opt == -2 || opt == -3);
            std::ofstream csv(csv_path, std::ios::out | std::ios::app);
            csv << system << "," << infeas_detected << "," << opt << ","
                << name << "," << sol.pmm_iter << "," << sol.ssn_iter << ","
                << sol.krylov_iter << "," << sol.fact << ","
                << sol.pmm_tol_achieved << "," << sol.ssn_tol_achieved << ","
                << sol.run_time << "," << sol.linesearch_fail << ","
                <<  sol.krylov_fail << "\n";
            csv.close();

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n";
            std::ofstream csv(csv_path, std::ios::out | std::ios::app);
            csv << e.what() << "," << 0 << "," << -1 << ","
                << name << "," << -1 << "," << -1 << ","
                << -1 << "," << -1 << ","
                << -1.0 << "," << -1.0 << ","
                << -1.0 << "," << 0 << ","
                << 0 << "\n";
            csv.close();
        }
    }
}


// ==================== A single Netlib infeasible problem ====================
/*
int main() {

    std::string root = "data/netlib_infeas/";  // run from the repo root
    std::string name = "GOSH";
    std::string filename = root + name + ".mps";

    // Parameters for KSP-QP solver
    T tol = 1e-6;
    int max_iter = 1000;
    PrintWhen when = PrintWhen::EVERY10;
    PrintWhat what = PrintWhat::TUNING;

    // Extract problem data from the mps file using our MpsFormatParser and construct solver
    MpsFormatParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    KSPQPdata<T> pd = parser.to_kspqp(model);

    Problem<T> prob(pd, tol, max_iter, when, what);
    KSP_QP<T> solver(prob);

    std::cout << "================================================ Solving " << name << " =================================================\n";

    // Solve the LP
    Solution<T> sol = solver.solve();
    sol.print_summary();

    return 0;
}
*/
