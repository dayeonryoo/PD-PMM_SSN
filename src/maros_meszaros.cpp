#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <chrono>
#include <ctime>
#include <thread>

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

// ----------------------- Solving individual problem -----------------------
/*
int main() {
    // std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/maros_meszaros/";
    std::string root = "/Users/dianaryoo/Desktop/KCL/PD-PMM_SSN/data/maros_meszaros/";
    std::string name = "QCAPRI";
    std::string filename = root + name + ".SIF";

    std::cout << "==================== Solving " + name << " ====================\n";
    
    MpsParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    PDPMMdata<T> pd = parser.to_pdpmm(model);

    T tol = 1e-6;
    int max_iter = 1000;
    double time_limit = 60.0; // in seconds
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::TUNING;

    Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
    SSN_PMM<T> solver(prob);

    // Solve:
    Solution<T> sol = solver.solve();
    sol.print_summary();
    std::cout << "\nPMM solver took " << sol.run_time << " s.\n";

    // Check feasibility
    // print_feasibility(pd, sol.x, tol);

    // Check convergence
    if (sol.opt == 0) {
        std::cout << "Solver converged!\n";
    } else if (sol.opt == 3) {
        std::cout << "Lineserach failed. Solver terminated.\n";
    } else if (sol.opt < 0) {
        std::cout << "Solver detected infeasibility.\n";
    } else  if (sol.opt == 4){
        std::cout << "Solver hit the time limit.\n";
    } else {
        std::cout << "Solver hit the max iteration before converging.\n";
    }
    if (sol.opt <= 0 && sol.pmm_tol_achieved > 1e0) {
        std::cout << "Solver possibly diverged.\n";
    }

    return 0;
}
*/


// ----------------------- Running Netlib problems -----------------------
void run_Netlib() {

    // Filenames and objective values of Netlib LPs
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
    
    // std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/";
    std::string root = "/Users/dianaryoo/Desktop/KCL/PD-PMM_SSN/";

    // Parameters in common
    T tol = 1e-6;
    int max_iter = 1000000000;
    double time_limit = 60.0; // in seconds

    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::TUNING;

    // Solver result
    std::string csv_path = root + "results/0727netlib.csv";
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

            Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
            SSN_PMM<T> solver(prob);

            std::time_t curr_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "Compuation started at " << std::ctime(&curr_time);

            // Solve the LP
            Solution<T> sol = solver.solve();
            // sol.print_summary();

            // Check convergence
            bool diverged = false;
            // if (sol.opt == 0) {
            //     std::cout << "Solver converged!\n";
            // } else if (sol.opt == 3) {
            //     std::cout << "Lineserach failed. Solver terminated.\n";
            // } else {
            //     std::cout << "Solver hit the max iteration before converging.\n";
            // }
            // if (sol.pmm_tol_achieved > 1e0) {
            //     diverged = true;
            //     std::cout << "Solver possibly diverged.\n";
            // }

            // Compare
            T abs_err = std::abs(sol.obj_val - ref_obj_val);
            T rel_err = abs_err / std::abs(ref_obj_val);
            T err_tol = 1e-2;
            bool abs_agree = abs_err < err_tol;
            bool rel_agree = rel_err < err_tol;
            bool agree = abs_agree || rel_agree;
            // if (agree) std::cout << "\nCORRECT! Absolute error = " << abs_err << ", Relative error = " << rel_err << "\n";
            // else std::cout << "\nIncorrect Absolute error = " << abs_err << ", Relative error = " << rel_err << "\n";

            std::string system = solver.ldlt_used? "L" : "S";

            // Store result
            TestResult<T> result = {
                system,
                agree, sol.opt, diverged, name,
                abs_err, rel_err, sol.obj_val,
                sol.pmm_iter, sol.ssn_iter, sol.krylov_iter, sol.fact, sol.smw_count,
                sol.pmm_tol_achieved, sol.ssn_tol_achieved,
                sol.run_time, sol.linesearch_fail, sol.krylov_fail
            };
            append_csv_result(csv_path, result);

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n";
            TestResult<T> result = {
                e.what(),
                false, -1, false, name,
                -1.0, -1.0, -1.0,
                -1, -1, -1, -1, -1,
                -1.0, -1.0,
                -1.0, -1, -1
            };
            append_csv_result(csv_path, result);
        }
    }

}

void run_Netlib_infeas() {
// Filenames and objective values of Netlib LPs
    
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
    int max_iter = 10000000;
    double time_limit = 60.0; // in seconds

    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::TUNING;

    // Solver result
    std::string csv_path = root + "results/0727infeas.csv";

    // Write header
    if (!std::filesystem::exists(std::filesystem::path(csv_path))
        || std::filesystem::is_empty(std::filesystem::path(csv_path))) {
        std::ofstream csv(csv_path);
        csv << "System,infeas_detected,opt_status,name,pmm_iter,ssn_iter,krylov_iter,fact,pmm_res,ssn_res,solving_time_sec,linesearch_fail\n";
    } else if (!std::filesystem::is_empty(std::filesystem::path(csv_path))) {
        std::ofstream csv(csv_path, std::ios::out | std::ios::app);
        csv << "\n";
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
                << name << "," << sol.pmm_iter << "," << sol.ssn_iter << ","
                << sol.krylov_iter << "," << sol.fact << ","
                << sol.pmm_tol_achieved << "," << sol.ssn_tol_achieved << ","
                << sol.run_time << "," << sol.linesearch_fail << "," << sol.krylov_fail << "\n";
            csv.close();

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n"; 
            std::ofstream csv(csv_path, std::ios::out | std::ios::app);
            csv << e.what() << "," << 0 << "," << -1 << ","
                << name << "," << -1 << "," << -1 << ","
                << -1 << "," << -1 << ","
                << -1.0 << "," << -1.0 << "," << -1.0 << "," << 0 << 0 << "\n";
            csv.close();
        }
    }

}

// ----------------------- Solving a set of problems -----------------------

int main() {

    run_Netlib_infeas();
    run_Netlib();

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
        {"CONT-300",   1.9151232e-01},
        {"CVXQP1L",    1.0870480e+08},
        {"CVXQP1M",    1.0875116e+06},
        {"CVXQP1S",    1.1590718e+04},
        {"CVXQP2L",    8.1842458e+07},
        {"CVXQP2M",    8.2015543e+05},
        {"CVXQP2S",    8.1209405e+03},
        {"CVXQP3L",    1.1571110e+08},
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
        {"EXDATA",    -1.4184343e+02},
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
        {"Q25FV47",    1.3744448e+07},
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
        {"QSHELL",     1.5726368e+12},
        {"QSHIP04L",   2.4200155e+06},
        {"QSHIP04S",   2.4249937e+06},
        {"QSHIP08L",   2.3760406e+06},
        {"QSHIP08S",   2.3857289e+06},
        {"QSHIP12L",   3.0188766e+06},
        {"QSHIP12S",   3.0569623e+06},
        {"QSIERRA",    2.3750458e+07},
        {"QSTAIR",     7.9854528e+06},
        {"QSTANDAT",   6.4118384e+03},
        {"S268",       5.7310705e-07},
        {"STADAT1",   -2.8526864e+07},
        {"STADAT2",   -3.2626665e+01},
        {"STADAT3",   -3.5779453e+01},
        {"STCQP1",     1.5514356e+05},
        {"STCQP2",     2.2327313e+04},
        {"TAME",       0.0000000e+00},
        {"UBH1",       1.1160008e+00},
        {"VALUES",    -1.3966211e+00},
        {"YAO",        1.9770426e+02},
        {"ZECEVIC2",  -4.1250000e+00}
    };

    // std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/";
    std::string root = "/Users/dianaryoo/Desktop/KCL/PD-PMM_SSN/";

    // Parameters
    T tol = 1e-6;
    int max_iter = 100000;
    double time_limit = 60.0; // in seconds

    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::TUNING;

    int cooldown_sec = 3;      // seconds to sleep between problems (prevents CPU throttling)

    // Solver result
    std::string csv_path = root + "results/0727mm.csv";
    write_csv_header(csv_path);

    for (const auto& [name, ref_obj_val] : QPs) {

        // Build full path and check if file exists
        std::string filename = root + "data/maros_meszaros/" + name + ".SIF";
        if (!std::filesystem::exists(filename)) {
            std::cerr << "SKIP: File not found: " << name << "\n";
            continue;
        }

        std::cout << "\n============================================= " << name << " =============================================\n";
        std::time_t curr_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::ctime(&curr_time);

        try {
            // Read problem data from the file
            MpsParser<T> parser;
            ParsedModel<T> model = parser.parse(filename);
            PDPMMdata<T> pd = parser.to_pdpmm(model);

            // Construct the problem and solver
            Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
            SSN_PMM<T> solver(prob);

            // Solve the QP
            Solution<T> sol = solver.solve();
            // sol.print_summary();
            
            // Check convergence
            bool diverged = false;
            // if (sol.opt == 0) {
            //     std::cout << "Solver converged!\n";
            // } else if (sol.opt == 3) {
            //     std::cout << "Lineserach failed. Solver terminated.\n";
            // } else if (sol.opt == 4) {
            //     std::cout << "Solver terminated by time limit.\n";
            // } else if (sol.opt == 1 || sol.opt == 2) {
            //     std::cout << "Solver hit the max iteration before converging.\n";
            // }
            // if (sol.pmm_tol_achieved > 1e0) {
            //     diverged = true;
            //     std::cout << "Solver possibly diverged.\n"; 
            // }

            // Compare
            T abs_err = std::abs(sol.obj_val - ref_obj_val);
            T rel_err = abs_err / std::abs(ref_obj_val);
            T err_tol = 1e-2;
            bool abs_agree = abs_err < err_tol;
            bool rel_agree = rel_err < err_tol;
            bool agree = abs_agree || rel_agree;
            // if (agree) std::cout << "CORRECT! Absolute error = " << abs_err << ", Relative error = " << rel_err << "\n\n";
            // else std::cout << "Incorrect. Absolute error = " << abs_err << ", Relative error = " << rel_err << "\n\n";

            // Record result
            std::string system = solver.ldlt_used? "L" : "S";

            TestResult<T> result = {
                system,
                agree, sol.opt, diverged, name,
                abs_err, rel_err,
                sol.obj_val, sol.pmm_iter, sol.ssn_iter, sol.krylov_iter, sol.fact, sol.smw_count,
                sol.pmm_tol_achieved, sol.ssn_tol_achieved,
                sol.run_time, sol.linesearch_fail, sol.krylov_fail
            };
            append_csv_result(csv_path, result);

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n";
            TestResult<T> result = {
                e.what(),
                false, -1, false, name,
                -1.0, -1.0,
                -1.0, -1, -1, -1, -1, -1,
                -1.0, -1.0, -1.0
            };
            append_csv_result(csv_path, result);
        }

        std::this_thread::sleep_for(std::chrono::seconds(cooldown_sec));
    }

    return 0;
}
