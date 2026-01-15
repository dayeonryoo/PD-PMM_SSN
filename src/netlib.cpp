#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <map>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "SSN_PMM.hpp"
#include "Problem.hpp"
#include "Printing.hpp"

#include "Highs.h"
#include "load_mps_lp.hpp"
#include "lp_to_pdpmm.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

struct NetlibTestResult {
    std::string name;
    bool agree;
    T abs_err;
};

int main() {

    std::string filename = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib/25FV47.mps";
    
    // Solving via HiGHS
    Highs h;
    h.setOptionValue("output_flag", true);
    h.readModel(filename);
    h.run();
    double ref_obj_val = h.getObjectiveValue();

    // Extract problem data from the mps file
    const HighsLp& lp = h.getLp();
    PDPMMdata<T> pd = lp_to_pdpmm<T>(lp);

    // Construct the problem and solver
    T tol = 1e-6;
    int max_iter = 1e3;
    PrintWhen PMM_print_when = PrintWhen::ALWAYS;
    PrintWhat PMM_print_what = PrintWhat::MINIMAL;
    PrintWhen SSN_print_when = PrintWhen::NEVER;
    PrintWhat SSN_print_what = PrintWhat::NONE;

    Problem<T> prob(pd.Q, pd.A, pd.B, pd.c, pd.b, pd.lx, pd.ux, pd.lw, pd.uw,
                    tol, max_iter, PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);
    SSN_PMM<T> solver(prob);
    

    // Solve the LP using PD-PMM_SSN solver
    Solution<T> sol = solver.solve();
    sol.print_summary();
    T obj_val = sol.obj_val;
    
    // Compare
    T abs_err = std::abs(obj_val - ref_obj_val);
    bool agree = abs_err <= 1e-4;
    if (agree) std::cout << "\nCORRECT!\n";
    else std::cout << "\nIncorrect. Absolute error = " << abs_err << "\n";

    return 0;
}

/*
int main() {

    // Filenames and corresponding optimal objective values
    std::map<std::string, T> netlib_opt = {
        {"25fv47",5501.845883}, {"80bau3b",987232.16}, {"adlittle",225494.96316},
        {"afiro",-464.75314286}, {"agg",-35995300.0}, {"agg2",-20239250.0},
        {"agg3",10312110.0}, {"bandm",-158.6}, {"beaconfd",33594.759},
        {"blend",-30.81214}, {"bnl1",1977.9496}, {"bnl2",1811.236},
        {"boeing1",-335.21356751}, {"boeing2",-315.013502}, {"bore3d",1373.080394},
        {"brandy",1518.0}, {"capri",2690.012003}, {"cycle",-5.226393024},
        {"czprob",2185023.0}, {"d2q06c",122700.0}, {"d6cube",315.49166667},
        {"degen2",-1435.178}, {"degen3",-987.294}, {"dfl001",11266000.0},
        {"e226",-18.75192666}, {"etamacro",-755.571999}, {"fffff800",555600.0},
        {"finnis",172769.96547}, {"fit1d",-9146.378923}, {"fit1p",9146.378923},
        {"fit2d",-68463.78}, {"fit2p",68463.783232}, {"forplan",-664.2187},
        {"ganges",-109586.3}, {"gfrd-pnc",6902235.999}, {"greenbea",-72460000.0},
        {"greenbeb",-4302000.0}, {"grow15",-106834000.0}, {"grow22",-160834300.0},
        {"grow7",-47778118.81}, {"israel",-896644.8}, {"kb2",-1749.900129},
        {"lotfi",-25.269}, {"maros",-58063.74}, {"maros-r7",1497185.0},
        {"modszk1",320.6197}, {"nesm",14076070.0}, {"perold",-9380.758},
        {"pilot",-557.4043}, {"pilot.ja",-6113.134411}, {"pilot.we",-2720043.0},
        {"pilot4",-2581.139264}, {"pilot87",301.71072827}, {"pilotnov",-4497.276},
        {"qap8",203.50}, {"qap12",522.89435056}, {"qap15",1040.9940410},
        {"recipe",-266.616}, {"sc105",-52.20206121}, {"sc205",-52.20206121},
        {"sc50a",-64.575077059}, {"sc50b",-70.0}, {"scagr25",-14753433.06},
        {"scagr7",-2331333.0}, {"scfxm1",18416.0}, {"scfxm2",36660.0},
        {"scfxm3",54901.25}, {"scorpion",1878.124}, {"scrs8",904.29998613},
        {"scsd1",8.6666666743}, {"scsd6",50.5}, {"scsd8",904.99999993},
        {"sctap1",1412.25}, {"sctap2",1724.8071429}, {"sctap3",1424.0},
        {"seba",15711.6}, {"share1b",-76589.31857}, {"share2b",-415.7322},
        {"shell",1208825346.0}, {"ship04l",1793324.38}, {"ship04s",1798714.70},
        {"ship08l",1909055.21}, {"ship08s",1920098.70}, {"ship12l",1470187.19},
        {"ship12s",1489236.14}, {"sierra",15394368.4}, {"stair",-251.2669},
        {"standata",1257.6995}, {"standmps",1406.0175}, {"stocfor1",-41131.97},
        {"stocfor2",-39024.40}, {"stocfor3",-39976.661576}, {"truss",458815.8471},
        {"tuff",0.292147747}, {"vtp.base",129814.6246}, {"wood1p",1.442902411},
        {"woodw",1.304476333}
    };

    // Parameters in common
    T tol = 1e-6;
    int max_iter = 1e3;
    PrintWhen PMM_print_when = PrintWhen::NEVER;
    PrintWhat PMM_print_what = PrintWhat::NONE;
    PrintWhen SSN_print_when = PrintWhen::NEVER;
    PrintWhat SSN_print_what = PrintWhat::NONE;

    // Solver result
    std::vector<NetlibTestResult> results;
    results.reserve(netlib_opt.size());

    // Looping over all LPs to solve and compare with the reference optimal objective values
    for (const auto& data : netlib_opt) {
        const std::string& name = data.first; // e.g. "afiro"
        T ref_obj_val = data.second; // optimal objective value

        // Build full path
        std::string filename = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib/" + name + ".mps";
        
        if (!std::filesystem::exists(filename)) {
            std::cerr << "SKIP: File not found: " << name << "\n";
            continue;
        }

        std::cout << "\n---Solving " << name << "---\n";
        
        try {
            // 1. Load LP data from MPS
            LPdata<T> lp = load_mps_lp<T>(filename);

            // 2. Convert to PD-PMM structure
            PDPMMdata<T> pd = lp_to_pdpmm<T>(lp);

            // 3. Construct the problem and solver
            Problem<T> prob(pd.Q, pd.A, pd.B, pd.c, pd.b, pd.lx, pd.ux, pd.lw, pd.uw,
                            tol, max_iter, PMM_print_when, PMM_print_what, SSN_print_when, SSN_print_what);
            SSN_PMM<T> solver(prob);

            // 4. Solve the LP
            Solution<T> sol = solver.solve();
            T obj_val = sol.obj_val;
            
            // 5. Compare
            T abs_err = std::abs(obj_val - ref_obj_val);
            bool agree = abs_err <= 1e-4;
            if (agree) std::cout << "CORRECT!\n";
            else std::cout << "Incorrect. Absolute error = " << abs_err << "\n";

            // 6. Store result
            results.push_back(NetlibTestResult{name, agree, abs_err});

        } catch (const std::exception& e) {
            std::cerr << "ERROR solving " << name << ": " << e.what() << "\n"; 
            results.push_back(NetlibTestResult{name, false, 1e100});
        }
    }

    // Write CSV file
    std::ofstream csv("netlib_results.csv");
    csv << "name, correct?, abs_err\n";
    for (const auto& r : results) {
        csv << r.name << ", " << (r.agree ? "1" : "0") << ", " << r.abs_err << "\n";
    }
    csv.close();
    std::cout << "\nResults written to netlib_results.csv\n";

    return 0;
}
*/