#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <string>
#include <vector>
#include <filesystem>
#include "MpsParser.hpp"
#include "highs.h"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

/*
#include "Highs.h"
int main() {
  std::string model_file = "qjh_quadobj.mps";
  Highs h;
  // Silence HiGHS with this option setting
  //  h.setOptionValue("output_flag", false);
  h.readModel(model_file);
  HighsModel model = h.getModel();
  // This solves the model with HiGHS
  h.run();
}
*/

int main() {
/*
    MpsParser<T> parser;
    std::string filename = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib/CYCLE.mps";
    ParsedModel<T> model = parser.parse(filename);

    Highs h;
    h.readModel(filename);
    const HighsLp& lp = h.getLp();

    std::cout << "Comparing parsed model with HiGHS model...\n";
    if (model.num_rows == lp.num_row_ && model.num_cols == lp.num_col_) {
        std::cout << "Dimensions match: " << model.num_rows << " rows, " << model.num_cols << " cols.\n";
    } else {
        std::cerr << "Dimension mismatch! Parsed: " << model.num_rows << " rows, " << model.num_cols << " cols; "
                  << "HiGHS: " << lp.num_row_ << " rows, " << lp.num_col_ << " cols.\n";
    }
    if (model.c.size() == lp.col_cost_.size()) {
        std::cout << "Objective coefficients size match: " << model.c.size() << "\n";
    } else {
        std::cerr << "Objective coefficients size mismatch! Parsed: " << model.c.size()
                  << "; HiGHS: " << lp.col_cost_.size() << "\n";
    }
    if (model.A.nonZeros() == lp.a_matrix_.value_.size()) {
        std::cout << "Number of nonzeros in A match: " << model.A.nonZeros() << "\n";
    } else {
        std::cerr << "Number of nonzeros in A mismatch! Parsed: " << model.A.nonZeros()
                  << "; HiGHS: " << lp.a_matrix_.value_.size() << "\n";
    }
*/
/*
    std::cout << "Testing MPS parser on Netlib problems...\n";
    // Filenames of Netlib test problems (without .mps extension)
    std::vector<std::string> netlib_names = {"25FV47","80BAU3B","ADLITTLE","AFIRO","AGG","AGG2","AGG3","BANDM","BEACONFD","BLEND","BNL1","BNL2","BOEING1","BOEING2","BORE3D","BRANDY","CAPRI","CYCLE","CZPROB","D2Q06C","D6CUBE","DEGEN2","DEGEN3","DFL001",
                                            "E226","ETAMACRO","FFFFF800","FINNIS","FIT1D","FIT1P","FIT2D","FIT2P","FORPLAN","GANGES","GFRD-PNC","GREENBEA","GREENBEB","GROW15","GROW22","GROW7","ISRAEL","KB2","LOTFI","MAROS","MAROS-R7","MODSZK1","NESM",
                                            "PEROLD","PILOT","PILOT.JA","PILOT.WE","PILOT4","PILOT87","PILOTNOV","QAP8","QAP12","QAP15","RECIPE","SC105","SC205","SC50A","SC50B","SCAGR25","SCAGR7","SCFXM1","SCFXM2","SCFXM3","SCORPION","SCRS8","SCSD1","SCSD6","SCSD8","SCTAP1","SCTAP2","SCTAP3",
                                            "SEBA","SHARE1B","SHARE2B","SHELL","SHIP04L","SHIP04S","SHIP08L","SHIP08S","SHIP12L","SHIP12S","SIERRA","STAIR","STANDATA","STANDGUB","STANDMPS","STOCFOR1","STOCFOR2","STOCFOR3","TRUSS","TUFF","VTP.BASE","WOOD1P","WOODW"};
    // std::vector<std::string> kennington_names = {"CRE-A","CRE-B","CRE-C","CRE-D","KEN-07","KEN-11","KEN-13","KEN-18","OSA-07","OSA-14","OSA-30","OSA-60","PDS-02","PDS-06","PDS-10","PDS-20"};

    // Root
    std::string root = "C:/Users/k24095864/C++project/PD-PMM_SSN/";

    for (const auto& name : netlib_names) {

        std::cout << "Testing " << name << "...\n";

        // Build full path and check if file exists
        std::string filename = root + "data/netlib/" + name + ".mps";
        if (!std::filesystem::exists(filename)) {
            std::cerr << "SKIP: File not found: " << name << "\n";
            continue;
        }

        Highs h;
        h.setOptionValue("output_flag", false);
        h.readModel(filename);
        const HighsLp& lp = h.getLp();

        MpsParser<T> parser;
        ParsedModel<T> model = parser.parse(filename);

        if (model.num_rows != lp.num_row_ || model.num_cols != lp.num_col_) {
            std::cerr << "FAIL: Dimension mismatch for " << name << "\n";
            continue;
        }
        if (model.A.nonZeros() != lp.a_matrix_.value_.size()) {
            std::cerr << "FAIL: Number of nonzeros in A mismatch for " << name << "\n";
            continue;
        }
        if (model.c.size() != lp.col_cost_.size()) {
            std::cerr << "FAIL: Objective coefficients size mismatch for " << name << "\n";
            continue;
        }
        if (model.row_lower.size() != lp.row_lower_.size() || model.row_upper.size() != lp.row_upper_.size()) {
            std::cerr << "FAIL: Row bounds size mismatch for " << name << "\n";
            continue;
        }
        if (model.col_lower.size() != lp.col_lower_.size() || model.col_upper.size() != lp.col_upper_.size()) {
            std::cerr << "FAIL: Column bounds size mismatch for " << name << "\n";
            continue;
        }
    }
    std::cout << "Testing completed.\n";
*/

    std::string filename = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/maros-meszaros/CVXQP2_M.SIF";

    MpsParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    std::cout << "Parsed model from " << filename << ":\n";
    std::cout << "  Number of rows: " << model.num_rows << "\n";
    std::cout << "  Number of cols: " << model.num_cols << "\n";
    std::cout << "  Number of nonzeros in A: " << model.A.nonZeros() << "\n";
    int Q_diag_nonzeros = 0;
    for (int i = 0; i < model.Q.rows(); ++i) {
        for (SpMat::InnerIterator it(model.Q, i); it; ++it) {
            if (it.row() == it.col()) {
                ++Q_diag_nonzeros;
            }
        }
    }
    std::cout << "  Number of diagonal nonzeros in Q: " << Q_diag_nonzeros << "\n";
    std::cout << "  Number of off-diagonal nonzeros in Q: " << model.Q.nonZeros() - Q_diag_nonzeros << "\n";

    return 0;
}