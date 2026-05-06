#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "SSN_PMM.hpp"
#include "Problem.hpp"
#include "MpsParser.hpp"

namespace py = pybind11;
using T      = double;
using Vec    = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat  = Eigen::SparseMatrix<T>;

// -----------------------------------------------------------------------
// Helper: Eigen compressed sparse (CSC) → Python dict of numpy arrays
// -----------------------------------------------------------------------
static py::dict eigen_sparse_to_dict(const SpMat& M_in, const std::string& key) {
    SpMat M = M_in;
    M.makeCompressed();

    int nnz  = M.nonZeros();
    int rows = (int)M.rows();
    int cols = (int)M.cols();

    py::array_t<double> data(nnz);
    py::array_t<int>    indices(nnz);
    py::array_t<int>    indptr(cols + 1);

    auto d = data.mutable_unchecked<1>();
    auto i = indices.mutable_unchecked<1>();
    auto p = indptr.mutable_unchecked<1>();

    for (int k = 0; k < nnz;    ++k) { d[k] = M.valuePtr()[k];      i[k] = M.innerIndexPtr()[k]; }
    for (int j = 0; j <= cols;  ++j)   p[j] = M.outerIndexPtr()[j];

    py::dict out;
    out[py::str(key + "_data")]    = data;
    out[py::str(key + "_indices")] = indices;
    out[py::str(key + "_indptr")]  = indptr;
    out[py::str(key + "_shape")]   = py::make_tuple(rows, cols);
    return out;
}

// -----------------------------------------------------------------------
// Helper: Eigen vector → 1-D numpy array
// -----------------------------------------------------------------------
static py::array_t<double> eigen_vec_to_array(const Vec& v) {
    py::array_t<double> arr(v.size());
    auto p = arr.mutable_unchecked<1>();
    for (int k = 0; k < (int)v.size(); ++k) p[k] = v[k];
    return arr;
}

// -----------------------------------------------------------------------
// parse_sif: parse a SIF/MPS file and return problem data as numpy arrays.
//
// Returned dict keys:
//   n, m, l                  – problem dimensions
//   Q_data/indices/indptr/shape  – CSC sparse Q (may be full symmetric)
//   A_data/indices/indptr/shape  – CSC sparse equality matrix
//   B_data/indices/indptr/shape  – CSC sparse general-inequality matrix
//   c, b                     – 1-D arrays
//   lx, ux, lw, uw           – 1-D bound arrays
//   obj_const                – scalar constant in objective
// -----------------------------------------------------------------------
py::dict parse_sif(const std::string& filename) {
    MpsParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    PDPMMdata<T>   pd    = parser.to_pdpmm(model);

    py::dict out;
    out["n"] = pd.n;
    out["m"] = pd.m;
    out["l"] = pd.l;

    // Sparse matrices
    for (auto& [M, key] : std::vector<std::pair<const SpMat*, std::string>>{
            {&pd.Q, "Q"}, {&pd.A, "A"}, {&pd.B, "B"}}) {
        py::dict d = eigen_sparse_to_dict(*M, key);
        for (auto item : d) out[item.first] = item.second;
    }

    // Dense vectors
    out["c"]         = eigen_vec_to_array(pd.c);
    out["b"]         = eigen_vec_to_array(pd.b);
    out["lx"]        = eigen_vec_to_array(pd.lx);
    out["ux"]        = eigen_vec_to_array(pd.ux);
    out["lw"]        = eigen_vec_to_array(pd.lw);
    out["uw"]        = eigen_vec_to_array(pd.uw);
    out["obj_const"] = (double)pd.obj_const;

    return out;
}

// -----------------------------------------------------------------------
// solve_from_sif: parse a SIF/MPS file and run the SSN-PMM solver.
//
// Returns dict:
//   status       – opt field (0 = optimal, <0 = infeasible, >0 = limit hit)
//   obj_val      – primal objective value
//   solving_time – wall-clock solving time in seconds
//   pmm_iter     – PMM outer iterations
//   ssn_iter     – total SSN inner iterations
//   krylov_iter  – total Krylov iterations
// -----------------------------------------------------------------------
py::dict solve_from_sif(const std::string& filename,
                        double tol        = 1e-6,
                        long long max_iter = 1'000'000'000LL,
                        double time_limit  = 600.0) {
    MpsParser<T>   parser;
    ParsedModel<T> model  = parser.parse(filename);
    PDPMMdata<T>   pd     = parser.to_pdpmm(model);

    Problem<T>  prob(pd, (T)tol, (int)max_iter, time_limit,
                     PrintWhen::NEVER, PrintWhat::NONE);
    SSN_PMM<T>  solver(prob);
    Solution<T> sol = solver.solve();

    py::dict out;
    out["status"]       = sol.opt;
    out["obj_val"]      = (double)sol.obj_val;
    out["solving_time"] = sol.solving_time;
    out["pmm_iter"]     = sol.PMM_iter;
    out["ssn_iter"]     = sol.SSN_iter;
    out["krylov_iter"]  = sol.Krylov_iter;
    return out;
}

// -----------------------------------------------------------------------
// Module
// -----------------------------------------------------------------------
PYBIND11_MODULE(ssn_pmm_bind, m) {
    m.doc() = "Python bindings for the SSN-PMM quadratic programming solver";

    m.def("parse_sif", &parse_sif,
          py::arg("filename"),
          R"(Parse a SIF/MPS file and return problem data as numpy arrays.

Returns a dict with keys: n, m, l, Q_*, A_*, B_*, c, b, lx, ux, lw, uw, obj_const.
The sparse matrices are in CSC format (data / indices / indptr / shape).)");

    m.def("solve_from_sif", &solve_from_sif,
          py::arg("filename"),
          py::arg("tol")        = 1e-6,
          py::arg("max_iter")   = 1'000'000'000LL,
          py::arg("time_limit") = 600.0,
          R"(Parse a SIF/MPS file and solve it with the SSN-PMM solver.

Returns a dict with keys: status, obj_val, solving_time, pmm_iter, ssn_iter, krylov_iter.
status == 0  → optimal solution found
status <  0  → infeasibility detected
status >  0  → iteration / time limit reached)");
}
