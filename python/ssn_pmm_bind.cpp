#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "SSN_PMM.hpp"
#include "Problem.hpp"
#include "MpsParser.hpp"
#include "PDEgenerator.hpp"

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
//   status            – opt field (0 = optimal, <0 = infeasible, >0 = limit hit)
//   obj_val           – primal objective value
//   run_time          – wall-clock solving time in seconds
//   pmm_iter          – PMM outer iterations
//   ssn_iter          – total SSN inner iterations
//   pmm_tol_achieved  – tolerance achieved by PMM at termination
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
    out["status"]           = sol.opt;
    out["obj_val"]          = (double)sol.obj_val;
    out["run_time"]         = sol.run_time;
    out["pmm_iter"]         = sol.pmm_iter;
    out["ssn_iter"]         = sol.ssn_iter;
    out["pmm_tol_achieved"] = (double)sol.pmm_tol_achieved;
    return out;
}

// -----------------------------------------------------------------------
// Helper: reconstruct PDPMMdata<T> from a parse_sif dict.
// -----------------------------------------------------------------------
static PDPMMdata<T> dict_to_pdpmm(const py::dict& d) {
    PDPMMdata<T> pd;
    pd.n         = d["n"].cast<int>();
    pd.m         = d["m"].cast<int>();
    pd.l         = d["l"].cast<int>();
    pd.obj_const = d["obj_const"].cast<double>();

    auto make_sparse = [&](const std::string& key) {
        auto shape_tup   = d[py::str(key + "_shape")].cast<py::tuple>();
        int rows = shape_tup[0].cast<int>(), cols = shape_tup[1].cast<int>();
        auto data_arr    = d[py::str(key + "_data")].cast<py::array_t<double>>();
        auto indices_arr = d[py::str(key + "_indices")].cast<py::array_t<int>>();
        auto indptr_arr  = d[py::str(key + "_indptr")].cast<py::array_t<int>>();
        auto dr = data_arr.unchecked<1>();
        auto ir = indices_arr.unchecked<1>();
        auto pr = indptr_arr.unchecked<1>();
        int nnz = (int)data_arr.size();
        std::vector<Eigen::Triplet<T>> trips;
        trips.reserve(nnz);
        for (int col = 0; col < cols; ++col)
            for (int k = pr[col]; k < pr[col + 1]; ++k)
                trips.emplace_back(ir[k], col, (T)dr[k]);
        SpMat M(rows, cols);
        M.setFromTriplets(trips.begin(), trips.end());
        M.makeCompressed();
        return M;
    };

    auto make_vec = [&](const std::string& key) {
        auto arr = d[py::str(key)].cast<py::array_t<double>>();
        auto r = arr.unchecked<1>();
        Vec v(r.size());
        for (int i = 0; i < (int)r.size(); ++i) v[i] = (T)r[i];
        return v;
    };

    pd.Q  = make_sparse("Q");
    pd.A  = make_sparse("A");
    pd.B  = make_sparse("B");
    pd.c  = make_vec("c");
    pd.b  = make_vec("b");
    pd.lx = make_vec("lx");
    pd.ux = make_vec("ux");
    pd.lw = make_vec("lw");
    pd.uw = make_vec("uw");
    return pd;
}

// -----------------------------------------------------------------------
// solve_from_data: run the SSN-PMM solver on already-parsed problem data.
//
// Takes a dict as returned by parse_sif (no file I/O).  Use this together
// with a Python wall-clock timer for fair benchmarking.
//
// Returns dict with the same keys as solve_from_sif.
// -----------------------------------------------------------------------
py::dict solve_from_data(const py::dict& pd_dict,
                         double tol         = 1e-6,
                         long long max_iter = 1'000'000'000LL,
                         double time_limit  = 600.0) {
    PDPMMdata<T> pd = dict_to_pdpmm(pd_dict);

    Problem<T>  prob(pd, (T)tol, (int)max_iter, time_limit,
                     PrintWhen::NEVER, PrintWhat::NONE);
    SSN_PMM<T>  solver(prob);
    Solution<T> sol = solver.solve();

    py::dict out;
    out["status"]           = sol.opt;
    out["obj_val"]          = (double)sol.obj_val;
    out["run_time"]         = sol.run_time;
    out["pmm_iter"]         = sol.pmm_iter;
    out["ssn_iter"]         = sol.ssn_iter;
    out["pmm_tol_achieved"] = (double)sol.pmm_tol_achieved;
    return out;
}

// -----------------------------------------------------------------------
// Helper: PDPMMdata<T> → Python dict (same format as parse_sif output)
// -----------------------------------------------------------------------
static py::dict pdpmm_to_dict(const PDPMMdata<T>& pd) {
    py::dict out;
    out["n"] = pd.n;
    out["m"] = pd.m;
    out["l"] = pd.l;

    for (auto& [M, key] : std::vector<std::pair<const SpMat*, std::string>>{
            {&pd.Q, "Q"}, {&pd.A, "A"}, {&pd.B, "B"}}) {
        py::dict d = eigen_sparse_to_dict(*M, key);
        for (auto item : d) out[item.first] = item.second;
    }

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
// generate_pde_problem: build a PDE-constrained QP from the C++ generators.
//
// choice = "poisson"  – Poisson control with L1/L2 regularisation
// choice = "convdiff" – Convection-diffusion control with L1/L2 regularisation
//
// Returns the same dict format as parse_sif(), so it can be passed directly
// to solve_from_data() and pdpmm_to_qpalm() in benchmark_pde.py.
// -----------------------------------------------------------------------
py::dict generate_pde_problem(const std::string& choice) {
    if (choice == "poisson")
        return pdpmm_to_dict(pdegen::make_poisson_L1L2_control_default<T>());
    if (choice == "convdiff")
        return pdpmm_to_dict(pdegen::make_convdiff_L1L2_control_default<T>());
    throw std::invalid_argument("choice must be 'poisson' or 'convdiff'");
}

// -----------------------------------------------------------------------
// generate_pde_problem_params: parametric Poisson problem generator.
//
// choice  = "poisson"
// nc      = grid exponent (grid size = 2^nc + 1 per direction)
// alpha1  = L1 regularisation weight
// alpha2  = L2 regularisation weight (0 is valid)
// -----------------------------------------------------------------------
py::dict generate_pde_problem_params(const std::string& choice,
                                      int nc, double alpha1, double alpha2) {
    if (choice == "poisson")
        return pdpmm_to_dict(
            pdegen::make_poisson_L1L2_control<T>(nc, (T)alpha1, (T)alpha2));
    if (choice == "convdiff")
        return pdpmm_to_dict(
            pdegen::make_convdiff_L1L2_control<T>(nc, (T)alpha1, (T)alpha2));
    throw std::invalid_argument("choice must be 'poisson' or 'convdiff'");
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

Returns a dict with keys: status, obj_val, run_time, pmm_iter, ssn_iter, pmm_tol_achieved.
status == 0  → optimal solution found
status <  0  → infeasibility detected
status >  0  → iteration / time limit reached)");

    m.def("generate_pde_problem", &generate_pde_problem,
          py::arg("choice"),
          R"(Generate a PDE-constrained QP using the built-in C++ problem generators.

choice = 'poisson'  – Poisson control with L1/L2 regularisation and control bounds
choice = 'convdiff' – Convection-diffusion control with L1/L2 regularisation and control bounds

Returns the same dict format as parse_sif(), so the result can be passed
directly to solve_from_data() and used with pdpmm_to_qpalm().)");

    m.def("generate_pde_problem_params", &generate_pde_problem_params,
          py::arg("choice"), py::arg("nc"), py::arg("alpha1"), py::arg("alpha2"),
          R"(Generate a parametric PDE-constrained QP with explicit parameters.

choice = 'poisson'  or  'convdiff'
nc     = grid exponent (grid size = 2^nc + 1 per direction; n_display = 2*(2^nc+1)^2)
alpha1 = L1 regularisation weight
alpha2 = L2 regularisation weight (0 is valid))");

    m.def("solve_from_data", &solve_from_data,
          py::arg("pd"),
          py::arg("tol")        = 1e-6,
          py::arg("max_iter")   = 1'000'000'000LL,
          py::arg("time_limit") = 600.0,
          R"(Solve with SSN-PMM using already-parsed problem data (dict from parse_sif).

No file I/O is performed.  Wrap this call with time.perf_counter() for fair
wall-clock benchmarking that is comparable to QPALM/OSQP setup+solve timing.

Returns a dict with keys: status, obj_val, pmm_iter, ssn_iter, pmm_tol_achieved.)");
}
