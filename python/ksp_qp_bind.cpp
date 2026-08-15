#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ksp_qp.hpp"
#include "problem.hpp"
#include "mps_format_parser.hpp"
#include "pde_generator.hpp"

namespace py = pybind11;
using T      = double;
using Vec    = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat  = Eigen::SparseMatrix<T>;

/*-----------------------------------------------------------------------
Helper: Eigen compressed sparse (CSC) → Python dict of numpy arrays
-----------------------------------------------------------------------*/
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

/*-----------------------------------------------------------------------
Helper: Eigen vector → 1D numpy array
-----------------------------------------------------------------------*/
static py::array_t<double> eigen_vec_to_array(const Vec& v) {
    py::array_t<double> arr(v.size());
    auto p = arr.mutable_unchecked<1>();
    for (int k = 0; k < (int)v.size(); ++k) p[k] = v[k];
    return arr;
}

/*-----------------------------------------------------------------------
parse_sif: parse a SIF/MPS file and return problem data as numpy arrays.

Returned dict keys:
  n, m, l                      – problem dimensions
  Q_data/indices/indptr/shape  – CSC sparse Q (may be full symmetric)
  A_data/indices/indptr/shape  – CSC sparse equality matrix
  B_data/indices/indptr/shape  – CSC sparse general-inequality matrix
  c, b                         – 1D arrays
  lx, ux, lw, uw               – 1D bound arrays
  obj_const                    – scalar constant in objective
-----------------------------------------------------------------------*/
py::dict parse_sif(const std::string& filename) {
    KSPQPdata<T> pd;
    {
        py::gil_scoped_release release;
        MpsFormatParser<T>   parser;
        ParsedModel<T> model = parser.parse(filename);
        pd = parser.to_kspqp(model);
    }

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

/*-----------------------------------------------------------------------
solve_from_sif: parse a SIF/MPS file and run the KSP-QP solver.

Returns dict:
  status            – opt field (0 = optimal, <0 = infeasible, >0 = limit hit)
  obj_val           – primal objective value
  setup_time        – wall-clock time in seconds spent in the KSP_QP constructor
  solve_time        – wall-clock time in seconds spent in solve()
  run_time          – setup_time + solve_time
  pmm_iter          – PMM outer iterations
  ssn_iter          – total SSN inner iterations
  pmm_tol_achieved  – tolerance achieved by PMM at termination
-----------------------------------------------------------------------*/
py::dict solve_from_sif(const std::string& filename,
                        double tol         = 1e-6,
                        long long max_iter = 1'000'000'000LL,
                        double time_limit  = 600.0) {
    int opt, pmm_iter, ssn_iter;
    double obj_val, setup_time, solve_time, run_time, pmm_tol_achieved;
    {
        py::gil_scoped_release release;
        MpsFormatParser<T>   parser;
        ParsedModel<T> model  = parser.parse(filename);
        KSPQPdata<T>   pd     = parser.to_kspqp(model);

        Problem<T>  prob(pd, (T)tol, (int)max_iter, time_limit,
                         PrintWhen::NEVER, PrintWhat::NONE);
        KSP_QP<T>  solver(prob);
        Solution<T> sol = solver.solve();
        opt              = static_cast<int>(sol.opt);
        obj_val          = (double)sol.obj_val;
        setup_time       = sol.setup_time;
        solve_time       = sol.solve_time;
        run_time         = sol.run_time;
        pmm_iter         = sol.pmm_iter;
        ssn_iter         = sol.ssn_iter;
        pmm_tol_achieved = (double)sol.pmm_tol_achieved;
    }

    py::dict out;
    out["status"]           = opt;
    out["obj_val"]          = obj_val;
    out["setup_time"]       = setup_time;
    out["solve_time"]       = solve_time;
    out["run_time"]         = run_time;
    out["pmm_iter"]         = pmm_iter;
    out["ssn_iter"]         = ssn_iter;
    out["pmm_tol_achieved"] = pmm_tol_achieved;
    return out;
}

/*-----------------------------------------------------------------------
Helper: reconstruct KSPQPdata<T> from a parse_sif dict.
-----------------------------------------------------------------------*/
static KSPQPdata<T> dict_to_kspqp(const py::dict& d) {
    KSPQPdata<T> pd;
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

/*-----------------------------------------------------------------------
solve_from_data: run the KSP-QP solver on already-parsed problem data.
Takes a dict as returned by parse_sif.
Returns dict with the same keys as solve_from_sif.
-----------------------------------------------------------------------*/
py::dict solve_from_data(const py::dict& pd_dict,
                         double tol         = 1e-6,
                         long long max_iter = 1'000'000'000LL,
                         double time_limit  = 600.0) {
    KSPQPdata<T> pd = dict_to_kspqp(pd_dict); // reads Python objects

    int opt, pmm_iter, ssn_iter;
    double obj_val, setup_time, solve_time, run_time, pmm_tol_achieved;
    {
        py::gil_scoped_release release;
        Problem<T>  prob(pd, (T)tol, (int)max_iter, time_limit,
                         PrintWhen::NEVER, PrintWhat::NONE);
        KSP_QP<T>  solver(prob);
        Solution<T> sol = solver.solve();
        opt              = static_cast<int>(sol.opt);
        obj_val          = (double)sol.obj_val;
        setup_time       = sol.setup_time;
        solve_time       = sol.solve_time;
        run_time         = sol.run_time;
        pmm_iter         = sol.pmm_iter;
        ssn_iter         = sol.ssn_iter;
        pmm_tol_achieved = (double)sol.pmm_tol_achieved;
    }

    py::dict out;
    out["status"]           = opt;
    out["obj_val"]          = obj_val;
    out["setup_time"]       = setup_time;
    out["solve_time"]       = solve_time;
    out["run_time"]         = run_time;
    out["pmm_iter"]         = pmm_iter;
    out["ssn_iter"]         = ssn_iter;
    out["pmm_tol_achieved"] = pmm_tol_achieved;
    return out;
}

/*-----------------------------------------------------------------------
Helper: KSPQPdata<T> → Python dict (same format as parse_sif output)
-----------------------------------------------------------------------*/
static py::dict kspqp_to_dict(const KSPQPdata<T>& pd) {
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

/*-----------------------------------------------------------------------
Generate L1/L2-regularised PDE-constrained QPs from (Gondzio, Pougkakiotis & Pearson 2022).

choice  = "poisson"  or  "convdiff"
nc      = grid exponent (grid size = 2^nc + 1 per direction)
alpha1  = L1 regularisation weight
alpha2  = L2 regularisation weight
y_lower/y_upper = state bounds (default ±inf)
-----------------------------------------------------------------------*/
py::dict generate_pde_l1l2_qp(const std::string& choice,
                              int nc, double alpha1, double alpha2,
                              double y_lower = -std::numeric_limits<double>::infinity(),
                              double y_upper =  std::numeric_limits<double>::infinity()) {
    if (choice != "poisson" && choice != "convdiff")
        throw std::invalid_argument("choice must be 'poisson' or 'convdiff'");

    KSPQPdata<T> pd;
    {
        py::gil_scoped_release release;
        pd = (choice == "poisson")
               ? pdegen::make_poisson_l1l2_control<T>(nc, (T)alpha1, (T)alpha2, T(-2), T(1.5),
                                                       (T)y_lower, (T)y_upper)
               : pdegen::make_convdiff_l1l2_control<T>(nc, (T)alpha1, (T)alpha2, T(-2), T(1.5), T(0.02),
                                                        (T)y_lower, (T)y_upper);
    }
    return kspqp_to_dict(pd);
}

/*-----------------------------------------------------------------------
Generate L2-regularised PDE-constrained QPs from (Pearson & Gondzio 2017).

choice = "poisson"       - 2D Poisson control (control-constrained)
choice = "poisson_state" - 2D Poisson control (state-constrained)
choice = "convdiff"      - 2D convection-diffusion control

y_lower/y_upper/u_lower/u_upper default to ±inf (unconstrained).
eps applies to "convdiff" only (diffusion coefficient).
-----------------------------------------------------------------------*/
py::dict generate_pde_l2_qp(const std::string& choice,
                                     int nc, double beta,
                                     double y_lower = -std::numeric_limits<double>::infinity(),
                                     double y_upper =  std::numeric_limits<double>::infinity(),
                                     double u_lower = -std::numeric_limits<double>::infinity(),
                                     double u_upper =  std::numeric_limits<double>::infinity(),
                                     double eps = 0.01) {
    if (choice != "poisson" && choice != "poisson_state" && choice != "convdiff")
        throw std::invalid_argument(
            "choice must be one of 'poisson', 'poisson_state', 'convdiff'");

    KSPQPdata<T> pd;
    {
        py::gil_scoped_release release;
        if (choice == "poisson") {
            pd = pdegen::make_poisson_l2_control<T>(nc, (T)beta, (T)y_lower, (T)y_upper,
                                                    (T)u_lower, (T)u_upper);
        } else if (choice == "poisson_state") {
            pd = pdegen::make_poisson_l2_state_control<T>(nc, (T)beta, (T)y_lower, (T)y_upper,
                                                        (T)u_lower, (T)u_upper);
        } else { // convdiff
            pd = pdegen::make_convdiff_l2_control<T>(nc, (T)beta, (T)y_lower, (T)y_upper,
                                                    (T)u_lower, (T)u_upper, (T)eps);
        }
    }
    return kspqp_to_dict(pd);
}

/*-----------------------------------------------------------------------
Module
-----------------------------------------------------------------------*/
PYBIND11_MODULE(ksp_qp_bind, m) {
    m.doc() = "Python bindings for the KSP-QP quadratic programming solver";

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
          R"(Parse a SIF/MPS file and solve it with the KSP-QP solver.

Returns a dict with keys: status, obj_val, setup_time, solve_time, run_time, pmm_iter, ssn_iter, pmm_tol_achieved.
status == 0  → optimal solution found
status <  0  → infeasibility detected
status >  0  → iteration / time limit reached)");

    m.def("generate_pde_l1l2_qp", &generate_pde_l1l2_qp,
          py::arg("choice"), py::arg("nc"), py::arg("alpha1"), py::arg("alpha2"),
          py::arg("y_lower") = -std::numeric_limits<double>::infinity(),
          py::arg("y_upper") =  std::numeric_limits<double>::infinity(),
          R"(Generate a L1/L2-regularized PDE-constrained QP via split control variables.

choice = 'poisson'  or  'convdiff'
nc     = grid exponent (grid size = 2^nc + 1 per direction; n_display = 2*(2^nc+1)^2)
alpha1 = L1 regularisation weight
alpha2 = L2 regularisation weight (0 is valid)
y_lower/y_upper = state bounds (default ±inf, i.e. control-constrained only;
                  pass finite bounds for a state- or jointly-constrained problem)

Uses Q1 finite-element discretisation with the consistent mass matrix.

Returns the same dict format as parse_sif(), so the result can be passed
directly to solve_from_data() and used with kspqp_to_qpalm().)");

    m.def("generate_pde_l2_qp", &generate_pde_l2_qp,
          py::arg("choice"), py::arg("nc"), py::arg("beta"),
          py::arg("y_lower") = -std::numeric_limits<double>::infinity(),
          py::arg("y_upper") =  std::numeric_limits<double>::infinity(),
          py::arg("u_lower") = -std::numeric_limits<double>::infinity(),
          py::arg("u_upper") =  std::numeric_limits<double>::infinity(),
          py::arg("eps") = 0.01,
          R"(Generate a L2-regularized PDE-constrained QP.

choice = 'poisson' (control-constrained)  or  'poisson_state' (state-constrained)  or  'convdiff' (control- and state-constrained)
nc     = grid exponent (grid size = 2^nc + 1 per direction)
beta   = L2 regularisation weight
y_lower/y_upper/u_lower/u_upper = box bounds on state/control (default ±inf)
eps = diffusion coefficient, 'convdiff' only.

Uses Q1 finite-element discretisation with the consistent mass matrix.

Returns the same dict format as parse_sif(), so the result can be passed
directly to solve_from_data() and used with kspqp_to_qpalm().)");

    m.def("solve_from_data", &solve_from_data,
          py::arg("pd"),
          py::arg("tol")        = 1e-6,
          py::arg("max_iter")   = 1'000'000'000LL,
          py::arg("time_limit") = 600.0,
          R"(Solve with KSP-QP using already-parsed problem data (dict from parse_sif).

Returns a dict with keys: status, obj_val, setup_time, solve_time, run_time, pmm_iter, ssn_iter, pmm_tol_achieved.)");
}
