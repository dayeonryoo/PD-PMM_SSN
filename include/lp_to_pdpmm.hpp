#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <vector>
#include <cmath>

template <typename T>
struct LPdata;

template <typename T>
struct PDPMMdata {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
};

template <typename T>
PDPMMdata<T> lp_to_pdpmm(const HighsLp& lp) {
    /*
    HiGHS LP data structure
    ------------------------------------------------
    min  c^T x
    s.t. row_lower <= A_full x <= row_upper
         col_lower <= x <= col_upper
    ------------------------------------------------
    num_col_ : number of variables (n)
    num_row_ : number of constraints, i.e. rows of A_full (m)
    col_cost_  : objective coefficients c
    col_lower_ : variable lower bounds
    col_upper_ : variable upper bounds
    row_lower_ : constraint lower bounds
    row_upper_ : constraint upper bounds
    a_matrix_  : constraint matrix
    sense_     : optimization sense (min)
    offset_    : objective offset (0.0)
    model_name_:  name of the model
    objective_name_: name of the objective
    col_names_ : names of the variables
    row_names_ : names of the constraints
    integer_columns_ : indices of integer variables
    ------------------------------------------------
    */
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;
    using Map = Eigen::Map<const Vec>;

    const int n = lp.num_col_; // number of variables (n)
    const int ml = lp.num_row_; // number of constraints (m + l)

    // Objective
    Vec c = Map(lp.col_cost_.data(), n);

    // Variable bounds
    Vec lx = Map(lp.col_lower_.data(), n);
    Vec ux = Map(lp.col_upper_.data(), n);

    // Constraints bounds
    Vec row_lower = Map(lp.row_lower_.data(), ml);
    Vec row_upper = Map(lp.row_upper_.data(), ml);

    // Constraint matrix
    const HighsSparseMatrix& A_highs = lp.a_matrix_;
    SpMat A_full(ml, n);
    A_full.reserve(A_highs.value_.size());
    for (int col = 0; col < n; ++col) { // HiGHS uses column-wise storage
        int k_start = A_highs.start_[col];
        int k_end   = A_highs.start_[col + 1];
        for (int k = k_start; k < k_end; ++k) {
            int row    = A_highs.index_[k];
            double val = A_highs.value_[k];
            A_full.insert(row, col) = val;
        }
    }
    A_full.makeCompressed();

    // Split A_full into equality and inequality rows
    const double tol = 1e-9;
    std::vector<int> eq_rows;
    std::vector<int> ineq_rows;
    eq_rows.reserve(ml);
    ineq_rows.reserve(ml);

    for (int i = 0; i < ml; ++i) {
        double lo = row_lower[i];
        double hi = row_upper[i];
        if (std::abs(hi - lo) < tol) {
            eq_rows.push_back(i);
        } else {
            ineq_rows.push_back(i);
        }
    }

    int m = static_cast<int>(eq_rows.size()); // number of equality constraints
    int l = static_cast<int>(ineq_rows.size()); // number of inequality constraints

    // Construct A and b (equality constraints)
    SpMat A(m, n);
    Vec b(m);
    {
        std::vector<Triplet> trips;
        trips.reserve(A_full.nonZeros());

        std::vector<int> row_map(ml, -1);
        for (int k = 0; k < m; ++k) {
            row_map[eq_rows[k]] = k;
            b[k] = row_lower[eq_rows[k]];
        }

        for (int col = 0; col < A_full.outerSize(); ++col) {
            for (typename SpMat::InnerIterator it(A_full, col); it; ++it) {
                int r = it.row();
                int loc = row_map[r];
                if (loc >= 0) {
                    trips.emplace_back(loc, col, it.value());
                }
            }
        }
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();
    }

    // Build B, lw and uw (inequalities)
    SpMat B(l, n);
    Vec lw(l), uw(l);
    {
        std::vector<Triplet> trips;
        trips.reserve(A_full.nonZeros());

        std::vector<int> row_map(ml, -1);
        for (int k = 0; k < l; ++k) {
            int r = ineq_rows[k];
            row_map[r] = k;
            lw[k] = row_lower[r];
            uw[k] = row_upper[r];
        }

        for (int col = 0; col < A_full.outerSize(); ++col) {
            for (typename SpMat::InnerIterator it(A_full, col); it; ++it) {
                int r = it.row();
                int loc = row_map[r];
                if (loc >= 0) {
                    trips.emplace_back(loc, col, it.value());
                }
            }
        }
        B.setFromTriplets(trips.begin(), trips.end());
        B.makeCompressed();
    }

    // Q = 0
    SpMat Q(n, n);
    Q.setZero();
    Q.makeCompressed();

    // Copy to PDPMM data structure
    PDPMMdata<T> pd;
    pd.Q = std::move(Q);
    pd.A = std::move(A);
    pd.B = std::move(B);
    pd.c = std::move(c);
    pd.b = std::move(b);
    pd.lx = std::move(lx);
    pd.ux = std::move(ux);
    pd.lw = std::move(lw);
    pd.uw = std::move(uw);

    return pd;
}