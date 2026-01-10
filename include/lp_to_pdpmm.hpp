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
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;
    using Map = Eigen::Map<const Vec>;

    const int n = lp.num_col_; // total number of variables (x + w)
    const int m = lp.num_row_; // number of constraints

    // Objective
    Vec c = Map(lp.col_cost_.data(), n);

    // Variable bounds
    Vec lx = Map(lp.col_lower_.data(), n);
    Vec ux = Map(lp.col_upper_.data(), n);

    // Constraint matrix
    const HighsSparseMatrix& A_highs = lp.a_matrix_;
    SpMat A_full(m, n);
    {
        std::vector<Triplet> trips;
        trips.reserve(A_highs.value_.size());

        if (A_highs.format_ == MatrixFormat::kColwise) {
            for (int col = 0; col < n; ++col) {
                int k_start = A_highs.start_[col];
                int k_end   = A_highs.start_[col + 1];
                for (int k = k_start; k < k_end; ++k) {
                    int row    = A_highs.index_[k];
                    double val = A_highs.value_[k];
                    trips.emplace_back(row, col, val);
                }
            }
        } else if (A_highs.format_ == MatrixFormat::kRowwise) {
            for (int row = 0; row < m; ++row) {
                int k_start = A_highs.start_[row];
                int k_end   = A_highs.start_[row + 1];
                for (int k = k_start; k < k_end; ++k) {
                    int col    = A_highs.index_[k];
                    double val = A_highs.value_[k];
                    trips.emplace_back(row, col, val);
                }
            }
        } else {
            throw std::runtime_error("Unknown matrix format in HiGHS");
        }

        A_full.setFromTriplets(trips.begin(), trips.end());
    }

    // Constraints
    Vec row_lower = Map(lp.row_lower_.data(), m);
    Vec row_upper = Map(lp.row_upper_.data(), m);

    // Split constraints into equalities and inequalities
    const double tol = 1e-9;
    std::vector<int> eq_rows;
    std::vector<int> ineq_rows;
    eq_rows.reserve(m);
    ineq_rows.reserve(m);
    
    for (int i = 0; i < m; ++i) {
        double lo = row_lower[i];
        double hi = row_upper[i];

        if (std::abs(hi - lo) < tol) {
            eq_rows.push_back(i);
        } else {
            ineq_rows.push_back(i);
        }
    }

    int m_eq = static_cast<int>(eq_rows.size());
    int m_ineq = static_cast<int>(ineq_rows.size());

    // Construct A and b (equality constraints)
    SpMat A(m_eq, n);
    Vec b(m_eq);
    {
        std::vector<Triplet> trips;
        trips.reserve(A_full.nonZeros());

        std::vector<int> row_map(m, -1);
        for (int k = 0; k < m_eq; ++k) {
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
    }

    // Build B, lw and uw (inequalities)
    SpMat B(m_ineq, n);
    Vec lw(m_ineq), uw(m_ineq);
    {
        std::vector<Triplet> trips;
        trips.reserve(A_full.nonZeros());

        std::vector<int> row_map(m, -1);
        for (int k = 0; k < m_ineq; ++k) {
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