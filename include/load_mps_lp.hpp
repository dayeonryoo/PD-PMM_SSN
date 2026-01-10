#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include <coin/CoinMpsIO.hpp>
#include <coin/CoinPackedMatrix.hpp>

template <typename T>
struct LPdata {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SpMat A;
    Vec rhs_lo; // rhs_lo <= Ax < rhs_hi
    Vec rhs_hi; 
    Vec c;
    Vec lb; // lb <= x <= ub
    Vec ub;
    std::vector<char> row_sense;
};

template <typename T>
LPdata<T> load_mps_lp(const std::string& filename) {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using Triplet = Eigen::Triplet<T>;

    CoinMpsIO mps;
    // mps.messageHandler()->setLogLevel(0);

    int status = mps.readMps(filename.c_str());
    if (status) {
        throw std::runtime_error("Failed to read MPS file: " + filename);
    }

    const int m = mps.getNumRows();
    const int n = mps.getNumCols();

    const CoinPackedMatrix* mat = mps.getMatrixByCol();
    if (!mat) {
        throw std::runtime_error("MPS matrix is null: " + filename);
    }

    if (mat->getNumRows() != m || mat->getNumCols() != n) {
        throw std::runtime_error("Matrix dimensions inconsistent in MPS file: " + filename);
    }

    // Build Eigen::SparseMatrix A from CoinPackedMatrix (column-wise)
    SpMat A(m, n);
    std::vector<Triplet> trpl;
    trpl.reserve(mat->getNumElements());

    const int* colStarts = mat->getVectorStarts();
    const int* colLens = mat->getVectorLengths();
    const int* rowIdx = mat->getIndices();
    const double* elements = mat->getElements();

    for (int j = 0; j < n; ++j) {
        int start = colStarts[j];
        int len = colLens[j];
        for (int k = start; k < start + len; ++k) {
            int i = rowIdx[k];
            T val = static_cast<T>(elements[k]);
            trpl.emplace_back(i, j, val);
        }
    }
    A.setFromTriplets(trpl.begin(), trpl.end());
    A.makeCompressed();

    // Objective
    Vec c(n);
    const double* obj = mps.getObjCoefficients();
    for (int j = 0; j < n; ++j) {
        c(j) = static_cast<T>(obj[j]);
    }

    // Infinity handling
    const double coinInf = mps.getInfinity();
    const double thresh = 0.5 * coinInf;
    const T plusInf  =  std::numeric_limits<T>::infinity();
    const T minusInf = -std::numeric_limits<T>::infinity();

    // Variable bounds
    Vec lb(n), ub(n);
    const double* colLower = mps.getColLower();
    const double* colUpper = mps.getColUpper();
    for (int j = 0; j < n; ++j) {
        double lo = colLower[j];
        double hi = colUpper[j];

        lb(j) = (lo <= -thresh) ? minusInf : static_cast<T>(lo);
        ub(j) = (hi >=  thresh) ? plusInf  : static_cast<T>(hi);
    }

    // Row bounds: rhs_lo <= Ax <= rhs_hi
    Vec rhs_lo(m), rhs_hi(m);
    const double* rowLower = mps.getRowLower();
    const double* rowUpper = mps.getRowUpper();
    for (int i = 0; i < m; ++i) {
        double lo = rowLower[i];
        double hi = rowUpper[i];

        rhs_lo(i) = (lo <= -thresh) ? minusInf : static_cast<T>(lo);
        rhs_hi(i) = (hi >=  thresh) ? plusInf  : static_cast<T>(hi);
    }

    // Row senses
    const char* sense = mps.getRowSense();
    std::vector<char> row_sense(m);
    for (int i = 0; i < m; ++i) {
        row_sense[i] = sense[i];
    }

    // Construct problem data
    LPdata<T> data;
    data.A         = std::move(A);
    data.rhs_lo    = std::move(rhs_lo);
    data.rhs_hi    = std::move(rhs_hi);
    data.c         = std::move(c);
    data.lb        = std::move(lb);
    data.ub        = std::move(ub);
    data.row_sense = std::move(row_sense);
    
    return data;
};