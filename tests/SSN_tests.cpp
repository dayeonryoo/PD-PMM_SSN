#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "SSN.hpp"
#include <gtest/gtest.h>

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

int main() {

    // Define problem data
    const int n = 2;
    const int m = 1;
    const int l = 2;

    // Q = 0
    SpMat Q(n, n);
    Q.setZero();

    // A = [1; 1]
    SpMat A(m, n);
    std::vector<Triplet> A_trpl;
    A_trpl.emplace_back(0, 0, 1.0);
    A_trpl.emplace_back(0, 1, 1.0);
    A.setFromTriplets(A_trpl.begin(), A_trpl.end());
    
    // B = I_2
    SpMat B(l, n);
    std::vector<Triplet> B_trpl;
    B_trpl.emplace_back(0, 0, 1.0);
    B_trpl.emplace_back(1, 1, 1.0);
    B.setFromTriplets(B_trpl.begin(), B_trpl.end());

    // c = [1; 2], b = 0
    Vec c(n);
    c << 1.0, 2.0;
    Vec b(m);
    b << 0.0;

    // 0 <= x, w <= 1
    Vec lx(n), ux(n);
    lx.setZero();
    ux.setOnes();
    Vec lw(l), uw(l);
    lw.setZero();
    uw.setOnes();

    Vec x = Vec::Zero(n);
    Vec y1 = Vec::Zero(m);
    Vec y2 = Vec::Zero(l);
    Vec z = Vec::Zero(n);

    T mu = 5e1;
    T rho = 1e2;
    T SSN_tol = 0.1;
    int SSN_max_in_iter = 5;

    SSN<T> NS(Q, A, B, c, b, lx, ux, lw, uw,
              x, y1, y2, z,
              mu, rho, n, m, l,
              SSN_tol, SSN_max_in_iter);

    return 0;
}