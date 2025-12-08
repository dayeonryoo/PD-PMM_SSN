#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "SSN.hpp"
using namespace std;
using namespace Eigen;

template <typename T>
void SSN<T>::solve() {
    iter = 0;
    tol_achieved = false;
}