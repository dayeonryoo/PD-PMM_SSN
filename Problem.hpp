#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <optional>
using namespace std;
using namespace Eigen;

// =============================================================
//      min  c^T x + (1/2) x^T Q x,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw
// =============================================================

template <typename T>
class Problem {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat = Eigen::SparseMatrix<T>;

    Mat Q, A, B;
    Vec c, b;
    Vec lx, ux;
    Vec lw, uw;
    int n, m, l;
    T tol;
    int max_it;

    Problem(const Mat& Q_, const Mat& A_, const Mat& B_,
            std::optional<Vec> c_  = std::nullopt,
            std::optional<Vec> b_  = std::nullopt,
            std::optional<Vec> lx_ = std::nullopt,
            std::optional<Vec> ux_ = std::nullopt,
            std::optional<Vec> lw_ = std::nullopt,
            std::optional<Vec> uw_ = std::nullopt,
            T tol_ = T(1e-4),
            int max_it_ = 200)
    : Q(Q_), A(A_), B(B_), tol(tol_), max_it(max_it_)
    {
        // Validate required matrices
        if (Q.rows() == 0 || Q.cols() == 0) {
            throw std::invalid_argument("Matrix Q must be provided with nonzero dimensions.");
        }
        if (A.rows() == 0 || A.cols() == 0) {
            throw std::invalid_argument("Matrix A must be provided with nonzero dimensions.");
        }
        if (B.rows() == 0 || B.cols() == 0) {
            throw std::invalid_argument("Matrix B must be provided with nonzero dimensions.");
        }

        // Dimensions
        n = Q.rows();
        m = A.rows();
        l = B.rows();

        // Validate dimensions
        if (Q.cols() != n) {
            throw std::invalid_argument("Matrix Q must be square (n x n).");
        }
        if (A.cols() != n) {
            throw std::invalid_argument("Matrix A must have n columns.");
        }
        if (B.cols() != n) {
            throw std::invalid_argument("Matrix B must have n columns.");
        }

        // Defaults
        c = c_.value_or(Vec::Zero(n));
        b = b_.value_or(Vec::Zero(m));

        T inf = std::numeric_limits<T>::infinity();
        lx = lx_.value_or(Vec::Constant(n, -inf));
        ux = ux_.value_or(Vec::Constant(n, inf));
        lw = lw_.value_or(Vec::Constant(l, -inf));
        uw = uw_.value_or(Vec::Constant(l, inf));

        // Validate vector dimensions
        if (c.size() != n) {
            throw std::invalid_argument("Vector c must have size n.");
        }
        if (b.size() != m) {
            throw std::invalid_argument("Vector b must have size m.");
        }
        if (lx.size() != n || ux.size() != n) {
            throw std::invalid_argument("Vectors lx and ux must have size n.");
        }
        if (lw.size() != l || uw.size() != l) {
            throw std::invalid_argument("Vectors lw and uw must have size l.");
        }

    }
};