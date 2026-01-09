#include <gtest/gtest.h>
#include <cassert>
#include "SSN.hpp"
#include "Printing.hpp"

template <typename T>
SSN<T> make_LP_test_instance() {
    /*
    minimize x1 + x2
    s.t. x1 + x2 = 1, 0 <= x1, x2 <= 1.

    c = [1; 1], Q = [0, 0; 0, 0], A = [1, 1], b = [1], B = -,
    lx = [0; 0], ux = [1; 1], lw = -, uw = -,
    x = [0; 0], y1 = [0], y2 = -, z = [0; 0],
    mu = 1.5, rho = 1.5

    expected solution: x = [0.5; 0.5], y1 = [1.5], y2 = -, z = [0; 0]
    */
    using Vec = Eigen::VectorXd;
    using SpMat = Eigen::SparseMatrix<T>;

    int n = 2;
    int m = 1;
    int l = 0;

    SSN<T> ssn;

    ssn.n = n;
    ssn.m = m;
    ssn.l = l;

    ssn.c = Vec::Ones(n);
    ssn.Q = SpMat(n, n); // zero matrix

    ssn.A = SpMat(m, n);
    ssn.A.insert(0, 0) = 1.0;
    ssn.A.insert(0, 1) = 1.0;
    ssn.b = Vec::Ones(m);

    ssn.B = SpMat(l, n); // zero dimension

    ssn.lx = Vec::Zero(n);
    ssn.ux = Vec::Ones(n);
    ssn.lw = Vec::Zero(l);
    ssn.uw = Vec::Ones(l);

    ssn.x = Vec::Zero(n);
    ssn.y1 = Vec::Zero(m);
    ssn.y2 = Vec::Zero(l);
    ssn.z = Vec::Zero(n);

    ssn.mu = 1.5;
    ssn.rho = 1.5;

    ssn.SSN_tol = 1e-3;
    ssn.SSN_max_in_iter = 100;

    ssn.Q_diag = ssn.Q.diagonal();
    ssn.A_tr = ssn.A.transpose();
    ssn.B_tr = ssn.B.transpose();
    ssn.ones_n = Vec::Ones(n);
    ssn.ones_m = Vec::Ones(m);
    ssn.ones_l = Vec::Ones(l);

    ssn.SSN_print_when = PrintWhen::ALWAYS;
    ssn.SSN_print_what = PrintWhat::FULL;

    return ssn;
}

template<typename T>
SSN<T> make_small_test_instance() {
    /*
    minimize x1 + x2 + 0.5(x1^2 + x2^2)
    s.t. x1 + x2 = 0, 0 <= x1, x2 <= 1.

    c = [1; 1], Q = [1, 0; 0, 1], A = [1, 1], b = [0], B = [1, 0],
    lx = [0; 0], ux = [1; 1], lw = [0], uw = [1],
    x = [0; 0], y1 = [0], y2 = [0], z = [0; 0],
    mu = 1, rho = 1

    expected solution: x = [0; 0], y1 = [1], y2 = [0], z = [0; 0]
    */
    using Vec = Eigen::VectorXd;
    using SpMat = Eigen::SparseMatrix<T>;

    int n = 2;
    int m = 1;
    int l = 1;

    SSN<T> ssn;

    ssn.n = n;
    ssn.m = m;
    ssn.l = l;

    ssn.c = Vec::Ones(n);
    ssn.Q = SpMat(n, n);
    ssn.Q.insert(0, 0) = 1.0;
    ssn.Q.insert(1, 1) = 1.0;

    ssn.A = SpMat(m, n);
    ssn.A.insert(0, 0) = 1.0;
    ssn.A.insert(0, 1) = 1.0;
    ssn.b = Vec::Zero(m);

    ssn.B = SpMat(l, n);
    ssn.B.insert(0, 0) = 1.0;

    ssn.lx = Vec::Zero(n);
    ssn.ux = Vec::Ones(n);
    ssn.lw = Vec::Zero(l);
    ssn.uw = Vec::Ones(l);

    ssn.x = Vec::Zero(n);
    ssn.y1 = Vec::Zero(m);
    ssn.y2 = Vec::Zero(l);
    ssn.z = Vec::Zero(n);

    ssn.mu = 1.5;
    ssn.rho = 1.5;

    ssn.SSN_tol = 1e-5;
    ssn.SSN_max_in_iter = 100;

    ssn.Q_diag = ssn.Q.diagonal();
    ssn.A_tr = ssn.A.transpose();
    ssn.B_tr = ssn.B.transpose();
    ssn.ones_n = Vec::Ones(n);
    ssn.ones_m = Vec::Ones(m);
    ssn.ones_l = Vec::Ones(l);

    ssn.SSN_print_when = PrintWhen::ALWAYS;
    ssn.SSN_print_what = PrintWhat::FULL;

    return ssn;
}

TEST(SSN_BoxProjection, ClipsCorrectly) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn;

    Vec v(3); v << -1.0, 0.5, 2.0;
    Vec lo(3); lo << 0.0, 0.0, 0.0;
    Vec hi(3); hi << 1.0, 1.0, 1.0;

    Vec proj_v = ssn.proj(v, lo, hi);

    Vec expected(3); expected << 0.0, 0.5, 1.0;
    EXPECT_TRUE(proj_v.isApprox(expected));
}

TEST(SSN_BoxDistance, DistanceIsZeroInsideBox) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn;

    Vec v(2); v << 0.2, 0.8;
    Vec lo = Vec::Zero(2);
    Vec hi = Vec::Ones(2);

    Vec dist = ssn.compute_dist_box(v, lo, hi);

    EXPECT_NEAR(dist.norm(), 0.0, 1e-14);
}

TEST(SSN_BoxDistance, DistanceOutsideBox) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn;

    Vec v(2); v << -1.0, 2.0;
    Vec lo = Vec::Zero(2);
    Vec hi = Vec::Ones(2);

    Vec dist = ssn.compute_dist_box(v, lo, hi);  

    Vec expected(2); expected << -1.0, 1.0;

    EXPECT_TRUE(dist.isApprox(expected));
}

TEST(SSN_Lagrangian, GradientMatchesFiniteDifference) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn = make_small_test_instance<T>();

    Vec x = ssn.x;
    Vec y2 = ssn.y2;

    Vec grad = ssn.compute_grad_Lagrangian(x, y2);

    T eps = 1e-6;
    Vec fd_x = Vec::Zero(x.size());
    Vec fd_y2 = Vec::Zero(y2.size());

    for (int i = 0; i < x.size(); ++i) {
        Vec x_p = x;
        Vec x_m = x;
        x_p(i) += eps;
        x_m(i) -= eps;

        T L_x_p = ssn.compute_Lagrangian(x_p, y2);
        T L_x_m = ssn.compute_Lagrangian(x_m, y2);

        fd_x(i) = (L_x_p - L_x_m) / (2 * eps);
    }

    for (int i = 0; i < y2.size(); ++i) {
        Vec y2_p = y2;
        Vec y2_m = y2;
        y2_p(i) += eps;
        y2_m(i) -= eps;

        T L_y2_p = ssn.compute_Lagrangian(x, y2_p);
        T L_y2_m = ssn.compute_Lagrangian(x, y2_m);

        fd_y2(i) = (L_y2_p - L_y2_m) / (2 * eps);
    }

    EXPECT_LT((fd_x - grad.head(x.size())).norm(), 1e-4);
    EXPECT_LT((fd_y2 - grad.tail(y2.size())).norm(), 1e-4);
}

TEST(SSN_Grad, AtInitialPoint) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn = make_small_test_instance<T>();

    Vec x_new = ssn.x;
    Vec y2_new = ssn.y2;

    Vec grad = ssn.compute_grad_Lagrangian(x_new, y2_new);

    EXPECT_NEAR(grad(0), 1.0, 1e-4);
    EXPECT_NEAR(grad(1), 1.0, 1e-4);
    EXPECT_NEAR(grad(2), 0.0, 1e-4);

}

TEST(SSN_Grad, AtNewPoint) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn = make_small_test_instance<T>();

    Vec x_new = ssn.x + Vec::Constant(ssn.n, 0.5);
    Vec y2_new = ssn.y2 + Vec::Constant(ssn.l, 0.5);
    T mu = ssn.mu;
    T rho = ssn.rho;

    Vec grad = ssn.compute_grad_Lagrangian(x_new, y2_new);

    EXPECT_NEAR(grad(0), 1.5 + mu + 0.5/rho, 1e-4);
    EXPECT_NEAR(grad(1), 1.5 + mu + 0.5/rho, 1e-4);
    EXPECT_NEAR(grad(2), 0.5*0.5/mu, 1e-4);

}

TEST(SSN_Grad, AtOptimalPoint) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn = make_small_test_instance<T>();
    ssn.y1 = Vec::Ones(ssn.m); // Set dual variable to optimal
    T rho = ssn.rho;

    Vec grad = ssn.compute_grad_Lagrangian(ssn.x, ssn.y2);

    EXPECT_NEAR(grad(0), 0.0, 1e-4);
    EXPECT_NEAR(grad(1), 0.0, 1e-4);
    EXPECT_NEAR(grad(2), 0.0, 1e-4);

}

TEST(SSN_ClarkeProj, Interior_Boundary_Exterior_100) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn;

    Vec u(5); u << -1.0, 0.0, 0.3, 1.0, 1.8;
    Vec lo = Vec::Zero(5);
    Vec hi = Vec::Ones(5);

    Vec grad = ssn.Clarke_subgrad_of_proj(u, lo, hi);

    Vec expected(5); expected << 0.0, 0.0, 1.0, 0.0, 0.0;

    EXPECT_TRUE(grad.isApprox(expected));
}

TEST(SSN_SeparateRowsVec, MixedMask) {
    using T = double;
    using Vec = Eigen::VectorXd;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    SSN<T> ssn;

    Vec u(5); u << 1.0, 2.0, 3.0, 4.0, 5.0;
    BoolArr mask(5); mask << true, false, true, false, true;

    Vec v = ssn.separate_rows(u, mask);

    Vec expected(5); expected << 1.0, 3.0, 5.0, 2.0, 4.0;

    EXPECT_TRUE(v.isApprox(expected));
}

TEST(SSN_SeparateRowsVec, AllTrue) {
    using T = double;
    using Vec = Eigen::VectorXd;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    SSN<T> ssn;

    Vec u(3); u << 1.0, 2.0, 3.0;
    BoolArr mask(3); mask << true, true, true;

    Vec v = ssn.separate_rows(u, mask);

    EXPECT_TRUE(v.isApprox(u));
}

TEST(SSN_SeparateRowsVec, AllFalse) {
    using T = double;
    using Vec = Eigen::VectorXd;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    SSN<T> ssn;

    Vec u(3); u << 1.0, 2.0, 3.0;
    BoolArr mask(3); mask << false, false, false;

    Vec v = ssn.separate_rows(u, mask);

    EXPECT_TRUE(v.isApprox(u));
}

TEST(SSN_SeparateRowsMat, MixedMask) {
    using T = double;
    using Mat = Eigen::MatrixXd;
    using SpMat = Eigen::SparseMatrix<T>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    SSN<T> ssn;

    SpMat M(3, 2);
    M.insert(0, 0) = 1.0;
    M.insert(1, 0) = 2.0;
    M.insert(2, 1) = 3.0;

    BoolArr mask(3); mask << false, true, false;

    SpMat M_sep = ssn.separate_rows(M, mask);
    Mat M_sep_dense = Mat(M_sep);

    Mat expected(3, 2);
    expected << 2.0,0.0,  1.0,0.0,  0.0,3.0;

    EXPECT_TRUE(M_sep_dense.isApprox(expected));
}

TEST(SSN_StackRows, SimpleCase) {
    using T = double;
    using Mat = Eigen::MatrixXd;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN<T> ssn;

    SpMat A(2, 2);
    A.insert(0, 0) = 1.0;
    A.insert(1, 1) = 2.0;

    SpMat B(1, 2);
    B.insert(0, 0) = 3.0;

    SpMat stack = ssn.stack_rows(A, B);
    Mat dense = Mat(stack);

    Mat expected(3, 2);
    expected << 1.0,0.0,  0.0,2.0,  3.0,0.0;

    EXPECT_TRUE(dense.isApprox(expected));
}

TEST(SSN_StackRows, EmptySecondBlock) {
    using T = double;
    using Mat = Eigen::MatrixXd;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN<T> ssn;

    SpMat A(2, 2);
    A.insert(0, 0) = 1.0;
    A.insert(1, 1) = 2.0;

    SpMat B(0, 2);

    SpMat stack = ssn.stack_rows(A, B);
    Mat dense = Mat(stack);

    Mat expected = Mat(A);

    EXPECT_TRUE(dense.isApprox(expected));
}


TEST(SSN_RetriveRowOrder, MixedMask) {
    using T = double;
    using Vec = Eigen::VectorXd;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    SSN<T> ssn;

    Vec u_sel(3); u_sel << 1.0, 3.0, 5.0;
    Vec u_unsel(2); u_unsel << 2.0, 4.0;
    BoolArr mask(5); mask << true, false, true, false, true;

    Vec u = ssn.retrive_row_order(u_sel, u_unsel, mask);

    Vec expected(5); expected << 1.0, 2.0, 3.0, 4.0, 5.0;

    EXPECT_TRUE(u.isApprox(expected));
}

TEST(SSN_SolveViaChol, SimplePositiveDefinite) {
    using T = double;
    using Vec = Eigen::VectorXd;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN<T> ssn;

    SpMat M(2, 2);
    M.insert(0, 0) = 4.0;
    M.insert(1, 1) = 5.0;
    M.insert(0, 1) = 1.0;
    M.insert(1, 0) = 1.0;

    Vec r(2); r << 1.0, 2.0;

    Vec sol = ssn.solve_via_chol(M, r);

    Vec res = M * sol - r; // Check residual

    EXPECT_NEAR(res.norm(), 0.0, 1e-10);
}

TEST(SSN_SolveViaChol, IdentityMatrix) {
    using T = double;
    using Vec = Eigen::VectorXd;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN<T> ssn;

    SpMat M(3, 3);
    M.insert(0, 0) = 1.0;
    M.insert(1, 1) = 1.0;
    M.insert(2, 2) = 1.0;

    Vec r(3); r << 5.0, -3.0, 2.0;

    Vec sol = ssn.solve_via_chol(M, r);

    EXPECT_TRUE(sol.isApprox(r));
}

TEST(SSN_SolveViaChol, NonPositiveDefiniteThrows) {
    using T = double;
    using Vec = Eigen::VectorXd;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN<T> ssn;

    SpMat M(2, 2);
    M.insert(0, 0) = 0.0;
    M.insert(1, 1) = -1.0;

    Vec r(2); r << 1.0, 2.0;

    EXPECT_THROW(ssn.solve_via_chol(M, r), std::runtime_error);
}

TEST(SSN_SolveViaChol, SchurLikeMatrix) {
    using T = double;
    using Vec = Eigen::VectorXd;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN<T> ssn;

    SpMat G(2, 2);
    G.insert(0, 0) = 2.0;
    G.insert(1, 1) = 3.0;
    SpMat G_tr = G.transpose();

    SpMat Hinv(2, 2);
    Hinv.insert(0, 0) = 0.5;
    Hinv.insert(1, 1) = 0.3;

    SpMat D(2, 2);
    D.insert(0, 0) = 1.0;
    D.insert(1, 1) = 1.0;

    SpMat S = G * Hinv * G_tr + D;

    Vec r(2); r << 1.0, 1.0;

    Vec sol = ssn.solve_via_chol(S, r);

    Vec res = S * sol - r; // Check residual

    EXPECT_NEAR(res.norm(), 0.0, 1e-10);
}

TEST(SSN_LineSearch, ArmijoCondition) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn = make_small_test_instance<T>();

    Vec dx = Vec::Random(ssn.n);
    Vec dy2 = Vec::Random(ssn.l);

    T L0 = ssn.compute_Lagrangian(ssn.x, ssn.y2);
    Vec grad_L0 = ssn.compute_grad_Lagrangian(ssn.x, ssn.y2);
    T grad_desc = grad_L0.head(ssn.n).dot(dx) + grad_L0.tail(ssn.l).dot(dy2);

    T alpha = ssn.backtracking_line_search(ssn.x, ssn.y2, dx, dy2);
    EXPECT_GT(alpha, 0.0);

    Vec x_new = ssn.x + alpha * dx;
    Vec y2_new = ssn.y2 + alpha * dy2;

    T L1 = ssn.compute_Lagrangian(x_new, y2_new);

    EXPECT_LE(L1 - L0, 1e-3); // Descent
    EXPECT_LE(L1, L0 + ssn.beta * alpha * grad_desc); // Armijo condition
}


TEST(SSN_SolveSSN, SmallTestInstance) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn = make_small_test_instance<T>();

    ssn.x = Vec::Ones(ssn.n) * 0.01; // close to optimal which is [0; 0]
    ssn.y1 = Vec::Ones(ssn.m); // optimal
    ssn.y2 = Vec::Zero(ssn.l); // optimal
    ssn.z = Vec::Zero(ssn.n); // optimal

    T eps = 1e-3;
    SSN_result<T> result = ssn.solve_SSN(eps);

    EXPECT_LT(result.SSN_tol_achieved, eps); // convergence
    EXPECT_LE(result.SSN_in_iter, ssn.SSN_max_in_iter); // reasonable iterations

    Vec expected_x = Vec::Zero(2); // optimal solution
    Vec expected_y2 = Vec::Zero(1);

    // EXPECT_NEAR(result.x(0), expected_x(0), eps);
    // EXPECT_NEAR(result.x(1), expected_x(1), eps);
    // EXPECT_NEAR(result.y2(0), expected_y2(0), eps);
}

TEST(SSN_SolveSSN, LPTestInstance) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN<T> ssn = make_LP_test_instance<T>();
    // Dual variables are close or at optimal
    ssn.x = Vec::Ones(ssn.n); // perturbed from optimal [0.5; 0.5]
    ssn.y1 = Vec::Ones(ssn.m); // perturbed from optimal [1.5]

    T eps = 1e-3;
    SSN_result<T> result = ssn.solve_SSN(eps);

    EXPECT_LT(result.SSN_tol_achieved, eps); // convergence
    EXPECT_LE(result.SSN_in_iter, ssn.SSN_max_in_iter); // reasonable iterations

    Vec expected_x = Vec::Ones(2) * 0.5; // optimal solution
    // EXPECT_NEAR(result.x(0), expected_x(0), eps);
    // EXPECT_NEAR(result.x(1), expected_x(1), eps);

}