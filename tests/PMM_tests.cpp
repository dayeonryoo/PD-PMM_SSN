#include <gtest/gtest.h>
#include <cassert>
#include "SSN_PMM.hpp"

template<typename T>
SSN_PMM<T> make_simple_ssn_pmm() {
    /*
    minimize x1 + x2 + 0.5(x1^2 + x2^2)
    s.t. x1 + x2 = 0, 0 <= x1, x2 <= 1.

    c = [1; 1], Q = [1, 0; 0, 1], A = [1, 1], b = [0], B = [1, 0],
    lx = [0; 0], ux = [1; 1], lw = [0], uw = [1],
    x = [0; 0], y1 = [1], y2 = [0], z = [0; 0],
    */
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN_PMM<T> pmm;

    int n = 2;
    int m = 1;
    int l = 1;

    pmm.n = n;
    pmm.m = m;
    pmm.l = l;

    // Objective
    pmm.c = Vec::Ones(n);
    pmm.Q = SpMat(n, n);
    pmm.Q.insert(0, 0) = 1.0;
    pmm.Q.insert(1, 1) = 1.0;

    // Constraints
    pmm.A = SpMat(m, n);
    pmm.A.insert(0, 0) = 1.0;
    pmm.A.insert(0, 1) = 1.0;
    pmm.b = Vec::Zero(m);

    pmm.B = SpMat(l, n);
    pmm.B.insert(0, 0) = 1.0;

    // Bounds
    pmm.lx = Vec::Zero(n);
    pmm.ux = Vec::Ones(n);
    pmm.lw = Vec::Zero(l);
    pmm.uw = Vec::Ones(l);

    // Variables (KKT-consistent)
    pmm.x  = Vec::Zero(n);
    pmm.y1 = Vec::Ones(m);
    pmm.y2 = Vec::Zero(l);
    pmm.z  = Vec::Zero(n);

    pmm.mu = 1.0;
    pmm.rho = 1.0;
    pmm.reg_limit = 100.0;

    pmm.tol = 1e-6;
    pmm.max_iter = 20;

    pmm.SSN_tol = 1e-6;
    pmm.SSN_max_iter = 50;
    pmm.SSN_max_in_iter = 20;

    pmm.PMM_print_what = PrintWhat::FULL;
    pmm.PMM_print_when = PrintWhen::ALWAYS;

    return pmm;
}

TEST(PMM_BoxProjection, ClipsCorrectly) {
    using T = double;
    using Vec = Eigen::VectorXd;

    SSN_PMM<T> pmm;

    Vec v(3); v << -1.0, 0.5, 2.0;
    Vec lo(3); lo << 0.0, 0.0, 0.0;
    Vec hi(3); hi << 1.0, 1.0, 1.0;

    Vec proj_v = pmm.proj(v, lo, hi);

    Vec expected(3); expected << 0.0, 0.5, 1.0;
    EXPECT_TRUE(proj_v.isApprox(expected));
}

TEST(PMM_Residuals, ZeroAtKKTPoint) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    SSN_PMM<T> pmm = make_simple_ssn_pmm<T>(); // KKT-consistent system

    Vec res = pmm.compute_residual_norms();

    Vec expected = Vec::Zero(4); // primal, dual, complementarity

    EXPECT_TRUE(res.isApprox(expected));
}

TEST(PMM_Residuals, NonZeroWhenPerturbed) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    SSN_PMM<T> pmm = make_simple_ssn_pmm<T>();

    // Perturb from KKT point
    pmm.x(0) = -1.0;
    pmm.x(1) = 2.0;
    pmm.y1(0) = 0.0;
    pmm.y2(0) = 2.0;
    pmm.z(0) = 3.0;
    pmm.z(1) = 2.0; 

    Vec res = pmm.compute_residual_norms();

    EXPECT_NEAR(res(0), 1.0, 1e-14); // primal residual
    EXPECT_NEAR(res(1), std::sqrt(26.0) / (1 + std::sqrt(2.0)), 1e-14); // dual residual
    EXPECT_NEAR(res(2), std::sqrt(5.0), 1e-14); // complementarity for x
    EXPECT_NEAR(res(3), 1.0, 1e-14); // complementarity for w
}

TEST(PMM_UpdateParams, AggressiveUpdate_Primal) {
    using T = double;

    SSN_PMM<T> pmm = make_simple_ssn_pmm<T>();

    T res_p = 10.0;
    T res_d = 10.0;
    T new_res_p = 8.0; // 0.95 * 10 = 9.5 > 8.0
    T new_res_d = 9.9; // 0.95 * 10 = 9.5 > 9.9 is false

    T old_mu = pmm.mu;
    T old_rho = pmm.rho;

    pmm.update_PMM_parameters(res_p, res_d, new_res_p, new_res_d);

    EXPECT_NEAR(pmm.mu, 1.2*old_mu, 1e-14);
    EXPECT_NEAR(pmm.rho, 1.4*old_rho, 1e-14);
}

TEST(PMM_UpdateParams, AggressiveUpdate_Dual) {
    using T = double;

    SSN_PMM<T> pmm = make_simple_ssn_pmm<T>();

    T res_p = 10.0;
    T res_d = 10.0;
    T new_res_p = 9.9; // 0.95 * 10 = 9.5 > 9.9 is false
    T new_res_d = 8.9; // 0.95 * 10 = 9.5 > 8.9

    T old_mu = pmm.mu;
    T old_rho = pmm.rho;

    pmm.update_PMM_parameters(res_p, res_d, new_res_p, new_res_d);

    EXPECT_NEAR(pmm.mu, 1.2*old_mu, 1e-14);
    EXPECT_NEAR(pmm.rho, 1.4*old_rho, 1e-14);
}

TEST(PMM_UpdateParams, SlowUpdate) {
    using T = double;

    SSN_PMM<T> pmm = make_simple_ssn_pmm<T>();

    T res_p = 10.0;
    T res_d = 10.0;
    T new_res_p = 9.5; // 0.95 * 10 = 9.5 > 9.5 is false
    T new_res_d = 9.9; // 0.95 * 10 = 9.5 > 9.9 is false

    T old_mu = pmm.mu;
    T old_rho = pmm.rho;

    pmm.update_PMM_parameters(res_p, res_d, new_res_p, new_res_d);

    EXPECT_NEAR(pmm.mu, 1.1*old_mu, 1e-14);
    EXPECT_NEAR(pmm.rho, 1.1*old_rho, 1e-14);
}

TEST(PMM_UpdateParams, SaturationAtRegLimit) {
    using T = double;

    SSN_PMM<T> pmm = make_simple_ssn_pmm<T>();

    pmm.mu = 0.99 * pmm.reg_limit;
    pmm.rho = 0.99 * 1e2 * pmm.reg_limit;

    pmm.update_PMM_parameters(10.0, 10.0, 8.9, 8.9); // aggressive update

    EXPECT_NEAR(pmm.mu, pmm.reg_limit, 1e-14);
    EXPECT_NEAR(pmm.rho, 1e2*pmm.reg_limit, 1e-14);
}

TEST(PMM_Solve, AlreadyOptimal) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    SSN_PMM<T> pmm = make_simple_ssn_pmm<T>();

    Solution<T> sol = pmm.solve();

    EXPECT_EQ(sol.opt, 0); // optimal solution found

    // Verify KKT conditions
    Vec res = pmm.compute_residual_norms();
    Vec expected = Vec::Zero(4); // primal, dual, complementarity
    EXPECT_TRUE(res.isApprox(expected));

    EXPECT_TRUE(sol.x.isApprox(pmm.x)); // solution matches initial KKT point
    EXPECT_TRUE(sol.y1.isApprox(pmm.y1));
    EXPECT_TRUE(sol.y2.isApprox(pmm.y2));
    EXPECT_TRUE(sol.z.isApprox(pmm.z));
}

TEST(PMM_Solve, SimpleProblem) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    SSN_PMM<T> pmm = make_simple_ssn_pmm<T>();

    // Perturb from KKT point
    pmm.x(0) = -1.0;
    pmm.x(1) = 2.0;
    pmm.y1(0) = 0.0;
    pmm.y2(0) = 2.0;
    pmm.z(0) = 3.0;
    pmm.z(1) = 2.0; 

    Solution<T> sol = pmm.solve();

    EXPECT_EQ(sol.opt, 0); // optimal solution found

    // Verify KKT conditions
    Vec res = pmm.compute_residual_norms();
    std::cout << "Final residuals: " << res.transpose() << std::endl;
    Vec expected = Vec::Zero(4); // primal, dual, complementarity
    EXPECT_NEAR(res(0), 0.0, pmm.tol); // primal
    EXPECT_NEAR(res(1), 0.0, pmm.tol); // dual
    EXPECT_NEAR(res(2), 0.0, pmm.tol); // complementarity for x
    EXPECT_NEAR(res(3), 0.0, pmm.tol); // complementarity for w

    // Verify solution (primal)
    EXPECT_NEAR(sol.x(0), 0.0, pmm.tol);
    EXPECT_NEAR(sol.x(1), 0.0, pmm.tol);

}

TEST(PMM_DetermineDimensions, FromC) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    SSN_PMM<T> pmm;
    pmm.c = Vec::Zero(5);

    pmm.determine_dimensions();

    EXPECT_EQ(pmm.n, 5);
    EXPECT_EQ(pmm.m, 0);
    EXPECT_EQ(pmm.l, 0);
}

TEST(PMM_DetermineDimensions, FromQ) {
    using T = double;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN_PMM<T> pmm;
    pmm.Q = SpMat(4, 4);

    pmm.determine_dimensions();

    EXPECT_EQ(pmm.n, 4);
    EXPECT_EQ(pmm.m, 0);
    EXPECT_EQ(pmm.l, 0);
}

TEST(PMM_DetermineDimensions, FromA) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN_PMM<T> pmm;
    pmm.A = SpMat(3, 6);
    pmm.determine_dimensions();

    EXPECT_EQ(pmm.n, 6);
    EXPECT_EQ(pmm.m, 3);
    EXPECT_EQ(pmm.l, 0);
}

TEST(PMM_DetermineDimensions, FromB) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN_PMM<T> pmm;
    pmm.B = SpMat(8, 7);
    pmm.determine_dimensions();

    EXPECT_EQ(pmm.n, 7);
    EXPECT_EQ(pmm.m, 0);
    EXPECT_EQ(pmm.l, 8);

}

TEST(PMM_DetermineDimensions, Fromlxux) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    SSN_PMM<T> pmm;
    pmm.lx = Vec::Zero(9);

    pmm.determine_dimensions();

    EXPECT_EQ(pmm.n, 9);
    EXPECT_EQ(pmm.m, 0);
    EXPECT_EQ(pmm.l, 0);
}

TEST(PMM_DefaultSettings, InitializesVariables) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN_PMM<T> pmm;
    pmm.c = Vec::Ones(3);
    
    pmm.determine_dimensions();
    pmm.set_default();

    EXPECT_EQ(pmm.mu, 50.0);
    EXPECT_EQ(pmm.rho, 100.0);
    EXPECT_EQ(pmm.tol, 1e-6);
    EXPECT_EQ(pmm.max_iter, 1e3);
    EXPECT_EQ(pmm.SSN_max_iter, 4000);
    EXPECT_EQ(pmm.SSN_max_in_iter, 40);
    EXPECT_EQ(pmm.SSN_tol, pmm.tol);
    EXPECT_EQ(pmm.reg_limit, 1e6);

    EXPECT_TRUE(pmm.x.isApprox(Vec::Zero(3)));
    EXPECT_TRUE(pmm.y1.isApprox(Vec::Zero(0)));
    EXPECT_TRUE(pmm.y2.isApprox(Vec::Zero(0)));
    EXPECT_TRUE(pmm.z.isApprox(Vec::Zero(3)));

    EXPECT_EQ(pmm.Q.rows(), 3);
    EXPECT_EQ(pmm.Q.cols(), 3);
    EXPECT_EQ(pmm.A.rows(), 0);
    EXPECT_EQ(pmm.A.cols(), 3);
    EXPECT_TRUE(pmm.b.isApprox(Vec::Zero(0)));
    EXPECT_EQ(pmm.B.rows(), 0);
    EXPECT_EQ(pmm.B.cols(), 3);

    T inf = std::numeric_limits<T>::infinity();
    EXPECT_EQ(pmm.lx.size(), 3);
    EXPECT_EQ(pmm.ux.size(), 3);
    EXPECT_EQ(pmm.lw.size(), 0);
    EXPECT_EQ(pmm.uw.size(), 0);
    EXPECT_EQ(pmm.lx(0), -inf);
    EXPECT_EQ(pmm.ux(0), inf);
}

TEST(PMM_Default_Solve, Given_c_lx) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    SSN_PMM<T> pmm;
    EXPECT_THROW(pmm.determine_dimensions(), std::invalid_argument);

    pmm.c = Vec::Ones(3);
    pmm.lx = Vec::Ones(3) * -1.0;
    pmm.determine_dimensions();
    pmm.set_default();

    Solution<T> sol = pmm.solve();
    EXPECT_EQ(sol.opt, 0);
    EXPECT_TRUE(sol.x.isApprox(pmm.lx, pmm.tol));
}

TEST(PMM_Default_Solve, Given_A_b) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN_PMM<T> pmm;
    pmm.A = SpMat(1, 2);
    pmm.A.insert(0, 0) = 1.0;
    pmm.A.insert(0, 1) = 1.0;
    pmm.b = Vec::Ones(1);
    
    pmm.determine_dimensions();
    pmm.set_default();

    Solution<T> sol = pmm.solve();
    Vec expected = Vec::Ones(2) / 2;
    EXPECT_EQ(sol.opt, 0);
    EXPECT_TRUE(sol.x.isApprox(expected, pmm.tol));
}

TEST(PMM_Default_Solve, Given_c_B_uw) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN_PMM<T> pmm;
    pmm.c = -1.0 * Vec::Ones(2);
    pmm.B = SpMat(1, 2); 
    pmm.B.insert(0, 0) = 1.0;
    pmm.B.insert(0, 1) = 1.0;
    pmm.uw = Vec::Ones(1);
    
    pmm.determine_dimensions();
    pmm.set_default();

    Solution<T> sol = pmm.solve();
    Vec expected = Vec::Ones(2) / 2;
    EXPECT_EQ(sol.opt, 0);
    EXPECT_TRUE(sol.x.isApprox(expected, pmm.tol));
}

TEST(PMM_CheckDimension, MixedCase) {
    using T = double;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    SSN_PMM<T> pmm;

    pmm.Q = SpMat(2, 2);
    pmm.A = SpMat(1, 3); // Wrong n
    
    pmm.determine_dimensions();
    pmm.set_default();

    EXPECT_THROW(pmm.check_dimensionality(), std::invalid_argument);

    pmm.A = SpMat(1, 2); // Correct n
    pmm.b = Vec::Zero(2); // Wrong m
    
    pmm.determine_dimensions();
    pmm.set_default();
    EXPECT_EQ(pmm.n, 2);
    EXPECT_THROW(pmm.check_dimensionality(), std::invalid_argument);

    pmm.b = Vec::Zero(1); // Correct m

    pmm.determine_dimensions();
    pmm.set_default();
    pmm.check_dimensionality();

    EXPECT_EQ(pmm.m, 1);
    EXPECT_EQ(pmm.l, 0);
    
}