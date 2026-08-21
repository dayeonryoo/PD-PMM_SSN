include(GoogleTest)

# One binary per header under test, mirroring include/. Keeps failures
# localized (a broken PDEgenerator test can't stop FemQ1 tests from running)
# and keeps individual binaries fast to rebuild while iterating.
function(ksp_qp_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ksp_qp GTest::gtest_main)
  gtest_discover_tests(${name})
endfunction()

ksp_qp_add_test(test_fem_q1)
ksp_qp_add_test(test_pde_generator)
ksp_qp_add_test(test_mps_format_parser)
ksp_qp_add_test(test_problem)
ksp_qp_add_test(test_solution)
ksp_qp_add_test(test_printing)
ksp_qp_add_test(test_schur_operator)
ksp_qp_add_test(test_schur_preconditioner)
ksp_qp_add_test(test_ssn)
ksp_qp_add_test(test_ksp_qp)
ksp_qp_add_test(test_kkt_residual)
