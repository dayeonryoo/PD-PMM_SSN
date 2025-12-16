cmake_minimum_required(VERSION 3.16)

project(SSN_PMM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------
# Dependencies
# ----------------------
include(FetchContent)

# Eigen
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.zip
)
FetchContent_MakeAvailable(eigen)

# GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# ----------------------
# Enable CTest
# ----------------------
enable_testing()

# ----------------------
# Solver library
# ----------------------
add_library(ssn_pmm INTERFACE)
target_include_directories(ssn_pmm INTERFACE include)
target_link_libraries(ssn_pmm INTERFACE Eigen3::Eigen)

# ----------------------
# Main executable
# ----------------------
add_executable(ssn_pmm_main src/main.cpp)
target_link_libraries(ssn_pmm_main ssn_pmm)

# ----------------------
# Tests
# ----------------------
add_executable(test_ssn tests/SSN_tests.cpp)
target_link_libraries(test_ssn
    ssn_pmm
    gtest
    gtest_main)
add_test(NAME SSNTests COMMAND test_ssn)

add_executable(test_pmm tests/PMM_tests.cpp)
target_link_libraries(test_pmm
    ssn_pmm
    gtest
    gtest_main)
add_test(NAME PMMTests COMMAND test_pmm)