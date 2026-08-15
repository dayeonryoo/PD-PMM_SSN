cmake_minimum_required(VERSION 3.16)

project(KSP_QP VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -march=native tunes for the CPU that runs *this* build, so binaries are not portable to
# other machines (they can crash with "Illegal instruction" on a different/older CPU). Safe as
# long as everyone builds from source on their own machine; do not reuse a compiled build/
# across machines, bake one into a Docker image that runs elsewhere, or ship it as a release
# binary without switching this to a portable baseline (e.g. -march=x86-64-v2).
add_compile_options(-O3 -march=native -funroll-loops)

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

# GoogleTest (only needed when tests are built)
option(KSP_QP_BUILD_TESTS "Build the gtest test suite" ON)
if (KSP_QP_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  )
  # Match the rest of the project against the parent compiler/runtime settings.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# ----------------------
# Solver library
# ----------------------
add_library(ksp_qp INTERFACE)
target_include_directories(ksp_qp INTERFACE include)
# Eigen's unsupported modules (including MINRES in unsupported/Eigen/IterativeSolvers)
# are bundled in the Eigen download above. This explicit include ensures the path resolves.
target_include_directories(ksp_qp INTERFACE $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
target_link_libraries(ksp_qp
    INTERFACE
        Eigen3::Eigen)

# ----------------------
# Main executable
# ----------------------
add_executable(ksp_qp_netlib src/netlib.cpp)
target_link_libraries(ksp_qp_netlib ksp_qp)

add_executable(mps_parser src/reading_mps.cpp)
target_link_libraries(mps_parser ksp_qp)

add_executable(ksp_qp_maros_meszaros src/maros_meszaros.cpp)
target_link_libraries(ksp_qp_maros_meszaros ksp_qp)

add_executable(ksp_qp_pde src/pde.cpp)
target_link_libraries(ksp_qp_pde ksp_qp)

# ----------------------
# Tests
# ----------------------
if (KSP_QP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
