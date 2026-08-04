cmake_minimum_required(VERSION 3.16)

project(SSN_PMM VERSION 0.1.0 LANGUAGES CXX)

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

# ----------------------
# Solver library
# ----------------------
add_library(ssn_pmm INTERFACE)
target_include_directories(ssn_pmm INTERFACE include)
# Eigen's unsupported modules (including MINRES in unsupported/Eigen/IterativeSolvers)
# are bundled in the Eigen download above. This explicit include ensures the path resolves.
target_include_directories(ssn_pmm INTERFACE $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
target_link_libraries(ssn_pmm
    INTERFACE
        Eigen3::Eigen)

# ----------------------
# Main executable
# ----------------------
add_executable(ssn_pmm_netlib src/netlib.cpp)
target_link_libraries(ssn_pmm_netlib ssn_pmm)

add_executable(mps_parser src/readingMps.cpp)
target_link_libraries(mps_parser ssn_pmm)

add_executable(ssn_pmm_maros_meszaros src/maros_meszaros.cpp)
target_link_libraries(ssn_pmm_maros_meszaros ssn_pmm)

add_executable(ssn_pmm_pde src/PDE.cpp)
target_link_libraries(ssn_pmm_pde ssn_pmm)
