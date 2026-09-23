cmake_minimum_required(VERSION 3.16)
project(ksp_qp_python_bindings VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native -funroll-loops)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# ----------------------
# Python (must be found before pybind11 to pick the right interpreter)
# ----------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ----------------------
# pybind11
# ----------------------
# Use the modern find_package(Python3) path instead of the legacy
# FindPythonLibsNew.cmake (which tries to execute the framework's
# 'python' shared library and fails on macOS).
set(PYBIND11_FINDPYTHON ON)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.1
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

# ----------------------
# Eigen (same version as the main project)
# ----------------------
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(eigen)

# ----------------------
# Python extension module
# ----------------------
pybind11_add_module(ksp_qp_bind ksp_qp_bind.cpp)

target_include_directories(ksp_qp_bind
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${eigen_SOURCE_DIR}
    ${eigen_SOURCE_DIR}/unsupported   # for MINRES / iterative solvers
)

target_link_libraries(ksp_qp_bind
  PRIVATE
    Eigen3::Eigen
)

# Output the .so next to this CMakeLists.txt so `import ksp_qp_bind` works
# from the python/ directory without any installation step.
set_target_properties(ksp_qp_bind PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
