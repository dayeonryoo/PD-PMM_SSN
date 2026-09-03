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
# METIS (nested-dissection ordering candidate for ordering_select.hpp) + its dependency GKlib.
# Mirrors the top-level CMakeLists.txt setup so the Python extension gets the same ordering
# candidates as the native build. See that file for rationale on the manual xinclude step.
# ----------------------
option(KSP_QP_BUILD_METIS "Fetch and build METIS from GitHub for a 3rd (nested-dissection) ordering candidate" ON)
if (KSP_QP_BUILD_METIS)
  FetchContent_Declare(
    gklib
    URL https://github.com/KarypisLab/GKlib/archive/3b7d61b9f885063c89901f3901fb4426f9cfb58f.zip
  )
  FetchContent_MakeAvailable(gklib)

  set(GKLIB_PATH ${gklib_SOURCE_DIR} CACHE PATH "GKlib root for METIS's build" FORCE)

  FetchContent_Declare(
    metis
    URL https://github.com/KarypisLab/METIS/archive/refs/tags/v5.2.1.zip
  )
  FetchContent_GetProperties(metis)
  if (NOT metis_POPULATED)
    FetchContent_Populate(metis)
    file(MAKE_DIRECTORY ${metis_SOURCE_DIR}/build/xinclude)
    file(READ ${metis_SOURCE_DIR}/include/metis.h _metis_header)
    file(WRITE ${metis_SOURCE_DIR}/build/xinclude/metis.h
      "#define IDXTYPEWIDTH 32\n#define REALTYPEWIDTH 32\n${_metis_header}")
    file(COPY ${metis_SOURCE_DIR}/include/CMakeLists.txt DESTINATION ${metis_SOURCE_DIR}/build/xinclude)
    add_subdirectory(${metis_SOURCE_DIR} ${metis_BINARY_DIR})
  endif()
endif()

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

if (KSP_QP_BUILD_METIS)
  target_include_directories(ksp_qp_bind PRIVATE ${metis_SOURCE_DIR}/build/xinclude)
  target_link_libraries(ksp_qp_bind PRIVATE metis GKlib::GKlib)
  target_compile_definitions(ksp_qp_bind PRIVATE KSP_QP_HAVE_METIS)
endif()

# Output the .so next to this CMakeLists.txt so `import ksp_qp_bind` works
# from the python/ directory without any installation step.
set_target_properties(ksp_qp_bind PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
