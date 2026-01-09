# Install script for directory: C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/k24095864/C++project/PD-PMM_SSN/out/install/preset_SSN_PMM")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/Users/k24095864/msys2/ucrt64/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/AdolcForward"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/AlignedVector3"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/ArpackSupport"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/AutoDiff"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/BVH"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/EulerAngles"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/FFT"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/IterativeSolvers"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/KroneckerProduct"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/LevenbergMarquardt"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/MatrixFunctions"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/MPRealSupport"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/NNLS"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/NonLinearOptimization"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/NumericalDiff"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/OpenGLSupport"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/Polynomials"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/SparseExtra"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/SpecialFunctions"
    "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-src/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/_deps/eigen-build/unsupported/Eigen/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
