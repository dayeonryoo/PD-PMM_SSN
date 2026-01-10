# CMake generated Testfile for 
# Source directory: C:/Users/k24095864/C++project/PD-PMM_SSN
# Build directory: C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(SSNTests "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/test_ssn.exe")
set_tests_properties(SSNTests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/k24095864/C++project/PD-PMM_SSN/CMakeLists.txt;88;add_test;C:/Users/k24095864/C++project/PD-PMM_SSN/CMakeLists.txt;0;")
add_test(PMMTests "C:/Users/k24095864/C++project/PD-PMM_SSN/out/build/preset_SSN_PMM/test_pmm.exe")
set_tests_properties(PMMTests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/k24095864/C++project/PD-PMM_SSN/CMakeLists.txt;95;add_test;C:/Users/k24095864/C++project/PD-PMM_SSN/CMakeLists.txt;0;")
subdirs("_deps/eigen-build")
subdirs("_deps/googletest-build")
subdirs("_deps/highs-build")
