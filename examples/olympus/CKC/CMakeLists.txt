
project(robotoc_examplel_olympus_ckc CXX)
cmake_minimum_required(VERSION 3.1)
#set(CTEST_MEMORYCHECK_COMMAND "valgrind")


# Enable testing
#enable_testing()

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Find required packages
find_package(pinocchio REQUIRED)
find_package(robotoc REQUIRED)
#include (CTest)


# Define a macro to add examples
macro(add_example EXAMPLE)
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} PRIVATE pinocchio::pinocchio robotoc::robotoc)
  # Add test
  #add_test(NAME ${EXAMPLE}_valgrind COMMAND ${EXAMPLE})
  #set_tests_properties(${EXAMPLE}_valgrind PROPERTIES
endmacro()

# Add examples
add_example(test_rna_derivatives)