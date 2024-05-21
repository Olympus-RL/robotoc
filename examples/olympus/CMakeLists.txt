project(robotoc_examplel_olympus_ckc CXX)
cmake_minimum_required(VERSION 3.1)
set(CTEST_MEMORYCHECK_COMMAND "valgrind")


# Enable testing
enable_testing()

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Find required packages
find_package(robotoc REQUIRED)
find_package(towr REQUIRED)
find_package(ifopt 2.0.1 REQUIRED)
include (CTest)


# Define a macro to add examples
macro(add_example EXAMPLE)
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} PRIVATE robotoc::robotoc towr::towr ifopt::ifopt_ipopt ifopt::ifopt_snopt) 
  # Add test
  add_test(NAME ${EXAMPLE}_valgrind COMMAND ${EXAMPLE})
  # Set test properties including Valgrind memory checking
  set_tests_properties(${EXAMPLE}_valgrind PROPERTIES
    LABELS "valgrind"
    TIMEOUT 1000
    # Specify Valgrind options for memory checking
    ENVIRONMENT "CTEST_MEMORYCHECK_COMMAND_OPTIONS=--leak-check=full --error-exitcode=1"
)

endmacro()

# Add examples
add_example(refrence_tracking)
add_example(jump_sto_open_chain)