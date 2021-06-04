
cmake_minimum_required(VERSION 3.10)

project(numerial_optimization
        LANGUAGES CXX C
        VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)

find_package(Msnhnet REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(UnconstrainedMethod/Steepest_Descent)
add_subdirectory(UnconstrainedMethod/Newton)
add_subdirectory(UnconstrainedMethod/DampedNewton)
add_subdirectory(UnconstrainedMethod/NewtonLM)
add_subdirectory(UnconstrainedMethod/QuasiNewton)
