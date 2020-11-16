
cmake_minimum_required(VERSION 3.7)
project(Petsc)

message("-- Loading PETSc...")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

set (PETSC_DIR "$ENV{PETSC_DIR}")
set (PETSC_ARCH "$ENV{PETSC_ARCH}")

message("-- PETSC_DIR: ${PETSC_DIR}")
message("-- PETSc_ARCH: ${PETSC_ARCH}")

find_package (PETSc COMPONENTS CXX)
message("-- PETSc Loaded!")
include_directories(${PETSC_INCLUDES})

add_executable(poisson poisson.cc)
target_link_libraries(poisson ${PETSC_LIBRARIES})


