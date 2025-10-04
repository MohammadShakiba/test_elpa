cmake_minimum_required(VERSION 3.20)
project(test_diag)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_C_COMPILER mpicc)
#set(CMAKE_CXX_COMPILER mpicxx)

find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy)
find_package(MPI REQUIRED)
if (MPI_FOUND)
message("FOUND MPI")
message(STATUS "MPI libraries: ${MPI_C_LIBRARIES}")
message(STATUS "MPI libraries: ${MPI_CXX_LIBRARIES}")
else()
message("NOT FOUND MPI")
endif()
#find_package(MKL REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
#find_package(ScaLAPACK required)

#include_directories(${ScaLAPACK_INCLUDE_PATH}) 
include_directories(${MPI_INCLUDE_PATH}) 

set(MKL_ROOT $ENV{MKLROOT})
set(MKL_INCLUDE_DIR ${MKL_ROOT}/include)
set(MKL_LIB_DIR ${MKL_ROOT}/lib/intel64)
include_directories(${MKL_INCLUDE_DIR})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(/projects/academic/alexeyak/mohammad/software/elpa_install/include/elpa-2024.03.001)
#include_directories(/projects/academic/cyberwksp21/SOFTWARE_NEW_ENV/cp2k/cp2k_v24/tools/toolchain/install/elpa-2023.05.001/cpu/include/elpa_openmp-2023.05.001)
message(STATUS "Python3 include directories: ${Python3_INCLUDE_DIRS}") 
message(STATUS "Python3 lib directories: ${Python3_LIBRARIES}") 
message(STATUS "MPI lib directories: ${MPI_CXX_LIBRARIES}") 
message(STATUS "MPI include directories: ${MPI_CXX_INCLUDE_PATH}") 
message(STATUS "ScaLapack_INCLUDE_DIRS : ${ScaLapack_INCLUDE_DIRS}") 
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}") 
message(STATUS "Python3 numpy directories: ${Python3_NumPy_INCLUDE_DIRS}") 
message(STATUS "Path to cnpy: ${CMAKE_CURRENT_SOURCE_DIR}/cnpy") 

#find_package(ZLIB REQUIRED)
add_subdirectory(cnpy)

# Add the cnpy library
#add_library(cnpy cnpy/cnpy.cpp)

#target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)

add_executable(diagonalize src/read_numpy.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cnpy)

#target_include_directories(diagonalize PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cnpy ${MPI_INCLUDE_DIRS} ${ScaLapack_INCLUDE_DIRS})
#find_library(ELPA_LIBRARY NAMES elpa_openmp HINTS /projects/academic/cyberwksp21/SOFTWARE_NEW_ENV/cp2k/cp2k_v24/tools/toolchain/install/elpa-2023.05.001/cpu/lib)
find_library(ELPA_LIBRARY NAMES elpa HINTS /projects/academic/alexeyak/mohammad/software/elpa_install/lib)
#set(ELPA_LIB_DIR "/projects/academic/alexeyak/mohammad/software/elpa_install/lib")
#link_directories(/projects/academic/alexeyak/mohammad/software/elpa_install/lib)
message(STATUS "ELPA LIB: ${ELPA_LIBRARY}")
target_link_libraries(diagonalize PRIVATE ${Python3_LIBRARIES} cnpy ${MPI_LIBRARIES} -lm -ldl  -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_lp64 -lpthread -lstdc++ ${ELPA_LIBRARY} )
#                                    ${MKL_LIB_DIR}/libmkl_scalapack_lp64.a 
#                                    ${MKL_LIB_DIR}/libmkl_intel_lp64.a 
#                                    ${MKL_LIB_DIR}/libmkl_intel_thread.a 
#                                    ${MKL_LIB_DIR}/libmkl_core.a 
#                                    ${MKL_LIB_DIR}/libmkl_sequential.a
#                                    ${MKL_LIB_DIR}/libmkl_scalapack_lp64.a
#                                    ${MKL_LIB_DIR}/libmkl_blacs_openmpi_lp64.a
#                                    ${MKL_LIB_DIR}/libmkl_intel_thread.a
#                                    ${MKL_LIB_DIR}/libmkl_blacs_intelmpi_lp64.a 
#                                    -lpthread -lm -ldl -lmkl_core -lmkl_intel_thread -lmkl_blacs_openmpi_lp64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread)
# BLAS::BLAS LAPACK::LAPACK ScaLAPACK::ScaLAPACK)


