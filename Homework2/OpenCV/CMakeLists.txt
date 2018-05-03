cmake_minimum_required(VERSION 2.8)
project( OpenCVExample )

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "-O3 -w -g ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "-O3 -w -g ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "-O3 -w -g ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( OpenCV REQUIRED )

add_executable( Pipeline opencv_batch.cpp )
target_link_libraries( Pipeline ${OpenCV_LIBS} )

add_executable( Filtering opencv_smooth.cpp )
target_link_libraries( Filtering ${OpenCV_LIBS} )

add_executable( Histogram opencv_hist.cpp )
target_link_libraries( Histogram ${OpenCV_LIBS} )
