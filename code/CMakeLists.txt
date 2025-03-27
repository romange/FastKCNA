cmake_minimum_required (VERSION 2.6)
project (KGRAPH)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++11")


find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

MESSAGE(STATUS ${CMAKE_CXX_FLAGS})

include_directories(${KGRAPH_SOURCE_DIR}/include)
include_directories(${KGRAPH_SOURCE_DIR}/src)
set(LIB_SRC src/kgraph.cpp src/metric.cpp)
add_library(kgraph SHARED ${LIB_SRC})
add_library(kgrapha STATIC ${LIB_SRC})
set_target_properties(kgrapha PROPERTIES OUTPUT_NAME kgraph)

add_executable(build_index build_index.cpp)

TARGET_LINK_LIBRARIES(build_index kgrapha rt ${Boost_LIBRARIES})

install (FILES kgraph.h kgraph-data.h DESTINATION include) 
install (TARGETS kgraph kgrapha
	 LIBRARY DESTINATION lib
	 ARCHIVE DESTINATION lib)