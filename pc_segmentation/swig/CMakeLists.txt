cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

#SET(CMAKE_BUILD_TYPE debug)
SET(CMAKE_BUILD_TYPE release)#relwithdebinfo)
#SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
#SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")

project(swig_for_pc)

set(OPENNI_INCLUDE_DIRS /usr/include/ni)

find_package(PCL 1.7 REQUIRED COMPONENTS)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${PROJECT_SOURCE_DIR}../include)
include_directories(../include)
include_directories(/usr/include/python2.7)

add_library(save_pc SHARED ../src/pcSavePython.cpp pcSavePython_wrap.cxx)

target_link_libraries(save_pc
  /home/baris/codes/pc-segmentation/build/libpcseg.so
  ${PCL_LIBRARIES}
	/opt/ros/groovy/lib/libroscpp.so
	/opt/ros/groovy/lib/libroscpp_serialization.so
	/opt/ros/groovy/lib/librosconsole.so
	/opt/ros/groovy/lib/libroslib.so )
