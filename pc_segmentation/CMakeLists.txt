cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pc_segmentation)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")

SET(CMAKE_BUILD_TYPE release)#relwithdebinfo)
#SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
#SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")

set(OPENNI_INCLUDE_DIRS /usr/include/ni)
set(ROS_DISTRO kinetic)
set(ROS_LIBRARIES /opt/ros/${ROS_DISTRO}/lib/libroscpp.so
	/opt/ros/${ROS_DISTRO}/lib/libroscpp_serialization.so
	/opt/ros/${ROS_DISTRO}/lib/librosconsole.so
	/opt/ros/${ROS_DISTRO}/lib/libroslib.so
  /opt/ros/${ROS_DISTRO}/lib/librostime.so
  /opt/ros/${ROS_DISTRO}/lib/libtf2.so
  /opt/ros/${ROS_DISTRO}/lib/libtf2_ros.so)

set(QHULL_LIBRARY /usr/lib/x86_64-linux-gnu/libqhull.so)


#set(BOOST_INCLUDEDIR "/usr/lib/x86_64-linux-gnu")
#IF (Boost_FOUND)
#    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
#    ADD_DEFINITIONS( "-DHAS_BOOST" )
#ENDIF()
set(BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu/")
#include("/home/eng230/software/pcl/cmake/pcl_find_boost.cmake")
FIND_PACKAGE(Boost)
find_package(PCL 1.7 REQUIRED COMPONENTS)
find_package(freenect2 REQUIRED)
find_package(message_generation)
find_package(std_msgs)
find_package(geometry_msgs)
find_package(sensor_msgs)

set(BOOST_THREAD_LIBRARY /usr/lib/x86_64-linux-gnu/libboost_thread.so)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${FREENECT2_INCLUDE_DIRS})

add_message_files(
		FILES
		PcFeatures.msg
		PcBoundingBox.msg
		PcObjects.msg
		PcSegmented.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
   geometry_msgs
   sensor_msgs
)

link_directories(${FREENECT2_PATH}/lib)

find_package(OpenCV REQUIRED)

include_directories(${freenect2_INCLUDE_DIRS})
include_directories(include)
include_directories(/opt/ros/${ROS_DISTRO}/include)

#SET(CMAKE_BUILD_TYPE Release)#relwithdebinfo)
#SET(CMAKE_CXX_FLAGS -march=corei7-avx)

#add_library(pcseg SHARED src/pc_segmentation.cpp src/rotcalipers.cpp src/utils.cpp src/region_growing_custom_color.cpp src/utils_pcl_ros.cpp)
add_library(pcseg SHARED src/pc_segmentation.cpp src/rotcalipers.cpp src/utils.cpp src/utils_pcl_ros.cpp)
add_dependencies(pcseg ${catkin_EXPORTED_TARGETS})
target_link_libraries(pcseg ${PCL_LIBRARIES} ${ROS_LIBRARIES})

add_executable(offlineProcV2 src/offlineProcV2.cpp)
target_link_libraries(offlineProcV2 ${PCL_LIBRARIES} pcseg)

add_executable(def_loop src/def_loop.cpp)
target_link_libraries(def_loop ${PCL_LIBRARIES} pcseg ${freenect2_LIBRARY} ${ROS_LIBRARIES} ${OpenCV_LIBS})
