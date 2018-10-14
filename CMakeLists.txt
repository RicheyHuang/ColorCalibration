cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(OpenCV_DIR "C://Program Files//OpenCV-3.4.1//build")

project(ColorCalibration)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (ColorCalibration src/main.cpp)
target_link_libraries (ColorCalibration ${PCL_LIBRARIES} ${OpenCV_LIBS})