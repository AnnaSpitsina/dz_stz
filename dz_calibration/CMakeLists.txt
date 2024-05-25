#cmake_minimum_required(VERSION 3.0)
#project(dz_calibration CXX)

## Использование стандарта c++11
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED YES)

## Поиск пакета Eigen
#find_package(Eigen3 REQUIRED)
#add_executable(${PROJECT_NAME} "main.cpp")

cmake_minimum_required(VERSION 3.0)

project(dz_calibration)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Core)

add_executable(${PROJECT_NAME} "main.cpp")
find_package(Eigen3 REQUIRED)

target_link_libraries(${PROJECT_NAME} Qt5::Core)
