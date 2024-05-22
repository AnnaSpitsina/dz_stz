cmake_minimum_required(VERSION 3.0)
project(dz_SLAU CXX)

# Использование стандарта c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Поиск пакета Eigen
find_package(Eigen3 REQUIRED)
add_executable(${PROJECT_NAME} "main.cpp")


