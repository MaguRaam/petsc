cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Tutorial VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add the executable
add_executable(wave wave.cpp)

include_directories(/home/ratnesh/blitz/include)
link_directories(/home/ratnesh/blitz/lib)

target_link_libraries(wave blitz)
