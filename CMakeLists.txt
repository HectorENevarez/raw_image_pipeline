cmake_minimum_required(VERSION 3.3)

project(optical_flow)

set(CMAKE_C_FLAGS "-std=gnu99 -Wall -Wextra -Wuninitialized \
	-Wunused-variable -Wdouble-promotion -Wmissing-prototypes \
	-Wmissing-declarations -Werror=undef -Wno-unused-function ${CMAKE_C_FLAGS}")


file(GLOB OPENCV_LIBS "/usr/lib/x86_64-linux-gnu/libopencv*.so")

add_subdirectory (src)