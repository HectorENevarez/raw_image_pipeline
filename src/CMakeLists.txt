cmake_minimum_required(VERSION 3.3)

set(TARGET img_proc)

file(GLOB_RECURSE all_src_files "*.c*")
add_executable(${TARGET} ${all_src_files})

include_directories(
    ../include/
    /usr/include/opencv4
)

target_link_libraries(${TARGET}
	${OPENCV_LIBS}
)

target_include_directories(${TARGET} PUBLIC /usr/include/opencv4)