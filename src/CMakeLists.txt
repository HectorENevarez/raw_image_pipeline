cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(TARGET img_proc)

# Add cuda if we decide to build with it
if (ENABLE_CUDA)
    find_package(CUDA 9.0 REQUIRED)
    enable_language(CUDA)
    add_definitions(-DPROC_BACKEND=cuda)
else()
    add_definitions(-DPROC_BACKEND=cpu)
endif()

file(GLOB_RECURSE all_src_files "*.c*")
add_executable(${TARGET} ${all_src_files})

include_directories(
    ../include/
    /usr/include/opencv4
)

target_link_libraries(${TARGET}
	${OPENCV_LIBS}
    ${CUDA_LIBRARIES}
)

target_include_directories(${TARGET} PUBLIC /usr/include/opencv4)