cmake_minimum_required(VERSION 3.15)
project(UTA VERSION 0.1.0 LANGUAGES CXX ASM)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(UTA_BUILD_TESTS "Build tests" ON)
option(UTA_BUILD_EXAMPLES "Build examples" ON)
option(UTA_ENABLE_CUDA "Enable CUDA support" ON)
option(UTA_ENABLE_ROCM "Enable ROCm support" ON)
option(UTA_ENABLE_ONEAPI "Enable OneAPI support" ON)

# Find dependencies
find_package(Threads REQUIRED)
if(UTA_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
endif()

# Core library
add_library(uta_core
    src/core/device_manager.cpp
    src/core/memory_manager.cpp
    src/core/scheduler.cpp
    src/core/ptx/ptx_compiler.cpp
)

target_include_directories(uta_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# PTX compilation
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/tensor_ops.cubin
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx ${CMAKE_CURRENT_SOURCE_DIR}/src/core/ptx/tensor_ops.ptx -o ${CMAKE_BINARY_DIR}/tensor_ops.cubin
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/core/ptx/tensor_ops.ptx
)

# Python bindings
add_subdirectory(python)

# Tests
if(UTA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(UTA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS uta_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)
