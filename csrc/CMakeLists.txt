cmake_minimum_required(VERSION 3.24)

# CUDA is optional so the tree configures and builds on machines without a
# toolkit (CI, laptops).  When nvcc is present the device kernels are compiled
# in and BPDECODE_WITH_CUDA is defined.
project(bpdecode_core LANGUAGES CXX)

option(BPDECODE_BUILD_TESTS "Build C++ unit tests" ON)
option(BPDECODE_BUILD_BENCH "Build nvbench microbenchmarks (needs CUDA)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 89 90)
    endif()
    message(STATUS "bpdecode: CUDA enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "bpdecode: no CUDA toolkit found - building host code only")
endif()

set(_bpdecode_sources src/mask_cpu.cpp)
if(CMAKE_CUDA_COMPILER)
    list(APPEND _bpdecode_sources src/mask_cuda.cu)
endif()

add_library(bpdecode_core STATIC ${_bpdecode_sources})
target_include_directories(bpdecode_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(bpdecode_core PUBLIC BPDECODE_WITH_CUDA)
    set_target_properties(bpdecode_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BPDECODE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(bpdecode_tests tests/test_mask.cpp)
    target_link_libraries(bpdecode_tests PRIVATE bpdecode_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(bpdecode_tests)
endif()
