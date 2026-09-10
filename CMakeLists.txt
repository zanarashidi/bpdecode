cmake_minimum_required(VERSION 3.24)

# Top-level build driven by scikit-build-core: compiles the csrc core and the
# torch custom-op library (loaded via torch.ops.load_library) into the wheel.
# The csrc tree also builds standalone -- see csrc/CMakeLists.txt.
project(bpdecode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'share' / 'cmake')"
  OUTPUT_VARIABLE _torch_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_dir}")
find_package(Torch REQUIRED)

set(BPDECODE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(csrc)
target_compile_options(bpdecode_core PUBLIC ${TORCH_CXX_FLAGS})

Python_add_library(_C MODULE WITH_SOABI bindings/torch_ops.cpp)
set_target_properties(_C PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_compile_options(_C PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(_C PRIVATE bpdecode_core "${TORCH_LIBRARIES}")
target_include_directories(_C PRIVATE "${TORCH_INCLUDE_DIRS}")

# Resolve libtorch at load time from its installed location; keep the build-tree
# rpath so CMake does no install-time surgery (which trips over torch on macOS).
set(_torch_lib "${TORCH_INSTALL_PREFIX}/lib")
set_target_properties(_C PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${_torch_lib}")
if(APPLE)
  target_link_options(_C PRIVATE "LINKER:-rpath,${_torch_lib}")
endif()

install(TARGETS _C DESTINATION bpdecode)
