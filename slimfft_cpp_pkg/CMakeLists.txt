cmake_minimum_required(VERSION 3.18)
project(slimfft_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

pybind11_add_module(_slimfft
    src/slimfft_module.cpp
    cpp/slimfft_profile.cpp
)

target_compile_definitions(_slimfft PRIVATE SLIMFFT_BUILD_PYTHON)
target_link_libraries(_slimfft PRIVATE PkgConfig::FFTW3)

# THIS IS THE IMPORTANT PART:
install(TARGETS _slimfft DESTINATION slimfft_cpp)