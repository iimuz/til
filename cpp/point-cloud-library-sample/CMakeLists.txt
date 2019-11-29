# cmake build file for Point Cloud Library (PCL) Samples

cmake_minimum_required(VERSION 2.8)

project(PCLSamples C CXX)

# use vcpkg
set(VCPKG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/vcpkg")
set(VCPKG_INSTALLED "${VCPKG_ROOT}/installed/x64-windows")

# use PCL
set(PCL_DIR "${VCPKG_INSTALLED}/share/pcl")
find_package(pcl 1.8.1 REQUIRED common io)

# set build flags
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  add_definitions(-D_WIN32_WINNT=0x600)

  # Visual Studio では utf-8 の BOM なしを CP932 で扱うので、
  # BOM なしを強制的に utf-8 に認識させる
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /source-charset:utf-8")
endif()

# targets
add_subdirectory(cmd/correspondence_grouping)

