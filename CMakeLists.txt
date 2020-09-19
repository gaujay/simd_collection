cmake_minimum_required (VERSION 3.10)

project (simd_collection LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ot -DNDEBUG /arch:AVX2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG -march=native -mtune=native")
endif()

set(BENCHMARK_ENABLE_TESTING 
    OFF 
    CACHE BOOL "Disable benchmark testing" FORCE
)

#
add_subdirectory(bench)
add_subdirectory(test)

add_subdirectory(lib/benchmark)
add_subdirectory(lib/benchmark/googletest)
