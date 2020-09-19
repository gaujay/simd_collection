#
set(INCLUDE_FILES
    ${CMAKE_SOURCE_DIR}/src/NetSort/nsort_8_i8.h
    ${CMAKE_SOURCE_DIR}/src/NetSort/nsort_8_i16.h
    ${CMAKE_SOURCE_DIR}/src/NetSort/nsort_8_i32.h
    ${CMAKE_SOURCE_DIR}/src/NetSort/nsort_8_flt.h
    ${CMAKE_SOURCE_DIR}/src/NetSort/nsort_8_dbl.h
    benchmark_nsort_8_i8.h
    benchmark_nsort_8_i16.h
    benchmark_nsort_8_i32.h
    benchmark_nsort_8_flt.h
    benchmark_nsort_8_dbl.h
)

set(SOURCE_FILES
    benchmark_main.cpp
)

add_executable(NetSort_benchmark
    ${INCLUDE_FILES}
    ${SOURCE_FILES}
)

target_include_directories(NetSort_benchmark
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

#
target_link_libraries(NetSort_benchmark
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
