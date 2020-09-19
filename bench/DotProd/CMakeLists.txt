#
set(INCLUDE_FILES
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_i8.h
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_i8ui8.h
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_i16i8.h
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_i16.h
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_i32i16.h
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_i32.h
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_flt.h
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_dbl.h
    ${CMAKE_SOURCE_DIR}/src/DotProd/dotp_simd.h
    benchmark_dotp_i8.h
    benchmark_dotp_i8ui8.h
    benchmark_dotp_i16i8.h
    benchmark_dotp_i16.h
    benchmark_dotp_i32i16.h
    benchmark_dotp_i32.h
    benchmark_dotp_flt.h
    benchmark_dotp_dbl.h
)

set(SOURCE_FILES
    benchmark_main.cpp
)

add_executable(DotProd_benchmark
    ${INCLUDE_FILES}
    ${SOURCE_FILES}
)

target_include_directories(DotProd_benchmark
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

#
target_link_libraries(DotProd_benchmark
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
