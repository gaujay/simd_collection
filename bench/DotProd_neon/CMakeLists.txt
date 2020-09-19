#
set(INCLUDE_FILES
    ${CMAKE_SOURCE_DIR}/src/DotProd_neon/dotp_i8_neon.h
    ${CMAKE_SOURCE_DIR}/src/DotProd_neon/dotp_i16i8_neon.h
    ${CMAKE_SOURCE_DIR}/src/DotProd_neon/dotp_i16_neon.h
    ${CMAKE_SOURCE_DIR}/src/DotProd_neon/dotp_i32i16_neon.h
    ${CMAKE_SOURCE_DIR}/src/DotProd_neon/dotp_i32_neon.h
    ${CMAKE_SOURCE_DIR}/src/DotProd_neon/dotp_flt_neon.h
    ${CMAKE_SOURCE_DIR}/src/DotProd_neon/dotp_simd_neon.h
    benchmark_dotp_i8_neon.h
    benchmark_dotp_i16i8_neon.h
    benchmark_dotp_i16_neon.h
    benchmark_dotp_i32i16_neon.h
    benchmark_dotp_i32_neon.h
    benchmark_dotp_flt_neon.h
)

set(SOURCE_FILES
    benchmark_neon_main.cpp
)

add_executable(DotProd_neon_benchmark
    ${INCLUDE_FILES}
    ${SOURCE_FILES}
)

target_include_directories(DotProd_neon_benchmark
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

#
target_link_libraries(DotProd_neon_benchmark
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
