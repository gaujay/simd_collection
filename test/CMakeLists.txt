#
include(GoogleTest)

#
set(SOURCE_FILES
    test_dotprod_main.cpp
)
set(SOURCE_FILES_NEON
    test_dotprod_neon_main.cpp
)
set(SOURCE_FILES_NSORT
    test_netsort_main.cpp
)

set(gtest_force_shared_crt
    ON CACHE BOOL "" FORCE
)

#
add_executable(DotProd_tests
    ${SOURCE_FILES}
)
add_executable(DotProd_neon_tests
    ${SOURCE_FILES_NEON}
)
add_executable(NetSort_tests
    ${SOURCE_FILES_NSORT}
)

target_include_directories(DotProd_tests 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/lib/benchmark/googletest/googletest/include
)
target_include_directories(DotProd_neon_tests 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/lib/benchmark/googletest/googletest/include
)
target_include_directories(NetSort_tests 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/lib/benchmark/googletest/googletest/include
)

target_link_libraries(DotProd_tests 
    PUBLIC 
        gtest
        gtest_main
)
target_link_libraries(DotProd_neon_tests 
    PUBLIC 
        gtest
        gtest_main
)
target_link_libraries(NetSort_tests 
    PUBLIC 
        gtest
        gtest_main
)

#
gtest_discover_tests(DotProd_tests)
gtest_discover_tests(DotProd_neon_tests)
gtest_discover_tests(NetSort_tests)
