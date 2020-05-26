project(furry)

if (MSVC)
    # Solves complications that arrise in the runtime when compiling
    # gtest using msvc.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif (MSVC)

add_executable(benchmark benchmark.cpp)
target_compile_options(benchmark PRIVATE -mavx2)

if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "GCC detected, adding compile flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra")
endif(CMAKE_COMPILER_IS_GNUCXX)