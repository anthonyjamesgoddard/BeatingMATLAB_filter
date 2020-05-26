project(furry)

if (MSVC)
    # Solves complications that arrise in the runtime when compiling
    # gtest using msvc.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif (MSVC)

add_executable(benchmark benchmark.cpp)
target_compile_options(benchmark PRIVATE -Wall -Wextra -mavx2)
