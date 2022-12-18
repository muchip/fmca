if(NOT OpenMP_FOUND)

FetchContent_Declare(OpenMP
    URL
       https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.5/openmp-15.0.5.src.tar.xz
)

set(OPENMP_STANDALONE_BUILD TRUE)
set(LIBOMP_INSTALL_ALIASES OFF)

FetchContent_MakeAvailable(OpenMP)
set(OpenMP_AVAILABLE TRUE)


target_link_directories(omp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/openmp-build/runtime/src>
        $<INSTALL_INTERFACE:/lib>
)

install(TARGETS omp
    LIBRARY
    DESTINATION
        "${LIBRARY_DIST}"
)
endif()