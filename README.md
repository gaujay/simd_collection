# SIMD-Collection


A collection of highly optimized, SIMD-accelerated (SSE, AVX, FMA, NEON) functions written in C.

Includes Google Benchmark and Google Test support (C++).

### Categories

- Dot product
	- using SSE, AVX, FMA and NEON intrinsics
	- for every data type combinaison: (u)int8, int16, int32, float, double
	- comparison with compiler auto-vectorized and naive implementations
	- optimization options: data alignement, vector size multiple, number of accumulators

- Sort 8-elements
	- based on optimal sorting networks
	- using SSE, AVX intrinsics
	- for every data type: int8, int16, int32, float, double
	- comparison with 'qsort' and 'std::sort' implementations (already ordered and random inputs)
	- optimization options: data alignement, early exit check
	
### Benchmark results

See corresponding 'bench' sub-folders for graphs.

### Dependencies

This project uses git submodule to include Google Benchmark and Google Test repositories:

    $ git clone https://github.com/gaujay/simd_collection.git
    $ cd simd_collection
    $ git submodule init && git submodule update
    $ git clone https://github.com/google/googletest lib/benchmark/googletest

### Building

Support GCC/MinGW and MSVC in release mode only (see 'CMakeLists.txt' and 'src/Utils/compiler_utils.h').

On Linux/Unix:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make <target> -j

On Windows:

You can use command line or QtCreator for simplicity:

    $ open 'CMakeLists.txt'
    $ in 'Projects', select each kit 'Build' tab
    $ under 'Build Steps->Targets', uncheck 'all' and check 'Current executable' (avoid building both x86/ARM targets)

### License

Apache License 2.0
