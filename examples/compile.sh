gcc -c -O3 pardiso_interface.c -fopenmp

g++ -I../ -O3 -c example_InversionFEM.cpp -fopenmp

g++ example_InversionFEM.o pardiso_interface.o ./libpardiso720-GNU840-X86-64.so \
-fopenmp -lgfortran  -lmkl_intel_lp64 -lmkl_lapack95_lp64 -lmkl_sequential \
-lmkl_core  -lgfortran   -lmkl_intel_lp64 -lmkl_lapack95_lp64 -lmkl_sequential \
-lmkl_core -lm -lmetis


