echo Loading modules
module load gcc
module load eigen
module load intel-mkl

MainFile=benchmark_Inversion
echo Compiling 
gcc -c -O3 pardiso_interface.c -fopenmp

g++ -I../ -O3 -c $MainFile.cpp -fopenmp

g++ $MainFile.o pardiso_interface.o ./libpardiso720-GNU840-X86-64.so \
-fopenmp -lgfortran  -lmkl_intel_lp64 -lmkl_lapack95_lp64 -lmkl_sequential \
-lmkl_core  -lgfortran   -lmkl_intel_lp64 -lmkl_lapack95_lp64 -lmkl_sequential \
-lmkl_core -lm ./libSamplet.so -oa.out
echo done
