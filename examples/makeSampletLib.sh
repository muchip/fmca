g++ -std=c++11 -O3 -fopenmp -fPIC -I../ sampletMatrixGenerator.cpp -c
g++ -std=c++11 -O3 -fopenmp sampletMatrixGenerator.o -shared -o libSamplets.so
