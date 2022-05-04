#ifndef PARDISO_INTERFACE_
#define PARDISO_INTERFACE_

#include <cstdio>
#include <cstdlib>

extern "C" {
int pardiso_interface(int *ia, int *ja, double *a, int n);
}
#endif
