#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* PARDISO prototype. */
void pardisoinit(void *, int *, int *, int *, double *, int *);
void pardiso(void *, int *, int *, int *, int *, int *, double *, int *, int *,
             int *, int *, int *, int *, double *, double *, int *, double *);
void pardiso_chkmatrix(int *, int *, double *, int *, int *, int *);
void pardiso_chkvec(int *, int *, double *, int *);
void pardiso_printstats(int *, int *, double *, int *, int *, int *, double *,
                        int *);

int pardiso_interface(int *ia, int *ja, double *a, int n) {
  //////////////////////////////////////////////////////////////////////////////
  fflush(stdout);
  int i = 0;
  int j = 0;
  int mtype = -2; /* Real symmetric matrix */
  /* RHS and solution vectors. */
  double *b = NULL;
  double *x = NULL;
  int nrhs = 1; /* Number of right hand sides. */
  /* Internal solver memory pointer pt,                  */
  /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
  /* or void *pt[64] should be OK on both architectures  */
  void *pt[64];
  /* Pardiso control parameters. */
  int iparm[64];
  double dparm[64];
  int maxfct = 0;
  int mnum = 0;
  int phase = 0;
  int error = 0;
  int msglvl = 0;
  int solver = 0;
  /* Number of processors. */
  int num_procs = 0;
  /* Auxiliary variables. */
  char *var = NULL;
  int k = 0;
  double ddum = 0; /* Double dummy */
  int idum = 0;    /* Integer dummy. */
  b = (double *)calloc(1, n * sizeof(double));
  x = (double *)calloc(1, n * sizeof(double));
  assert(b && x && "Nullptr in pardiso");
  printf("ia=%p ja=%p a=%p n=%i nnz=%i\n", ia, ja, a, n, ia[n]);
  printf("b=%p x=%p\n", b, x);
  fflush(stdout);
  /* -------------------------------------------------------------------- */
  /* ..  Setup Pardiso control parameters.                                */
  /* -------------------------------------------------------------------- */

  error = 0;
  solver = 0; /* use sparse direct solver */
  pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);
  printf("pardisoinit done.\n");
  fflush(stdout);

  if (error != 0) {
    if (error == -10) printf("No license file found \n");
    if (error == -11) printf("License is expired \n");
    if (error == -12) printf("Wrong username or hostname \n");
    return 1;
  } else
    printf("[PARDISO]: License check was successful ... \n");

  /* Numbers of processors, value of OMP_NUM_THREADS */
  var = getenv("OMP_NUM_THREADS");
  if (var != NULL)
    sscanf(var, "%d", &num_procs);
  else {
    printf("Set environment OMP_NUM_THREADS to 1");
    exit(1);
  }
  iparm[2] = num_procs;
  // iparm[2] = 2;

  maxfct = 1; /* Maximum number of numerical factorizations.  */
  mnum = 1;   /* Which factorization to use. */

  msglvl = 1; /* Print statistical information  */
  error = 0;  /* Initialize error flag */

  /* -------------------------------------------------------------------- */
  /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
  /*     notation.                                                        */
  /* -------------------------------------------------------------------- */

  int nnz = ia[n];
  for (i = 0; i < n + 1; i++) {
    ia[i] += 1;
  }
  for (i = 0; i < nnz; i++) {
    ja[i] += 1;
  }
  printf("matrix converted from 0-based to 1-based.\n");
  fflush(stdout);
  /* Set right hand side to i. */
  for (i = 0; i < n; i++) {
    b[i] = i + 1;
  }
  printf("rhs set to i.\n");
  fflush(stdout);

  /* -------------------------------------------------------------------- */
  /*  .. pardiso_chk_matrix(...)                                          */
  /*     Checks the consistency of the given matrix.                      */
  /*     Use this functionality only for debugging purposes               */
  /* -------------------------------------------------------------------- */

  pardiso_chkmatrix(&mtype, &n, a, ia, ja, &error);
  printf("pardiso_chkmatrix done.\n");
  fflush(stdout);
  if (error != 0) {
    printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
  }

  /* -------------------------------------------------------------------- */
  /* ..  pardiso_chkvec(...)                                              */
  /*     Checks the given vectors for infinite and NaN values             */
  /*     Input parameters (see PARDISO user manual for a description):    */
  /*     Use this functionality only for debugging purposes               */
  /* -------------------------------------------------------------------- */

  pardiso_chkvec(&n, &nrhs, b, &error);
  printf("pardiso_chkvec done.\n");
  fflush(stdout);
  if (error != 0) {
    printf("\nERROR  in right hand side: %d", error);
    exit(1);
  }

  /* -------------------------------------------------------------------- */
  /* .. pardiso_printstats(...)                                           */
  /*    prints information on the matrix to STDOUT.                       */
  /*    Use this functionality only for debugging purposes                */
  /* -------------------------------------------------------------------- */

  pardiso_printstats(&mtype, &n, a, ia, ja, &nrhs, b, &error);
  printf("pardiso_printstats done.\n");
  fflush(stdout);
  if (error != 0) {
    printf("\nERROR right hand side: %d", error);
    exit(1);
  }

  /* -------------------------------------------------------------------- */
  /* ..  Reordering and Symbolic Factorization.  This step also allocates */
  /*     all memory that is necessary for the factorization.              */
  /* -------------------------------------------------------------------- */
  phase = 11;
  printf("***phase=11 pardiso\n");
  iparm[1] = 3; /* compute determinant */
  fflush(stdout);
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error, dparm);
  printf(" done.\n");
  fflush(stdout);
  if (error != 0) {
    printf("\nERROR during symbolic factorization: %d", error);
    exit(1);
  }
  printf("\nReordering completed ... ");
  printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
  printf("\nNumber of factorization GFLOPS = %d", iparm[18]);
  fflush(stdout);
  /* -------------------------------------------------------------------- */
  /* ..  Numerical factorization.                                         */
  /* -------------------------------------------------------------------- */
  phase = 22;
  iparm[32] = 1; /* compute determinant */
  printf("***phase=22 pardiso\n");
  fflush(stdout);
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error, dparm);
  printf(" done.\n");
  fflush(stdout);
  if (error != 0) {
    printf("\nERROR during numerical factorization: %d", error);
    exit(2);
  }
  printf("\nFactorization completed ...\n ");
  fflush(stdout);
#if 0
  /* -------------------------------------------------------------------- */
  /* ..  Back substitution and iterative refinement.                      */
  /* -------------------------------------------------------------------- */
  phase = 33;

  iparm[7] = 1; /* Max numbers of iterative refinement steps. */

  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
          iparm, &msglvl, b, x, &error, dparm);

  if (error != 0) {
    printf("\nERROR during solution: %d", error);
    exit(3);
  }

    printf("\nSolve completed ... ");
    printf("\nThe solution of the system is: ");
    for (i = 0; i < n; i++) {
        printf("\n x [%d] = % f", i, x[i] );
    }
    printf ("\n\n");
#endif

  /* -------------------------------------------------------------------- */
  /* ... Inverse factorization.                                           */
  /* -------------------------------------------------------------------- */

  /*FILE *mat_file = NULL;
  mat_file = fopen("inverse.iajaa", "w");
*/
  if (solver == 0) {
    printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    fflush(stdout);
    phase = -22;
    iparm[35] = 1; /*  no not overwrite internal factor L */
    printf("***phase=-22 pardiso\n");
    fflush(stdout);
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
            iparm, &msglvl, b, x, &error, dparm);
    printf(" done.\n");
    fflush(stdout);
#if 0
    /* print diagonal elements */
    for (k = 0; k < n; k++) {
      int j = ia[k] - 1;
      printf("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j] - 1,
             a[j]);
    }
#endif
  }
#if 0
  for (i = 0; i < n; ++i)
    for (j = ia[i]; j <= ia[i + 1] - 1; ++j)
      fprintf(mat_file, "%d %d %32.24e\n", i + 1, ja[j - 1], a[j - 1]);
  // printf(" i %d ja %d a %lf\n", i+1, ja[j-1], a[j-1]);
#endif

  /* -------------------------------------------------------------------- */
  /* ..  Convert matrix back to 0-based C-notation.                       */
  /* -------------------------------------------------------------------- */
  printf("matrix converted from 1-based to 0-based.\n");
  fflush(stdout);
  for (i = 0; i < n + 1; i++) {
    ia[i] -= 1;
  }
  for (i = 0; i < nnz; i++) {
    ja[i] -= 1;
  }

  /* -------------------------------------------------------------------- */
  /* ..  Termination and release of memory.                               */
  /* -------------------------------------------------------------------- */
  phase = -1; /* Release internal memory. */
  printf("phase=-1 pardiso");
  fflush(stdout);
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, ia, ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error, dparm);
  printf(" done.\n");
  fflush(stdout);
  return 0;
}
