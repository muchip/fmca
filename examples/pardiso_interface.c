#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int pardiso_interface(int *ia, int *ja, double *a, int m, int n) {
  //////////////////////////////////////////////////////////////////////////////
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
  int maxfct, mnum, phase, error, msglvl, solver;
  /* Number of processors. */
  int num_procs;
  /* Auxiliary variables. */
  char *var;
  int k;
  double ddum; /* Double dummy */
  int idum;    /* Integer dummy. */
  b = (double *)calloc(1, m * sizeof(double));
  x = (double *)calloc(1, n * sizeof(double));
  /* -------------------------------------------------------------------- */
  /* ..  Setup Pardiso control parameters.                                */
  /* -------------------------------------------------------------------- */
  error = 0;
  solver = 0; /* use sparse direct solver */
  pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);

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

  maxfct = 1; /* Maximum number of numerical factorizations.  */
  mnum = 1;   /* Which factorization to use. */

  msglvl = 1; /* Print statistical information  */
  error = 0;  /* Initialize error flag */

  /* -------------------------------------------------------------------- */
  /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
  /*     notation.                                                        */
  /* -------------------------------------------------------------------- */
  for (i = 0; i < n + 1; i++) {
    ia[i] += 1;
  }
  int nnz = ia[n];
  for (i = 0; i < nnz; i++) {
    ja[i] += 1;
  }

  /* Set right hand side to i. */
  for (i = 0; i < n; i++) {
    b[i] = i + 1;
  }

  /* -------------------------------------------------------------------- */
  /*  .. pardiso_chk_matrix(...)                                          */
  /*     Checks the consistency of the given matrix.                      */
  /*     Use this functionality only for debugging purposes               */
  /* -------------------------------------------------------------------- */

  pardiso_chkmatrix(&mtype, &n, a, ia, ja, &error);
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
  if (error != 0) {
    printf("\nERROR right hand side: %d", error);
    exit(1);
  }

  /* -------------------------------------------------------------------- */
  /* ..  Reordering and Symbolic Factorization.  This step also allocates */
  /*     all memory that is necessary for the factorization.              */
  /* -------------------------------------------------------------------- */
  phase = 11;

  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error, dparm);

  if (error != 0) {
    printf("\nERROR during symbolic factorization: %d", error);
    exit(1);
  }
  printf("\nReordering completed ... ");
  printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
  printf("\nNumber of factorization GFLOPS = %d", iparm[18]);

  /* -------------------------------------------------------------------- */
  /* ..  Numerical factorization.                                         */
  /* -------------------------------------------------------------------- */
  phase = 22;
  iparm[32] = 1; /* compute determinant */

  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error, dparm);

  if (error != 0) {
    printf("\nERROR during numerical factorization: %d", error);
    exit(2);
  }
  printf("\nFactorization completed ...\n ");

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

#if 0
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

  FILE *mat_file = NULL;
  mat_file = fopen("inverse.iajaa", "w");

  if (solver == 0) {
    printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    phase = -22;
    iparm[35] = 1; /*  no not overwrite internal factor L */

    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
            iparm, &msglvl, b, x, &error, dparm);
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

  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, ia, ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error, dparm);

  return 0;
}
