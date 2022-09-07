// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_TICTOC__
#define FMCA_UTIL_TICTOC__

#include <iostream>
#include <string>
#include <iostream>
#include <string>
#ifdef _WIN32
#include <sys/timeb.h>
#include <sys/types.h>
#include <winsock2.h>
int gettimeofday(struct timeval* t, void* timezone)
{
    struct _timeb timebuffer;
    _ftime(&timebuffer);
    t->tv_sec = timebuffer.time;
    t->tv_usec = 1000 * timebuffer.millitm;
    return 0;
}

/* Structure describing CPU time used by a process and its children.  */
struct tms
{
    clock_t tms_utime;          /* User CPU time.  */
    clock_t tms_stime;          /* System CPU time.  */

    clock_t tms_cutime;         /* User CPU time of dead children.  */
    clock_t tms_cstime;         /* System CPU time of dead children.  */
  };

/* Store the CPU time used by this process and all its
   dead children (and their dead children) in BUFFER.
   Return the elapsed real time, or (clock_t) -1 for errors.
   All times are in CLK_TCKths of a second.  */
clock_t times(struct tms* __buffer) {

    __buffer->tms_utime = clock();
    __buffer->tms_stime = 0;
    __buffer->tms_cstime = 0;
    __buffer->tms_cutime = 0;
    return __buffer->tms_utime;
}
typedef long long suseconds_t;

#else
#include <sys/time.h>
#endif 

#include "Macros.h"

namespace FMCA {
class Tictoc {
public:
  void tic(void) { gettimeofday(&start, NULL); }
  Scalar toc(void) {
    gettimeofday(&stop, NULL);
    Scalar dtime =
        stop.tv_sec - start.tv_sec + 1e-6 * (stop.tv_usec - start.tv_usec);
    return dtime;
  }
  Scalar toc(const std::string &message) {
    gettimeofday(&stop, NULL);
    Scalar dtime =
        stop.tv_sec - start.tv_sec + 1e-6 * (stop.tv_usec - start.tv_usec);
    std::cout << message << " " << dtime << "sec.\n";
    return dtime;
  }

private:
  struct timeval start; /* variables for timing */
  struct timeval stop;
};
} // namespace FMCA
#endif
