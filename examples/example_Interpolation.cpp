// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/Interpolators>
#include <FMCA/src/util/HaltonSet.h>
#include <FMCA/src/Interpolators/interpolationPointsGenerator.h>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Tictoc.h>

using Interpolator = FMCA::TotalDegreeInterpolator;

int main(int argc, char *argv[]) {
  const FMCA::Index dim = atoi(argv[1]);
  const FMCA::Index mp_deg = 5;
  Interpolator td_interp;
  td_interp.init(dim, mp_deg);
  FMCA::Matrix pts = FMCA::interpolationPointsGenerator(td_interp);
  return 0;
}
