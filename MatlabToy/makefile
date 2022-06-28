################################################################################
#                                                                              #
#   makefile						         	       #
#   SParse Quadrature Routines (SPQR)				               # 
#                                                                              #
#   created 9/11/2016                                                          #
#                                                                              #
################################################################################
#
VERSION=1.0
SHELL=/bin/sh

# set your MEX compiler
MEX= /Users/mmulterer/Applications/MATLAB_R2021a.app/bin/mex

# set C++ compile flags
MEXFLAGS= -I./
#
#

OBJ = MXsparseIndexSet.cpp

all:			$(OBJ)
				$(MEX) $(OBJ) $(MEXFLAGS)
				

# tell make how to create a .o from a .cpp

%.o:%.cpp
	$(CPP)  $(CPPFLAGS) -I$(EIGENPATH) -c $<


.PHONY: clean

clean:
	rm -f *.o

# DO NOT DELETE THIS LINE -- make depends on it.
