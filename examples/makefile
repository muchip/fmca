################################################################################
#                                                                              #
#   makefile						         	       																			 #
#   FMCA                                                  			               # 
#                                                                              #
#   created 9/11/2016                                                          #
#                                                                              #
################################################################################
#
VERSION=1.0
SHELL=/bin/sh

# set your C++ compiler
CPP= clang++ -std=c++11
# set your MEX compiler

# set C++ compile flags
CPPFLAGS= -O3
# set Eigen include path
#EIGENPATH= /usr/local/include/eigen3/
EIGENPATH= /opt/homebrew/include/eigen3/
INCLUDE= -I/Users/mmulterer/Desktop/Repos/github/libigl/include -I$(EIGENPATH) -I../
#
#
OBJ = example_SampletArithmetic.o
all:	$(OBJ)
			$(CPP) $(OBJ)				

# tell make how to create a .o from a .cpp
%.o:%.cpp
	$(CPP)  $(CPPFLAGS) $(INCLUDE) -c $<


.PHONY: clean

clean:
	rm -f *.o

# DO NOT DELETE THIS LINE -- make depends on it.
