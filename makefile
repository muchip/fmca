################################################################################
#                                                                              #
#   FMCA                                                                       #
#                                                                              #
#   created April '19                                                          #
#                                                                              #
################################################################################
#
PROG= a.out
CPP = clang++ -fPIC -std=c++11
CPPFLAGS = -g
INCLUDE = /usr/include/eigen3
LDFLAGS = -lm
OBJECTS = main.o

all: $(OBJECTS)
	$(CPP) $(OBJECTS) -o$(PROG) $(LDFLAGS)  

#
# tell make how to create a .o from a .c
#

%.o:%.cpp
	$(CPP)  $(CPPFLAGS) -I$(INCLUDE) -c $<
#
.PHONY: clean
#
clean:
	rm -f *.o
	rm -f *.vtk
	rm -f $(PROG)

#
# DO NOT DELETE THIS LINE -- make depend depends on it.
