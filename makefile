########################################################################
#                                                                      #
#   FMCA                                                               #
#                                                                      #
#   created June 2021                                                  #
#                                                                      #
########################################################################
PROG= a.out
CPP = clang++ -std=c++11
CPPFLAGS = -O3 -fPIC
INCLUDE = -I/opt/homebrew/include/eigen3 -I/opt/homebrew/include\
					-I/usr/local/include/eigen3
#INCLUDE = -I/usr/include/eigen3
LDFLAGS = -lm #-L/opt/homebrew/lib -lmetis
OBJECTS = mainDebug.o
all: $(OBJECTS)
	$(CPP) $(OBJECTS) -o$(PROG) $(LDFLAGS)  
#
# tell make how to create a .o from a .c
#
%.o:%.cpp
	$(CPP)  $(CPPFLAGS) $(INCLUDE) -c $<
#
.PHONY: clean
#
clean:
	rm -f *.o
	rm -f *.vtk
	rm -f $(PROG)
#
# DO NOT DELETE THIS LINE -- make depend depends on it.
