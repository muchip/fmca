g++-11 -g -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) \
 -I/opt/homebrew/include/eigen3 -I../ \
FMCA.cpp -o FMCA$(python3-config --extension-suffix) \
-L/opt/homebrew/Cellar/python@3.9/3.9.10/\
Frameworks/Python.framework/Versions/3.9/lib/ -lpython3.9 \
 -I/usr/local/include/eigen3\
 -L/usr/local/Cellar/python@3.9/3.9.13_1/Frameworks/Python.framework/Versions/3.9/lib
