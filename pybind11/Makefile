INCLUDES := $(shell python3 -m pybind11 --includes)
EXT_SUFFIX := $(shell python3-config --extension-suffix)
CXX := c++
CXXFLAGS := -O3 -Wall -shared -std=c++11 -I../clbk -fPIC

.PHONY: clbk$(EXT_SUFFIX)
clbk$(EXT_SUFFIX):
	$(CXX) $(CXXFLAGS) $(INCLUDES) clbk.cpp -o $@

.PHONY: clean
clean:
	rm -f clbk$(EXT_SUFFIX)
