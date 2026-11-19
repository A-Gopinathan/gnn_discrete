CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
LDLIBS = -larmadillo

.PHONY: all clean
all: potts_dynamics_ed

potts_dynamics_ed: simulation_ed.o potts_model.o util.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

simulation_ed.o potts_model.o: potts_model.hpp util.hpp
util.o: util.hpp

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f potts_dynamics_ed simulation_ed.o potts_model.o util.o
