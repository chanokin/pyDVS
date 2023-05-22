LIBS = -lstdc++ $(shell pkg-config --libs opencv4)
CFLAGS = -g -Wall -std=gnu++0x  
CFLAGS += $(shell pkg-config --cflags opencv4)

test: test.cpp dvs_emu.cpp
	$(CXX) -o $@ $@.cpp dvs_emu.cpp dvs_op.cpp $(LIBS) $(CFLAGS) 
	./$@

clean:
	rm test

