LIBS = -lstdc++ $(shell pkg-config --libs opencv)
CFLAGS = -g -Wall -std=gnu++0x  
CFLAGS += $(shell pkg-config --cflags opencv)

test: test.cpp dvs_emu.cpp
	$(CXX) -o $@ $@.cpp dvs_emu.cpp dvs_op.cpp $(LIBS) $(CFLAGS) 
	./$@

clean:
	rm test

