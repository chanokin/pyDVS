LIBS = -lstdc++ $(shell pkg-config --libs opencv4)
CFLAGS = -g -Wall -std=gnu++0x  
CFLAGS += $(shell pkg-config --cflags opencv4)

test: test.cpp nvs_emu.cpp
	$(CXX) -o $@ $@.cpp nvs_emu.cpp nvs_op.cpp $(LIBS) $(CFLAGS) 
	./$@

clean:
	rm test

