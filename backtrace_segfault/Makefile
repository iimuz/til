BUILD_DIR ?= ../_bin

BIN_PATH := $(BUILD_DIR)/$(shell basename `pwd`).out
CPP_FLAGS := -std=c++14 -O0 -rdynamic

all: build
.PHONY: all

build:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPP_FLAGS) -o $(BIN_PATH) segfault.cpp
.PHONY: build

$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(TEST_INC) $(CPP_FLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_PATH)
.PHONY: clean

run-catchsegv:
	catchsegv $(BIN_PATH)
.PHONY: run-catchsegv

run-preload:
	LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so $(BIN_PATH)
.PHONY: run-preload
