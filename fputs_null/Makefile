BUILD_DIR ?= ../_bin

BIN_PATH := $(BUILD_DIR)/$(shell basename `pwd`).out
CPP_FLAGS := -std=c++14 -O2

all: build
.PHONY: all

build:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPP_FLAGS) -o $(BIN_PATH) fputs_null.cpp
.PHONY: main

$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(TEST_INC) $(CPP_FLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_PATH)
.PHONY: clean

run:
	$(BIN_PATH)
.PHONY: run
