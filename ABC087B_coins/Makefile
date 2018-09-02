BUILD_DIR ?= ../bin

BIN_PATH := $(BUILD_DIR)/$(shell basename `pwd`).out
CPP_FLAGS := -std=c++14 -O2

TEST_PATH := $(BUILD_DIR)/$(shell basename `pwd`)_test.out
TEST_SRC_DIRS := ./test
TEST_SRCS := $(shell find $(TEST_SRC_DIRS) -regex ".*\.cpp")
TEST_OBJS := $(TEST_SRCS:%=$(BUILD_DIR)/%.o)
TEST_INC := -I../vendor/googletest/googletest/include
TEST_LIB := -L../vendor/googletest/build/googlemock/gtest -lgtest

all: build
.PHONY: all

build:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPP_FLAGS) -o $(BIN_PATH) main.cpp
.PHONY: main

build-gtest:
	cd ../vendor/googletest; mkdir -p build; cd build; cmake ..; make;
.PHONY: build-gtest

build-test: $(TEST_OBJS)
	echo $(TEST_SRCS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(TEST_INC) $(CPP_FLAGS) $(TEST_OBJS) -o $(TEST_PATH) $(TEST_LIB)
.PHONY: build-test

$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(TEST_INC) $(CPP_FLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_PATH)
	rm -rf $(TEST_PATH)
	rm -rf $(BUILD_DIR)/$(TEST_SRC_DIRS)
.PHONY: clean

run:
	$(BIN_PATH)
.PHONY: run

run-test:
	$(TEST_PATH)
.PHONY: run-test
