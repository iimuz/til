CPP_FLAGS := -std=c++14 -O2

BUILD_DIR ?= ../bin
BIN_PATH := $(BUILD_DIR)/$(shell basename `pwd`).out
SRCS := $(shell find . -regex ".*\.cpp")
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
INC := -I../vendor/googletest/googletest/include
LIB := -L../vendor/googletest/build/googlemock/gtest -lgtest

all: build/gtest build
.PHONY: all

build/gtest:
	cd ../vendor/googletest; mkdir -p build; cd build; cmake ..; make;
.PHONY: build-gtest

build: $(OBJS)
	echo $(SRCS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(INC) $(CPP_FLAGS) $(OBJS) -o $(BIN_PATH) $(LIB)
.PHONY: build

run:
	$(BIN_PATH)
.PHONY: run

clean:
	rm -rf $(BIN_PATH)
	rm -rf $(OBJS)
.PHONY: clean

$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(INC) $(CPP_FLAGS) -c $< -o $@
