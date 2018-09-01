BUILD_DIR ?= ../bin

BIN_PATH := ${BUILD_DIR}/$(shell basename `pwd`).out
CPP_FLAGS := -std=c++14 -O2

all: main
.PHONY: all

clean:
	rm -rf $(BIN_PATH)
.PHONY: clean

main:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(INC_FLAGS) $(LIB_FLAGS) $(CPP_FLAGS) -o ${BIN_PATH} main.cpp
.PHONY: main

run:
	${BIN_PATH}
