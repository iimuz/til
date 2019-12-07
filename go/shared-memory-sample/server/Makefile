SRC_DIRS ?= ./src
BUILD_DIR ?= ../../build/shm-go-server

NAME := shm-go-server
VERSION := $(shell git describe --tags --abbrev=0)
REVISION := $(shell git rev-parse --short HEAD)
SRCS := $(shell find . -type f -name '*.go')
LDFLAGS := -X 'main.version=$(VERSION)' \
	-X 'main.revision=$(REVISION)'

all: server

clean:
	$(RM) -r $(BUILD_DIR)

run:
	$(BUILD_DIR)/server

server: ${SRC_DIRS}/*.go
	CGO_ENABLED=0 go build -ldflags "$(LDFLAGS)" -o ${BUILD_DIR}/$@ $^

.PHONY: all server clean

MKDIR_P ?= mkdir -p

