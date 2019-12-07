SRC_DIRS ?= ./src
BUILD_DIR ?= ../../build/shm-go-client

NAME := shm-go-client
VERSION := $(shell git describe --tags --abbrev=0)
REVISION := $(shell git rev-parse --short HEAD)
SRCS := $(shell find . -type f -name '*.go')
LDFLAGS := -X 'main.version=$(VERSION)' \
	-X 'main.revision=$(REVISION)'

all: client

clean:
	$(RM) -r $(BUILD_DIR)

run:
	$(BUILD_DIR)/client

client: ${SRC_DIRS}/*.go
	CGO_ENABLED=0 go build -ldflags "$(LDFLAGS)" -o ${BUILD_DIR}/$@ $^

.PHONY: all client clean

MKDIR_P ?= mkdir -p

