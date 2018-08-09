# メタ情報
NAME     := go-vue-example
VERSION  := v0.1.0
REVISION := $(shell git rev-parse --short HEAD)
LDFLAGS := -ldflags="-s -w -X \"main.Version=$(VERSION)\" -X \"main.Revision=$(REVISION)\" -extldflags \"-static\""

SRCS    := $(shell find . -type f -name '*.go')


# build all
all:

## Build.
## Usage: make bin/govue
bin/%: cmd/%/main.go $(SRCS)
	go build -a -tags netgo -installsuffix netgo $(LDFLAGS) -o $@ $<

## Clean build data.
clean:
	rm -rf bin/

## Install dependencies.
deps:
	dep ensure

## Run formatting.
fmt:

## Show help.
.DEFAULT_GOAL := help
help:
	@make2help $(MAKEFILE_LIST)

## Run lint.
lint:

## Run tests.
test:

.PHONY: all clean deps fmt help lint test
