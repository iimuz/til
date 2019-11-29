BIN_DIR := ../bin
GRPC_DIR := ../vendor/grpc
PROTOBUF_DIR := $(GRPC_DIR)/third_party/protobuf
HELLO_DIR := $(GRPC_DIR)/examples/cpp/helloworld

all:
.PHONY: all

build-grpc: build-protobuf
	cd $(GRPC_DIR) && git submodule update --init
	cd $(GRPC_DIR) && $(MAKE) -j4
.PHONY: build-grpc

build-hello:
	cd $(HELLO_DIR) && $(MAKE)
	mkdir -p $(BIN_DIR)
	cp $(HELLO_DIR)/greeter_server $(BIN_DIR)/
	cp $(HELLO_DIR)/greeter_client $(BIN_DIR)/
	cp $(HELLO_DIR)/greeter_async_server $(BIN_DIR)/
	cp $(HELLO_DIR)/greeter_async_client $(BIN_DIR)/
	cp $(HELLO_DIR)/greeter_async_client2 $(BIN_DIR)/
	cp /usr/local/lib/libprotobuf.so.17 $(BIN_DIR)/
	cp /usr/local/lib/libgrpc++.so.1 $(BIN_DIR)/
	cp /usr/local/lib/libgrpc.so.6 $(BIN_DIR)/
	cp /usr/local/lib/libgrpc++_reflection.so.1 $(BIN_DIR)/
	cp /usr/local/lib/libgpr.so.6 $(BIN_DIR)/
.PHONY: build-hello

build-protobuf:
	cd $(PROTOBUF_DIR) && ./autogen.sh
	cd $(PROTOBUF_DIR) && ./configure CC=$(CC) CXX=$(CXX)
	cd $(PROTOBUF_DIR) && $(MAKE) -j4
	cd $(PROTOBUF_DIR) && $(MAKE) install
.PHONY: build-protobuf

clean:
	cd $(PROTOBUF_DIR) && make clean
	cd $(GRPC_DIR) && make clean
.PHONY: clean

clean-hello:
	cd $(HELLO_DIR) && $(MAKE) clean
	rm -rf $(BIN_DIR)/*
.PHONY: clean-hello

install-grpc: install-protobuf
	cd $(GRPC_DIR) && $(MAKE) install
	ldconfig
.PHONY: install-grpc

install-protobuf: build-protobuf
	cd $(PROTOBUF_DIR) && $(MAKE) install
	ldconfig
.PHONY: install-protobuf

run-hello/server:
	LD_LIBRARY_PATH=$(BIN_DIR) $(BIN_DIR)/greeter_server
