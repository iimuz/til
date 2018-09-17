GRPC_DIR := ../vendor/grpc
PROTOBUF_DIR := $(GRPC_DIR)/third_party/protobuf

all:
.PHONY: all

build-grpc:
	cd $(GRPC_DIR) && git submodule update --init
	cd $(GRPC_DIR) && $(MAKE) -j4
.PHONY: build-grpc

build-protobuf:
	cd $(PROTOBUF_DIR) && ./autogen.sh
	cd $(PROTOBUF_DIR) && ./configure CC=$(CC) CXX=$(CXX)
	cd $(PROTOBUF_DIR) && $(MAKE) -j4
	cd $(PROTOBUF_DIR) && $(MAKE) install
.PHONY: build-protobuf

install-grpc: build-grpc
	cd $(GRPC_DIR) && $(MAKE) install
	ldconfig
.PHONY: install-grpc

install-protobuf: build-protobuf
	cd $(PROTOBUF_DIR) && $(MAKE) install
	ldconfig
.PHONY: install-protobuf

clean:
	cd $(GRPC_DIR)/third_party/protobuf && make clean
	cd $(GRPC_DIR) && make clean
