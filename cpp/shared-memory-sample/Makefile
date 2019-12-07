all: build

build:
	make -C server server
	make -C client client

clean:
	make -C server clean
	make -C client clean

format:
	make -C server format
	make -C client format

run/client:
	make -C client run

run/server:
	make -C server run

