build/gtest:
	cd vendor/googletest; mkdir -p build; cd build; cmake ..; make;
.PHONY: build-gtest

test:
	python -m unittest discover -v
.PHONY: test

test/%:
	python -m unittest $*/test_main.py -v
