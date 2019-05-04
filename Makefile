all:
.PHONY: all

build/%:
	cd $* && $(MAKE) build

run/%:
	cd $* && $(MAKE) run

test:
	python -m unittest discover -v
.PHONY: test

test/%:
	python -m unittest $*/test_main.py -v
