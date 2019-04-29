test:
	python -m unittest discover -v
.PHONY: test

test/%:
	python -m unittest $*/test_main.py -v

