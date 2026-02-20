default: run

run: format

format:
	python -m isort *.py
	python -m black *.py
