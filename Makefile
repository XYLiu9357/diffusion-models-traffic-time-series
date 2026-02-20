default: run-pipeline

run-pipeline: format
	python -m scripts.base_pipeline

format:
	find . -name "*.py" -not -path "./venv/*" -not -path "./.env/*" | xargs isort
	find . -name "*.py" -not -path "./venv/*" -not -path "./.env/*" | xargs black
