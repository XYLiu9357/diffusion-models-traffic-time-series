default: run-csdi

run-pipeline: format
	python -m scripts.base_pipeline

train-csdi-forecast: format
	python -m scripts.train_csdi --task forecast

train-csdi-imputation: format
	python -m scripts.train_csdi --task imputation --corruption_rate 0.15

train-timegrad: format
	python -m scripts.train_timegrad

eval-csdi-forecast: format
	python -m scripts.evaluate_csdi --task forecast --checkpoint csdi_forecast.pth

eval-csdi-imputation: format
	python -m scripts.evaluate_csdi --task imputation --checkpoint csdi_imputation.pth

eval-timegrad: format
	python -m scripts.evaluate_timegrad

compare-all-forecast: format
	python -m scripts.compare_all --task forecast

compare-all-imputation: format
	python -m scripts.compare_all --task imputation

format:
	find . -name "*.py" -not -path "./venv/*" -not -path "./.env/*" | xargs isort
	find . -name "*.py" -not -path "./venv/*" -not -path "./.env/*" | xargs black
