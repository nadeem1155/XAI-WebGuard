.PHONY: install test smoke serve clean
install:
	python -m pip install -r requirements.txt
test:
	pytest -q
smoke:
	python scripts/make_synthetic_data.py && python scripts/smoke_test.py
serve:
	XWG_TREE=results/model_a.pkl uvicorn xai_webguard.serve.app:app --port 8000
clean:
	rm -rf results/*.pkl results/*.json __pycache__ */__pycache__ .pytest_cache
