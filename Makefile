.PHONY: test lint coverage clean

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=dravex --cov-branch --cov-config=.coveragerc --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov .pytest_cache

lint:
	python -m py_compile dravex/__init__.py
