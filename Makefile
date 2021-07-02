.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

COMMAND_PREFIX = poetry run

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

push: ## push code and tags to remote repository
	git push && git push --tag

lint: format ## check style with flake8
	$(COMMAND_PREFIX) flake8 textnets tests

format: ## format code with black
	$(COMMAND_PREFIX) black textnets tests

test: ## run tests quickly with the default Python
	$(COMMAND_PREFIX) pytest
	$(COMMAND_PREFIX) mypy

coverage: ## check code coverage quickly with the default Python
	$(COMMAND_PREFIX) coverage run --source textnets -m pytest
	$(COMMAND_PREFIX) coverage report -m
	$(COMMAND_PREFIX) coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	$(COMMAND_PREFIX) $(MAKE) -C docs clean
	$(COMMAND_PREFIX) $(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

api-docs:
	rm -f docs/reference.rst
	$(COMMAND_PREFIX) sphinx-apidoc -T -M -H "API Reference" -o docs/ textnets
	mv docs/textnets.rst docs/reference.rst

servedocs: docs ## compile the docs watching for changes
	$(COMMAND_PREFIX) watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

install: clean ## install the package and its dependencies
	poetry install
