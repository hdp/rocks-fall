.DEFAULT_GOAL := test

export VIRTUAL_ENV := $(shell sh -c '. env/bin/activate; echo $${VIRTUAL_ENV}')
export PATH := $(shell sh -c '. env/bin/activate; echo $${PATH}')

.PHONY: coverage
coverage:
	coverage erase
	coverage run --include rocks_fall/* -m pytest -ra
	coverage report -m

.PHONY: deps
deps:
	python -m pip install --upgrade pip
	python -m pip install \
		black \
		coverage \
		flake8 \
		mccabe \
		mypy \
		parameterized \
		pylint \
		pytest \
		tox \
		tox-gh-actions

.PHONY: lint
lint:
	python -m flake8 rocks_fall
	python -m pylint rocks_fall
	python -m mypy rocks_fall

.PHONY: publish
publish:
	# python -m flit publish

.PHONY: push
push:
	git push && git push --tags

.PHONY: test
test:
	python -m pytest -ra

.PHONY: tox
tox:
	python -m tox
