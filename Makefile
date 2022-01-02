.DEFAULT_GOAL := test
.PHONY: coverage deps help lint publish push test

export VIRTUAL_ENV := $(shell sh -c '. env/bin/activate; echo $${VIRTUAL_ENV}')
export PATH := $(shell sh -c '. env/bin/activate; echo $${PATH}')

coverage:
	coverage erase
	coverage run --include rocks_fall/* -m pytest -ra
	coverage report -m

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

lint:
	python -m flake8 rocks_fall
	python -m pylint rocks_fall
	python -m mypy rocks_fall

publish:
	# python -m flit publish

push:
	# git push && git push --tags

test:
	python -m pytest -ra

tox:
	python -m tox
