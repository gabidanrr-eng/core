.PHONY: install lint fmt fmt-check typecheck test test-all test-browser selftest build check

install:
	uv sync

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

fmt-check:
	uv run ruff format --check src tests

typecheck:
	uv run mypy

test:
	uv run pytest -m "not network and not live and not browser"

test-browser:
	uv run pytest -m browser

test-all:
	uv run pytest -m "not live"

selftest:
	uv run core selftest

build:
	uv build

check: lint fmt-check typecheck test selftest
