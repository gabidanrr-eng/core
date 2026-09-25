---
name: python-projects
description: Conventions for working in Python repositories - environments, pytest, typing, packaging and common pitfalls. Use when the project is Python.
license: MIT
metadata:
  core-languages: python
  core-stage: implementation, correction, debugging, review
  core-triggers: python, pytest, pyproject, django, flask, fastapi, pip, uv, poetry
  core-priority: "4"
  core-version: "1"
---

# Python projects

- **Environment**: use the project's tool (`uv run`, `poetry run`, an existing `.venv`) instead of
  the system interpreter; never `pip install` globally. Respect `requires-python`.
- **Tests**: follow the existing layout (`tests/`, `conftest.py` fixtures, markers). Run the
  narrowest target first (`pytest path::test_name -q`), then the suite. Exit code 5 means no
  tests were collected — that is not a pass.
- **Imports**: respect the package layout (`src/` layout needs the package installed or on the
  path); do not add `sys.path` hacks to production code.
- **Typing and linting**: keep type hints consistent with the codebase; run the configured
  `ruff`/`mypy`/`pyright` when present.
- **Pitfalls**: mutable default arguments, bare `except`, catching `Exception` to hide bugs,
  naive datetimes, blocking calls inside `async` code, resource leaks without context managers.
