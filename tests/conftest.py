from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import CALC_REPO, Harness, make_repo


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.fixture
def calc_repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "proj", CALC_REPO)
