"""Load repo `.env` before test collection so credential-gated tests
(e.g. live LLM checks) see keys without requiring manual exports.
Plain environment still takes precedence; nothing here runs live
unless explicitly enabled (see tests/test_llm_live.py gates).
"""
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pytest


@pytest.fixture(scope="module", autouse=True)
def _isolated_db(tmp_path_factory):
    """Every test module gets a fresh SQLite file: no repo vigil.db
    pollution, no cross-module decisions/cases/users leaking."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    mp.setenv("VIGIL_DB_PATH", str(tmp_path_factory.mktemp("db") / "t.db"))
    yield
    mp.undo()
