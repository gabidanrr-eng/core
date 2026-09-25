"""Checksummed, forward-only schema migrations with automatic pre-upgrade backup."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from coremain.errors import MigrationError
from coremain.store.db import Database
from coremain.util.jsonutil import sha256_hex

_NAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return sha256_hex(self.sql)


@dataclass
class MigrationStatus:
    current: int
    latest: int
    pending: list[int] = field(default_factory=list)
    newer_than_code: list[int] = field(default_factory=list)
    checksum_mismatches: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.pending or self.newer_than_code or self.checksum_mismatches)


@dataclass
class MigrationReport:
    applied: list[int]
    current: int
    backup_path: Path | None = None


def load_migrations() -> list[Migration]:
    pkg = resources.files("coremain.store") / "migrations"
    found: list[Migration] = []
    for entry in pkg.iterdir():
        m = _NAME.match(entry.name)
        if m:
            found.append(Migration(int(m.group(1)), m.group(2), entry.read_text(encoding="utf-8")))
    found.sort(key=lambda mig: mig.version)
    versions = [mig.version for mig in found]
    if versions != list(range(1, len(versions) + 1)):
        raise MigrationError(f"migration versions must be contiguous from 1, found {versions}")
    return found


def _ensure_table(db: Database) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
        "checksum TEXT NOT NULL, applied_at REAL NOT NULL)"
    )


def status(db: Database) -> MigrationStatus:
    _ensure_table(db)
    known = {m.version: m for m in load_migrations()}
    have = {int(r["version"]): str(r["checksum"]) for r in db.query("SELECT version, checksum FROM schema_migrations")}
    latest = max(known) if known else 0
    return MigrationStatus(
        current=max(have) if have else 0,
        latest=latest,
        pending=sorted(v for v in known if v not in have),
        newer_than_code=sorted(v for v in have if v not in known),
        checksum_mismatches=sorted(v for v, c in have.items() if v in known and known[v].checksum != c),
    )


def migrate(db: Database, *, backup_dir: Path | None = None) -> MigrationReport:
    st = status(db)
    if st.newer_than_code:
        raise MigrationError(
            f"database schema version {st.current} is newer than this Core Main build supports ({st.latest})",
            hint="Upgrade Core Main, or restore a backup made by this version (`core db restore`).",
        )
    if st.checksum_mismatches:
        raise MigrationError(
            f"applied migrations {st.checksum_mismatches} differ from the shipped migration files",
            hint="The database or installation may be corrupted. Run `core doctor` and restore from backup.",
        )
    if not st.pending:
        return MigrationReport([], st.current)
    backup_path: Path | None = None
    if st.current > 0 and backup_dir is not None:
        backup_path = backup_dir / f"core-v{st.current}-pre-v{st.latest}-{int(time.time())}.db"
        db.backup_to(backup_path)
    known = {m.version: m for m in load_migrations()}
    conn = db.conn()
    applied: list[int] = []
    for version in st.pending:
        mig = known[version]
        script = (
            "BEGIN IMMEDIATE;\n"
            + mig.sql
            + f"\nINSERT INTO schema_migrations(version, name, checksum, applied_at) VALUES "
            f"({mig.version}, '{mig.name}', '{mig.checksum}', {time.time()});\nCOMMIT;\n"
        )
        try:
            conn.executescript(script)
        except Exception as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise MigrationError(f"migration {mig.version}_{mig.name} failed: {exc}") from exc
        applied.append(version)
    return MigrationReport(applied, max(applied), backup_path)
