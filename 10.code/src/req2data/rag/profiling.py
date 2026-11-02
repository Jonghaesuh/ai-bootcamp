"""원천 CSV → DuckDB 적재와 컬럼 자동 프로파일."""

import json
import logging
from pathlib import Path
from typing import Any

import duckdb

from req2data.rag.sources import CatalogSpec

log = logging.getLogger(__name__)

NUMERIC = ("INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL")


def load_csvs(spec: CatalogSpec, raw_dir: Path, db_path: Path) -> dict[str, int]:
    """카탈로그의 source_file을 테이블로 적재한다. 타입은 카탈로그 정의를 따른다."""
    missing = [
        t.source_file for t in spec.tables.values() if not (raw_dir / t.source_file).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{raw_dir}에 원천 CSV가 없습니다: {', '.join(missing)}\n다운로드: {spec.source}"
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with duckdb.connect(str(db_path)) as con:
        for t in spec.tables.values():
            types = {c.name: c.type for c in t.columns}
            con.execute(
                f"CREATE OR REPLACE TABLE {t.name} AS "
                f"SELECT * FROM read_csv(?, header=true, columns={_duck_struct(types)})",
                [str(raw_dir / t.source_file)],
            )
            counts[t.name] = con.execute(f"SELECT count(*) FROM {t.name}").fetchone()[0]
            log.info("적재 %s: %d행", t.name, counts[t.name])
    return counts


def _duck_struct(types: dict[str, str]) -> str:
    return "{" + ", ".join(f"'{k}': '{v}'" for k, v in types.items()) + "}"


def profile_tables(con: duckdb.DuckDBPyConnection, spec: CatalogSpec) -> dict[str, dict[str, Any]]:
    """{"table.column": {row_count, null_ratio, distinct, min, max, examples}} 와 {"table": {row_count}}."""
    profiles: dict[str, dict[str, Any]] = {}
    for t in spec.tables.values():
        n = con.execute(f"SELECT count(*) FROM {t.name}").fetchone()[0]
        profiles[t.name] = {"row_count": n}
        for c in t.columns:
            ranged = c.type in NUMERIC or c.type == "TIMESTAMP"
            nulls, distinct, lo, hi = con.execute(
                f"SELECT count(*) - count({c.name}), count(DISTINCT {c.name}), "
                + (f"min({c.name}), max({c.name}) " if ranged else "NULL, NULL ")
                + f"FROM {t.name}"
            ).fetchone()
            examples = [
                r[0]
                for r in con.execute(
                    f"SELECT {c.name} FROM {t.name} WHERE {c.name} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY count(*) DESC LIMIT 3"
                ).fetchall()
            ]
            profiles[f"{t.name}.{c.name}"] = {
                "row_count": n,
                "null_ratio": round(nulls / n, 4) if n else 0.0,
                "distinct": int(distinct),
                "min": lo,
                "max": hi,
                "examples": examples,
            }
    return profiles


def save_profiles(profiles: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profiles, ensure_ascii=False, default=str, indent=1), encoding="utf-8"
    )


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
