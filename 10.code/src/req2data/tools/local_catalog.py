"""로컬 CatalogClient: DuckDB(읽기 전용) + RAG 지식 베이스 + FK 그래프.

MCP 카탈로그 서버(PRD 2.5 B)는 이 클래스를 도구로 감싸 노출한다.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from req2data.config import Settings, get_settings
from req2data.rag.knowledge_base import KnowledgeBase
from req2data.tools.catalog import SQLExecutionError
from req2data.tools.sql_guard import check_readonly_sql


class LocalCatalogClient:
    def __init__(
        self,
        db_path: Path,
        kb: KnowledgeBase,
        *,
        max_rows: int = 10000,
        timeout_seconds: float = 30,
    ):
        self.db_path = Path(db_path)
        self.kb = kb
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self._con: duckdb.DuckDBPyConnection | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, kb: KnowledgeBase | None = None
    ) -> "LocalCatalogClient":
        s = settings or get_settings()
        sql = s.policy.get("sql", {})
        return cls(
            s.db_path,
            kb or KnowledgeBase.from_settings(s),
            max_rows=sql.get("max_rows", 10000),
            timeout_seconds=sql.get("timeout_seconds", 30),
        )

    # ---- 카탈로그 ----

    def list_tables(self) -> list[dict[str, Any]]:
        return [
            {"table": t.name, "description": t.description, "row_count": self._row_count(t.name)}
            for t in self.kb.spec.tables.values()
        ]

    def describe_table(self, table: str) -> dict[str, Any]:
        spec = self.kb.spec.tables.get(table)
        if spec is None:
            raise KeyError(f"카탈로그에 없는 테이블: {table}")
        return {
            "table": table,
            "description": spec.description,
            "grain": spec.grain,
            "row_count": self._row_count(table),
            "columns": [
                {
                    "name": c.name,
                    "type": c.type,
                    "description": c.description,
                    "is_pk": c.pk,
                    "fk": c.fk,
                    "is_time": c.is_time,
                    "time_rank": c.time_rank,
                    "available_at": c.available_at,
                    "note": c.note,
                }
                for c in spec.columns
            ],
        }

    def search_catalog(
        self, query: str, k: int = 8, source: str = "catalog"
    ) -> list[dict[str, Any]]:
        return self.kb.search(query, k=k, source=source)

    def profile_column(self, table: str, column: str) -> dict[str, Any]:
        cached = self.kb.profiles.get(f"{table}.{column}")
        if cached:
            return cached
        if self.kb.spec.column(f"{table}.{column}") is None:
            raise KeyError(f"카탈로그에 없는 컬럼: {table}.{column}")
        row = self.run_readonly_sql(
            f"SELECT count(*), count(*) - count({column}), count(DISTINCT {column}) FROM {table}"
        )["rows"][0]
        n, nulls, distinct = row
        return {
            "row_count": n,
            "null_ratio": round(nulls / n, 4) if n else 0.0,
            "distinct": distinct,
        }

    def find_join_path(self, tables: list[str]) -> list[dict[str, str]]:
        return self.kb.fk_graph.join_path(tables)

    # ---- SQL ----

    def run_readonly_sql(self, sql: str, max_rows: int | None = None) -> dict[str, Any]:
        check_readonly_sql(sql, set(self.kb.spec.tables))
        limit = min(max_rows or self.max_rows, self.max_rows)
        with self._lock:  # 연결을 얻는 것부터 잠가야 close()·재적재와 겹치지 않는다
            con = self._connection()
            timer = threading.Timer(self.timeout_seconds, con.interrupt)
            timer.start()
            try:
                cur = con.execute(sql)
                rows = cur.fetchmany(limit + 1)
                columns = [d[0] for d in cur.description]
            except duckdb.InterruptException as exc:
                raise SQLExecutionError(f"실행 시간 {self.timeout_seconds}초 초과") from exc
            except duckdb.Error as exc:
                raise SQLExecutionError(str(exc)) from exc
            finally:
                timer.cancel()
        return {
            "columns": columns,
            "rows": [list(r) for r in rows[:limit]],
            "row_count": min(len(rows), limit),
            "truncated": len(rows) > limit,
        }

    def close(self) -> None:
        """읽기 전용 연결을 닫는다 (다음 조회 때 다시 연다)."""
        with self._lock:
            self._close()

    @contextmanager
    def offline(self) -> Iterator[None]:
        """DB 재적재 동안 연결을 닫고 조회를 막는다. 블록이 끝나야 다음 조회가 연결을 다시 연다."""
        with self._lock:
            self._close()
            yield

    def _close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def _connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            if not self.db_path.exists():
                raise SQLExecutionError(f"DB 파일이 없습니다: {self.db_path} (make data로 적재)")
            self._con = duckdb.connect(
                str(self.db_path), read_only=True, config={"enable_external_access": False}
            )
        return self._con

    def _row_count(self, table: str) -> int | None:
        cached = self.kb.profiles.get(table, {}).get("row_count")
        if cached is not None or not self.db_path.exists():
            return cached
        return self.run_readonly_sql(f"SELECT count(*) FROM {table}")["rows"][0][0]
