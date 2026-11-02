"""Agent가 데이터에 접근하는 유일한 경계. MCP 카탈로그 서버 클라이언트가 이 프로토콜을 구현한다."""

import json
from typing import Any, Protocol

from langchain_core.tools import BaseTool, tool


class SQLExecutionError(RuntimeError):
    pass


class CatalogClient(Protocol):
    def list_tables(self) -> list[dict[str, Any]]:
        """[{"table", "description", "row_count"}]"""
        ...

    def describe_table(self, table: str) -> dict[str, Any]:
        """{"table", "description", "row_count", "columns": [{"name", "type", "description",
        "is_pk", "fk" ("table.column"|None), "is_time", "time_rank" (시점 컬럼의 업무 흐름상 순서),
        "available_at" (값이 확정되는 시점 컬럼 "table.column")}]}"""
        ...

    def search_catalog(
        self, query: str, k: int = 8, source: str = "catalog"
    ) -> list[dict[str, Any]]:
        """[{"doc_id", "table", "column", "text", "score"}]"""
        ...

    def profile_column(self, table: str, column: str) -> dict[str, Any]:
        """{"row_count", "null_ratio", "distinct", "min", "max", "examples"}"""
        ...

    def find_join_path(self, tables: list[str]) -> list[dict[str, str]]:
        """[{"left": "a.x", "right": "b.y"}]"""
        ...

    def run_readonly_sql(self, sql: str, max_rows: int | None = None) -> dict[str, Any]:
        """{"columns", "rows", "row_count", "truncated"}. 실패 시 SQLExecutionError."""
        ...


def schema_index(client: CatalogClient, tables: list[str]) -> dict[str, dict[str, Any]]:
    """{"table.column": 컬럼 메타데이터}. 존재 확인·시점 판정에 쓴다."""
    index: dict[str, dict[str, Any]] = {}
    for table in tables:
        try:
            desc = client.describe_table(table)
        except Exception:
            continue
        for col in desc.get("columns", []):
            index[f"{table}.{col['name']}"] = col
    return index


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def build_catalog_tools(client: CatalogClient, *, max_rows: int = 50) -> list[BaseTool]:
    """ReAct Agent용 LangChain 도구. 결과는 LLM이 읽을 JSON 문자열로 돌려준다."""

    @tool
    def search_catalog(query: str) -> str:
        """데이터 카탈로그에서 질의와 관련된 테이블·컬럼 문서를 검색한다."""
        return _dump(client.search_catalog(query))

    @tool
    def describe_table(table: str) -> str:
        """테이블의 컬럼, 타입, PK/FK, 시점 컬럼 정보를 조회한다."""
        return _dump(client.describe_table(table))

    @tool
    def profile_column(table: str, column: str) -> str:
        """컬럼의 행 수, 결측률, 고유값 수, 범위를 조회한다. 키 컬럼의 단위 확인에 쓴다."""
        return _dump(client.profile_column(table, column))

    @tool
    def find_join_path(tables: list[str]) -> str:
        """주어진 테이블들을 잇는 FK 조인 경로를 찾는다."""
        return _dump(client.find_join_path(tables))

    @tool
    def run_readonly_sql(sql: str) -> str:
        """읽기 전용 SELECT 문을 실행하고 결과 일부를 돌려준다."""
        try:
            return _dump(client.run_readonly_sql(sql, max_rows=max_rows))
        except SQLExecutionError as exc:
            return _dump({"error": str(exc)})

    return [search_catalog, describe_table, profile_column, find_join_path, run_readonly_sql]
