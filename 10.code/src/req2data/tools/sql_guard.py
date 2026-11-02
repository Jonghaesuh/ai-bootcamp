"""읽기 전용 SQL 검사. DuckDB의 enable_external_access=false와 함께 이중으로 막는다."""

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from req2data.tools.catalog import SQLExecutionError

# 파일·네트워크·확장에 접근하는 함수
BLOCKED_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "parquet_scan",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_text",
    "read_blob",
    "glob",
    "sniff_csv",
    "query_table",
    "query",
}


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return node.sql_name().lower()


def check_readonly_sql(sql: str, allowed_tables: set[str]) -> None:
    """단일 SELECT(CTE·UNION 포함)이고, 카탈로그 테이블만 읽고, 파일 함수를 쓰지 않는지 확인한다."""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="duckdb") if s is not None]
    except ParseError as exc:
        raise SQLExecutionError(f"SQL 파싱 실패: {exc}") from exc
    if len(statements) != 1:
        raise SQLExecutionError("SQL 문은 하나만 실행할 수 있습니다.")
    stmt = statements[0]
    if not isinstance(stmt, exp.Query):
        raise SQLExecutionError(f"SELECT 문만 허용됩니다: {stmt.key.upper()}")

    for func in stmt.find_all(exp.Func):
        name = _function_name(func).replace("_", "")
        if name in {b.replace("_", "") for b in BLOCKED_FUNCTIONS}:
            raise SQLExecutionError(f"허용되지 않는 함수: {_function_name(func)}")

    ctes = {c.alias_or_name for c in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue  # range() 같은 테이블 함수는 위 함수 검사로 처리
        if table.name not in ctes and table.name not in allowed_tables:
            raise SQLExecutionError(f"카탈로그에 없는 테이블: {table.name}")
