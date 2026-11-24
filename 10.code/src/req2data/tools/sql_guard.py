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

    # DuckDB 식별자는 대소문자를 구분하지 않는다 (FROM Orders == FROM orders)
    allowed = {t.lower() for t in allowed_tables}
    ctes = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            # 테이블 자리의 함수는 range()·generate_series()만 허용한다.
            # duckdb_settings()·duckdb_secrets() 같은 시스템 함수로 설정을 읽지 못하게 막는다.
            if isinstance(table.this, exp.GenerateSeries):
                continue
            raise SQLExecutionError(
                f"허용되지 않는 테이블 함수: {table.this.sql(dialect='duckdb')}"
            )
        name = table.name.lower()
        if name not in ctes and name not in allowed:
            raise SQLExecutionError(f"카탈로그에 없는 테이블: {table.name}")
