"""테스트용 CatalogClient: Olist 구조를 흉내 낸 작은 DuckDB와 카탈로그 메타데이터."""

import random
from datetime import datetime, timedelta
from typing import Any

import duckdb
import pandas as pd

from req2data.tools.catalog import SQLExecutionError

TIME_RANK = {
    "orders.order_purchase_timestamp": 1,
    "orders.order_approved_at": 2,
    "orders.order_delivered_customer_date": 4,
    "order_reviews.review_creation_date": 5,
}

# (컬럼, 타입, 설명, PK, FK, available_at)
CATALOG: dict[str, dict[str, Any]] = {
    "customers": {
        "description": "고객 테이블. customer_id는 주문마다 새로 발급되고 customer_unique_id가 실제 고객",
        "columns": [
            ("customer_id", "VARCHAR", "주문별 고객 키", True, None, None),
            ("customer_unique_id", "VARCHAR", "실제 고객 고유 ID", False, None, None),
            ("customer_state", "VARCHAR", "고객 지역(주)", False, None, None),
        ],
    },
    "orders": {
        "description": "주문 테이블. 주문 상태와 구매·승인·배송 시각",
        "columns": [
            ("order_id", "VARCHAR", "주문 ID", True, None, None),
            ("customer_id", "VARCHAR", "주문 고객 키", False, "customers.customer_id", None),
            (
                "order_status",
                "VARCHAR",
                "주문 상태 (delivered, canceled 등) 취소",
                False,
                None,
                None,
            ),
            (
                "order_purchase_timestamp",
                "TIMESTAMP",
                "구매 시각 첫 구매 재구매 주문",
                False,
                None,
                None,
            ),
            ("order_approved_at", "TIMESTAMP", "결제 승인 시각", False, None, None),
            (
                "order_delivered_customer_date",
                "TIMESTAMP",
                "고객 배송 완료 시각",
                False,
                None,
                None,
            ),
        ],
    },
    "order_payments": {
        "description": "주문 결제 정보. 결제 수단, 할부, 금액",
        "columns": [
            ("order_id", "VARCHAR", "주문 ID", False, "orders.order_id", None),
            (
                "payment_type",
                "VARCHAR",
                "결제 수단",
                False,
                None,
                "orders.order_purchase_timestamp",
            ),
            (
                "payment_installments",
                "INTEGER",
                "할부 개월 수",
                False,
                None,
                "orders.order_purchase_timestamp",
            ),
            (
                "payment_value",
                "DOUBLE",
                "결제 금액 주문 금액",
                False,
                None,
                "orders.order_purchase_timestamp",
            ),
        ],
    },
    "order_reviews": {
        "description": "주문 리뷰. 배송 후 작성되는 리뷰 점수",
        "columns": [
            ("review_id", "VARCHAR", "리뷰 ID", True, None, None),
            ("order_id", "VARCHAR", "주문 ID", False, "orders.order_id", None),
            (
                "review_score",
                "INTEGER",
                "리뷰 점수 1~5 만족도",
                False,
                None,
                "order_reviews.review_creation_date",
            ),
            ("review_creation_date", "TIMESTAMP", "리뷰 작성 시각", False, None, None),
        ],
    },
}


def _bigrams(text: str) -> set[str]:
    s = "".join(text.lower().split())
    return {s[i : i + 2] for i in range(len(s) - 1)}


def build_db(n_customers: int = 1500, seed: int = 7) -> duckdb.DuckDBPyConnection:
    rng = random.Random(seed)
    start, span = datetime(2017, 1, 1), (datetime(2018, 8, 1) - datetime(2017, 1, 1)).days
    customers, orders, payments, reviews = [], [], [], []

    def add_order(uid: str, ts: datetime, status: str = "delivered") -> None:
        n = len(orders)
        cid, oid = f"c{n}", f"o{n}"
        delivered = ts + timedelta(days=rng.randint(3, 20)) if status == "delivered" else None
        customers.append((cid, uid, rng.choice(["SP", "RJ", "MG"])))
        orders.append((oid, cid, status, ts, ts + timedelta(hours=1), delivered))
        payments.append(
            (
                oid,
                rng.choice(["credit_card", "boleto"]),
                rng.randint(1, 10),
                round(rng.uniform(20, 500), 2),
            )
        )
        if delivered:
            reviews.append(
                (f"r{n}", oid, rng.randint(1, 5), delivered + timedelta(days=rng.randint(1, 5)))
            )

    for i in range(n_customers):
        uid = f"u{i}"
        first = start + timedelta(days=rng.randint(0, span), hours=rng.randint(0, 23))
        add_order(uid, first, "canceled" if rng.random() < 0.03 else "delivered")
        r = rng.random()
        if r < 0.12:
            add_order(uid, first + timedelta(days=rng.randint(1, 80)))
        elif r < 0.17:
            add_order(uid, first + timedelta(days=rng.randint(120, 200)))

    con = duckdb.connect(":memory:")
    frames = {
        "customers": pd.DataFrame(
            customers, columns=["customer_id", "customer_unique_id", "customer_state"]
        ),
        "orders": pd.DataFrame(
            orders,
            columns=[
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_customer_date",
            ],
        ),
        "order_payments": pd.DataFrame(
            payments, columns=["order_id", "payment_type", "payment_installments", "payment_value"]
        ),
        "order_reviews": pd.DataFrame(
            reviews, columns=["review_id", "order_id", "review_score", "review_creation_date"]
        ),
    }
    for name, df in frames.items():
        con.register(f"_{name}", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
        con.unregister(f"_{name}")
    return con


class FakeCatalog:
    def __init__(self) -> None:
        self.con = build_db()
        self.calls: list[tuple[str, Any]] = []

    def _row_count(self, table: str) -> int:
        return self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def list_tables(self) -> list[dict[str, Any]]:
        return [
            {"table": t, "description": m["description"], "row_count": self._row_count(t)}
            for t, m in CATALOG.items()
        ]

    def describe_table(self, table: str) -> dict[str, Any]:
        self.calls.append(("describe_table", table))
        if table not in CATALOG:
            raise KeyError(table)
        cols = []
        for name, typ, desc, pk, fk, avail in CATALOG[table]["columns"]:
            ref = f"{table}.{name}"
            cols.append(
                {
                    "name": name,
                    "type": typ,
                    "description": desc,
                    "is_pk": pk,
                    "fk": fk,
                    "is_time": typ == "TIMESTAMP",
                    "time_rank": TIME_RANK.get(ref),
                    "available_at": avail,
                }
            )
        return {
            "table": table,
            "description": CATALOG[table]["description"],
            "row_count": self._row_count(table),
            "columns": cols,
        }

    def search_catalog(
        self, query: str, k: int = 8, source: str = "catalog"
    ) -> list[dict[str, Any]]:
        self.calls.append(("search_catalog", query))
        q = _bigrams(query)
        hits = []
        for table, meta in CATALOG.items():
            for name, _, desc, *_ in meta["columns"]:
                text = f"{table} {name} {desc} {meta['description']}"
                score = len(q & _bigrams(text))
                if score:
                    hits.append(
                        {
                            "doc_id": f"{table}.{name}",
                            "table": table,
                            "column": name,
                            "text": desc,
                            "score": float(score),
                        }
                    )
        return sorted(hits, key=lambda h: -h["score"])[:k]

    def profile_column(self, table: str, column: str) -> dict[str, Any]:
        n, nulls, distinct = self.con.execute(
            f"SELECT count(*), count(*) - count({column}), count(DISTINCT {column}) FROM {table}"
        ).fetchone()
        return {"row_count": n, "null_ratio": nulls / n if n else 0, "distinct": distinct}

    def find_join_path(self, tables: list[str]) -> list[dict[str, str]]:
        wanted = set(tables)
        joins = []
        for table, meta in CATALOG.items():
            for name, _, _, _, fk, _ in meta["columns"]:
                if fk and table in wanted and fk.split(".")[0] in wanted:
                    joins.append({"left": f"{table}.{name}", "right": fk})
        return joins

    def run_readonly_sql(self, sql: str, max_rows: int | None = None) -> dict[str, Any]:
        self.calls.append(("run_readonly_sql", sql))
        head = sql.lstrip().split(None, 1)[0].upper()
        if head not in ("SELECT", "WITH"):
            raise SQLExecutionError("SELECT 문만 허용됩니다.")
        try:
            cur = self.con.execute(sql)
            rows = cur.fetchall()
        except duckdb.Error as exc:
            raise SQLExecutionError(str(exc)) from exc
        columns = [d[0] for d in cur.description]
        truncated = max_rows is not None and len(rows) > max_rows
        return {
            "columns": columns,
            "rows": rows[:max_rows] if max_rows else rows,
            "row_count": len(rows),
            "truncated": truncated,
        }
