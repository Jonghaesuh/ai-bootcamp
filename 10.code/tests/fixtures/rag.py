"""RAG 테스트 도구: 결정적 해시 임베딩, 실제 카탈로그 스키마를 가진 테스트 DuckDB."""

import math
import zlib
from pathlib import Path

import duckdb
from langchain_core.embeddings import Embeddings

from req2data.rag.sources import CatalogSpec
from req2data.rag.tokenize import tokenize
from tests.fixtures.fake_catalog import build_db

DIM = 256


class HashEmbeddings(Embeddings):
    """토큰 해시 bag-of-words 벡터. 네트워크 없이 의미 검색 경로를 재현한다."""

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_local_db(path: Path, spec: CatalogSpec) -> Path:
    """fake_catalog의 합성 데이터에 카탈로그의 나머지 테이블·컬럼을 빈 값으로 채워 파일 DB로 저장한다."""
    mem = build_db()
    with duckdb.connect(str(path)) as con:
        for t in spec.tables.values():
            existing = (
                {r[0] for r in mem.execute(f"DESCRIBE {t.name}").fetchall()}
                if _has(mem, t.name)
                else set()
            )
            select = ", ".join(
                c.name if c.name in existing else f"CAST(NULL AS {c.type}) AS {c.name}"
                for c in t.columns
            )
            rows = mem.execute(
                f"SELECT {select} FROM {t.name}" if existing else f"SELECT {select} LIMIT 0"
            ).df()
            con.register("_src", rows)
            con.execute(f"CREATE TABLE {t.name} AS SELECT * FROM _src")
            con.unregister("_src")
    return path


def _has(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()[0]
    )
