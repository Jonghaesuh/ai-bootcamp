"""Olist CSV → DuckDB 적재 후 컬럼 프로파일을 data/catalog/profiles.json에 저장한다."""

import logging
import sys

import duckdb

from req2data.config import get_settings
from req2data.rag.knowledge_base import KnowledgePaths
from req2data.rag.profiling import load_csvs, profile_tables, save_profiles
from req2data.rag.sources import load_catalog_spec


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    s = get_settings()
    paths = KnowledgePaths.under(s.data_dir)
    spec = load_catalog_spec(paths.catalog)
    try:
        counts = load_csvs(spec, s.raw_data_dir, s.db_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    with duckdb.connect(str(s.db_path), read_only=True) as con:
        profiles = profile_tables(con, spec)
    save_profiles(profiles, paths.profiles)
    print(f"적재 완료: {s.db_path} {counts}")
    print(f"프로파일: {paths.profiles} ({len(profiles)}개 항목)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
