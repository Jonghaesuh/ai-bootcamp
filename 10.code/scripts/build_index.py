"""카탈로그(+프로파일)·용어집·가이드·사례를 임베딩해 Chroma 인덱스를 다시 만든다."""

import argparse
import logging
import sys

from req2data.config import get_settings
from req2data.rag.knowledge_base import KnowledgeBase


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="인덱스 구축 후 확인용 검색어")
    parser.add_argument("--source", default=None, help="catalog | glossary | guide | case")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    s = get_settings()
    kb = KnowledgeBase.from_settings(s)
    if not kb.profiles:
        print("프로파일 없음: 카탈로그 정의만 색인합니다 (make data로 적재하면 프로파일이 추가됨).")
    counts = kb.rebuild()
    print(f"인덱스 구축 완료: {s.chroma_dir} {counts}")

    if args.query:
        for h in kb.search(args.query, source=args.source):
            print(f"{h['score']:.4f} {h['ranks']} {h['doc_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
