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
    parser.add_argument(
        "--if-stale",
        action="store_true",
        help="임베딩 설정과 지식 소스가 그대로면 건너뛰고, 바뀌었으면 다시 만든다 (컨테이너 시작 시 사용)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    s = get_settings()
    kb = KnowledgeBase.from_settings(s)
    if args.if_stale:
        ok = kb.ensure_current()
        state = "최신" if ok else "재구축 실패, 기존 인덱스 유지"
        print(
            f"인덱스 {state}: {s.chroma_dir} ({kb.index.built_with()}, {kb.index.count()}개 문서)"
        )
        return 0 if ok else 1
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
