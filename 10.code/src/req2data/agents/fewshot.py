"""동적 Few-shot: 입력과 가장 비슷한 예시를 골라 프롬프트에 넣는다.

RAG 인덱스가 준비되기 전까지는 문자 bigram BM25로 고른다(한글은 형태소 분석 없이도 bigram이 잘 맞음).
"""

import json
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from rank_bm25 import BM25Okapi


def _bigrams(text: str) -> list[str]:
    s = "".join(text.lower().split())
    return [s[i : i + 2] for i in range(len(s) - 1)] or [s]


class FewShotSelector:
    def __init__(self, fewshot_dir: Path, k: int = 2, exclude_domains: tuple[str, ...] = ()):
        self.fewshot_dir = Path(fewshot_dir)
        self.k = k
        self.exclude_domains = set(exclude_domains)

    def examples(self, kind: str) -> list[dict[str, Any]]:
        path = self.fewshot_dir / f"{kind}.yaml"
        if not path.exists():
            return []
        return [e for e in _load(path) if e.get("domain") not in self.exclude_domains]

    def select(self, kind: str, query: str, k: int | None = None) -> list[dict[str, Any]]:
        examples = self.examples(kind)
        k = self.k if k is None else k
        if len(examples) <= k:
            return examples
        bm25 = BM25Okapi([_bigrams(e["input"]) for e in examples])
        scores = bm25.get_scores(_bigrams(query))
        ranked = sorted(range(len(examples)), key=lambda i: scores[i], reverse=True)
        return [examples[i] for i in ranked[:k]]

    def render(self, kind: str, query: str) -> str:
        chosen = self.select(kind, query)
        if not chosen:
            return ""
        blocks = [
            f"### 예시 {i} ({e.get('domain', '')})\n[입력]\n{e['input'].strip()}\n[출력]\n"
            f"{json.dumps(e['output'], ensure_ascii=False)}"
            for i, e in enumerate(chosen, 1)
        ]
        return "## 참고 예시 (형식과 판단 기준만 참고하고 내용을 베끼지 말 것)\n" + "\n\n".join(
            blocks
        )


@cache
def _load(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or []
