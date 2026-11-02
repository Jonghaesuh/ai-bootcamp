"""BM25용 토크나이저. 한글은 문자 bigram, 영문·컬럼명은 단어와 밑줄 조각으로 쪼갠다."""

import re

_ASCII = re.compile(r"[a-z0-9_]+")
_HANGUL = re.compile(r"[가-힣]+")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for word in _ASCII.findall(text):
        tokens.append(word)
        parts = [p for p in word.split("_") if p]
        if len(parts) > 1:
            tokens += parts
    for run in _HANGUL.findall(text):
        tokens += [run] if len(run) == 1 else [run[i : i + 2] for i in range(len(run) - 1)]
    return tokens
