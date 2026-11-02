"""데모 모드용 의미 검색 기록·재생.

LLM 프롬프트에는 RAG 검색 결과가 들어가므로, 임베딩 서버(Ollama)가 없는 환경에서 키워드 검색만으로 순위가 바뀌면
프롬프트가 녹화와 달라져 LLM 응답 재생이 맞지 않는다. 그래서 의미 검색 순위도 LLM 응답처럼 기록하고 재생한다.

- record: 실제 검색 결과를 기록한다.
- replay: 기록이 있으면 그대로 쓰고, 없으면 실제 검색을 시도해 성공하면 기록에 더한다 (Ollama가 있는 곳에서 한 번 돌리면 녹화가 채워짐).
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Literal

RECORD_FILE = "retrieval_recordings.json"


class DenseRecorder:
    def __init__(self, path: Path, mode: Literal["record", "replay"]):
        self.path = Path(path)
        self.mode = mode
        self._lock = threading.Lock()
        self._data: dict[str, list[str]] | None = None

    @staticmethod
    def key(query: str, sources: list[str] | None, k: int) -> str:
        payload = json.dumps([query, sorted(sources or []), k], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _load(self) -> dict[str, list[str]]:
        if self._data is None:
            self._data = (
                json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            )
        return self._data

    def get(self, key: str) -> list[str] | None:
        return self._load().get(key) if self.mode == "replay" else None

    def put(self, key: str, ids: list[str]) -> None:
        with self._lock:
            data = self._load()
            if data.get(key) == ids:
                return
            data[key] = ids
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=0, sort_keys=True) + "\n",
                encoding="utf-8",
            )
