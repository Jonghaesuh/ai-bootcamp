"""데모 녹화의 사용자 입력(정의서·사용자 유형·확인 질문 답변).

LLM 응답 재생은 입력이 녹화 때와 완전히 같을 때만 맞는다. 그래서 기록 모드에서 사용자 입력도 함께 저장하고,
데모 모드에서는 이 값을 UI에 미리 채워 버튼만 눌러도 녹화가 그대로 재생되게 한다.
"""

import json
import threading
from pathlib import Path
from typing import Any, Literal

INPUTS_FILE = "demo_inputs.json"
DemoRole = Literal["record", "replay"]

_lock = threading.Lock()


class DemoInputs:
    def __init__(self, path: Path, role: DemoRole | None = None):
        self.path = Path(path)
        self.role = role

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"projects": [], "answers": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.setdefault("projects", [])
        data.setdefault("answers", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )

    # ---- 기록 모드 ----

    def record_project(self, document: str, user_type: str) -> None:
        if self.role != "record":
            return
        with _lock:
            data = self.load()
            entry = {"document": document, "user_type": user_type}
            if entry not in data["projects"]:
                data["projects"].append(entry)
            self._save(data)

    def record_answers(self, questions: list[dict], answers: dict[str, str]) -> None:
        """질문 문구 → 실제로 반영된 답(빈 답은 기본값)을 저장한다."""
        if self.role != "record":
            return
        with _lock:
            data = self.load()
            for q in questions:
                answer = answers.get(q["id"]) or q.get("default_option")
                if answer:
                    data["answers"][q["question"]] = answer
            self._save(data)

    # ---- 데모 모드 ----

    def answers_for(self, questions: list[dict]) -> dict[str, str]:
        """대기 중인 질문 ID → 녹화 당시 답. 녹화에 없는 질문은 빠진다."""
        if self.role != "replay":
            return {}
        recorded = self.load()["answers"]
        return {q["id"]: recorded[q["question"]] for q in questions if q["question"] in recorded}
