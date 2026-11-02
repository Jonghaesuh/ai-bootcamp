from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AgentPrompt:
    """역할 기반 프롬프트. `prompts/<agent>.yaml` 하나에 대응한다."""

    agent: str
    version: int
    persona: str
    responsibilities: list[str] = field(default_factory=list)
    prohibitions: list[str] = field(default_factory=list)
    procedure: list[str] = field(default_factory=list)
    user_type: dict[str, str] = field(default_factory=dict)
    tasks: dict[str, str] = field(default_factory=dict)

    def system(
        self,
        *,
        user_type: str | None = None,
        feedback: list[str] | None = None,
        extra: str | None = None,
    ) -> str:
        parts = [f"# 역할\n{self.persona.strip()}"]
        if self.responsibilities:
            parts.append("## 책임\n" + _bullets(self.responsibilities))
        if self.prohibitions:
            parts.append("## 금지 사항\n" + _bullets(self.prohibitions))
        if self.procedure:
            steps = "\n".join(f"{i}. {s}" for i, s in enumerate(self.procedure, 1))
            parts.append(f"## 사고 절차 (이 순서대로 생각하고 rationale에 요약)\n{steps}")
        if user_type and user_type in self.user_type:
            parts.append(f"## 사용자 유형별 지침\n{self.user_type[user_type].strip()}")
        if feedback:
            parts.append("## 이전 검증 피드백 (같은 실수를 반복하지 말 것)\n" + _bullets(feedback))
        if extra:
            parts.append(extra.strip())
        return "\n\n".join(parts)

    def task(self, name: str, **variables: Any) -> str:
        return self.tasks[name].format(**variables)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


class PromptRegistry:
    def __init__(self, prompt_dir: Path):
        self.prompt_dir = Path(prompt_dir)

    def get(self, agent: str) -> AgentPrompt:
        return _load(self.prompt_dir / f"{agent}.yaml", agent)


@cache
def _load(path: Path, agent: str) -> AgentPrompt:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AgentPrompt(agent=agent, **data)
