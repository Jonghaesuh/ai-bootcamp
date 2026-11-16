import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from req2data.agents.citations import citation_records, validate_citations
from req2data.agents.fewshot import FewShotSelector
from req2data.agents.prompts import PromptRegistry
from req2data.config import Settings, get_settings
from req2data.graph.state import Req2DataState
from req2data.llm import StructuredResult, get_chat_model, invoke_structured
from req2data.llm.chains import retrieval_chain
from req2data.tools.catalog import CatalogClient, CatalogRetriever

log = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    llm: BaseChatModel  # 도구 호출·자유 텍스트
    json_llm: BaseChatModel  # Structured Output 전용
    prompts: PromptRegistry
    fewshots: FewShotSelector
    policy: dict[str, Any]
    catalog: CatalogClient | None = None
    artifact_dir: Path = Path("./storage/artifacts")

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, catalog: CatalogClient | None = None
    ) -> "AgentDeps":
        s = settings or get_settings()
        policy = s.policy
        return cls(
            llm=get_chat_model(s),
            json_llm=get_chat_model(s, json_mode=True),
            prompts=PromptRegistry(s.prompt_dir),
            fewshots=FewShotSelector(
                s.prompt_dir / "fewshot",
                k=policy.get("rag", {}).get("fewshot_k", 2),
                exclude_domains=tuple(policy.get("fewshot", {}).get("exclude_domains", [])),
            ),
            policy=policy,
            catalog=catalog,
            artifact_dir=s.artifact_dir,
        )

    def cfg(self, path: str, default: Any = None) -> Any:
        node: Any = self.policy
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def as_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    return json.dumps(value, ensure_ascii=False, default=str, indent=1)


class BaseAgent:
    """역할 프롬프트 + Structured Output + Reflection + (선택) ReAct를 묶은 공통 골격."""

    name: str

    def __init__(self, deps: AgentDeps):
        self.deps = deps
        self.prompt = deps.prompts.get(self.name)

    @property
    def catalog(self) -> CatalogClient:
        if self.deps.catalog is None:
            raise RuntimeError(f"{self.name}: CatalogClient가 연결되지 않았습니다.")
        return self.deps.catalog

    def system(self, state: Req2DataState, extra: str | None = None) -> str:
        return self.prompt.system(
            user_type=state.get("user_type"),
            feedback=(state.get("feedback") or {}).get(self.name),
            extra=extra,
        )

    def structured[T: BaseModel](
        self, system: str, task: str, schema: type[T], fallback
    ) -> StructuredResult[T]:
        return invoke_structured(
            self.deps.json_llm,
            [SystemMessage(content=system), HumanMessage(content=task)],
            schema,
            retries=self.deps.cfg("workflow.structured_output_retries", 1),
            fallback=fallback,
        )

    def react(self, system: str, task: str, tools: list[BaseTool]) -> str:
        """도구로 추가 조사하고 발견한 사실을 메모로 돌려준다. 실패해도 설계를 막지 않는다."""
        if not self.deps.cfg("workflow.react_enabled", True) or not tools:
            return "(추가 조사 생략)"
        from langchain.agents import create_agent
        from langgraph.errors import GraphRecursionError

        max_steps = self.deps.cfg("workflow.max_react_steps", 12)
        agent = create_agent(self.deps.llm, tools, system_prompt=system)
        messages: list[BaseMessage] = []
        try:
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=task)]},
                {"recursion_limit": max_steps * 2 + 1},
                stream_mode="values",
            ):
                messages = chunk["messages"]
        except GraphRecursionError:
            log.info("%s: ReAct 단계 한도(%d) 도달", self.name, max_steps)
        except Exception as exc:
            log.warning("%s: ReAct 실패: %s", self.name, exc)
            return f"(추가 조사 실패: {exc})"
        return _summarize_react(messages)

    def knowledge(
        self,
        query: str,
        source: str,
        k: int = 3,
        title: str = "",
        into: list[Document] | None = None,
    ) -> str:
        """RAG 검색 결과를 프롬프트용 블록으로. 검색한 문서는 인용 검증에 쓰도록 `into`에 모은다.
        카탈로그가 없거나 검색이 실패하면 빈 문자열."""
        if self.deps.catalog is None or not query.strip():
            return ""
        chain = retrieval_chain(CatalogRetriever(client=self.deps.catalog, source=source, k=k))
        try:
            found = chain.invoke(query)
        except Exception as exc:
            log.warning("%s: %s 검색 실패: %s", self.name, source, exc)
            return ""
        if not found["docs"]:
            return ""
        if into is not None:
            into.extend(found["docs"])
        return (
            f"## {title or source} (검색 결과, 근거로 쓴 문서는 대괄호 안의 doc_id를 citations에 적을 것)\n"
            f"{found['context']}"
        )

    def cite(self, value: Any, retrieved: list[Document]) -> tuple[list[dict], list[str]]:
        """LLM 출력의 citations를 검색 문서와 대조한다. (State에 실을 기록, 버린 doc_id)를 돌려준다."""
        kept, rejected = validate_citations(value.citations, retrieved)
        value.citations = kept
        if rejected:
            log.info("%s: 검색되지 않은 문서 인용 제거: %s", self.name, rejected)
        return citation_records(self.name, kept, retrieved), rejected

    def result_event(self, result: StructuredResult, **detail: Any) -> dict:
        return {
            "agent": self.name,
            "event": "fallback" if result.used_fallback else "done",
            "attempts": result.attempts,
            **({"error": result.error} if result.used_fallback else {}),
            **detail,
        }


def _summarize_react(messages: list[BaseMessage], limit: int = 600) -> str:
    lines = []
    for m in messages:
        if isinstance(m, ToolMessage):
            lines.append(f"- [{m.name}] {str(m.content)[:limit]}")
    final = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None
    )
    if final and final.content:
        lines.append(f"\n결론:\n{final.content}")
    return "\n".join(lines) or "(추가 조사 결과 없음)"
