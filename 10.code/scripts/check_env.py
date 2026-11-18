"""실행 환경 점검: LLM·임베딩 호출과 RAG 인덱스를 한 번씩 확인한다. 키 값은 출력하지 않는다.

python scripts/check_env.py           # 호출 점검만 (수 초)
python scripts/check_env.py --full    # 샘플 정의서를 그래프로 끝까지 실행 (수 분)
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from req2data.agents import AgentDeps
from req2data.config import get_settings
from req2data.graph import build_graph, new_project_state
from req2data.llm import get_chat_model
from req2data.rag.embeddings import get_embeddings
from req2data.rag.knowledge_base import KnowledgeBase
from req2data.tools.local_catalog import LocalCatalogClient

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "01_repurchase.md"


LOCAL_HINT = "로컬 제공자(ollama·gemini)는 `uv sync --extra local`로 패키지를 설치하세요."


def step(name: str, fn: Callable[[], Any]) -> bool:
    started = time.time()
    try:
        print(f"[OK]   {name}: {fn()} ({time.time() - started:.1f}초)")
        return True
    except Exception as exc:
        hint = f" → {LOCAL_HINT}" if isinstance(exc, ModuleNotFoundError) else ""
        print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:300]}{hint}")
        return False


def _require(value: Any, what: str) -> Any:
    if value is None:
        raise RuntimeError(f"{what} 준비에 실패해 건너뜁니다.")
    return value


def run_full(deps: AgentDeps, document: str, project_id: str = "check_env") -> dict[str, Any]:
    """정의서 하나를 그래프로 끝까지 실행한다. 확인 질문에는 각 질문의 기본값으로 답한다."""
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": project_id}, "recursion_limit": 100}
    graph.invoke(new_project_state(project_id, document), config)
    for _ in range(10):
        if not graph.get_state(config).next:
            break
        graph.invoke(Command(resume={"answers": {}}), config)  # 빈 답변 → 기본값 (ask_user)
    snap = graph.get_state(config)
    values = snap.values
    design = values.get("dataset_design") or {}
    review = values.get("review_report") or {}
    return {
        "finished": not snap.next,
        "row_count": design.get("row_count") or 0,
        "target_ratio": design.get("target_ratio"),
        "passed": review.get("passed"),
        "failed_checks": [c["name"] for c in review.get("checks", []) if c["status"] == "fail"],
        "citations": len(values.get("citations") or []),
        "fallbacks": [e["agent"] for e in values.get("trace", []) if e.get("event") == "fallback"],
        "artifacts": values.get("artifacts") or {},
    }


def _full_summary(deps: AgentDeps, document: str) -> str:
    out = run_full(deps, document)
    if not out["finished"] or not out["artifacts"]:
        raise RuntimeError(f"끝까지 실행되지 않음: {out}")
    ratio = out["target_ratio"]
    return (
        f"행 수 {out['row_count']:,}, 타깃 비율 {'-' if ratio is None else f'{ratio:.2%}'}, "
        f"검증 {'통과' if out['passed'] else '미통과 ' + str(out['failed_checks'])}, "
        f"인용 {out['citations']}건, 기본값 대체 {out['fallbacks'] or '없음'}, "
        f"산출물 {Path(next(iter(out['artifacts'].values()))).parent}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="샘플 정의서를 끝까지 실행")
    parser.add_argument("--document", type=Path, default=SAMPLE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    s = get_settings()
    print(f"LLM {s.llm_label} | 임베딩 {s.embedding_label} | 데모 모드 {s.demo_mode}")
    made: dict[str, Any] = {}

    def make(key: str, factory: Callable[[], Any]) -> Callable[[], str]:
        def run() -> str:
            made[key] = factory()
            return type(made[key]).__name__

        return run

    results = [
        step("LLM 준비", make("llm", lambda: get_chat_model(s))),
        step("지식 베이스 준비", make("kb", lambda: KnowledgeBase.from_settings(s))),
    ]
    llm, kb = made.get("llm"), made.get("kb")

    @tool
    def describe_table(table: str) -> str:
        """테이블의 컬럼 정보를 조회한다."""
        return "{}"

    def tool_call() -> str:
        calls = (
            _require(llm, "LLM")
            .bind_tools([describe_table])
            .invoke([HumanMessage(content="orders 테이블의 컬럼을 조회하세요.")])
            .tool_calls
        )
        if not calls:
            raise RuntimeError("모델이 도구를 호출하지 않았습니다.")
        return f"{calls[0]['name']}({calls[0]['args']})"

    def index() -> str:
        if not _require(kb, "지식 베이스").ensure_current():
            raise RuntimeError("인덱스를 다시 만들지 못했습니다 (임베딩 호출 실패).")
        return f"{kb.index.count()}개 문서, {kb.index.built_with()}"

    results += [
        step(
            "채팅 호출",
            lambda: str(
                _require(llm, "LLM")
                .invoke([HumanMessage(content="한 단어로만 답하세요: 안녕")])
                .content
            )[:40],
        ),
        step(
            "JSON 출력",
            lambda: str(
                get_chat_model(s, json_mode=True)
                .invoke([HumanMessage(content='{"ok": true} 형태의 JSON 객체 하나만 출력하세요.')])
                .content
            )[:60],
        ),
        step("Tool 호출", tool_call),
        step("임베딩 호출", lambda: f"차원 {len(get_embeddings(s).embed_query('재구매 고객'))}"),
        step("RAG 인덱스", index),
    ]
    if args.full:

        def full() -> str:
            catalog = LocalCatalogClient.from_settings(s, kb=_require(kb, "지식 베이스"))
            deps = AgentDeps.from_settings(s, catalog=catalog)
            return _full_summary(deps, args.document.read_text(encoding="utf-8"))

        results.append(step(f"전체 실행 ({args.document.name})", full))
    print("결과: " + ("모두 정상" if all(results) else "실패 항목 있음"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
