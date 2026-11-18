"""LangGraph 실행을 프로젝트 단위로 감싼다: 시작·재개·대화형 수정, 진행 이벤트 스트림, 상태 요약."""

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langgraph.types import Command

from req2data.graph import build_graph, new_project_state
from req2data.service.demo import DemoInputs
from req2data.service.projects import Project, ProjectRepository

log = logging.getLogger(__name__)

STATE_KEYS = (
    "requirement_spec",
    "data_mapping",
    "dataset_design",
    "review_report",
    "artifacts",
    "pending_questions",
    "answered",
    "rework_counts",
    "design_version",
    "trace",
)


class ProjectBusy(RuntimeError):
    pass


class InvalidTransition(RuntimeError):
    pass


class GraphRunner:
    def __init__(
        self, deps, checkpointer, projects: ProjectRepository, demo: DemoInputs | None = None
    ):
        self.graph = build_graph(deps, checkpointer=checkpointer)
        self.projects = projects
        self.demo = demo or DemoInputs(Path("demo_inputs.json"))
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    @staticmethod
    def config(project_id: str) -> dict:
        return {"configurable": {"thread_id": project_id}}

    # ---- 시작·재개 ----

    def start(self, project: Project) -> Iterator[dict]:
        if self.graph.get_state(self.config(project.id)).values:
            raise InvalidTransition("이미 실행한 프로젝트입니다. 재개(resume)를 사용하세요.")
        payload = new_project_state(project.id, project.document, project.user_type)
        events = self._stream(project.id, payload)
        self._or_close(
            events, lambda: self.demo.record_project(project.document, project.user_type)
        )
        return events

    def resume(
        self,
        project: Project,
        answers: dict[str, str] | None = None,
        change_request: str | None = None,
    ) -> Iterator[dict]:
        snap = self.graph.get_state(self.config(project.id))
        if not snap.values:
            raise InvalidTransition("아직 실행하지 않은 프로젝트입니다.")
        if snap.interrupts:
            if change_request:
                raise InvalidTransition("확인 질문에 먼저 답해 주세요.")
            events = self._stream(project.id, Command(resume={"answers": answers or {}}))
            self._or_close(
                events,
                lambda: self.demo.record_answers(
                    snap.interrupts[0].value["questions"], answers or {}
                ),
            )
            return events
        if snap.next and not change_request:
            return self._stream(project.id, None)  # 중단된 실행을 마지막 체크포인트부터 이어서
        if not change_request:
            raise InvalidTransition(
                "완료된 프로젝트는 수정 요청(change_request)으로만 재개할 수 있습니다."
            )
        self.projects.update(project.id, approved=False)
        return self._stream(project.id, {"change_request": change_request})

    def _stream(self, project_id: str, payload: Any) -> Iterator[dict]:
        lock = self._lock_for(project_id)
        if not lock.acquire(blocking=False):
            raise ProjectBusy("이 프로젝트는 이미 실행 중입니다.")
        events = self._events(project_id, payload, lock)
        # 시작하지 않은 제너레이터는 닫아도 finally가 돌지 않는다. 첫 yield까지 진행해 두면
        # 이후 close()·GC(SSE 본문을 끝내 읽지 않은 경우 포함)에서 finally가 잠금을 푼다.
        next(events)
        return events

    @staticmethod
    def _or_close(events: Iterator[dict], action) -> None:
        """실행 전 부수 작업(데모 녹화)이 실패하면 스트림을 닫아 잠금을 풀고 예외를 그대로 올린다."""
        try:
            action()
        except BaseException:
            events.close()  # type: ignore[attr-defined]
            raise

    def _events(self, project_id: str, payload: Any, lock: threading.Lock) -> Iterator[dict]:
        config = self.config(project_id)
        started = False
        try:
            yield {}  # _stream이 소비하는 준비 단계 (클라이언트에 나가지 않음)
            started = True
            self.projects.update(project_id, status="running", error=None)
            for chunk in self.graph.stream(payload, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        continue
                    for event in (update or {}).get("trace", []):
                        yield {"event": "trace", "data": event}
                    yield {"event": "node", "data": {"node": node}}
            snap = self.graph.get_state(config)
            if snap.interrupts:
                self.projects.update(project_id, status="waiting")
                yield {"event": "interrupt", "data": snap.interrupts[0].value}
            else:
                self.projects.update(project_id, status="done")
                yield {"event": "done", "data": {"artifacts": snap.values.get("artifacts") or {}}}
        except (
            GeneratorExit
        ):  # 클라이언트 연결 종료: 체크포인트는 남아 있어 resume으로 이어갈 수 있다
            if started:
                self.projects.update(
                    project_id,
                    status="error",
                    error="실행이 중단되었습니다. 재개하면 이어서 실행합니다.",
                )
            raise
        except Exception as exc:  # 실패도 이벤트로 알리고 상태에 남긴다
            log.exception("프로젝트 %s 실행 실패", project_id)
            self.projects.update(project_id, status="error", error=str(exc))
            yield {"event": "error", "data": {"message": str(exc)}}
        finally:
            lock.release()

    def _lock_for(self, project_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(project_id, threading.Lock())

    # ---- 조회 ----

    def state(self, project_id: str) -> dict[str, Any]:
        snap = self.graph.get_state(self.config(project_id))
        values = snap.values or {}
        state = {k: values.get(k) for k in STATE_KEYS}
        state["interrupt"] = snap.interrupts[0].value if snap.interrupts else None
        state["demo_answers"] = (
            self.demo.answers_for(state["interrupt"]["questions"]) if state["interrupt"] else {}
        )
        return state
