from collections.abc import Iterator

import streamlit as st
from api_client import ApiError, Req2DataClient

from req2data.config import get_settings

AGENT_LABELS = {
    "supervisor": "Supervisor",
    "requirement_analyst": "요구사항 분석",
    "data_explorer": "데이터 탐색",
    "dataset_designer": "데이터셋 설계",
    "reviewer": "검증",
    "report": "리포트",
    "ask_user": "확인 질문",
    "end": "종료",
}
STATUS_LABELS = {
    "new": "실행 전",
    "running": "실행 중",
    "waiting": "확인 대기",
    "done": "완료",
    "error": "오류",
}
USER_TYPES = {
    "practitioner": "실무자 (데이터 분석가·사이언티스트)",
    "planner": "기획자 (PM·컨설턴트)",
}


@st.cache_resource
def client() -> Req2DataClient:
    return Req2DataClient(get_settings().api_base_url)


def label(agent: str) -> str:
    return AGENT_LABELS.get(agent, agent)


def format_trace(e: dict) -> str:
    agent, kind = e.get("agent", ""), e.get("event", "")
    if kind == "route":
        return f"🧭 **Supervisor** → {label(e['next'])} — {e.get('reason', '')}"
    if kind == "change_scope":
        return f"✏️ 수정 범위 판단 → {label(e['restart_from'])}부터 재실행 ({', '.join(e.get('changed_fields', []))})"
    if kind == "answered":
        return f"💬 답변 반영: {e.get('answers')}"
    if kind == "fallback":
        if "데모 녹화와 다른 입력" in (e.get("error") or ""):
            return f"🎬 **{label(agent)}**: 데모 녹화와 다른 입력이라 기록된 응답이 없어 규칙 기반 기본값으로 진행"
        return f"⚠️ **{label(agent)}**: LLM 출력 실패로 규칙 기반 기본값 사용"
    details = []
    for key, text in (
        ("questions", "질문 {}개"),
        ("tables", "테이블 {}"),
        ("rejected", "제거한 참조 {}"),
        ("version", "설계 v{}"),
        ("row_count", "{}행"),
        ("target_ratio", "타깃 비율 {}"),
        ("sql_error", "SQL 오류: {}"),
        ("failed", "미통과 {}"),
        ("feasibility", "모델링 실익 {}"),
    ):
        if e.get(key) not in (None, [], ""):
            details.append(text.format(e[key]))
    if agent == "reviewer":
        details.insert(0, "통과" if e.get("passed") else "미통과")
    return f"✅ **{label(agent)}** 완료" + (f" — {', '.join(details)}" if details else "")


def run_stream(events: Iterator[tuple[str, dict]]) -> None:
    """SSE 이벤트를 진행 화면에 실시간으로 보여 주고, 끝나면 페이지를 새로 그린다."""
    with st.status("Agent가 작업 중입니다...", expanded=True) as status:
        try:
            for event, data in events:
                if event == "trace":
                    st.markdown(format_trace(data))
                elif event == "node" and data.get("node") not in ("supervisor",):
                    status.update(label=f"{label(data['node'])} 단계 완료, 다음 단계 진행 중...")
                elif event == "interrupt":
                    status.update(label="확인 질문에 답해 주세요", state="complete", expanded=False)
                elif event == "done":
                    status.update(label="완료되었습니다", state="complete", expanded=False)
                elif event == "error":
                    status.update(label="오류가 발생했습니다", state="error")
                    st.error(data.get("message"))
                    return
        except ApiError as exc:
            status.update(label="요청을 처리할 수 없습니다", state="error")
            st.error(str(exc))
            return
    st.rerun()


def demo_info() -> dict:
    """데모 모드면 녹화 입력 정보, 아니면 {"demo_mode": False}."""
    try:
        return client().demo()
    except Exception:
        return {"demo_mode": False, "projects": []}


def api_ok() -> bool:
    try:
        client().health()
        return True
    except Exception:
        st.error(
            f"API 서버({get_settings().api_base_url})에 연결할 수 없습니다. `make api`로 실행하세요."
        )
        return False
