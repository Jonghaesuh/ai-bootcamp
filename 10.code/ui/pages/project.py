"""진행 화면(실시간 단계·확인 질문 카드) + 결과 탭 + 대화로 수정 + 승인."""

import io

import pandas as pd
import streamlit as st
from common import STATUS_LABELS, USER_TYPES, api_ok, client, format_trace, label, run_stream

FIELD_LABELS = {
    "business_goal": "비즈니스 목표",
    "prediction_target": "예측 대상",
    "analysis_unit": "분석 단위",
    "prediction_time": "예측 시점",
    "target_definition": "타깃 정의",
    "success_metric": "성공 지표",
}
CHECK_ICONS = {"pass": "🟢 통과", "warn": "🟡 경고", "fail": "🔴 미통과"}
DIRECT = "직접 입력"

if not api_ok():
    st.stop()

projects = client().list_projects()
if not projects:
    st.info("프로젝트가 없습니다. 새 프로젝트를 만들어 주세요.")
    st.stop()

ids = [p["id"] for p in projects]
current = st.session_state.get("project_id")
pid = st.sidebar.selectbox(
    "프로젝트",
    ids,
    index=ids.index(current) if current in ids else 0,
    # 제목이 같은 프로젝트가 여러 개일 수 있어 ID를 붙인다. 표시 문자열이 겹치면 Streamlit이 다른 항목을 고른다.
    format_func=lambda i: f"{next(p['title'] for p in projects if p['id'] == i)} · {i}",
)
st.session_state["project_id"] = pid

detail = client().project(pid)
project, state = detail["project"], detail["state"]
planner = project["user_type"] == "planner"

st.title(project["title"])
st.caption(
    f"`{pid}` · {USER_TYPES[project['user_type']]} · 상태: **{STATUS_LABELS[project['status']]}**"
    + (" · ✅ 승인됨" if project["approved"] else "")
)
with st.expander("요구사항 정의서 원문"):
    st.markdown(project["document"])

# ---- 실행 ----

if st.session_state.pop("autorun", None) == pid and project["status"] == "new":
    run_stream(client().run(pid))
if project["status"] == "new":
    if st.button("▶ 실행", type="primary"):
        run_stream(client().run(pid))
    st.stop()
if project["status"] == "error":
    st.error(f"실행 중 오류: {project['error']}")
    if st.button("이어서 실행"):
        run_stream(client().resume(pid))
if project["status"] == "running":
    st.info("다른 창에서 실행 중입니다. 잠시 후 새로고침하세요.")

# ---- 확인 질문 카드 ----

interrupt = state.get("interrupt")
if interrupt:
    st.subheader("❓ 확인이 필요합니다")
    st.caption(
        "고객에게 그대로 전달할 수 있는 질문입니다."
        if planner
        else "답하지 않은 항목은 기본값으로 진행합니다."
    )
    demo_answers = state.get("demo_answers") or {}
    if demo_answers:
        st.info(
            "🎬 데모 모드: 녹화 당시 답변이 채워져 있습니다. 그대로 '답변하고 계속'을 누르면 녹화가 재생됩니다."
        )
    with st.form("answers"):
        answers = {}
        for q in interrupt["questions"]:
            recorded = demo_answers.get(q["id"])
            with st.container(border=True):
                st.markdown(f"**{q['question']}**")
                if q.get("reason"):
                    st.caption(f"이유: {q['reason']}")
                options = q.get("options") or []
                if options:
                    default = recorded or q.get("default_option")
                    if default in options:
                        index = options.index(default)
                    else:
                        index = (
                            len(options) if recorded else 0
                        )  # 녹화 답이 선택지에 없으면 직접 입력
                    choice = st.radio(
                        "답변", [*options, DIRECT], index=index, key=f"r_{q['id']}", horizontal=True
                    )
                    custom = st.text_input(
                        "직접 입력",
                        value=recorded if recorded and recorded not in options else "",
                        key=f"t_{q['id']}",
                    )
                    answers[q["id"]] = custom if choice == DIRECT else choice
                else:
                    answers[q["id"]] = st.text_input(
                        "답변",
                        value=recorded or "",
                        key=f"t_{q['id']}",
                        placeholder=q.get("default_option") or "",
                    )
        if st.form_submit_button("답변하고 계속", type="primary"):
            run_stream(client().resume(pid, answers={k: v for k, v in answers.items() if v}))

# ---- 결과 탭 ----

spec, mapping = state.get("requirement_spec"), state.get("data_mapping")
design, review = state.get("dataset_design"), state.get("review_report")
artifacts = client().artifacts(pid) if state.get("artifacts") else {}

tab_names = [
    "요약",
    "요구사항 명세",
    "데이터 매핑",
    "설계·SQL",
    "검증 리포트",
    "추적표",
    "진행 로그",
]
tabs = dict(zip(tab_names, st.tabs(tab_names), strict=True))

with tabs["요약"]:
    if "summary.md" in artifacts:
        st.markdown(client().download(artifacts["summary.md"]).decode("utf-8"))
    else:
        st.info("산출물이 아직 없습니다.")
    if review:
        c = st.columns(4)
        c[0].metric("검증", "통과" if review["passed"] else "미통과")
        c[1].metric("모델링 실익", review["feasibility"])
        c[2].metric("행 수", f"{(design or {}).get('row_count') or 0:,}")
        ratio = (design or {}).get("target_ratio")
        c[3].metric("타깃 비율", f"{ratio:.2%}" if ratio is not None else "-")
    if artifacts:
        st.markdown("**산출물 다운로드**")
        cols = st.columns(4)
        for i, (name, path) in enumerate(artifacts.items()):
            cols[i % 4].download_button(
                name, client().download(path), file_name=name, key=f"dl_{name}"
            )

with tabs["요구사항 명세"]:
    if spec:
        st.table(
            pd.DataFrame(
                [{"항목": v, "내용": spec.get(k) or "(미정)"} for k, v in FIELD_LABELS.items()]
            )
        )
        if spec.get("constraints"):
            st.markdown("**제약·가정**\n" + "\n".join(f"- {c}" for c in spec["constraints"]))
        st.dataframe(pd.DataFrame(spec.get("requirement_items", [])), hide_index=True)
        if state.get("answered"):
            st.markdown("**확인 질문 답변**")
            for a in state["answered"]:
                st.markdown(f"- {a['question']} → **{a['answer']}**")
        if spec.get("rationale") and not planner:
            st.caption(spec["rationale"])

with tabs["데이터 매핑"]:
    if mapping:
        key, ref = mapping.get("analysis_key"), mapping.get("reference_time")
        if key:
            st.markdown(f"- 분석 키: `{key['table']}.{key['column']}`")
        if ref:
            st.markdown(f"- 기준 시점 후보: `{ref['table']}.{ref['column']}`")
        rows = [
            {
                "요구사항": m["requirement_id"],
                "역할": m["role"],
                "컬럼": ", ".join(f"{c['table']}.{c['column']}" for c in m["columns"]),
                "근거": ", ".join(m.get("evidence", [])),
                "메모": m.get("note", ""),
            }
            for m in mapping.get("items", [])
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if mapping.get("join_paths"):
            st.markdown(
                "**조인 경로**\n"
                + "\n".join(
                    f"- `{j['left']['table']}.{j['left']['column']}` = `{j['right']['table']}.{j['right']['column']}`"
                    for j in mapping["join_paths"]
                )
            )
        if mapping.get("rejected_refs"):
            st.warning(f"스키마에 없어 제거한 참조: {', '.join(mapping['rejected_refs'])}")
        if mapping.get("unmapped_requirements"):
            st.warning(f"매핑하지 못한 요구사항: {', '.join(mapping['unmapped_requirements'])}")
        for p in mapping.get("proxy_suggestions", []):
            st.info(f"대체 데이터 제안: {p}")

with tabs["설계·SQL"]:
    if design:
        t = design["target"]
        st.markdown(
            f"- 분석 단위: **{design['analysis_unit']}** (`{design['entity_key']}`)\n"
            f"- 기준 시점: `{design['reference_time']}`\n"
            f"- 타깃: {t['description']} (**{t['window_days']}일**, 조건: {t['positive_condition']})\n"
            f"- 분할: {design['split']['method']} (train < {design['split'].get('train_end')} ≤ val < {design['split'].get('val_end')} ≤ test)"
        )
        st.dataframe(pd.DataFrame(design.get("features", [])), hide_index=True)
        if design.get("population_filters"):
            st.markdown(
                "**모집단 필터**\n" + "\n".join(f"- {f}" for f in design["population_filters"])
            )
        st.markdown("**설계 근거**")
        st.write(design.get("rationale", ""))
        if not planner or st.toggle("SQL 보기"):
            st.code(design["sql"], language="sql")
        if design.get("sql_error"):
            st.error(f"SQL 오류: {design['sql_error']}")

with tabs["검증 리포트"]:
    if review:
        st.markdown(review.get("summary", ""))
        st.dataframe(
            pd.DataFrame(
                [
                    {"검사": c["name"], "판정": CHECK_ICONS[c["status"]], "근거": c["evidence"]}
                    for c in review["checks"]
                ]
            ),
            hide_index=True,
        )
        if eda := review.get("eda"):
            c1, c2 = st.columns(2)
            c1.markdown("**분할별 행 수·타깃 비율**")
            c1.dataframe(
                pd.DataFrame(
                    {"행 수": eda["split_counts"], "타깃 비율": eda["split_target_ratio"]}
                ),
            )
            c2.markdown("**피처 결측률·타깃 상관**")
            c2.dataframe(
                pd.DataFrame({"결측률": eda["null_ratio"], "타깃 상관": eda["target_correlation"]}),
            )
        if state.get("rework_counts"):
            st.caption(f"재작업 횟수: {state['rework_counts']}")

with tabs["추적표"]:
    if "traceability.csv" in artifacts:
        csv = client().download(artifacts["traceability.csv"]).decode("utf-8")
        st.dataframe(pd.read_csv(io.StringIO(csv)), hide_index=True)
    else:
        st.info("산출물이 생성되면 표시됩니다.")

with tabs["진행 로그"]:
    for e in state.get("trace") or []:
        st.markdown(format_trace(e))

# ---- 대화로 수정 · 승인 ----

if project["status"] == "done":
    st.divider()
    left, right = st.columns([3, 1])
    left.markdown(
        '**대화로 수정** — 예: "타깃 기간을 60일로 바꿔줘", "결제 테이블도 써줘". 영향받는 단계만 다시 실행합니다.'
    )
    passed = bool(review and review.get("passed"))
    if not passed:
        right.caption("검증을 통과해야 승인할 수 있습니다.")
    elif not project["approved"] and right.button(
        "✅ 설계 승인", help="승인한 설계는 다음 프로젝트의 참고 사례로 저장됩니다."
    ):
        res = client().approve(pid)
        st.toast(f"승인 완료: 사례 문서 {len(res['case_documents'])}개 저장")
        st.rerun()
    if request := st.chat_input("수정 요청을 입력하세요"):
        st.chat_message("user").write(request)
        run_stream(client().resume(pid, change_request=request))

st.sidebar.caption(f"마지막 단계: {label((state.get('trace') or [{}])[-1].get('agent', '-'))}")
