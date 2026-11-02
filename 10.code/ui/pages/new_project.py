import streamlit as st
from common import STATUS_LABELS, USER_TYPES, api_ok, client, demo_info

from req2data.config import get_settings

st.title("📝 새 프로젝트")
st.caption(
    "요구사항 정의서를 올리면 Agent가 요구사항 구조화 → 데이터 매핑 → 데이터셋 설계 → 검증까지 진행합니다."
)

if not api_ok():
    st.stop()

samples = {p.stem: p for p in sorted((get_settings().data_dir / "samples").glob("*.md"))}

# 데모 모드: 재생은 녹화와 입력이 같을 때만 맞으므로 녹화된 정의서·사용자 유형을 미리 고른다
demo = demo_info()
recorded = demo["projects"][0] if demo.get("demo_mode") and demo.get("projects") else None
recorded_sample = next(
    (
        stem
        for stem, path in samples.items()
        if recorded and path.read_text(encoding="utf-8") == recorded["document"]
    ),
    None,
)
if demo.get("demo_mode"):
    st.info(
        "🎬 데모 모드: LLM 없이 녹화된 응답을 재생합니다. 녹화와 같은 정의서·사용자 유형·답변일 때만 재생되며, "
        "아래 입력과 확인 질문 답변이 녹화 값으로 채워져 있습니다."
        if recorded
        else "🎬 데모 모드지만 녹화된 입력 정보가 없습니다 (demo/recordings/demo_inputs.json)."
    )

inputs = ["직접 입력", "파일 업로드", "샘플 선택"]
source = st.radio("정의서 입력", inputs, index=2 if recorded_sample else 0, horizontal=True)
text = ""
if source == "직접 입력":
    text = st.text_area(
        "요구사항 정의서",
        value=recorded["document"] if recorded and not recorded_sample else "",
        height=260,
        placeholder="예: 첫 구매 고객 중 재구매 가능성이 낮은 고객에게 쿠폰을 보내고 싶다...",
    )
elif source == "파일 업로드":
    uploaded = st.file_uploader("정의서 파일 (.md, .txt)", type=["md", "txt"])
    if uploaded:
        text = uploaded.getvalue().decode("utf-8")
        with st.expander("내용 보기"):
            st.markdown(text)
elif samples:
    names = list(samples)
    name = st.selectbox(
        "샘플 정의서", names, index=names.index(recorded_sample) if recorded_sample else 0
    )
    text = samples[name].read_text(encoding="utf-8")
    with st.container(border=True, height=260):
        st.text(text)
else:
    st.info("data/samples/에 샘플 정의서가 없습니다.")

types = list(USER_TYPES)
user_type = st.radio(
    "사용자 유형",
    types,
    index=types.index(recorded["user_type"]) if recorded else 0,
    format_func=USER_TYPES.get,
    horizontal=True,
)
title = st.text_input("프로젝트 이름 (비우면 정의서 첫 줄)")

if st.button("프로젝트 만들고 실행", type="primary"):
    if len(text.strip()) < 10:
        st.warning("정의서 내용을 입력해 주세요.")
    else:
        project = client().create_project(text, user_type, title or None)
        st.session_state["project_id"] = project["id"]
        st.session_state["autorun"] = project["id"]
        st.switch_page("pages/project.py")

st.divider()
st.subheader("최근 프로젝트")
projects = client().list_projects()
if not projects:
    st.info("아직 프로젝트가 없습니다.")
for p in projects[:10]:
    cols = st.columns([5, 2, 2, 1])
    cols[0].markdown(
        f"**{p['title']}**  \n`{p['id']}` · {USER_TYPES[p['user_type']].split(' ')[0]}"
    )
    cols[1].markdown(
        STATUS_LABELS.get(p["status"], p["status"]) + (" · ✅ 승인" if p["approved"] else "")
    )
    cols[2].caption(p["updated_at"].replace("T", " ")[:16])
    if cols[3].button("열기", key=f"open_{p['id']}"):
        st.session_state["project_id"] = p["id"]
        st.switch_page("pages/project.py")
