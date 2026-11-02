"""데이터 카탈로그 탐색: 하이브리드 검색, 테이블·컬럼(프로파일), 조인, 용어집."""

import pandas as pd
import streamlit as st
from common import api_ok, client

st.title("📚 데이터 카탈로그")
if not api_ok():
    st.stop()

SOURCES = {
    "전체": None,
    "카탈로그": "catalog",
    "용어집": "glossary",
    "설계 가이드": "guide",
    "과거 사례": "case",
}

st.subheader("검색")
c1, c2 = st.columns([4, 1])
query = c1.text_input("검색어", placeholder="예: 재구매 고객 키, 배송 지연, 리뷰 점수 누수")
source = c2.selectbox("소스", list(SOURCES))
if query:
    hits = client().search(query, SOURCES[source], k=8)
    if not hits:
        st.info("결과가 없습니다.")
    for h in hits:
        with st.expander(f"{h['doc_id']}  ·  점수 {h['score']:.4f}  ·  순위 {h['ranks']}"):
            st.text(h["text"])

st.divider()
st.subheader("테이블")
tables = client().tables()
st.dataframe(pd.DataFrame(tables), hide_index=True)
name = st.selectbox("테이블 상세", [t["table"] for t in tables])
if name:
    desc = client().table(name)
    st.markdown(f"**{desc['table']}** — {desc['description']}  \n행 단위: {desc.get('grain', '')}")
    rows = []
    for c in desc["columns"]:
        p = c.get("profile") or {}
        rows.append(
            {
                "컬럼": c["name"],
                "타입": c["type"],
                "설명": c["description"],
                "키": "PK" if c["is_pk"] else (f"FK→{c['fk']}" if c["fk"] else ""),
                "시점 순서": c.get("time_rank"),
                "값 확정 시점": c.get("available_at"),
                "결측률": p.get("null_ratio"),
                "고유값": p.get("distinct"),
                "범위": f"{p['min']} ~ {p['max']}" if p.get("min") is not None else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if desc.get("joins"):
        st.markdown(
            "**조인**\n" + "\n".join(f"- `{j['left']}` = `{j['right']}`" for j in desc["joins"])
        )

st.divider()
with st.expander("비즈니스 용어집"):
    st.dataframe(pd.DataFrame(client().glossary()), hide_index=True)

with st.expander("관리"):
    reload_data = st.checkbox("CSV 재적재·프로파일까지 다시 하기")
    if st.button("인덱스 재구축"):
        with st.spinner("재구축 중..."):
            st.json(client().reindex(reload_data))
