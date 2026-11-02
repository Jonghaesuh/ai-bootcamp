"""Req2Data Agent UI: 새 프로젝트 / 프로젝트(진행·질문·결과·대화 수정) / 카탈로그 탐색."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))  # api_client, common을 페이지에서 import

st.set_page_config(page_title="Req2Data Agent", page_icon="🧭", layout="wide")

pages = st.navigation(
    [
        st.Page("pages/new_project.py", title="새 프로젝트", icon="📝", default=True),
        st.Page("pages/project.py", title="프로젝트", icon="🧭"),
        st.Page("pages/catalog.py", title="데이터 카탈로그", icon="📚"),
    ]
)
pages.run()
