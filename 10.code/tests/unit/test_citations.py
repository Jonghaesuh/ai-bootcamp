"""검색 근거 인용: LLM이 적은 인용을 검색 결과와 대조해 검증하고 State·리포트에 싣는다."""

from dataclasses import replace

from langchain_core.documents import Document

from req2data.agents import RequirementAnalyst
from req2data.agents.citations import citation_records, validate_citations
from req2data.graph import new_project_state
from req2data.reporting.render import citations_md
from req2data.schemas import Citation

RETRIEVED = [
    Document(
        id="glossary:재구매", page_content="...", metadata={"source": "glossary", "title": "재구매"}
    )
]


def test_validate_citations_drops_unknown_and_duplicate():
    cites = [
        Citation(doc_id="glossary:재구매", used_for="타깃 정의"),
        Citation(doc_id="glossary:재구매"),
        Citation(doc_id="glossary:없는용어"),
    ]
    kept, rejected = validate_citations(cites, RETRIEVED)
    assert [c.doc_id for c in kept] == ["glossary:재구매"] and rejected == ["glossary:없는용어"]


def test_citation_records_carry_source_and_title():
    kept = [Citation(doc_id="glossary:재구매", used_for="타깃 정의")]
    assert citation_records("requirement_analyst", kept, RETRIEVED) == [
        {
            "agent": "requirement_analyst",
            "doc_id": "glossary:재구매",
            "source": "glossary",
            "title": "재구매",
            "used_for": "타깃 정의",
        }
    ]


def test_requirement_analyst_records_validated_citations(deps, llm, client):
    deps = replace(deps, catalog=client)
    llm.script(
        "RequirementSpec",
        {
            "business_goal": "재구매 예측",
            "citations": [
                {"doc_id": "glossary:재구매", "used_for": "타깃 정의"},
                {"doc_id": "glossary:없는용어"},
            ],
        },
    )
    out = RequirementAnalyst(deps)(
        new_project_state("p1", "첫 구매 고객의 재구매 가능성을 예측한다.")
    )
    assert [c["doc_id"] for c in out["citations"]] == ["glossary:재구매"]
    assert out["trace"][0]["rejected_citations"] == ["glossary:없는용어"]
    assert out["requirement_spec"]["citations"] == [
        {"doc_id": "glossary:재구매", "used_for": "타깃 정의"}
    ]


def test_citations_md_table_and_empty():
    md = citations_md(
        [
            {
                "agent": "requirement_analyst",
                "doc_id": "glossary:재구매",
                "source": "glossary",
                "title": "재구매",
                "used_for": "타깃 정의",
            }
        ]
    )
    assert "## 참고 근거" in md
    assert "| requirement_analyst | 재구매 | glossary | 타깃 정의 |" in md
    assert "인용된 문서가 없습니다" in citations_md([])
