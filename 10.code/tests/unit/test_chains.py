from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableSequence

from req2data.llm.chains import rag_chain, retrieval_chain, structured_chain
from req2data.schemas import GroundedAnswer, RouteDecision
from req2data.tools.catalog import CatalogRetriever


def test_structured_chain_is_lcel_and_parses(llm):
    llm.script("RouteDecision", {"next": "end", "reason": "완료"})
    chain = structured_chain(llm, RouteDecision)
    assert isinstance(chain, RunnableSequence)
    out = chain.invoke({"system": "역할", "messages": [HumanMessage(content="과제")]})
    assert out == RouteDecision(next="end", reason="완료")
    system, human = llm.calls[0]["messages"]
    assert system.content.startswith("역할\n\n## 출력 형식\n출력 스키마: RouteDecision")
    assert human.content == "과제"


def test_prose_and_fenced_json_still_parse(llm):
    llm.script("RouteDecision", '설명입니다.\n```json\n{"next": "end", "reason": "r"}\n```\n끝')
    assert structured_chain(llm, RouteDecision).invoke({"system": "", "messages": []}).next == "end"


def test_retrieval_chain_returns_docs_and_context(client):
    out = retrieval_chain(CatalogRetriever(client=client, source="glossary", k=2)).invoke("재구매")
    assert out["docs"] and f"[{out['docs'][0].id}]" in out["context"]


def test_rag_chain_grounds_answer_in_retrieved_context(llm, client):
    llm.script("GroundedAnswer", {"answer": "customer_unique_id 기준", "citations": []})
    chain = rag_chain(
        CatalogRetriever(client=client, source="glossary", k=2),
        llm,
        GroundedAnswer,
        "문맥만 근거로 답하세요.",
    )
    assert chain.invoke("재구매는 어떻게 정의하나").answer == "customer_unique_id 기준"
    assert "[glossary:" in llm.calls[0]["messages"][-1].content
