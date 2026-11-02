"""정의서 하나를 API로 끝까지 실행하고 결과를 요약한다 (브라우저 없이 동작 확인용).

확인 질문에는 데모 녹화 답변이 있으면 그 값으로, 없으면 질문의 기본값으로 답한다.

    python scripts/run_scenario.py data/samples/01_repurchase.md [--api http://localhost:8000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
from api_client import Req2DataClient  # noqa: E402

KEYS = (
    "next",
    "questions",
    "version",
    "row_count",
    "target_ratio",
    "failed",
    "passed",
    "feasibility",
    "error",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--user-type", default="practitioner", choices=["practitioner", "planner"])
    args = parser.parse_args()

    c = Req2DataClient(args.api)
    print(f"API {c.health()}")
    pid = c.create_project(args.document.read_text(encoding="utf-8"), args.user_type)["id"]
    print(f"프로젝트 {pid} 생성: {args.document.name}")
    t0 = time.time()

    def run(events):
        last = None
        for ev, data in events:
            if ev == "trace" and data.get("event") != "route":
                detail = {k: data[k] for k in KEYS if data.get(k) not in (None, [], "")}
                print(
                    f"  [{time.time() - t0:5.1f}s] {data.get('agent')}: {data.get('event')} {detail}"
                )
            elif ev in ("interrupt", "done", "error"):
                last = (ev, data)
        return last

    last = run(c.run(pid))
    while last and last[0] == "interrupt":
        prefill = c.project(pid)["state"].get("demo_answers") or {}
        answers = {}
        for q in last[1]["questions"]:
            answers[q["id"]] = prefill.get(q["id"]) or q.get("default_option") or ""
            print(f"  ? {q['question']}\n    → {answers[q['id']] or '(기본값)'}")
        last = run(c.resume(pid, answers={k: v for k, v in answers.items() if v}))

    if not last or last[0] != "done":
        print(f"실패: {last}")
        return 1
    state = c.project(pid)["state"]
    design, review = state["dataset_design"], state["review_report"]
    print("\n== 결과 ==")
    print(
        f"분석 단위: {design['analysis_unit']} ({design['entity_key']}), 기준 시점: {design['reference_time']}"
    )
    print(f"타깃: {design['target']['description']} ({design['target']['window_days']}일)")
    print(f"행 수: {design['row_count']:,}, 타깃 비율: {design['target_ratio']:.2%}")
    print(f"검증: {'통과' if review['passed'] else '미통과'} (모델링 실익 {review['feasibility']})")
    for check in review["checks"]:
        print(f"  - {check['name']}: {check['status']}  {check['evidence'][:80]}")
    print(f"산출물: {', '.join(c.artifacts(pid))}")
    print(f"소요 시간: {time.time() - t0:.1f}s")
    return 0 if review["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
