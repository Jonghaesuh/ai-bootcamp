from typing import Any

from req2data.agents.base import BaseAgent, as_json
from req2data.graph.state import Req2DataState, load
from req2data.schemas import ColumnRef, DataMapping, JoinStep, RequirementSpec
from req2data.tools.catalog import build_catalog_tools, schema_index

MAX_TABLES = 8
MAX_KEY_PROFILES = 12


def requirement_queries(spec: RequirementSpec) -> list[str]:
    fields = [
        spec.prediction_target,
        spec.analysis_unit,
        spec.prediction_time,
        spec.target_definition,
    ]
    return [i.text for i in spec.requirement_items] + [f for f in fields if f]


class DataExplorer(BaseAgent):
    """고정 선행 조회 → (선택) ReAct 추가 조사 → DataMapping → 스키마 대조로 환각 제거."""

    name = "data_explorer"

    def __call__(self, state: Req2DataState) -> dict:
        spec = load(state, "requirement_spec", RequirementSpec)
        assert spec is not None, "requirement_spec이 필요합니다."
        prefetch = self.prefetch(spec)
        requirements = as_json(spec.model_copy(update={"open_questions": []}))
        prefetch_text = as_json(prefetch)
        system = self.system(state)

        notes = self.react(
            system,
            self.prompt.task("investigate", requirements=requirements, prefetch=prefetch_text),
            build_catalog_tools(self.catalog),
        )
        result = self.structured(
            system,
            self.prompt.task("map", requirements=requirements, prefetch=prefetch_text, notes=notes),
            DataMapping,
            fallback=lambda: fallback_mapping(spec, prefetch["join_paths"]),
        )
        mapping = self.drop_hallucinations(result.value, list(prefetch["tables"]))
        return {
            "data_mapping": mapping.model_dump(mode="json"),
            "trace": [
                self.result_event(
                    result,
                    tables=list(prefetch["tables"]),
                    rejected=mapping.rejected_refs,
                    unmapped=mapping.unmapped_requirements,
                )
            ],
        }

    def prefetch(self, spec: RequirementSpec) -> dict[str, Any]:
        """7B 모델이 매번 반복하는 앞부분 조회는 코드로 고정한다."""
        k = self.deps.cfg("rag.top_k", 8)
        hits: dict[str, dict] = {}
        table_score: dict[str, float] = {}
        for query in requirement_queries(spec):
            for h in self.catalog.search_catalog(query, k=k):
                key = h.get("doc_id") or f"{h.get('table')}.{h.get('column')}"
                if key not in hits or h.get("score", 0) > hits[key].get("score", 0):
                    hits[key] = h
                table_score[h["table"]] = table_score.get(h["table"], 0) + h.get("score", 1)
        tables = sorted(table_score, key=table_score.get, reverse=True)[:MAX_TABLES]

        described = {t: self.catalog.describe_table(t) for t in tables}
        profiles = {}
        for t, desc in described.items():
            for col in desc.get("columns", []):
                is_key = col.get("is_pk") or col.get("fk") or col["name"].endswith("_id")
                if is_key and len(profiles) < MAX_KEY_PROFILES:
                    p = self.catalog.profile_column(t, col["name"])
                    profiles[f"{t}.{col['name']}"] = {
                        "row_count": p.get("row_count"),
                        "distinct": p.get("distinct"),
                        "null_ratio": p.get("null_ratio"),
                    }
        return {
            "search_hits": sorted(hits.values(), key=lambda h: -h.get("score", 0))[:20],
            "tables": {
                t: {
                    "description": d.get("description"),
                    "row_count": d.get("row_count"),
                    "columns": [
                        {
                            k: c.get(k)
                            for k in ("name", "type", "description", "is_pk", "fk", "is_time")
                        }
                        for c in d.get("columns", [])
                    ],
                }
                for t, d in described.items()
            },
            "key_profiles": profiles,
            "join_paths": self.catalog.find_join_path(tables) if len(tables) > 1 else [],
        }

    def drop_hallucinations(self, mapping: DataMapping, known_tables: list[str]) -> DataMapping:
        tables = set(known_tables) | {c.table for c in _all_refs(mapping)}
        index = schema_index(self.catalog, sorted(tables))
        rejected: set[str] = set(mapping.rejected_refs)

        def ok(c: ColumnRef | None) -> bool:
            if c is None:
                return True
            if c.ref in index:
                return True
            rejected.add(c.ref)
            return False

        unmapped = list(mapping.unmapped_requirements)
        items = []
        for item in mapping.items:
            item.columns = [c for c in item.columns if ok(c)]
            if item.columns:
                items.append(item)
            elif item.requirement_id not in unmapped:
                unmapped.append(item.requirement_id)
        return mapping.model_copy(
            update={
                "items": items,
                "analysis_key": mapping.analysis_key if ok(mapping.analysis_key) else None,
                "reference_time": mapping.reference_time if ok(mapping.reference_time) else None,
                "join_paths": [j for j in mapping.join_paths if ok(j.left) and ok(j.right)],
                "unmapped_requirements": unmapped,
                "rejected_refs": sorted(rejected),
            }
        )


def _all_refs(mapping: DataMapping) -> list[ColumnRef]:
    refs = [c for item in mapping.items for c in item.columns]
    refs += [c for c in (mapping.analysis_key, mapping.reference_time) if c]
    refs += [c for j in mapping.join_paths for c in (j.left, j.right)]
    return refs


def fallback_mapping(spec: RequirementSpec, join_paths: list[dict]) -> DataMapping:
    return DataMapping(
        join_paths=[
            JoinStep(left=ColumnRef.parse(j["left"]), right=ColumnRef.parse(j["right"]))
            for j in join_paths
        ],
        unmapped_requirements=[i.id for i in spec.requirement_items],
        rationale="LLM 매핑 실패로 조인 경로만 남겼습니다. 검증 후 재작업합니다.",
    )
