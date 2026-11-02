from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from req2data.api.deps import get_services
from req2data.service import Services

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/search")
def search_catalog(
    q: str = Query(min_length=1),
    source: Literal["catalog", "glossary", "guide", "case"] | None = None,
    k: int = Query(8, ge=1, le=50),
    services: Services = Depends(get_services),
) -> list[dict[str, Any]]:
    return services.kb.search(q, k=k, source=source)


@router.get("/tables")
def list_tables(services: Services = Depends(get_services)) -> list[dict[str, Any]]:
    return services.catalog.list_tables()


@router.get("/tables/{table}")
def describe_table(table: str, services: Services = Depends(get_services)) -> dict[str, Any]:
    try:
        desc = services.catalog.describe_table(table)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    for col in desc["columns"]:
        col["profile"] = services.kb.profiles.get(f"{table}.{col['name']}")
    desc["joins"] = [
        {"left": left, "right": right}
        for left, right in services.kb.spec.foreign_keys()
        if table in (left.split(".")[0], right.split(".")[0])
    ]
    return desc


@router.get("/glossary")
def glossary(services: Services = Depends(get_services)) -> list[dict[str, Any]]:
    return [t.model_dump() for t in services.kb.glossary]
