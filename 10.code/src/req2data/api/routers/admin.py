import duckdb
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from req2data.api.deps import get_services
from req2data.config import get_settings
from req2data.rag.profiling import load_csvs, load_profiles, profile_tables, save_profiles
from req2data.service import Services

router = APIRouter(prefix="/admin", tags=["admin"])


class ReindexRequest(BaseModel):
    reload_data: bool = False  # true면 CSV 재적재·프로파일까지 다시 한다


@router.post("/reindex")
def reindex(body: ReindexRequest | None = None, services: Services = Depends(get_services)) -> dict:
    """데이터 적재(선택)·프로파일·RAG 인덱스 재구축."""
    loaded = None
    if body and body.reload_data:
        s = get_settings()
        services.catalog.close()
        try:
            loaded = load_csvs(services.kb.spec, s.raw_data_dir, s.db_path)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
        with duckdb.connect(str(s.db_path), read_only=True) as con:
            save_profiles(profile_tables(con, services.kb.spec), services.kb.paths.profiles)
        services.kb.profiles = load_profiles(services.kb.paths.profiles)
    return {"loaded": loaded, "indexed": services.kb.rebuild()}
