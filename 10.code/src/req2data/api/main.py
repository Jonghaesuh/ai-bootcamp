import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from req2data.api.routers import admin, catalog, projects
from req2data.config import get_settings
from req2data.service import Services, build_services


def create_app(services: Services | None = None) -> FastAPI:
    """`services`를 넘기면 그대로 쓰고(테스트), 없으면 설정에서 조립한다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "services", None) is None:
            app.state.services = build_services()
        yield

    app = FastAPI(title="Req2Data Agent API", version="0.1.0", lifespan=lifespan)
    app.state.services = services
    app.include_router(projects.router)
    app.include_router(catalog.router)
    app.include_router(admin.router)

    @app.get("/health", tags=["admin"])
    def health() -> dict:
        s = get_settings()
        return {"status": "ok", "llm": f"{s.llm_provider}:{s.llm_model}", "demo_mode": s.demo_mode}

    return app


logging.basicConfig(
    level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
app = create_app()
