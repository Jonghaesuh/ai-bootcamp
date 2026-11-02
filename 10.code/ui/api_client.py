"""Streamlit → FastAPI 클라이언트. 실행·재개는 SSE 이벤트를 (event, data)로 흘려준다."""

import json
from collections.abc import Iterator
from typing import Any

import httpx


class ApiError(RuntimeError):
    pass


class Req2DataClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=30)

    def _json(self, method: str, path: str, **kwargs) -> Any:
        r = self._http.request(method, path, **kwargs)
        if r.status_code >= 400:
            raise ApiError(_detail(r))
        return r.json()

    # ---- 프로젝트 ----

    def health(self) -> dict:
        return self._json("GET", "/health")

    def demo(self) -> dict:
        """데모 모드면 녹화된 정의서·사용자 유형 ({"demo_mode", "projects"})."""
        return self._json("GET", "/admin/demo")

    def create_project(self, document: str, user_type: str, title: str | None = None) -> dict:
        return self._json(
            "POST", "/projects", json={"document": document, "user_type": user_type, "title": title}
        )

    def list_projects(self) -> list[dict]:
        return self._json("GET", "/projects")

    def project(self, project_id: str) -> dict:
        return self._json("GET", f"/projects/{project_id}")

    def run(self, project_id: str) -> Iterator[tuple[str, dict]]:
        return self._stream(f"/projects/{project_id}/run", None)

    def resume(
        self, project_id: str, answers: dict | None = None, change_request: str | None = None
    ) -> Iterator[tuple[str, dict]]:
        return self._stream(
            f"/projects/{project_id}/resume", {"answers": answers, "change_request": change_request}
        )

    def artifacts(self, project_id: str) -> dict[str, str]:
        return self._json("GET", f"/projects/{project_id}/artifacts")

    def download(self, path: str) -> bytes:
        r = self._http.get(path)
        if r.status_code >= 400:
            raise ApiError(_detail(r))
        return r.content

    def approve(self, project_id: str) -> dict:
        return self._json("POST", f"/projects/{project_id}/approve")

    # ---- 카탈로그 ----

    def search(self, q: str, source: str | None = None, k: int = 8) -> list[dict]:
        params = {"q": q, "k": k} | ({"source": source} if source else {})
        return self._json("GET", "/catalog/search", params=params)

    def tables(self) -> list[dict]:
        return self._json("GET", "/catalog/tables")

    def table(self, name: str) -> dict:
        return self._json("GET", f"/catalog/tables/{name}")

    def glossary(self) -> list[dict]:
        return self._json("GET", "/catalog/glossary")

    def reindex(self, reload_data: bool = False) -> dict:
        return self._json("POST", "/admin/reindex", json={"reload_data": reload_data}, timeout=600)

    # ---- SSE ----

    def _stream(self, path: str, body: dict | None) -> Iterator[tuple[str, dict]]:
        # LLM 단계가 길 수 있어 읽기 타임아웃을 두지 않는다 (서버가 15초마다 ping).
        with self._http.stream("POST", path, json=body, timeout=httpx.Timeout(30, read=None)) as r:
            if r.status_code >= 400:
                r.read()
                raise ApiError(_detail(r))
            event, data = None, []
            for line in r.iter_lines():
                if not line:
                    if event:
                        yield event, json.loads("\n".join(data) or "{}")
                    event, data = None, []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail", r.text))
    except ValueError:
        return r.text
