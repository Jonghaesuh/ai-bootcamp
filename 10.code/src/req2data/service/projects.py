"""프로젝트 메타데이터 저장소. 그래프 상태 자체는 LangGraph 체크포인터(스레드 = 프로젝트 ID)에 있다."""

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ProjectStatus = Literal["new", "running", "waiting", "done", "error"]


class Project(BaseModel):
    id: str
    title: str
    user_type: str
    document: str
    status: ProjectStatus = "new"
    approved: bool = False
    error: str | None = None
    created_at: str
    updated_at: str


_COLUMNS = (
    "id",
    "title",
    "user_type",
    "document",
    "status",
    "approved",
    "error",
    "created_at",
    "updated_at",
)


class ProjectRepository:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS projects ("
            "id TEXT PRIMARY KEY, title TEXT, user_type TEXT, document TEXT, status TEXT,"
            " approved INTEGER DEFAULT 0, error TEXT, created_at TEXT, updated_at TEXT)"
        )
        self._con.commit()

    def create(self, document: str, user_type: str, title: str | None = None) -> Project:
        now = _now()
        project = Project(
            id=uuid.uuid4().hex[:10],
            title=title or _default_title(document),
            user_type=user_type,
            document=document,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._con.execute(
                f"INSERT INTO projects ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
                [getattr(project, c) for c in _COLUMNS],
            )
            self._con.commit()
        return project

    def get(self, project_id: str) -> Project | None:
        with self._lock:
            row = self._con.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM projects WHERE id = ?", [project_id]
            ).fetchone()
        return Project(**dict(zip(_COLUMNS, row, strict=True))) if row else None

    def list(self) -> list[Project]:
        with self._lock:
            rows = self._con.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM projects ORDER BY created_at DESC"
            ).fetchall()
        return [Project(**dict(zip(_COLUMNS, r, strict=True))) for r in rows]

    def mark_interrupted(self) -> int:
        """서버가 실행 도중 종료되면 상태가 running으로 남는다. 시작 시 error로 바꿔 재개할 수 있게 한다."""
        with self._lock:
            cur = self._con.execute(
                "UPDATE projects SET status = 'error', error = ?, updated_at = ? WHERE status = 'running'",
                ["서버 재시작으로 실행이 중단되었습니다. 재개하면 이어서 실행합니다.", _now()],
            )
            self._con.commit()
        return cur.rowcount

    def update(self, project_id: str, **fields) -> None:
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._con.execute(
                f"UPDATE projects SET {assignments} WHERE id = ?", [*fields.values(), project_id]
            )
            self._con.commit()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _default_title(document: str) -> str:
    first = next((line.strip() for line in document.splitlines() if line.strip()), "새 프로젝트")
    return first.lstrip("#").strip()[:40] or "새 프로젝트"  # Markdown 제목 기호 제거
