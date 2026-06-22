from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.core.settings import settings


class SessionStore:
    def __init__(self, root: Path = settings.output_dir):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> tuple[str, Path]:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        (session_dir / "variations").mkdir()
        return session_id, session_dir

    async def save_upload(self, upload: UploadFile, session_dir: Path) -> Path:
        suffix = Path(upload.filename or "input.png").suffix.lower() or ".png"
        target = session_dir / f"input{suffix}"
        with target.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        return target

    def write_record(self, session_dir: Path, record: dict[str, Any]) -> Path:
        path = session_dir / "record.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return path

    def read_record(self, session_id: str) -> dict[str, Any]:
        path = self.root / session_id / "record.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def update_record(self, session_id: str, updates: dict[str, Any]) -> Path:
        session_dir = self.root / session_id
        record = self.read_record(session_id)
        record.update(updates)
        return self.write_record(session_dir, record)


def public_output_url(path: Path) -> str:
    rel = path.relative_to(settings.output_dir.parent).as_posix()
    return f"/outputs/{rel}"

