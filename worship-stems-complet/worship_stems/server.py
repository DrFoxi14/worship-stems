"""Local web server.

A browser UI rather than PyQt: nothing to install beyond Python, it looks the
same on every machine, and the person running sound on Sunday just opens a
page. Nothing leaves the computer - the server binds to localhost only.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .analyze import Analysis, Section
from .config import Options
from .pipeline import Pipeline
from .separate import SeparationEngine

log = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent
WEB_DIR = APP_DIR / "web"


def workspace() -> Path:
    root = Path.home() / "WorshipStems"
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class Job:
    def __init__(self, job_id: str, source: Path):
        self.id = job_id
        self.source = source
        self.title = source.stem
        self.state = "uploaded"
        self.stage = ""
        self.fraction = 0.0
        self.message = ""
        self.analysis: Analysis | None = None
        self.result = None
        self.error: str | None = None
        self.events: queue.Queue = queue.Queue()

    def emit(self, stage: str, fraction: float, message: str) -> None:
        self.stage, self.fraction, self.message = stage, fraction, message
        self.events.put({"stage": stage, "fraction": fraction, "message": message, "state": self.state})

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "state": self.state,
            "stage": self.stage,
            "fraction": self.fraction,
            "message": self.message,
            "error": self.error,
        }


JOBS: dict[str, Job] = {}


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------


class SectionIn(BaseModel):
    start: float
    end: float
    label: str
    index: int = 1


class RenderIn(BaseModel):
    job_id: str
    sections: list[SectionIn] | None = None
    bpm: float | None = None
    key: str | None = None
    mode: str | None = None
    options: dict = {}


def build_app() -> FastAPI:
    app = FastAPI(title="Worship Stems")
    root = workspace()

    # ---- pages ----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict:
        return {
            "separator_installed": SeparationEngine.available(),
            "separator_error": SeparationEngine.import_error(),
            "device": SeparationEngine.describe_device(),
            "output_dir": str(root / "output"),
            "tts": __import__("worship_stems.guide", fromlist=["available_tts"]).available_tts(),
            "allin1": _has_allin1(),
            "patches": __import__("worship_stems.render", fromlist=["list_patches"]).list_patches(),
        }

    # ---- upload ---------------------------------------------------------

    @app.post("/api/upload")
    async def upload(file: UploadFile) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job_dir = root / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        name = Path(file.filename or "song.mp3").name
        dest = job_dir / name
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        job = Job(job_id, dest)
        JOBS[job_id] = job
        return {"job_id": job_id, "title": job.title, "size": dest.stat().st_size}

    # ---- analyse --------------------------------------------------------

    @app.post("/api/analyze")
    def start_analyze(payload: dict) -> dict:
        job = _job(payload.get("job_id"))
        if job.state in {"analyzing", "rendering"}:
            raise HTTPException(409, "job already running")
        job.state = "analyzing"

        def work() -> None:
            try:
                pipe = Pipeline(Options())
                job.analysis = pipe.analyze_only(job.source, progress=job.emit)
                job.state = "analyzed"
                job.emit("analyze", 1.0, "Analysis complete")
            except Exception as exc:  # noqa: BLE001
                job.error = str(exc)
                job.state = "error"
                job.emit("error", 1.0, str(exc))

        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}

    @app.get("/api/analysis/{job_id}")
    def get_analysis(job_id: str) -> dict:
        job = _job(job_id)
        if job.analysis is None:
            raise HTTPException(404, "not analysed yet")
        return job.analysis.to_dict()

    # ---- render ---------------------------------------------------------

    @app.post("/api/render")
    def start_render(payload: RenderIn) -> dict:
        job = _job(payload.job_id)
        if job.state == "rendering":
            raise HTTPException(409, "already rendering")

        analysis = job.analysis
        if analysis is not None:
            if payload.sections:
                analysis.sections = [
                    Section(s.start, s.end, s.label, s.index)
                    for s in sorted(payload.sections, key=lambda x: x.start)
                ]
            if payload.bpm:
                analysis.bpm = payload.bpm
            if payload.key:
                analysis.key = payload.key
            if payload.mode:
                analysis.mode = payload.mode

        opts = Options(**{k: v for k, v in payload.options.items() if k in Options.__dataclass_fields__})
        job.state = "rendering"

        def work() -> None:
            try:
                pipe = Pipeline(opts)
                job.result = pipe.run(job.source, root / "output", progress=job.emit, analysis=analysis)
                job.state = "done"
                job.emit("done", 1.0, f"{len(job.result.files)} tracks ready")
            except Exception as exc:  # noqa: BLE001
                log.exception("render failed")
                job.error = str(exc)
                job.state = "error"
                job.emit("error", 1.0, str(exc))

        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}

    @app.get("/api/result/{job_id}")
    def result(job_id: str) -> dict:
        job = _job(job_id)
        if job.result is None:
            raise HTTPException(404, "no result yet")
        r = job.result
        return {
            "folder": str(r.folder),
            "folder_name": r.folder.name,
            "seconds": round(r.seconds, 1),
            "warnings": r.warnings,
            "metrics": r.metrics,
            "analysis": r.analysis.to_dict(),
            "inventory": r.inventory,
            "quality": r.quality,
            "spatial": r.spatial,
            "restore": r.restore,
            "difficulty": r.difficulty,
            "files": [
                {"key": f.key, "label": f.label, "name": f.path.name, "peak_db": round(f.peak_db, 2)}
                for f in r.files
            ],
        }

    # ---- progress stream -------------------------------------------------

    @app.get("/api/progress/{job_id}")
    def progress(job_id: str) -> StreamingResponse:
        job = _job(job_id)

        def stream():
            yield f"data: {json.dumps(job.snapshot())}\n\n"
            while True:
                try:
                    event = job.events.get(timeout=20)
                    event["state"] = job.state
                    yield f"data: {json.dumps(event)}\n\n"
                    if job.state in {"done", "error"} and event.get("stage") in {"done", "error"}:
                        break
                except queue.Empty:
                    yield ": keepalive\n\n"
                    if job.state in {"done", "error"}:
                        break

        return StreamingResponse(stream(), media_type="text/event-stream")

    # ---- files ----------------------------------------------------------

    @app.get("/api/audio/{job_id}/{name}")
    def audio(job_id: str, name: str) -> FileResponse:
        job = _job(job_id)
        if job.result is None:
            raise HTTPException(404, "no result")
        target = (job.result.folder / name).resolve()
        if not str(target).startswith(str(job.result.folder.resolve())) or not target.exists():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    @app.get("/api/source/{job_id}")
    def source(job_id: str) -> FileResponse:
        job = _job(job_id)
        return FileResponse(job.source)

    @app.get("/api/download/{job_id}")
    def download(job_id: str) -> FileResponse:
        job = _job(job_id)
        if job.result is None:
            raise HTTPException(404, "no result")
        zip_path = job.result.folder.parent / f"{job.result.folder.name}.zip"
        if not zip_path.exists():
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                for f in sorted(job.result.folder.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(job.result.folder.parent))
        return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [j.snapshot() for j in JOBS.values()]

    return app


def _job(job_id: str | None) -> Job:
    if not job_id or job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


def _has_allin1() -> bool:
    try:
        import allin1  # noqa: F401

        return True
    except Exception:
        return False


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    if open_browser:
        def _open() -> None:
            time.sleep(1.2)
            import webbrowser

            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(build_app(), host=host, port=port, log_level="warning")
