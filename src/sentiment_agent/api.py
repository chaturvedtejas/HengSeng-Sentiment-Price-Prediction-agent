"""FastAPI: serves the dashboard, the report JSON, live run progress (SSE) and a background refresher.

The dashboard is never blocked by a slow run: the last good report keeps being served while a new one is built,
and /api/status + the SSE stream expose step-by-step progress (source, assets, posts, Exa, model) and any error."""
from __future__ import annotations
import asyncio, json, logging, os, threading
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from .config import ROOT, load_config, resolve_mode

log = logging.getLogger("api")
OUT = ROOT / "outputs"
LATEST, STATUS = OUT / "latest.json", OUT / "status.json"
app = FastAPI(title="HSI Sentiment Agent", version="2.0.0")
state = {"running": False, "last_error": None}
_lock = threading.Lock()


def _claim() -> bool:
    """Atomic check-and-set: two refresh clicks can never start two pipelines."""
    with _lock:
        if state["running"]:
            return False
        state["running"] = True
        return True


def _has_report() -> bool:
    return LATEST.exists() and LATEST.stat().st_size > 2


def _status() -> dict:
    s = {}
    if STATUS.exists():
        try:
            s = json.loads(STATUS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            s = {}
    s.update(running=state["running"], last_error=state["last_error"], has_report=_has_report(), mode=resolve_mode(),
             report_mtime=LATEST.stat().st_mtime if _has_report() else None)
    return s


def _run_pipeline(claimed: bool = False):
    from .pipeline import run
    if not claimed and not _claim():
        return
    try:
        run()
        state["last_error"] = None
    except Exception as e:  # keep serving the previous report; the error is shown with a retry button
        log.exception("pipeline failed")
        state["last_error"] = f"{type(e).__name__}: {e}"
    finally:
        state["running"] = False


async def _scheduler():
    cfg = load_config()
    loop = asyncio.get_running_loop()
    if not _has_report():
        await loop.run_in_executor(None, _run_pipeline)
    while True:
        await asyncio.sleep(cfg["refresh_minutes"] * 60)
        await loop.run_in_executor(None, _run_pipeline)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_scheduler())


@app.get("/api/health")
def health():
    return {"ok": True, "mode": resolve_mode(), "running": state["running"], "last_error": state["last_error"], "has_report": _has_report()}


@app.get("/api/status")
def status():
    return _status()


@app.get("/api/report")
def report():
    if not _has_report():
        raise HTTPException(503, "report is being generated, retry in a few seconds")
    try:
        return json.loads(LATEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(503, "report is being rewritten, retry in a few seconds")


@app.post("/api/refresh")
async def refresh(request: Request, x_admin_key: str | None = Header(default=None)):
    """Refresh costs Apify/Exa/Claude credits. With ADMIN_KEY set, the key is required from everyone.
    Without one, only requests from this machine (loopback) may trigger a run, so a public deployment is never open."""
    key = os.getenv("ADMIN_KEY")
    host = request.client.host if request.client else ""
    if key and x_admin_key != key:
        raise HTTPException(401, "admin key required (enter the ADMIN_KEY from .env when the dashboard asks)")
    if not key and host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "refresh from other machines needs ADMIN_KEY set in .env")
    if not _claim():
        return JSONResponse({"started": False, "reason": "a run is already in progress"}, status_code=409)
    asyncio.get_running_loop().run_in_executor(None, _run_pipeline, True)
    return {"started": True}


@app.get("/api/stream")
async def stream():
    async def gen():
        last = None
        while True:
            s = _status()
            m = s.get("report_mtime")
            ev = "update" if m != last else "status"
            last = m
            yield f"event: {ev}\ndata: {json.dumps(s, ensure_ascii=False)}\n\n"
            await asyncio.sleep(1.5)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
