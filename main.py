"""FastAPI entry point: uvicorn main:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal

from services.workbench import ARTIFACTS, MAX_FILE, MAX_TEXT, ROOT, Problem, Workbench, read


class ChunkOptions(BaseModel):
    chunk_size: int = Field(200, ge=1, le=256)
    overlap: int = Field(30, ge=0, le=255)
    max_input_tokens: int = Field(256, ge=8, le=256)
    include_references: bool = False

    @model_validator(mode="after")
    def check_overlap(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        return self


class SearchOptions(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    method: Literal["bm25", "dense", "hybrid", "hybrid_reranker"] = "bm25"
    top_k: int = Field(5, ge=1, le=20)
    document_ids: list[str] | None = None
    dataset: str = "openalex"

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("请输入问题或关键词")
        return value.strip()


class PipelineOptions(BaseModel):
    chunk: ChunkOptions = Field(default_factory=ChunkOptions)
    search: SearchOptions


class TextInput(BaseModel):
    title: str = Field("Pasted text", min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=MAX_TEXT)


def create_app(data_root=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.workbench = Workbench(data_root)
        yield
        app.state.workbench.close()

    app = FastAPI(title="PDF Processing API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(Problem)
    async def problem_handler(request, exc):
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def wb(request):
        return request.app.state.workbench

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse(url="/docs")

    @app.get("/guide", include_in_schema=False)
    def guide():
        return FileResponse(ROOT / "WEB_GUIDE.md", media_type="text/plain; charset=utf-8")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "paper-lab"}

    @app.get("/api/config")
    def capabilities(request: Request):
        return wb(request).capabilities()

    @app.get("/api/runs")
    def runs(request: Request):
        return {"items": wb(request).list_runs()}

    @app.post("/api/runs", status_code=201)
    async def upload(request: Request, file: UploadFile = File(...), title: str = Form("")):
        content = await file.read(MAX_FILE + 1)
        await file.close()
        return wb(request).create_pdf(file.filename or "paper.pdf", title, content)

    @app.post("/api/runs/import-cleaned", status_code=201)
    async def import_cleaned(request: Request, file: UploadFile = File(...), title: str = Form("")):
        content = await file.read(MAX_FILE + 1)
        await file.close()
        if len(content) > MAX_FILE:
            raise Problem("清洗 JSON 不能超过 20 MB。", 413)
        import json
        try:
            data = json.loads(content)
        except (ValueError, UnicodeDecodeError) as exc:
            raise Problem("无法读取 JSON，请上传 cleaned.json。", 422) from exc
        return wb(request).create_cleaned(data, title)

    @app.post("/api/runs/text", status_code=201)
    def paste_text(body: TextInput, request: Request):
        return wb(request).create_text(body.title, body.text)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str, request: Request):
        return wb(request).public(run_id)

    @app.post("/api/runs/{run_id}/extract", status_code=202)
    def extract(run_id: str, request: Request):
        store = wb(request)
        if store.state(run_id)["source_kind"] != "pdf":
            raise Problem("这是文本来源，已经可以直接切块。")
        return store.submit("extract", lambda task: store.extract(run_id, task), run_id)

    @app.post("/api/runs/{run_id}/chunks", status_code=202)
    def chunk(run_id: str, body: ChunkOptions, request: Request):
        store = wb(request)
        if not store.state(run_id)["_cleaned"]:
            raise Problem("请先清洗 PDF，或导入清洗后的 JSON。")
        return store.submit("chunk", lambda task: store.chunk(run_id, body.model_dump(), task), run_id)

    @app.post("/api/runs/{run_id}/search", status_code=202)
    def search(run_id: str, body: SearchOptions, request: Request):
        store = wb(request)
        if not store.state(run_id)["_corpus"]:
            raise Problem("请先生成 chunks。")
        return store.submit("search", lambda task: store.search(run_id, body.model_dump(), task), run_id)

    @app.post("/api/runs/{run_id}/pipeline", status_code=202)
    def pipeline(run_id: str, body: PipelineOptions, request: Request):
        store = wb(request)
        store.state(run_id)
        return store.submit("pipeline", lambda task: store.pipeline(run_id, body.chunk.model_dump(),
                            body.search.model_dump(), task), run_id)

    @app.post("/api/search", status_code=202)
    def search_dataset(body: SearchOptions, request: Request):
        store = wb(request)
        return store.submit("search_dataset", lambda task: store.search(None, body.model_dump(), task))

    @app.get("/api/tasks/{task_id}")
    def task(task_id: str, request: Request):
        return wb(request).task(task_id)

    @app.get("/api/runs/{run_id}/cleaned")
    def cleaned(run_id: str, request: Request):
        return read(wb(request).artifact(run_id, "cleaned.json"))

    @app.get("/api/runs/{run_id}/chunks")
    def chunk_list(run_id: str, request: Request, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
        import json
        store = wb(request)
        path = store.artifact(run_id, "chunks.jsonl")
        chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        maps = {r["chunk_id"]: r for r in map(json.loads, (path.parent / "id_map.jsonl").read_text(encoding="utf-8").splitlines())}
        return {"total": len(chunks), "offset": offset, "items": [{**c, "section": maps[c["chunk_id"]]["section"],
                "input_token_count": maps[c["chunk_id"]]["input_token_count"]} for c in chunks[offset:offset + limit]]}

    @app.get("/api/runs/{run_id}/results")
    def results(run_id: str, request: Request):
        return read(wb(request).artifact(run_id, "result.json"))

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    def artifact(run_id: str, name: str, request: Request, inline: bool = False):
        if name not in ARTIFACTS:
            raise Problem("产物不存在。", 404)
        path = wb(request).artifact(run_id, name)
        media = "application/pdf" if name.endswith(".pdf") else ("text/plain; charset=utf-8" if name.endswith((".txt", ".jsonl")) else "application/json")
        return FileResponse(path, filename=name, media_type=media,
                            content_disposition_type="inline" if inline else "attachment")

    return app


app = create_app()
