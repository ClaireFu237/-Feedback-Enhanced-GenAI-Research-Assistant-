"""Local, single-process workbench: isolated runs and a bounded serial job queue."""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import re
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from services.ingestion.chunk import TokenBudget, make_chunks
from services.ingestion.extract import extract_pdf
from services.ingestion.pipeline import OUTPUTS, ROOT, build_recipe, digest, process_document
from services.retrieval.data import Corpus, load_config, read_jsonl
from services.retrieval.engine import LocalRetriever, resolve_model
from storage.local.library import LibraryStorage, LocalLibraryStorage

MAX_FILE = 20 * 1024 * 1024
MAX_TEXT = 2_000_000
ARTIFACTS = set(OUTPUTS) | {"source.pdf", "cleaned.txt", "cleaned.json", "pages.json", "result.json"}


class Problem(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validated_cleaned(data):
    """Accept saved normalization, never a server path or caller-supplied IDs."""
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise Problem("清洗文件必须是包含 text 和 paragraphs 的 JSON 对象。", 422)
    text, paragraphs = data["text"], data.get("paragraphs")
    if not text.strip() or len(text) > MAX_TEXT:
        raise Problem(f"文本不能为空，且最多 {MAX_TEXT:,} 个字符。", 422)
    if not isinstance(paragraphs, list) or not 1 <= len(paragraphs) <= 50000:
        raise Problem("paragraphs 必须是非空段落数组。", 422)
    result, cursor, page_kinds = [], 0, set()
    for i, p in enumerate(paragraphs):
        if not isinstance(p, dict):
            raise Problem("段落必须是 JSON 对象。", 422)
        a, b, page = p.get("char_start"), p.get("char_end"), p.get("page_number")
        if (type(a) is not int or type(b) is not int or not cursor <= a < b <= len(text)
                or text[cursor:a].strip()):
            raise Problem("段落字符位置无效，或与 text 不匹配；请重新导出清洗 JSON。", 422)
        if page is not None and (type(page) is not int or page < 1):
            raise Problem("page_number 必须是正整数或 null。", 422)
        page_kinds.add(page is not None)
        section_id = p.get("section_id", 0)
        if type(section_id) is not int or type(p.get("searchable", True)) is not bool:
            raise Problem("section_id 必须是整数，searchable 必须是布尔值。", 422)
        result.append({"char_start": a, "char_end": b, "page_number": page,
                       "block_id": str(p.get("block_id", f"text:{i}"))[:200],
                       "section_id": section_id, "section": str(p.get("section", "Text"))[:500],
                       "searchable": p.get("searchable", True), "is_heading": bool(p.get("is_heading", False))})
        cursor = b
    if len(page_kinds) > 1 or text[cursor:].strip():
        raise Problem("段落来源页码不一致，或有未映射的正文。", 422)
    return text, result


class Workbench:
    def __init__(self, root=None, library: LibraryStorage | None = None):
        self.root = Path(root or ROOT / "data_local/web_runs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        (self.root / "tasks").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-workbench")
        self.pending = 0
        self.closed = False
        self.retrievers = OrderedDict()
        self.library = library if library is not None else LocalLibraryStorage(self.root / "library")
        # A server restart cannot resume a Python thread or trust its old busy flag.
        for path in (self.root / "tasks").glob("*.json"):
            task = read(path)
            if task["status"] in ("queued", "running"):
                task.update(status="failed", error="服务已重启，请重新运行这一步。", updated_at=now())
                write(path, task)
        for path in (self.root / "runs").glob("*/state.json"):
            state = read(path)
            if state.get("active_task"):
                state.update(active_task=None, last_error="上次任务因服务重启而中断。")
                write(path, state)

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)
        self.retrievers.clear()

    def folder(self, run_id):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise Problem("测试记录不存在。", 404)
        folder = self.root / "runs" / run_id
        if not (folder / "state.json").is_file():
            raise Problem("测试记录不存在。", 404)
        return folder

    def state(self, run_id):
        with self.lock:
            return read(self.folder(run_id) / "state.json")

    def update(self, run_id, **fields):
        with self.lock:
            state = self.state(run_id)
            state.update(fields, updated_at=now())
            write(self.folder(run_id) / "state.json", state)
            return state

    def public(self, run_id):
        state = {k: v for k, v in self.state(run_id).items() if not k.startswith("_")}
        state["artifacts"] = {name: f"/api/runs/{run_id}/artifacts/{name}" for name in sorted(ARTIFACTS)
                              if self.artifact(run_id, name, optional=True) is not None}
        return state

    def list_runs(self):
        ids = [p.parent.name for p in (self.root / "runs").glob("*/state.json")]
        states = [self.public(run_id) for run_id in ids]
        return sorted(states, key=lambda s: s["created_at"], reverse=True)[:50]

    def create(self, filename, title, kind, data):
        title = title.strip() or Path(filename).stem
        if not 1 <= len(title) <= 500:
            raise Problem("论文标题需要 1–500 个字符。", 422)
        run_id = uuid.uuid4().hex
        folder = self.root / "runs" / run_id
        folder.mkdir()
        source = "source.pdf" if kind == "pdf" else "source.txt"
        (folder / source).write_bytes(data)
        state = {"run_id": run_id, "filename": filename, "title": title, "source_kind": kind,
                 "phase": "uploaded", "created_at": now(), "updated_at": now(), "active_task": None,
                 "last_error": None, "report": {}, "chunk_count": 0, "chunk_config": None,
                 "_source": source, "_cleaned": None, "_corpus": None, "_result": None}
        write(folder / "state.json", state)
        return run_id

    def create_pdf(self, filename, title, content):
        filename = Path(filename.replace("\\", "/")).name[:255]
        if not filename.lower().endswith(".pdf") or b"%PDF-" not in content[:1024]:
            raise Problem("请上传有效的 .pdf 文件。", 422)
        if not content or len(content) > MAX_FILE:
            raise Problem("PDF 不能超过 20 MB。", 413)
        return self.public(self.create(filename, title, "pdf", content))

    def create_cleaned(self, payload, title=""):
        text, paragraphs = validated_cleaned(payload)
        run_id = self.create("imported-cleaned.txt", title or str(payload.get("title") or "Imported text"),
                             "cleaned_json", text.encode("utf-8"))
        pages = [p["page_number"] for p in paragraphs if p["page_number"] is not None]
        report = {"page_count": max(pages, default=0), "low_text_pages": [],
                  "warnings": ["使用导入的页码/段落映射；本次未上传或重新解析原始 PDF。"],
                  "excluded_reference_paragraphs": sum(not p["searchable"] for p in paragraphs)}
        self.save_cleaned(run_id, text, paragraphs, [], report)
        return self.public(run_id)

    def create_text(self, title, text):
        paragraphs = []
        for i, match in enumerate(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S)):
            paragraphs.append({"char_start": match.start(), "char_end": match.end(), "page_number": None,
                               "block_id": f"text:{i}", "section_id": 0, "section": "Text", "searchable": True})
        validated_cleaned({"text": text, "paragraphs": paragraphs})
        run_id = self.create("pasted-text.txt", title, "text", text.encode("utf-8"))
        self.save_cleaned(run_id, text, paragraphs, [], {"page_count": 0, "low_text_pages": [],
                          "warnings": ["纯文本来源仅提供字符位置，不提供 PDF 页码。"], "excluded_reference_paragraphs": 0})
        return self.public(run_id)

    def save_cleaned(self, run_id, text, paragraphs, pages, report):
        if len(text) > MAX_TEXT:
            raise Problem("抽取文本超过 2,000,000 字符，请拆分文档后测试。", 422)
        folder, state = self.folder(run_id), self.state(run_id)
        data = {"title": state["title"], "text": text, "paragraphs": paragraphs, "report": report}
        write(folder / "cleaned.json", data)
        write(folder / "pages.json", pages)
        (folder / "cleaned.txt").write_text(text, encoding="utf-8")
        self.update(run_id, phase="extracted", _cleaned="cleaned.json", _corpus=None, _result=None,
                    report=report, char_count=len(text), chunk_count=0, chunk_config=None)

    def artifact(self, run_id, name, optional=False):
        state, folder = self.state(run_id), self.folder(run_id)
        path = None
        if name == "source.pdf" and state["source_kind"] == "pdf":
            path = folder / state["_source"]
        elif name in ("cleaned.json", "cleaned.txt", "pages.json") and state["_cleaned"]:
            path = folder / name
        elif name in OUTPUTS and state["_corpus"]:
            path = folder / state["_corpus"] / name
        elif name == "result.json" and state["_result"]:
            path = folder / state["_result"]
        if path is not None and path.is_file():
            return path
        if optional:
            return None
        raise Problem("该产物尚未生成，请先执行对应步骤。", 404)

    def task(self, task_id):
        if not re.fullmatch(r"[0-9a-f]{32}", task_id) or not (self.root / "tasks" / f"{task_id}.json").is_file():
            raise Problem("任务不存在。", 404)
        with self.lock:
            return read(self.root / "tasks" / f"{task_id}.json")

    def progress(self, task_id, step):
        with self.lock:
            task = self.task(task_id)
            task.update(step=step, updated_at=now())
            task["history"].append({"step": step, "at": now()})
            write(self.root / "tasks" / f"{task_id}.json", task)

    def submit(self, operation, fn, run_id=None, input_data=None):
        with self.lock:
            if self.closed:
                raise Problem("服务正在关闭。", 503)
            if self.pending >= 8:
                raise Problem("任务队列已满，请稍后再试。", 429)
            if run_id and self.state(run_id)["active_task"]:
                raise Problem("这篇论文已有任务在运行，请等待它完成。")
            task_id = uuid.uuid4().hex
            task = {"task_id": task_id, "run_id": run_id, "operation": operation, "status": "queued",
                    "step": "queued", "created_at": now(), "updated_at": now(), "error": None,
                    "history": [], "result": None}
            if input_data is not None:
                task["input"] = copy.deepcopy(input_data)
            write(self.root / "tasks" / f"{task_id}.json", task)
            if run_id:
                self.update(run_id, active_task=task_id, last_error=None)
            self.pending += 1
            self.executor.submit(self.execute, task_id, fn)
            return task

    def execute(self, task_id, fn):
        try:
            with self.lock:
                task = self.task(task_id)
                task.update(status="running", updated_at=now())
                write(self.root / "tasks" / f"{task_id}.json", task)
            result = fn(task_id)
            with self.lock:
                task = self.task(task_id)
                task.update(status="completed", step="completed", result=result, updated_at=now())
                write(self.root / "tasks" / f"{task_id}.json", task)
        except Exception as exc:
            if isinstance(exc, (Problem, ValueError, OSError, ImportError)):
                logging.warning("Workbench task %s: %s", task_id, exc)
            else:
                logging.exception("Workbench task %s failed", task_id)
            with self.lock:
                task = self.task(task_id)
                task.update(status="failed", error=str(exc), updated_at=now())
                write(self.root / "tasks" / f"{task_id}.json", task)
        finally:
            with self.lock:
                task = self.task(task_id)
                if task["run_id"]:
                    self.update(task["run_id"], active_task=None, last_error=task["error"])
                self.pending -= 1

    def extract(self, run_id, task_id):
        state, folder = self.state(run_id), self.folder(run_id)
        if state["source_kind"] != "pdf":
            raise Problem("此记录来自文本，可直接切块，无需 PDF 解析。")
        self.progress(task_id, "extracting")
        self.update(run_id, phase="uploaded", _cleaned=None, _corpus=None, _result=None, chunk_count=0,
                    chunk_config=None, char_count=0, report={}, index_version=None)
        self.drop_retriever(run_id)
        pages, text, paragraphs, report = extract_pdf(folder / state["_source"])
        self.save_cleaned(run_id, text, paragraphs, pages, report)
        return {"char_count": len(text), "page_count": len(pages)}

    def chunk(self, run_id, options, task_id):
        state, folder = self.state(run_id), self.folder(run_id)
        if not state["_cleaned"]:
            raise Problem("请先执行 PDF 清洗，或导入清洗后的 JSON。")
        self.progress(task_id, "chunking")
        self.update(run_id, phase="extracted", _corpus=None, _result=None, chunk_count=0,
                    chunk_config=None, index_version=None)
        self.drop_retriever(run_id)
        data = read(folder / state["_cleaned"])
        paragraphs = copy.deepcopy(data["paragraphs"])
        if options["include_references"]:
            for p in paragraphs:
                p["searchable"] = True
        budget = TokenBudget(chunk_size=options["chunk_size"], overlap=options["overlap"],
                             max_input=options["max_input_tokens"])
        recipe = build_recipe(budget, options["include_references"])
        recipe.update(web_stage_version=1, cleaned_content_hash=digest(data["text"]),
                      paragraph_map_hash=digest(json.dumps(paragraphs, sort_keys=True, ensure_ascii=False)))
        relative = "corpora/" + uuid.uuid4().hex
        output = folder / relative
        output.mkdir(parents=True)
        pages = read(folder / "pages.json")
        report = copy.deepcopy(data["report"])
        if options["include_references"]:
            report["excluded_reference_paragraphs"] = 0
        timestamp = now()
        records = process_document(folder / state["_source"], {"title": state["title"]}, budget, recipe,
            timestamp, output, extract_pdf, make_chunks, extraction=(pages, data["text"], paragraphs, report))
        document, page_rows, normalized, chunks, mappings, report = records
        document["filename"] = state["filename"]
        batches = {"documents.jsonl": [document], "pages.jsonl": page_rows, "normalized_texts.jsonl": [normalized],
                   "chunks.jsonl": chunks, "id_map.jsonl": mappings}
        for name, values in batches.items():
            (output / name).write_text("".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values), encoding="utf-8")
        write(output / "quality_report.json", {"status": "complete", "documents": [report], "failed_document_count": 0})
        import hashlib
        write(output / "manifest.json", {"schema_version": 1, "status": "complete",
              "stage": "pdf_to_chunks" if state["source_kind"] == "pdf" else "text_to_chunks", "recipe": recipe,
              "created_at": timestamp, "counts": {"documents": 1, "pages": len(pages), "chunks": len(chunks)},
              "files": {n: hashlib.sha256((output / n).read_bytes()).hexdigest() for n in OUTPUTS if n != "manifest.json"}})
        self.update(run_id, phase="chunked", _corpus=relative, chunk_count=len(chunks), chunk_config=options,
                    index_version=document["index_version"], report=report)
        return {"chunk_count": len(chunks), "index_version": document["index_version"]}

    def drop_retriever(self, key):
        self.retrievers.pop(key, None)

    def publish(self, run_id, task_id):
        state = self.state(run_id)
        if not state["_corpus"]:
            raise Problem("请先生成 chunks，再发布论文。")
        self.progress(task_id, "publishing")
        result = self.library.publish(self.folder(run_id) / state["_corpus"])
        self.drop_retriever("dataset:library")
        return result

    def library_documents(self):
        snapshot = self.library.snapshot()
        if snapshot is None:
            return {"total": 0, "items": []}
        documents = read_jsonl(snapshot / "documents.jsonl")
        return {"total": len(documents), "items": [
            {key: row[key] for key in ("document_id", "title", "filename", "chunk_count", "index_version")}
            for row in documents]}

    def search(self, run_id, options, task_id):
        method = options["method"]
        config = load_config()
        if method != "bm25":
            if not all(importlib.util.find_spec(m) for m in ("torch", "transformers")):
                raise Problem("未安装神经检索依赖。请运行 python -m pip install -r requirements.txt；BM25 可直接使用。")
            for name in (["dense", "reranker"] if method == "hybrid_reranker" else ["dense"]):
                try:
                    resolve_model(config[name], config.get("model_cache_root"))
                except FileNotFoundError as exc:
                    raise Problem("缺少本地模型。请运行 python scripts/download_retrieval_models.py --config config/retrieval.json --method " + method) from exc
        if run_id:
            state = self.state(run_id)
            if not state["_corpus"]:
                raise Problem("请先生成 chunks，再运行检索。")
            corpus = self.folder(run_id) / state["_corpus"]
            cache_key = run_id
            dataset = "web_" + run_id
        else:
            dataset = options["dataset"]
            if dataset not in config["datasets"]:
                raise Problem("未知语料库。", 422)
            if dataset == "library":
                corpus = self.library.snapshot()
                if corpus is None:
                    raise Problem("资料库为空，请先上传、切块并发布论文。")
            else:
                corpus = ROOT / config["datasets"][dataset]
            cache_key = "dataset:" + dataset
        # Detect publications by another process as well as this workbench.
        if dataset == "library" and cache_key in self.retrievers:
            if self.retrievers[cache_key]._library_snapshot != corpus:
                self.drop_retriever(cache_key)
        self.progress(task_id, "indexing" if method == "bm25" else "embedding")
        if cache_key not in self.retrievers:
            config = copy.deepcopy(config)
            config["cache_root"] = str(self.root / "embeddings")
            self.retrievers[cache_key] = LocalRetriever(config=config, corpus=Corpus.load(dataset, corpus))
            if dataset == "library":
                self.retrievers[cache_key]._library_snapshot = corpus
            while len(self.retrievers) > 2:
                self.retrievers.popitem(last=False)
        retriever = self.retrievers[cache_key]
        self.retrievers.move_to_end(cache_key)
        retriever.prepare(method)
        self.progress(task_id, "searching")
        result = retriever.search(options["query"], options["top_k"], options.get("document_ids"), method)
        result["setup_timings"] = dict(retriever.timings)
        result["dataset"] = dataset
        result["document_ids"] = options.get("document_ids")
        if run_id:
            relative = "results/" + uuid.uuid4().hex + ".json"
            write(self.folder(run_id) / relative, result)
            self.update(run_id, _result=relative)
        return result

    def pipeline(self, run_id, chunk_options, search_options, task_id):
        state = self.state(run_id)
        if state["source_kind"] == "pdf" and not state["_cleaned"]:
            self.extract(run_id, task_id)
        state = self.state(run_id)
        if not state["_corpus"] or state["chunk_config"] != chunk_options:
            self.chunk(run_id, chunk_options, task_id)
        return self.search(run_id, search_options, task_id)

    def capabilities(self):
        config, methods = load_config(), {"bm25": {"available": True, "reason": ""}}
        dependencies = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))
        for method in ("dense", "hybrid", "hybrid_reranker"):
            reason = ""
            if not dependencies:
                reason = "先安装 requirements.txt 中的依赖"
            else:
                try:
                    for name in (["dense", "reranker"] if method == "hybrid_reranker" else ["dense"]):
                        resolve_model(config[name], config.get("model_cache_root"))
                except FileNotFoundError:
                    reason = "先准备本地模型权重"
            methods[method] = {"available": not reason, "reason": reason}
        return {"methods": methods, "datasets": [{"id": name, "available":
                self.library.snapshot() is not None if name == "library" else (ROOT / path / "manifest.json").is_file()}
                for name, path in config["datasets"].items()], "max_upload_mb": 20, "max_text_characters": MAX_TEXT}
