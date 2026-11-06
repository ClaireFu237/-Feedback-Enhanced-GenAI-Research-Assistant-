"""Small, offline functional checks; no retrieval-quality evaluation."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

from main import create_app
from services.ingestion.extract import extract_pdf as real_extract
from services.workbench import Workbench, write


def paper_bytes():
    pdf = pymupdf.open()
    for heading, body in [
        ("Abstract", "Retrieval augmented generation retrieves evidence from a knowledge corpus. " * 25),
        ("References", "Excluded bibliography evidence. " * 10),
        ("Appendix A", "Appendix evidence describes evaluating grounded responses. " * 20),
    ]:
        page = pdf.new_page()
        page.insert_text((40, 70), heading, fontsize=16, fontname="hebo")
        page.insert_textbox(pymupdf.Rect(40, 100, 550, 740), body, fontsize=11)
    content = pdf.tobytes()
    pdf.close()
    return content


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.app = create_app(Path(self.temporary.name))
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    def upload(self):
        response = self.client.post("/api/runs", files={"file": ("paper.pdf", paper_bytes(), "application/pdf")}, data={"title": "Evidence paper"})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["run_id"]

    def wait(self, response, failed=False):
        self.assertEqual(response.status_code, 202, response.text)
        task_id = response.json()["task_id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            task = self.client.get("/api/tasks/" + task_id).json()
            if task["status"] in ("completed", "failed"):
                if task["run_id"] and self.client.get("/api/runs/" + task["run_id"]).json()["active_task"]:
                    time.sleep(.01)
                    continue
                self.assertEqual(task["status"], "failed" if failed else "completed", task)
                return task
            time.sleep(.02)
        self.fail("Task did not finish")

    def pipeline(self, run_id):
        return self.wait(self.client.post(f"/api/runs/{run_id}/pipeline", json={"search": {"query": "retrieval evidence"}}))

    def test_page_health_and_api_docs(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/docs")
        self.assertIn("SwaggerUIBundle", self.client.get("/docs").text)
        for path in ("/guide", "/docs", "/redoc", "/openapi.json", "/api/health", "/api/config"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_separate_pdf_stages_and_provenance(self):
        run_id = self.upload()
        base = f"/api/runs/{run_id}"
        self.assertEqual(self.client.post(base + "/chunks", json={}).status_code, 409)
        self.wait(self.client.post(base + "/extract"))
        cleaned = self.client.get(base + "/cleaned").json()
        self.assertIn("Retrieval augmented", cleaned["text"])
        self.assertEqual(self.client.get(base).json()["chunk_count"], 0)
        with patch("services.workbench.extract_pdf", side_effect=AssertionError("Chunking reparsed PDF")):
            self.wait(self.client.post(base + "/chunks", json={}))
        chunks = self.client.get(base + "/chunks?limit=100").json()["items"]
        body = " ".join(c["text"] for c in chunks)
        self.assertNotIn("Excluded bibliography", body)
        self.assertIn("Appendix evidence", body)
        for c in chunks:
            loc = c["source_location"]
            self.assertEqual(cleaned["text"][loc["char_start"]:loc["char_end"]], c["text"])
            self.assertLessEqual(c["token_count"], 200)
            self.assertLessEqual(c["input_token_count"], 256)
            self.assertTrue(1 <= loc["page_start"] <= loc["page_end"] <= 3)
        result = self.wait(self.client.post(base + "/search", json={"query": "retrieval evidence"}))["result"]
        self.assertTrue(result["results"])
        self.assertIn(result["results"][0]["chunk_id"], {c["chunk_id"] for c in chunks})
        self.assertEqual(self.client.get(base + "/artifacts/source.pdf").content[:5], b"%PDF-")

    def test_pipeline_reuses_text_and_chunks_then_invalidates_downstream(self):
        run_id = self.upload()
        base = f"/api/runs/{run_id}"
        self.pipeline(run_id)
        first = self.client.get(base + "/chunks?limit=100").json()
        with patch("services.workbench.extract_pdf", side_effect=AssertionError("Unexpected extraction")), \
             patch.object(self.app.state.workbench, "chunk", side_effect=AssertionError("Unexpected chunking")):
            self.pipeline(run_id)
        self.wait(self.client.post(base + "/chunks", json={"chunk_size": 80, "overlap": 10}))
        self.assertEqual(self.client.get(base + "/results").status_code, 404)
        second = self.client.get(base + "/chunks?limit=100").json()
        self.assertNotEqual(first["items"][0]["index_version"], second["items"][0]["index_version"])
        self.wait(self.client.post(base + "/chunks", json={"chunk_size": 80, "overlap": 10}))
        repeated = self.client.get(base + "/chunks?limit=100").json()
        self.assertEqual([c["chunk_id"] for c in second["items"]], [c["chunk_id"] for c in repeated["items"]])
        self.wait(self.client.post(base + "/extract"))
        self.assertEqual(self.client.get(base + "/chunks").status_code, 404)

    def test_import_cleaned_json_and_plain_text(self):
        original = self.upload()
        self.wait(self.client.post(f"/api/runs/{original}/extract"))
        cleaned = self.client.get(f"/api/runs/{original}/cleaned").json()
        response = self.client.post("/api/runs/import-cleaned", files={"file": ("cleaned.json", json.dumps(cleaned).encode(), "application/json")})
        self.assertEqual(response.status_code, 201, response.text)
        imported = response.json()["run_id"]
        self.assertEqual(self.client.post(f"/api/runs/{imported}/extract").status_code, 409)
        self.pipeline(imported)
        first = self.client.post("/api/runs/text", json={"title": "Text A", "text": "Alpha evidence.\n\nSecond paragraph."}).json()
        second = self.client.post("/api/runs/text", json={"title": "Text B", "text": "Different beta evidence."}).json()
        for run in (first, second):
            self.pipeline(run["run_id"])
            chunks = self.client.get(f"/api/runs/{run['run_id']}/chunks").json()["items"]
            self.assertTrue(chunks)
            self.assertTrue(all("page_start" not in c["source_location"] for c in chunks))
        a = self.client.get(f"/api/runs/{first['run_id']}/results").json()
        self.assertTrue(all("Different beta" not in c["text"] for c in a["results"]))
        self.assertEqual(self.client.get(f"/api/runs/{first['run_id']}/artifacts/source.pdf").status_code, 404)

    def test_parameters_files_and_invalid_mapping(self):
        run = self.upload()
        self.assertEqual(self.client.post(f"/api/runs/{run}/chunks", json={"chunk_size": 10, "overlap": 10}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/runs/{run}/search", json={"query": " "}).status_code, 422)
        self.assertEqual(self.client.post("/api/runs", files={"file": ("fake.pdf", b"no PDF")}).status_code, 422)
        bad = {"text": "hello", "paragraphs": [{"char_start": 0, "char_end": 50}]}
        self.assertEqual(self.client.post("/api/runs/import-cleaned", files={"file": ("bad.json", json.dumps(bad))}).status_code, 422)
        self.assertEqual(self.client.get(f"/api/runs/{run}/artifacts/state.json").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/not-a-run-id").status_code, 404)

    def test_blank_pdf_fails_as_a_task_without_chunks(self):
        pdf = pymupdf.open(); pdf.new_page(); data = pdf.tobytes(); pdf.close()
        run_id = self.client.post("/api/runs", files={"file": ("blank.pdf", data)}).json()["run_id"]
        task = self.wait(self.client.post(f"/api/runs/{run_id}/extract"), failed=True)
        self.assertIn("No usable text", task["error"])
        state = self.client.get(f"/api/runs/{run_id}").json()
        self.assertEqual(state["phase"], "uploaded")
        self.assertNotIn("chunks.jsonl", state["artifacts"])

    def test_include_references_does_not_mutate_cleaned_source(self):
        run_id = self.upload()
        base = f"/api/runs/{run_id}"
        self.wait(self.client.post(base + "/extract"))
        self.wait(self.client.post(base + "/chunks", json={"include_references": True}))
        chunks = self.client.get(base + "/chunks?limit=100").json()["items"]
        self.assertIn("Excluded bibliography", " ".join(c["text"] for c in chunks))
        self.assertTrue(any(not p["searchable"] for p in self.client.get(base + "/cleaned").json()["paragraphs"]))

    def test_busy_run_is_rejected_and_health_remains_responsive(self):
        run_id = self.upload()
        entered, release = threading.Event(), threading.Event()

        def delayed(*args, **kwargs):
            entered.set()
            release.wait(5)
            return real_extract(*args, **kwargs)

        with patch("services.workbench.extract_pdf", side_effect=delayed):
            response = self.client.post(f"/api/runs/{run_id}/extract")
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(self.client.get("/api/health").status_code, 200)
                self.assertEqual(self.client.post(f"/api/runs/{run_id}/extract").status_code, 409)
            finally:
                release.set()
            self.wait(response)

    def test_restart_marks_interrupted_jobs_and_preserves_cleaned_text(self):
        run_id = self.client.post("/api/runs/text", json={"text": "Persisted evidence."}).json()["run_id"]
        task = self.pipeline(run_id)
        task["status"] = "running"
        store = self.app.state.workbench
        write(store.root / "tasks" / (task["task_id"] + ".json"), task)
        store.update(run_id, active_task=task["task_id"])
        restarted = Workbench(store.root)
        try:
            self.assertEqual(restarted.task(task["task_id"])["status"], "failed")
            self.assertIsNone(restarted.public(run_id)["active_task"])
            self.assertTrue(restarted.artifact(run_id, "cleaned.json").is_file())
        finally:
            restarted.close()


if __name__ == "__main__":
    unittest.main()
