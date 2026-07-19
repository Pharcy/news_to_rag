"""
FastAPI backend for the PDF -> Articles -> Chat showcase.

Run with:  python server.py   (from the webapp/ directory)

Endpoints:
  GET  /                     the single-page frontend
  GET  /api/sample           whether an example PDF is available
  POST /api/process          multipart PDF upload -> starts a job
  POST /api/process-sample   process the bundled example PDF
  GET  /api/status/{job_id}  polled by the frontend during processing
  POST /api/chat/{job_id}    {"question": ...} -> RAG answer + sources
"""

import shutil
import tempfile
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from pipeline import JOBS, active_job_count, prewarm_sample, start_job
from rag import OllamaError

app = FastAPI(title="Newspaper OCR + RAG showcase")


@app.on_event("startup")
def _prewarm_sample_cache():
    """Build the sample's embeddings in the background at startup so the
    'Try the example newspaper' button responds instantly."""
    sample = _find_sample_pdf()
    if sample is not None:
        threading.Thread(target=prewarm_sample, args=(sample,), daemon=True).start()

STATIC_DIR = Path(__file__).parent / "static"


def _find_sample_pdf() -> Path | None:
    """First *.pdf dropped into webapp/sample/ is offered as the example."""
    if config.SAMPLE_DIR.is_dir():
        pdfs = sorted(config.SAMPLE_DIR.glob("*.pdf")) + sorted(config.SAMPLE_DIR.glob("*.PDF"))
        if pdfs:
            return pdfs[0]
    return None


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/sample")
def sample_info():
    sample = _find_sample_pdf()
    return {"available": sample is not None,
            "name": sample.name if sample else None}


@app.get("/api/sample-pdf")
def sample_pdf():
    """Serve the example PDF inline so users can view (and save) it."""
    sample = _find_sample_pdf()
    if sample is None:
        return JSONResponse(status_code=404, content={
            "error": "No example PDF is installed on this server yet."})
    return FileResponse(sample, media_type="application/pdf",
                        filename=sample.name, content_disposition_type="inline")


@app.post("/api/process")
async def process_upload(file: UploadFile, fast: str = Form("0")):
    if not (file.filename or "").lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={
            "error": "That doesn't look like a PDF. Please upload a .pdf file."})
    fast_mode = fast.strip().lower() in ("1", "true", "on", "yes")
    if active_job_count() >= config.MAX_ACTIVE_JOBS:
        return JSONResponse(status_code=429, content={
            "error": "The server is busy processing other documents. "
                     "Please try again in a few minutes."})

    # Copy the upload to a temp file the background thread can own; a plain
    # ASCII stem keeps ocr_pdf's <stem>.json output name predictable. The
    # copy is size-capped so an oversized upload can't fill the disk.
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    tmpdir = Path(tempfile.mkdtemp(prefix="newsrag_upload_"))
    pdf_path = tmpdir / "upload.pdf"
    received = 0
    with open(pdf_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            received += len(chunk)
            if received > max_bytes:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return JSONResponse(status_code=413, content={
                    "error": f"That file is too large. The limit is "
                             f"{config.MAX_UPLOAD_MB} MB."})
            out.write(chunk)

    job = start_job(pdf_path, fast=fast_mode)
    return {"job_id": job.id}


@app.post("/api/process-sample")
def process_sample():
    sample = _find_sample_pdf()
    if sample is None:
        return JSONResponse(status_code=404, content={
            "error": "No example PDF is installed on this server yet."})
    if active_job_count() >= config.MAX_ACTIVE_JOBS:
        return JSONResponse(status_code=429, content={
            "error": "The server is busy processing other documents. "
                     "Please try again in a few minutes."})
    job = start_job(sample, is_sample=True)
    return {"job_id": job.id}


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Unknown job."})
    return job.status()


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat/{job_id}")
def chat(job_id: str, req: ChatRequest):
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Unknown job."})
    if job.stage != "ready":
        return JSONResponse(status_code=409, content={
            "error": "This document isn't ready to chat with yet."})

    question = req.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Please type a question."})
    if len(question) > config.MAX_QUESTION_CHARS:
        return JSONResponse(status_code=400, content={
            "error": f"That question is too long — please keep it under "
                     f"{config.MAX_QUESTION_CHARS} characters."})

    try:
        result = job.index.answer(question, job.chat_history)
    except OllamaError as e:
        print(f"[job {job_id}] chat failure: {e}")
        return JSONResponse(status_code=502, content={
            "error": "The language model didn't respond. Please try again in a moment."})

    # Remember the exchange so follow-up questions have context. The stored
    # history is bounded so a long-lived chat can't grow memory forever.
    job.chat_history.append({"role": "user", "content": question})
    job.chat_history.append({"role": "assistant", "content": result["answer"]})
    del job.chat_history[:-config.MAX_CHAT_HISTORY]
    return result


# Mounted last so /api/* and / take precedence.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
