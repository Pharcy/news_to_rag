"""
Glue between the webapp and the existing OCR_LIB scripts.

How it hooks into the existing pipeline:

1. OCR_LIB/Pdf_to_json.py -> ocr_pdf(pdf_path, output_dir, progress_callback)
   Rasterizes the PDF at 300 dpi, OCRs each page with Tesseract, and writes
   `<pdf_stem>.json` into output_dir ({"pages": [{"page_number", "text", ...}]}).
   We call it directly (it was already an importable function) and read that
   JSON back. The progress_callback parameter was added (backward-compatibly)
   so the UI can show per-page OCR progress.

2. OCR_LIB/article_seperate.py -> call_ollama_llm() + parse_articles()
   We deliberately call these two lower-level functions instead of its
   process_json_file() wrapper: the wrapper writes one JSON file per article
   to disk and prints previews, none of which a web backend wants. We invoke
   the LLM once per OCR'd page (matching how parse_articles' source_metadata
   is shaped) and renumber article_ids globally afterwards.

Each upload becomes a Job processed on a background thread; the frontend
polls /api/status for the fields updated here. Jobs live only in memory.
"""

import json
import sys
import tempfile
import threading
import uuid
from pathlib import Path

# Make OCR_LIB importable regardless of the working directory the server
# was started from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OCR_LIB"))

from Pdf_to_json import ocr_pdf                       # noqa: E402
from article_seperate import call_ollama_llm, parse_articles  # noqa: E402

from config import SEGMENT_MODEL                      # noqa: E402
from rag import ArticleIndex, OllamaError             # noqa: E402


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.stage = "queued"        # queued -> ocr -> segmenting -> embedding -> ready | error
        self.message = "Waiting to start…"
        self.error = None            # plain-language error string when stage == "error"
        self.articles = []           # dicts from parse_articles
        self.index = None            # ArticleIndex once embeddings are built
        self.chat_history = []       # [{"role", "content"}] for follow-up context
        self.lock = threading.Lock()

    def set(self, stage=None, message=None, error=None):
        with self.lock:
            if stage:
                self.stage = stage
            if message:
                self.message = message
            if error:
                self.stage = "error"
                self.error = error

    def status(self):
        with self.lock:
            return {
                "job_id": self.id,
                "stage": self.stage,
                "message": self.message,
                "error": self.error,
                "article_count": len(self.articles),
                "articles": [
                    {"article_id": a["article_id"], "title": a["title"]}
                    for a in self.articles
                ] if self.stage == "ready" else [],
            }


JOBS: dict[str, Job] = {}


def start_job(pdf_path: Path) -> Job:
    """Create a Job for pdf_path and process it on a background thread."""
    job = Job()
    JOBS[job.id] = job
    threading.Thread(target=_run, args=(job, pdf_path), daemon=True).start()
    return job


def _run(job: Job, pdf_path: Path):
    try:
        _process(job, pdf_path)
    except OllamaError as e:
        job.set(error=f"Could not talk to the local language model. ({e})")
    except Exception as e:  # anything else -> plain-language message, no traceback in UI
        print(f"[job {job.id}] unexpected failure: {e!r}")
        job.set(error="Something went wrong while processing this file. "
                      "Please check it is a valid PDF and try again.")


def _process(job: Job, pdf_path: Path):
    # --- Stage 1a: OCR via Pdf_to_json.ocr_pdf -----------------------------
    job.set(stage="ocr", message="Scanning PDF…")
    workdir = Path(tempfile.mkdtemp(prefix=f"newsrag_{job.id}_"))

    def on_page(page_num, total):
        job.set(message=f"Scanning PDF — reading page {page_num} of {total}…")

    # ocr_pdf returns False on any failure (it swallows the exception and
    # prints it), and writes <stem>.json into workdir on success.
    if not ocr_pdf(str(pdf_path), str(workdir), progress_callback=on_page):
        job.set(error="Could not read this file. It may be corrupted, "
                      "password-protected, or not a real PDF.")
        return

    ocr_json_path = workdir / f"{pdf_path.stem}.json"
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr_data = json.load(f)

    pages = [p for p in ocr_data.get("pages", []) if p.get("text", "").strip()]
    if not pages:
        job.set(error="The scan finished but no readable text was found. "
                      "The PDF may be blank or too low-quality to OCR.")
        return

    # --- Stage 1b: LLM segmentation via article_seperate -------------------
    articles = []
    for i, page in enumerate(pages, start=1):
        job.set(stage="segmenting",
                message=f"Separating articles — page {i} of {len(pages)}…")

        llm_response = call_ollama_llm(page["text"], model=SEGMENT_MODEL)
        if llm_response is None:
            raise OllamaError("segmentation model returned no response")

        source_metadata = {
            "source_file": ocr_data.get("source_file", str(pdf_path)),
            "page_number": page.get("page_number", i),
            "total_pages": ocr_data.get("total_pages", len(pages)),
            "processed_date": ocr_data.get("processed_date", ""),
        }
        articles.extend(parse_articles(llm_response, source_metadata))

    if not articles:
        job.set(error="No articles could be identified in this document. "
                      "It may not contain newspaper-style text.")
        return

    # parse_articles numbers per call; renumber globally across pages.
    for n, art in enumerate(articles, start=1):
        art["article_id"] = n

    # --- Stage 2 prep: embed articles for RAG -------------------------------
    job.set(stage="embedding",
            message=f"Found {len(articles)} articles — preparing them for chat…")
    index = ArticleIndex(articles)

    with job.lock:
        job.articles = articles
        job.index = index
        job.stage = "ready"
        job.message = f"Found {len(articles)} articles. Ready to chat!"
