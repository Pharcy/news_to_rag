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
from article_seperate import (                        # noqa: E402
    call_ollama_llm,
    merge_continued_articles,
    parse_articles,
)

import config                                         # noqa: E402
from config import SEGMENT_MODEL                      # noqa: E402
from rag import ArticleIndex, OllamaError             # noqa: E402

# ---------------------------------------------------------------------------
# Sample-PDF result cache
#
# The bundled example PDF never changes between clicks, so its OCR + article
# separation results are saved to disk the first time it is processed
# (webapp/sample/sample_cache.json) and reused afterwards — clicking "Try the
# example newspaper" then takes seconds (just embeddings) instead of minutes.
# The embeddings themselves are additionally kept in memory (prewarmed at
# server startup), making subsequent clicks effectively instant.
#
# The cache is keyed on the sample PDF's name and size, so dropping a
# different PDF into webapp/sample/ automatically invalidates it.
# ---------------------------------------------------------------------------

SAMPLE_CACHE_FILE = config.SAMPLE_DIR / "sample_cache.json"

_sample_cache = {"fingerprint": None, "articles": None, "index": None}
_sample_lock = threading.Lock()


def _fingerprint(pdf_path: Path) -> dict:
    st = pdf_path.stat()
    return {"name": pdf_path.name, "size": st.st_size}


def _load_sample_cache(pdf_path: Path) -> bool:
    """Make the in-memory sample index available; True if cache was usable."""
    fp = _fingerprint(pdf_path)
    with _sample_lock:
        if _sample_cache["index"] is not None and _sample_cache["fingerprint"] == fp:
            return True
    if not SAMPLE_CACHE_FILE.exists():
        return False
    try:
        data = json.loads(SAMPLE_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if data.get("fingerprint") != fp or not data.get("articles"):
        return False
    # Merging is applied on load too, so caches seeded before the continued-
    # article merger existed still come out combined and with clean titles.
    articles = merge_continued_articles(data["articles"])
    # Only embeddings are recomputed (a few seconds); OCR + LLM are skipped.
    index = ArticleIndex(articles)
    with _sample_lock:
        _sample_cache.update(fingerprint=fp, articles=articles, index=index)
    return True


def _save_sample_cache(pdf_path: Path, articles: list, index: ArticleIndex):
    fp = _fingerprint(pdf_path)
    SAMPLE_CACHE_FILE.write_text(
        json.dumps({"fingerprint": fp, "articles": articles},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with _sample_lock:
        _sample_cache.update(fingerprint=fp, articles=articles, index=index)


def prewarm_sample(pdf_path: Path):
    """Called on server startup so the first sample click is instant."""
    try:
        if _load_sample_cache(pdf_path):
            print(f"Sample cache prewarmed for {pdf_path.name}")
    except Exception as e:
        print(f"Sample prewarm skipped: {e!r}")


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
                # Full content included so the UI can show each article's
                # text on click; only sent once processing is finished
                # (polling stops at "ready", so this isn't resent repeatedly).
                "articles": [
                    {
                        "article_id": a["article_id"],
                        "title": a["title"],
                        "content": a["content"],
                        "source_page": a.get("source_page"),
                    }
                    for a in self.articles
                ] if self.stage == "ready" else [],
            }


JOBS: dict[str, Job] = {}


def start_job(pdf_path: Path, is_sample: bool = False, fast: bool = False) -> Job:
    """Create a Job for pdf_path and process it on a background thread.
    `fast` scans at the lower FAST_OCR_DPI (quicker, possibly worse OCR)."""
    job = Job()
    JOBS[job.id] = job
    threading.Thread(target=_run, args=(job, pdf_path, is_sample, fast),
                     daemon=True).start()
    return job


def _run(job: Job, pdf_path: Path, is_sample: bool = False, fast: bool = False):
    try:
        # The sample PDF's results are cached — serve them without reprocessing.
        if is_sample:
            job.set(stage="embedding", message="Loading the example newspaper…")
            if _load_sample_cache(pdf_path):
                with _sample_lock:
                    articles = _sample_cache["articles"]
                    index = _sample_cache["index"]
                with job.lock:
                    job.articles = articles
                    job.index = index
                    job.stage = "ready"
                    job.message = f"Found {len(articles)} articles. Ready to chat!"
                return
            job.set(stage="ocr", message="Scanning PDF…")  # cache miss: full run

        _process(job, pdf_path, fast=fast)

        # First successful sample run seeds the cache for future clicks.
        if is_sample and job.stage == "ready":
            _save_sample_cache(pdf_path, job.articles, job.index)
    except OllamaError as e:
        job.set(error=f"Could not talk to the local language model. ({e})")
    except Exception as e:  # anything else -> plain-language message, no traceback in UI
        print(f"[job {job.id}] unexpected failure: {e!r}")
        job.set(error="Something went wrong while processing this file. "
                      "Please check it is a valid PDF and try again.")


def _process(job: Job, pdf_path: Path, fast: bool = False):
    # --- Stage 1a: OCR via Pdf_to_json.ocr_pdf -----------------------------
    dpi = config.FAST_OCR_DPI if fast else config.OCR_DPI
    mode = " (fast mode)" if fast else ""
    job.set(stage="ocr", message=f"Scanning PDF{mode}…")
    workdir = Path(tempfile.mkdtemp(prefix=f"newsrag_{job.id}_"))

    def on_page(page_num, total):
        job.set(message=f"Scanning PDF{mode} — reading page {page_num} of {total}…")

    # ocr_pdf returns False on any failure (it swallows the exception and
    # prints it), and writes <stem>.json into workdir on success.
    if not ocr_pdf(str(pdf_path), str(workdir), progress_callback=on_page, dpi=dpi):
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

    # Reunite stories that continue across pages ("… (Continued from Page 3)")
    # with their opening part; also renumbers article_ids globally.
    articles = merge_continued_articles(articles)

    # --- Stage 2 prep: embed articles for RAG -------------------------------
    job.set(stage="embedding",
            message=f"Found {len(articles)} articles — preparing them for chat…")
    index = ArticleIndex(articles)

    with job.lock:
        job.articles = articles
        job.index = index
        job.stage = "ready"
        job.message = f"Found {len(articles)} articles. Ready to chat!"
