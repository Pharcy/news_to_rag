"""
Central configuration for the webapp. Everything is overridable via
environment variables so nothing is hardcoded to one machine.
"""

import os
from pathlib import Path

# Where the local Ollama instance lives.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Model used by article_seperate.py to segment OCR text into articles.
SEGMENT_MODEL = os.environ.get("SEGMENT_MODEL", "llama3.1:8b")

# Model used to answer chat questions over the extracted articles.
# Deliberately lightweight so chat responses feel fast.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "phi4-mini:latest")

# Embedding model for RAG retrieval.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# How many articles to hand the chat model as context per question.
TOP_K = int(os.environ.get("TOP_K", "3"))

# OCR rasterization resolution. High quality is the default; "fast mode"
# (user-selectable on the upload screen) trades accuracy for ~4x less work.
OCR_DPI = int(os.environ.get("OCR_DPI", "600"))
FAST_OCR_DPI = int(os.environ.get("FAST_OCR_DPI", "300"))

# Directory holding the clickable example PDF (first *.pdf found is offered).
SAMPLE_DIR = Path(__file__).parent / "sample"

# Server bind settings.
HOST = os.environ.get("WEBAPP_HOST", "0.0.0.0")
PORT = int(os.environ.get("WEBAPP_PORT", "8000"))

# ---- Abuse guards -------------------------------------------------------
# Largest PDF upload accepted (a broadsheet scan is typically well under this).
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
# Longest chat question accepted.
MAX_QUESTION_CHARS = int(os.environ.get("MAX_QUESTION_CHARS", "2000"))
# How many uploads may be OCR'd/segmented at once before new ones get a 429.
MAX_ACTIVE_JOBS = int(os.environ.get("MAX_ACTIVE_JOBS", "3"))
# Total jobs kept in memory; oldest finished jobs are evicted past this.
MAX_KEPT_JOBS = int(os.environ.get("MAX_KEPT_JOBS", "50"))
# Chat messages remembered per job (user + assistant turns combined).
MAX_CHAT_HISTORY = int(os.environ.get("MAX_CHAT_HISTORY", "20"))
