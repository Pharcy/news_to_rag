# Newspaper Digitizer — OCR + Local RAG Showcase

A self-hosted, single-page webapp that turns a scanned newspaper PDF into a
chat session:

1. **Upload & Process** — the PDF is OCR'd with Tesseract
   (`OCR_LIB/Pdf_to_json.py`), then a local Ollama LLM segments the raw text
   into individual articles (`OCR_LIB/article_seperate.py`).
2. **Chat** — the extracted articles are embedded with a local Ollama
   embedding model and used as the knowledge base for a simple RAG chat.
   Each answer shows which articles were used as sources.

Everything runs locally: Tesseract for OCR, Ollama for the LLM and
embeddings. No cloud calls, no database — state is in-memory per session.

## Requirements

- Python with: `fastapi`, `uvicorn`, `requests`, `numpy`, `pytesseract`,
  `pdf2image`, `opencv-python`, `pillow`
  (on this server the `RAG` conda env has all of these)
- System packages: `tesseract-ocr`, `poppler-utils` (for `pdftoppm`)
- A running [Ollama](https://ollama.com) with the models pulled:
  ```
  ollama pull llama3.1:8b      # article segmentation (default)
  ollama pull phi4-mini        # chat answers (default; small = fast)
  ollama pull nomic-embed-text # embeddings
  ```

## Run it

```bash
cd webapp
/home/crothfu1/miniconda3/envs/RAG/bin/python server.py
```

Then open http://localhost:8000 (or the server's address). Port and bind
address are configurable — see below.

> Note: on this particular server ports 8000 and 8010 are already occupied
> by another app, so start with e.g. `WEBAPP_PORT=8137 python server.py`.

## Configuration (all via environment variables)

| Variable        | Default                  | Purpose                                   |
|-----------------|--------------------------|-------------------------------------------|
| `OLLAMA_HOST`   | `http://localhost:11434` | Where Ollama is listening                  |
| `SEGMENT_MODEL` | `llama3.1:8b`            | Model that splits OCR text into articles   |
| `CHAT_MODEL`    | `phi4-mini:latest`       | Model that answers chat questions (small = fast) |
| `EMBED_MODEL`   | `nomic-embed-text`       | Embedding model for retrieval              |
| `TOP_K`         | `3`                      | Articles retrieved per question            |
| `WEBAPP_HOST`   | `0.0.0.0`                | Bind address                               |
| `WEBAPP_PORT`   | `8000`                   | Port                                       |

Example — point at a remote Ollama box and a different chat model:

```bash
OLLAMA_HOST=http://10.0.0.5:11434 CHAT_MODEL=llama3.1:8b python server.py
```

`OLLAMA_HOST` is also respected by `OCR_LIB/article_seperate.py` itself, so
the CLI workflow follows the same setting.

## Where to drop the example PDF

Put a PDF at `webapp/sample/` (any `*.pdf`; the first one alphabetically is
used). The upload page then shows a **"Try the example newspaper"** button.
No PDF in that directory → the button is hidden. Nothing else to configure.

## How it hooks into the existing scripts

- `webapp/pipeline.py` imports `ocr_pdf()` from `OCR_LIB/Pdf_to_json.py` and
  runs it against a per-job temp directory, then reads back the
  `<pdf_stem>.json` it writes.
- It then calls `call_ollama_llm()` + `parse_articles()` from
  `OCR_LIB/article_seperate.py` once per OCR'd page and renumbers the
  resulting articles globally. (It bypasses `process_json_file()` because
  that wrapper writes per-article files to disk and prints previews —
  behaviour a web backend doesn't want.)
- `webapp/rag.py` embeds each article (title + content) via Ollama and does
  flat numpy cosine-similarity retrieval; top-k articles are passed as
  context to the chat model.

## Changes made to the existing scripts

Both changes are backward-compatible with CLI usage:

- `Pdf_to_json.py` — `ocr_pdf()` gained an optional `progress_callback`
  parameter so the UI can show per-page OCR progress.
- `article_seperate.py` — the Ollama URL is now read from the `OLLAMA_HOST`
  env var (default unchanged); the request got a 15-minute timeout so a hung
  Ollama can't block forever; and a `NameError` was fixed where
  `process_json_file()` printed `index_file`, a variable that only existed
  in a commented-out block (the function previously always crashed at the
  end, after saving the articles).

## Assumptions & fragility notes

- **Assumed interfaces**: `ocr_pdf(pdf_path, output_dir)` writing
  `<stem>.json` with a `pages[].text` structure, and `parse_articles()`
  consuming per-page `source_metadata` — both confirmed by reading the
  scripts, not guessed.
- **Per-page segmentation**: `process_json_file()` concatenates *all* pages
  into a single LLM prompt, which risks blowing the model's context window
  on multi-page issues. The webapp calls the LLM once per page instead —
  same functions, safer envelope. Worth adopting in the CLI script too.
- **`ocr_pdf` swallows exceptions**: it catches everything, prints, and
  returns `False`, so the webapp can only show a generic "couldn't read this
  file" message. Raising (or returning) the actual error would allow better
  diagnostics.
- **LLM output is parsed by regex**: if the segmentation model doesn't
  reproduce the `<<<ARTICLE_START>>>` delimiters faithfully, articles are
  silently dropped. The webapp reports "no articles found" in that case, but
  a retry or a stricter/structured output format would harden this.
- **No article-count sanity check**: the LLM may merge or over-split
  articles; there's no ground truth to validate against in a demo.
- Uploads and OCR output go to per-job directories under the system temp
  dir and are not persisted or cleaned until the OS clears temp storage.
