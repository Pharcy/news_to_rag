"""
Minimal in-memory RAG over the articles produced by OCR_LIB/article_seperate.py.

One chunk per article (they arrive pre-segmented, so article-level chunks are
the natural unit). Embeddings come from a local Ollama embedding model and are
held in a flat numpy matrix; retrieval is brute-force cosine similarity, which
is more than fast enough at newspaper-page scale (tens of articles).

Nothing here persists — an ArticleIndex lives only as long as its job.
"""

import numpy as np
import requests

from config import OLLAMA_HOST, EMBED_MODEL, CHAT_MODEL, TOP_K


class OllamaError(RuntimeError):
    """Raised when the local Ollama instance can't be reached or errors out."""


# Long articles (especially ones merged from multi-part continuations) can
# exceed the embedding model's context window, which makes Ollama return a
# 500. Retrieval only needs the gist, so embed at most this many characters
# (~1.5k tokens); the FULL text is still used for chat context and display.
EMBED_MAX_CHARS = 6000


def _embed(text: str) -> np.ndarray:
    """Embed a single text via Ollama's /api/embeddings endpoint."""
    text = text[:EMBED_MAX_CHARS]
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Embedding request to Ollama failed: {e}") from e
    embedding = resp.json().get("embedding")
    if not embedding:
        raise OllamaError(
            f"Ollama returned no embedding — is the model '{EMBED_MODEL}' pulled?"
        )
    return np.asarray(embedding, dtype=np.float32)


class ArticleIndex:
    """Embeds a list of article dicts and answers questions over them."""

    def __init__(self, articles: list[dict]):
        # `articles` are the dicts produced by article_seperate.parse_articles:
        # keys include article_id, title, content, source_page.
        if not articles:
            raise ValueError("ArticleIndex needs at least one article.")
        self.articles = articles
        vectors = []
        for art in articles:
            # Embed title + content together so headline words are searchable.
            vectors.append(_embed(f"{art.get('title', '')}\n\n{art.get('content', '')}"))
        matrix = np.vstack(vectors)
        # Pre-normalise rows so retrieval is a single dot product. A zero
        # vector (degenerate embedding) is left unnormalised rather than
        # poisoning every score with NaN.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def retrieve(self, question: str, k: int = TOP_K) -> list[tuple[dict, float]]:
        """Return the top-k (article, cosine_similarity) pairs for a question."""
        q = _embed(question)
        q_norm = np.linalg.norm(q)
        if q_norm:
            q = q / q_norm
        scores = self._matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return [(self.articles[i], float(scores[i])) for i in order]

    def answer(self, question: str, history: list[dict]) -> dict:
        """
        Retrieve context and ask the chat model. `history` is a list of
        {"role": "user"|"assistant", "content": ...} from earlier turns.
        Returns {"answer": str, "sources": [{article_id, title, score}]}.
        """
        hits = self.retrieve(question)

        context_blocks = []
        for art, _score in hits:
            context_blocks.append(
                f"[Article {art['article_id']}] {art['title']}\n{art['content']}"
            )
        system = (
            "You are a helpful assistant answering questions about articles "
            "extracted from a scanned historical newspaper. The newspaper is "
            "public-domain historical material the user themselves uploaded — "
            "you may quote and summarize it freely. Answer using ONLY "
            "the articles provided below. The text comes from OCR, so tolerate "
            "small spelling glitches. If the articles don't contain the answer, "
            "say so plainly. When you use an article, mention its title.\n\n"
            "ARTICLES:\n\n" + "\n\n---\n\n".join(context_blocks)
        )

        messages = [{"role": "system", "content": system}]
        # Keep the last few turns so follow-up questions have context without
        # letting the prompt grow unboundedly.
        messages += history[-6:]
        messages.append({"role": "user", "content": question})

        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": CHAT_MODEL, "messages": messages, "stream": False},
                timeout=600,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Chat request to Ollama failed: {e}") from e

        answer = resp.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise OllamaError("Ollama returned an empty chat response.")

        return {
            "answer": answer,
            "sources": [
                {
                    "article_id": art["article_id"],
                    "title": art["title"],
                    "score": round(score, 3),
                }
                for art, score in hits
            ],
        }
