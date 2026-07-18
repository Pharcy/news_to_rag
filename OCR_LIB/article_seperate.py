import json
import os
import requests
import re
from difflib import SequenceMatcher

# Ollama host is overridable via env var so this module works both on the CLI
# and from the webapp without code changes. Default preserves old behaviour.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

def call_ollama_llm(text, model="gpt-oss:latest", base_url=None):
    """
    Send text to Ollama LLM and get response
    """
    url = f"{base_url or OLLAMA_HOST}/api/generate"
    
    prompt = f"""You are tasked with separating newspaper articles from the following text. 
Each article should be clearly delimited and formatted with a title.

Format your response EXACTLY like this for each article:
<<<ARTICLE_START>>>
TITLE: [Extract the article's headline/title here]
CONTENT:
[Article content goes here]
<<<ARTICLE_END>>>

IMPORTANT: 
- Every article MUST have a TITLE: line at the start
- After TITLE:, add a CONTENT: line before the article text
- Use <<<ARTICLE_START>>> and <<<ARTICLE_END>>> as delimiters for EACH article

Here is the text to process:

{text}"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            # A full newspaper page of OCR text easily exceeds Ollama's
            # default context window, which silently truncates the input and
            # drops most of the articles. 16k covers a large page + output.
            "num_ctx": 16384,
            # Deterministic output: segmentation shouldn't vary between runs.
            "temperature": 0,
        },
    }
    
    try:
        # LLM segmentation of a full newspaper page can take minutes; the
        # timeout guards against a hung Ollama instance blocking forever.
        response = requests.post(url, json=payload, timeout=900)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

def parse_articles(llm_response, source_metadata):
    """
    Parse individual articles from LLM response using delimiters
    Extract title and content, and include source metadata
    """
    # Split by article markers
    article_pattern = r'<<<ARTICLE_START>>>(.*?)<<<ARTICLE_END>>>'
    articles = re.findall(article_pattern, llm_response, re.DOTALL)
    
    # Clean up each article and extract title
    parsed_articles = []
    for i, article in enumerate(articles, 1):
        article = article.strip()
        if not article:
            continue
        
        # Try to extract title and content
        title = f"Article {i}"  # Default title
        content = article
        
        # Look for TITLE: and CONTENT: markers
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', article, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            # Remove the TITLE: line from content
            content = re.sub(r'TITLE:\s*.+?(?:\n|$)', '', article, flags=re.IGNORECASE)
        
        # Remove CONTENT: marker if present
        content = re.sub(r'^\s*CONTENT:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = content.strip()
        
        parsed_articles.append({
            "article_id": i,
            "title": title,
            "content": content,
            "source_file": source_metadata.get("source_file", "Unknown"),
            "source_page": source_metadata.get("page_number", "Unknown"),
            "total_source_pages": source_metadata.get("total_pages", "Unknown"),
            "processed_date": source_metadata.get("processed_date", "Unknown")
        })
    
    return parsed_articles

# ---------------------------------------------------------------------------
# Merging continued articles
#
# Newspaper stories often run "SMITH WINS ELECTION" on page 1 and reappear as
# "Smith Wins Election (Continued from Page 8)" later. Exact title matching
# can't reunite them: continuation tags, OCR misreads, and LLM re-wording all
# make the titles differ. This merger therefore combines three signals:
#   1. titles are normalised (continuation tags stripped, case/punctuation
#      folded) before comparison,
#   2. similarity is fuzzy (difflib ratio) rather than exact, and
#   3. an explicit "(continued …)" marker in the title or opening text lowers
#      the similarity bar and requires merging backward into an earlier page.
# ---------------------------------------------------------------------------

_CONTINUATION_RE = re.compile(
    r"[\(\[]?\s*(?:continued|cont'?d|cont\.)\s*(?:from|on)?\s*"
    r"(?:page\s*\d+|p\.?\s*\d+)?\s*[\)\]]?",
    re.IGNORECASE,
)


def _strip_continuation(title):
    """Remove '(continued from page N)'-style tags and tidy the remains."""
    cleaned = _CONTINUATION_RE.sub(" ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—:,.")
    return cleaned


def _normalize_title(title):
    """Fold a title down to lowercase alphanumerics for comparison."""
    t = _strip_continuation(title).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_continuation(article):
    """True if the title or the opening of the content carries a
    continuation marker."""
    if _CONTINUATION_RE.search(article.get("title", "")):
        return True
    head = article.get("content", "")[:200]
    return bool(re.search(r"continued\s+from", head, re.IGNORECASE))


def merge_continued_articles(articles, fuzzy_threshold=0.75,
                             continued_threshold=0.5):
    """
    Merge multi-page continuations into their parent articles.

    `articles` must be in reading order (page, then position). Each candidate
    is compared against every already-merged article:
      - identical normalised titles always merge (old exact behaviour);
      - otherwise a fuzzy match needs `fuzzy_threshold` similarity, a
        reasonably long title, and a *different* source page;
      - articles explicitly marked "(continued)" merge at the lower
        `continued_threshold`, but only into an article from an earlier page.
    Unmatched articles are kept as-is (with any continuation tag stripped
    from their title). Article IDs are renumbered sequentially.
    """
    merged = []

    for art in articles:
        key = _normalize_title(art.get("title", ""))
        is_cont = _is_continuation(art)
        page = art.get("source_page")

        best, best_score = None, 0.0
        for prev in merged:
            score = SequenceMatcher(None, key, prev["_key"]).ratio() if key else 0.0
            if score <= best_score:
                continue
            same_page = page is not None and page in prev["source_pages"]
            if score >= 1.0 and prev["_key"]:
                best, best_score = prev, score        # exact: always eligible
            elif is_cont and not same_page and score >= continued_threshold:
                best, best_score = prev, score        # marked continuation
            elif not same_page and len(key) >= 8 and score >= fuzzy_threshold:
                best, best_score = prev, score        # fuzzy cross-page match
        # A marked continuation must flow backward: never merge into an
        # article that only exists on the same or a later page.
        if best is not None and is_cont and page is not None:
            if not any(p is not None and p < page for p in best["source_pages"]) \
               and page not in best["source_pages"]:
                best = None

        if best is not None:
            best["content"] = best["content"].rstrip() + "\n\n" + art["content"].strip()
            if page is not None and page not in best["source_pages"]:
                best["source_pages"].append(page)
        else:
            copy = dict(art)
            copy["title"] = _strip_continuation(art.get("title", "")) or art.get("title", "")
            copy["_key"] = key
            copy["source_pages"] = [page] if page is not None else []
            merged.append(copy)

    for i, art in enumerate(merged, 1):
        art["article_id"] = i
        art.pop("_key", None)
    return merged


def extract_text_from_json(data):
    """
    Extract text from the JSON structure
    Handles the nested pages structure
    """
    # Check if 'pages' exists and has content
    if 'pages' in data and isinstance(data['pages'], list) and len(data['pages']) > 0:
        # Concatenate text from all pages
        all_text = []
        for page in data['pages']:
            if 'text' in page:
                all_text.append(page['text'])
        
        return '\n\n'.join(all_text)
    
    # Fallback: check for direct 'text' field
    elif 'text' in data:
        return data['text']
    
    else:
        return None

def process_json_file(json_file_path, output_dir="output_articles"):
    """
    Main function to process JSON file and extract articles
    Saves each article as a separate JSON file
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file_path}'")
        return
    
    # Extract text from JSON
    text_content = extract_text_from_json(data)
    
    if not text_content:
        print("Error: Could not find text content in JSON")
        print("Available keys:", list(data.keys()))
        return
    
    # Prepare source metadata
    source_metadata = {
        "source_file": data.get('source_file', 'Unknown'),
        "total_pages": data.get('total_pages', 'Unknown'),
        "processed_date": data.get('processed_date', 'Unknown'),
        "page_number": data.get('pages', [{}])[0].get('page_number', 1) if data.get('pages') else 1
    }
    
    print(f"Extracted {len(text_content.split())} words from JSON")
    print(f"Source file: {source_metadata['source_file']}")
    print(f"Total pages: {source_metadata['total_pages']}")
    
    # Send to LLM
    print("\nSending text to Ollama LLM (gpt-oss)...")
    llm_response = call_ollama_llm(text_content)
    
    if not llm_response:
        print("Failed to get response from LLM")
        return
    
    print("Received response from LLM")
    
    # Parse articles
    print("\nParsing articles...")
    articles = parse_articles(llm_response, source_metadata)
    
    print(f"\nFound {len(articles)} articles")
    
    # Save each article as a separate JSON file
    saved_files = []
    for article in articles:
        # Create safe filename from title
        safe_title = re.sub(r'[^\w\s-]', '', article['title'])
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        safe_title = safe_title[:50]  # Limit length
        
        filename = f"article_{article['article_id']}_{safe_title}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
        
        saved_files.append(filename)
        
        # Display preview
        preview = article['content'][:150] + "..." if len(article['content']) > 150 else article['content']
        print(f"\n--- Article {article['article_id']}: {article['title']} ---")
        print(f"File: {filename}")
        print(preview)
    
    # # Also save a master index file
    # index_file = os.path.join(output_dir, "_index.json")
    # index_data = {
    #     "source_file": source_metadata['source_file'],
    #     "total_pages": source_metadata['total_pages'],
    #     "processed_date": source_metadata['processed_date'],
    #     "total_articles": len(articles),
    #     "articles": [
    #         {
    #             "article_id": a['article_id'],
    #             "title": a['title'],
    #             "filename": saved_files[i]
    #         }
    #         for i, a in enumerate(articles)
    #     ]
    # }
    
    # with open(index_file, 'w', encoding='utf-8') as f:
    #     json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ {len(articles)} articles saved to '{output_dir}/' directory")
    
    return articles

if __name__ == "__main__":
    # Replace with your JSON file path
    json_file_path = "output/wyubdi_20040901_0006.json"
    
    # Process the file (will save to 'output_articles' directory by default)
    articles = process_json_file(json_file_path)
