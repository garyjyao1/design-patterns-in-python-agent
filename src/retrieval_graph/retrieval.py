"""Document loading and retrieval utilities."""

from __future__ import annotations

import json
import re
from io import BytesIO
from dataclasses import dataclass
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

from langchain_core.documents import Document

GITHUB_API = "https://api.github.com"
TARGET_OWNER = "garyjyao"
TARGET_REPO = "design-patterns-in-python"
TARGET_BRANCH = "main"
GITHUB_ZIP_URL = (
    f"https://codeload.github.com/{TARGET_OWNER}/{TARGET_REPO}/zip/refs/heads/{TARGET_BRANCH}"
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]{2,}", text.lower()))


def _api_get_json(url: str) -> object:
    request = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=30) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _iter_repo_files(path: str = "") -> list[dict]:
    url = f"{GITHUB_API}/repos/{TARGET_OWNER}/{TARGET_REPO}/contents/{path}?ref={TARGET_BRANCH}"
    payload = _api_get_json(url)
    if isinstance(payload, dict):
        payload = [payload]
    files: list[dict] = []
    for item in payload:
        item_type = item.get("type")
        item_path = item.get("path", "")
        if item_type == "dir":
            files.extend(_iter_repo_files(item_path))
        elif item_type == "file" and item_path.endswith((".md", ".py", ".txt")):
            files.append(item)
    return files


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if end >= len(text):
            break
    return chunks


def load_documents() -> list[Document]:
    """Load and chunk design pattern docs from the target GitHub repository."""
    docs: list[Document] = []
    try:
        with urlopen(GITHUB_ZIP_URL, timeout=30) as response:  # nosec B310
            zip_bytes = response.read()
        with ZipFile(BytesIO(zip_bytes)) as archive:
            for name in archive.namelist():
                if not name.endswith((".md", ".py", ".txt")):
                    continue
                if name.endswith("/"):
                    continue
                path = name.split("/", 1)[1] if "/" in name else name
                text = archive.read(name).decode("utf-8", errors="ignore")
                for idx, chunk in enumerate(_chunk_text(text)):
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": f"https://github.com/{TARGET_OWNER}/{TARGET_REPO}/blob/{TARGET_BRANCH}/{path}",
                                "path": path,
                                "chunk": idx,
                            },
                        )
                    )
    except (HTTPError, URLError, TimeoutError):
        for item in _iter_repo_files():
            download_url = item.get("download_url")
            path = item.get("path", "")
            if not download_url:
                continue
            text = urlopen(download_url, timeout=30).read().decode("utf-8", errors="ignore")  # nosec B310
            for idx, chunk in enumerate(_chunk_text(text)):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": f"https://github.com/{TARGET_OWNER}/{TARGET_REPO}/blob/{TARGET_BRANCH}/{path}",
                            "path": path,
                            "chunk": idx,
                        },
                    )
                )
    return docs


@dataclass
class SimpleKeywordRetriever:
    """Simple lexical retriever for offline and deterministic behavior."""

    documents: list[Document]

    def invoke(self, query: str, k: int = 4) -> list[Document]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return self.documents[:k]
        scored: list[tuple[int, Document]] = []
        for doc in self.documents:
            score = len(query_tokens & _tokenize(doc.page_content))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]


@lru_cache(maxsize=1)
def get_default_retriever() -> SimpleKeywordRetriever:
    """Create and cache the default retriever for the target repository docs."""
    return SimpleKeywordRetriever(load_documents())
