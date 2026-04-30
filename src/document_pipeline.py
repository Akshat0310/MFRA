from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from src.corpus_manager import ROOT_DIR, CorpusAsset, corpus_summary, discover_corpus_assets


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "during",
    "under",
    "month",
    "period",
    "schemes",
    "scheme",
    "fund",
    "funds",
}


@dataclass(frozen=True)
class DocumentPage:
    doc_id: str
    title: str
    path: str
    page_number: int
    text: str
    word_count: int
    source_id: str
    source_label: str
    source_type: str
    trust_tier: str
    source_url: str
    domain: str
    document_kind: str
    tags: tuple[str, ...]
    published_date: str | None


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    path: str
    page_numbers: tuple[int, ...]
    text: str
    word_count: int
    source_id: str
    source_label: str
    source_type: str
    trust_tier: str
    source_url: str
    domain: str
    document_kind: str
    tags: tuple[str, ...]
    published_date: str | None


@dataclass(frozen=True)
class DocumentSearchResult:
    chunk: DocumentChunk
    score: float
    matched_terms: tuple[str, ...]


def list_document_paths(root_dir: str = str(ROOT_DIR)) -> tuple[Path, ...]:
    return tuple(Path(asset.path) for asset in discover_corpus_assets(root_dir))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def document_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return str(relative).replace("/", "_").replace("\\", "_").replace(".", "_").lower()


def load_markdown_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace("```", " ")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    return normalize_whitespace(text)


def load_plain_text(path: Path) -> str:
    return normalize_whitespace(path.read_text(encoding="utf-8"))


def load_html_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = HTML_TAG_PATTERN.sub(" ", text)
    return normalize_whitespace(text)


def build_page(asset: CorpusAsset, root: Path, page_number: int, text: str) -> DocumentPage:
    return DocumentPage(
        doc_id=document_id(Path(asset.path), root),
        title=asset.title,
        path=asset.path,
        page_number=page_number,
        text=text,
        word_count=len(text.split()),
        source_id=asset.source_id,
        source_label=asset.source_label,
        source_type=asset.source_type,
        trust_tier=asset.trust_tier,
        source_url=asset.source_url,
        domain=asset.domain,
        document_kind=asset.document_kind,
        tags=asset.tags,
        published_date=asset.published_date,
    )


@lru_cache(maxsize=2)
def load_document_pages(root_dir: str = str(ROOT_DIR)) -> tuple[DocumentPage, ...]:
    root = Path(root_dir)
    pages: list[DocumentPage] = []

    for asset in discover_corpus_assets(root_dir):
        path = Path(asset.path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = normalize_whitespace(page.extract_text() or "")
                if text:
                    pages.append(build_page(asset, root, page_number, text))
            continue

        if suffix == ".md":
            text = load_markdown_text(path)
        elif suffix == ".txt":
            text = load_plain_text(path)
        elif suffix in {".html", ".htm"}:
            text = load_html_text(path)
        else:
            continue

        if text:
            pages.append(build_page(asset, root, 1, text))

    return tuple(pages)


def chunk_words(words: list[str], chunk_size: int = 140, overlap: int = 35) -> list[list[str]]:
    if not words:
        return []

    chunks: list[list[str]] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


@lru_cache(maxsize=2)
def load_document_chunks(
    root_dir: str = str(ROOT_DIR),
    chunk_size: int = 140,
    overlap: int = 35,
) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    pages = load_document_pages(root_dir)

    for page in pages:
        page_chunks = chunk_words(page.text.split(), chunk_size=chunk_size, overlap=overlap)
        for index, words in enumerate(page_chunks, start=1):
            text = " ".join(words)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{page.doc_id}-p{page.page_number}-c{index}",
                    doc_id=page.doc_id,
                    title=page.title,
                    path=page.path,
                    page_numbers=(page.page_number,),
                    text=text,
                    word_count=len(words),
                    source_id=page.source_id,
                    source_label=page.source_label,
                    source_type=page.source_type,
                    trust_tier=page.trust_tier,
                    source_url=page.source_url,
                    domain=page.domain,
                    document_kind=page.document_kind,
                    tags=page.tags,
                    published_date=page.published_date,
                )
            )

    return tuple(chunks)


def document_library_summary(
    pages: tuple[DocumentPage, ...] | None = None,
    chunks: tuple[DocumentChunk, ...] | None = None,
) -> dict[str, object]:
    page_catalog = pages if pages is not None else load_document_pages()
    chunk_catalog = chunks if chunks is not None else load_document_chunks()

    docs: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "title": "",
            "path": "",
            "page_count": 0,
            "chunk_count": 0,
            "word_count": 0,
            "source_label": "",
            "source_type": "",
            "trust_tier": "",
            "domain": "",
            "document_kind": "",
        }
    )

    for page in page_catalog:
        entry = docs[page.doc_id]
        entry["title"] = page.title
        entry["path"] = page.path
        entry["page_count"] = int(entry["page_count"]) + 1
        entry["word_count"] = int(entry["word_count"]) + page.word_count
        entry["source_label"] = page.source_label
        entry["source_type"] = page.source_type
        entry["trust_tier"] = page.trust_tier
        entry["domain"] = page.domain
        entry["document_kind"] = page.document_kind

    for chunk in chunk_catalog:
        docs[chunk.doc_id]["chunk_count"] = int(docs[chunk.doc_id]["chunk_count"]) + 1

    ordered_docs = sorted(docs.values(), key=lambda item: str(item["title"]).lower())
    source_type_distribution = Counter(page.source_type for page in page_catalog)
    trust_distribution = Counter(page.trust_tier for page in page_catalog)
    corpus_meta = corpus_summary()

    return {
        "document_count": len(ordered_docs),
        "page_count": len(page_catalog),
        "chunk_count": len(chunk_catalog),
        "source_count": len({page.source_id for page in page_catalog}),
        "documents": ordered_docs,
        "source_type_distribution": sorted(source_type_distribution.items()),
        "trust_distribution": sorted(trust_distribution.items()),
        "configured_source_count": corpus_meta["configured_source_count"],
        "indexed_asset_count": corpus_meta["indexed_asset_count"],
    }


def search_document_chunks(
    query: str,
    chunks: tuple[DocumentChunk, ...] | None = None,
    top_n: int = 5,
) -> list[DocumentSearchResult]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    chunk_catalog = chunks if chunks is not None else load_document_chunks()
    results: list[DocumentSearchResult] = []

    for chunk in chunk_catalog:
        text_lower = chunk.text.lower()
        token_counts = Counter(tokenize(chunk.text))
        matched_terms = sorted(token for token in query_tokens if token in token_counts)
        if not matched_terms:
            continue

        score = 0.0
        for term in matched_terms:
            score += 1.5 + token_counts[term]

        phrase_bonus = query.lower().strip()
        if phrase_bonus and phrase_bonus in text_lower:
            score += 4.0

        title_tokens = set(tokenize(chunk.title))
        score += 0.5 * len(title_tokens.intersection(query_tokens))

        results.append(
            DocumentSearchResult(
                chunk=chunk,
                score=round(score, 2),
                matched_terms=tuple(matched_terms),
            )
        )

    results.sort(key=lambda item: (item.score, item.chunk.trust_tier, item.chunk.title), reverse=True)
    return results[:top_n]
