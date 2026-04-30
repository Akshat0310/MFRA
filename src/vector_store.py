from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from src.document_pipeline import DocumentChunk, ROOT_DIR, tokenize


DEFAULT_VECTOR_STORE_PATH = ROOT_DIR / "data" / "vector_store" / "document_chunk_index.json"
DEFAULT_VECTOR_DIMENSION = 384

DOMAIN_EXPANSIONS = {
    "aum": ("assets", "under", "management"),
    "nav": ("net", "asset", "value"),
    "idcw": ("dividend", "payout"),
    "debt": ("income", "debt"),
    "equity": ("growth", "equity"),
    "liquidity": ("liquid",),
    "inflow": ("net", "inflow"),
    "outflow": ("net", "outflow"),
    "riskometer": ("risk", "meter"),
    "sid": ("scheme", "information", "document"),
    "kim": ("key", "information", "memorandum"),
}

TRUST_MULTIPLIERS = {
    "regulatory": 1.18,
    "official": 1.14,
    "amc": 1.08,
    "reference": 1.04,
    "internal": 0.96,
}

SOURCE_TYPE_MULTIPLIERS = {
    "regulator": 1.10,
    "industry_body": 1.08,
    "amc": 1.05,
    "central_bank": 1.05,
    "government_education": 1.04,
    "project": 0.96,
    "project_data": 0.94,
}

DOCUMENT_KIND_MULTIPLIERS = {
    "pdf_snapshot": 1.05,
    "html_snapshot": 1.03,
    "markdown_snapshot": 1.0,
    "text_snapshot": 0.98,
}


@dataclass(frozen=True)
class ChunkEmbeddingRecord:
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
    vector: tuple[float, ...]


@dataclass(frozen=True)
class VectorSearchResult:
    record: ChunkEmbeddingRecord
    score: float
    matched_terms: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class ChunkVectorIndex:
    dimension: int
    chunk_count: int
    idf_map: dict[str, float]
    records: tuple[ChunkEmbeddingRecord, ...]


def hashed_index(token: str, dimension: int) -> tuple[int, int]:
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=16).digest()
    raw = int.from_bytes(digest[:8], "big")
    sign = 1 if (raw & 1) == 0 else -1
    return raw % dimension, sign


def expand_tokens(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        if token in DOMAIN_EXPANSIONS:
            expanded.extend(DOMAIN_EXPANSIONS[token])
    return expanded


def feature_tokens(text: str) -> list[str]:
    tokens = expand_tokens(tokenize(text))
    if not tokens:
        return []

    features = list(tokens)
    features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def compute_idf_map(chunks: tuple[DocumentChunk, ...]) -> dict[str, float]:
    doc_frequency: Counter[str] = Counter()
    total_documents = max(len(chunks), 1)

    for chunk in chunks:
        doc_frequency.update(set(feature_tokens(chunk.text)))

    return {
        token: math.log((1 + total_documents) / (1 + freq)) + 1.0
        for token, freq in doc_frequency.items()
    }


def l2_normalize(vector: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(0.0 for _ in vector)
    return tuple(value / norm for value in vector)


def embed_text(text: str, idf_map: dict[str, float], dimension: int) -> tuple[float, ...]:
    features = feature_tokens(text)
    if not features:
        return tuple(0.0 for _ in range(dimension))

    counts = Counter(features)
    vector = [0.0] * dimension
    for token, count in counts.items():
        index, sign = hashed_index(token, dimension)
        tf = 1.0 + math.log(count)
        weight = tf * idf_map.get(token, 1.0)
        vector[index] += sign * weight

    return l2_normalize(vector)


def dot_product(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def build_chunk_vector_index(
    chunks: tuple[DocumentChunk, ...],
    dimension: int = DEFAULT_VECTOR_DIMENSION,
) -> ChunkVectorIndex:
    idf_map = compute_idf_map(chunks)
    records = tuple(
        ChunkEmbeddingRecord(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            path=chunk.path,
            page_numbers=chunk.page_numbers,
            text=chunk.text,
            word_count=chunk.word_count,
            source_id=chunk.source_id,
            source_label=chunk.source_label,
            source_type=chunk.source_type,
            trust_tier=chunk.trust_tier,
            source_url=chunk.source_url,
            domain=chunk.domain,
            document_kind=chunk.document_kind,
            tags=chunk.tags,
            published_date=chunk.published_date,
            vector=embed_text(chunk.text, idf_map, dimension),
        )
        for chunk in chunks
    )
    return ChunkVectorIndex(
        dimension=dimension,
        chunk_count=len(records),
        idf_map=idf_map,
        records=records,
    )


def export_chunk_vector_index(index: ChunkVectorIndex, output_path: str = str(DEFAULT_VECTOR_STORE_PATH)) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dimension": index.dimension,
        "chunk_count": index.chunk_count,
        "idf_map": index.idf_map,
        "records": [asdict(record) for record in index.records],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)

    return str(path)


def load_exported_chunk_vector_index(path: str) -> ChunkVectorIndex | None:
    target = Path(path)
    if not target.exists():
        return None

    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    records = tuple(
        ChunkEmbeddingRecord(
            chunk_id=item["chunk_id"],
            doc_id=item["doc_id"],
            title=item["title"],
            path=item["path"],
            page_numbers=tuple(item["page_numbers"]),
            text=item["text"],
            word_count=item["word_count"],
            source_id=item.get("source_id", ""),
            source_label=item.get("source_label", ""),
            source_type=item.get("source_type", ""),
            trust_tier=item.get("trust_tier", ""),
            source_url=item.get("source_url", ""),
            domain=item.get("domain", ""),
            document_kind=item.get("document_kind", ""),
            tags=tuple(item.get("tags", [])),
            published_date=item.get("published_date"),
            vector=tuple(item["vector"]),
        )
        for item in payload["records"]
    )
    return ChunkVectorIndex(
        dimension=payload["dimension"],
        chunk_count=payload["chunk_count"],
        idf_map={key: float(value) for key, value in payload["idf_map"].items()},
        records=records,
    )


@lru_cache(maxsize=2)
def load_or_build_chunk_vector_index(
    chunks: tuple[DocumentChunk, ...],
    output_path: str = str(DEFAULT_VECTOR_STORE_PATH),
    dimension: int = DEFAULT_VECTOR_DIMENSION,
) -> ChunkVectorIndex:
    cached = load_exported_chunk_vector_index(output_path)
    if cached is not None and cached.dimension == dimension and cached.chunk_count == len(chunks):
        return cached

    index = build_chunk_vector_index(chunks, dimension=dimension)
    export_chunk_vector_index(index, output_path=output_path)
    return index


def retrieval_weight(record: ChunkEmbeddingRecord) -> float:
    trust_weight = TRUST_MULTIPLIERS.get(record.trust_tier, 1.0)
    type_weight = SOURCE_TYPE_MULTIPLIERS.get(record.source_type, 1.0)
    kind_weight = DOCUMENT_KIND_MULTIPLIERS.get(record.document_kind, 1.0)
    return trust_weight * type_weight * kind_weight


def search_vector_index(
    query: str,
    index: ChunkVectorIndex,
    top_n: int = 5,
) -> list[VectorSearchResult]:
    query_terms = tuple(sorted(set(tokenize(query))))
    if not query_terms:
        return []

    query_vector = embed_text(query, index.idf_map, index.dimension)
    scored: list[VectorSearchResult] = []
    for record in index.records:
        score = dot_product(query_vector, record.vector) * retrieval_weight(record)
        if score <= 0:
            continue
        scored.append(
            VectorSearchResult(
                record=record,
                score=round(score, 4),
                matched_terms=query_terms,
                mode="semantic",
            )
        )

    scored.sort(
        key=lambda item: (item.score, item.record.trust_tier, item.record.source_label, item.record.chunk_id),
        reverse=True,
    )
    return scored[:top_n]


def hybrid_search_vector_index(
    query: str,
    index: ChunkVectorIndex,
    top_n: int = 5,
) -> list[VectorSearchResult]:
    semantic_results = search_vector_index(query, index, top_n=max(top_n * 3, 10))
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    lexical_scores: dict[str, float] = {}
    lexical_terms: dict[str, tuple[str, ...]] = {}
    for record in index.records:
        terms = Counter(tokenize(record.text))
        matched = sorted(token for token in query_terms if token in terms)
        if not matched:
            continue
        score = (sum(1.0 + terms[token] for token in matched)) * retrieval_weight(record)
        lexical_scores[record.chunk_id] = score
        lexical_terms[record.chunk_id] = tuple(matched)

    semantic_lookup = {item.record.chunk_id: item.score for item in semantic_results}
    semantic_max = max(semantic_lookup.values(), default=1.0)
    lexical_max = max(lexical_scores.values(), default=1.0)

    combined: list[VectorSearchResult] = []
    for record in index.records:
        semantic_score = semantic_lookup.get(record.chunk_id, 0.0)
        lexical_score = lexical_scores.get(record.chunk_id, 0.0)
        if semantic_score == 0.0 and lexical_score == 0.0:
            continue

        combined_score = 0.62 * (semantic_score / semantic_max if semantic_max else 0.0) + 0.38 * (
            lexical_score / lexical_max if lexical_max else 0.0
        )
        combined.append(
            VectorSearchResult(
                record=record,
                score=round(combined_score, 4),
                matched_terms=lexical_terms.get(record.chunk_id, tuple(sorted(query_terms))),
                mode="hybrid",
            )
        )

    combined.sort(
        key=lambda item: (item.score, item.record.trust_tier, item.record.source_label, item.record.chunk_id),
        reverse=True,
    )
    return combined[:top_n]


def vector_index_summary(index: ChunkVectorIndex) -> dict[str, object]:
    source_counts = Counter(record.source_label for record in index.records)
    trust_counts = Counter(record.trust_tier for record in index.records)
    return {
        "dimension": index.dimension,
        "chunk_count": index.chunk_count,
        "feature_count": len(index.idf_map),
        "store_path": str(DEFAULT_VECTOR_STORE_PATH),
        "source_count": len(source_counts),
        "active_sources": sorted(source_counts.items(), key=lambda item: (item[1], item[0]), reverse=True),
        "trust_distribution": sorted(trust_counts.items()),
    }
