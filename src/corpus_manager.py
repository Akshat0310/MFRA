from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT_DIR / "data" / "corpus"
CORPUS_RAW_DIR = CORPUS_DIR / "raw"
SOURCE_REGISTRY_PATH = CORPUS_DIR / "source_registry.json"

SUPPORTED_CORPUS_SUFFIXES = (".md", ".txt", ".html", ".htm", ".pdf")


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    label: str
    source_type: str
    trust_tier: str
    domain: str
    base_url: str
    description: str
    tags: tuple[str, ...]
    enabled: bool
    ingestion_mode: str


@dataclass(frozen=True)
class CorpusAsset:
    asset_id: str
    source_id: str
    source_label: str
    source_type: str
    trust_tier: str
    domain: str
    source_url: str
    title: str
    path: str
    file_type: str
    document_kind: str
    tags: tuple[str, ...]
    published_date: str | None


def humanize_label(text: str) -> str:
    return text.replace("_", " ").replace("-", " ").strip().title()


def normalize_tags(tags: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not tags:
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for item in tags:
        cleaned = " ".join(str(item).strip().lower().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return tuple(ordered)


def project_sources() -> tuple[CorpusSource, ...]:
    return (
        CorpusSource(
            source_id="project_docs",
            label="Project Documentation",
            source_type="project",
            trust_tier="internal",
            domain="local.project",
            base_url="",
            description="Internal project docs, README notes, and implementation write-ups.",
            tags=("project", "documentation", "architecture"),
            enabled=True,
            ingestion_mode="local",
        ),
        CorpusSource(
            source_id="legacy_project_assets",
            label="Legacy Project Assets",
            source_type="project_data",
            trust_tier="internal",
            domain="local.project",
            base_url="",
            description="Local PDFs and other reference files already bundled in the repository.",
            tags=("project", "reference", "assets"),
            enabled=True,
            ingestion_mode="local",
        ),
    )


@lru_cache(maxsize=2)
def load_corpus_sources(registry_path: str = str(SOURCE_REGISTRY_PATH)) -> tuple[CorpusSource, ...]:
    sources = list(project_sources())
    path = Path(registry_path)
    if not path.exists():
        return tuple(sources)

    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload:
        sources.append(
            CorpusSource(
                source_id=item["source_id"],
                label=item["label"],
                source_type=item["source_type"],
                trust_tier=item["trust_tier"],
                domain=item["domain"],
                base_url=item.get("base_url", ""),
                description=item.get("description", ""),
                tags=normalize_tags(item.get("tags")),
                enabled=bool(item.get("enabled", True)),
                ingestion_mode=item.get("ingestion_mode", "local_snapshot"),
            )
        )
    return tuple(sources)


def build_source_lookup(sources: tuple[CorpusSource, ...]) -> dict[str, CorpusSource]:
    return {item.source_id: item for item in sources}


def list_corpus_candidate_paths(root_dir: str = str(ROOT_DIR)) -> tuple[Path, ...]:
    root = Path(root_dir)
    candidates: set[Path] = set()

    for suffix in SUPPORTED_CORPUS_SUFFIXES:
        candidates.update(path for path in (root / "data" / "corpus" / "raw").glob(f"**/*{suffix}") if path.is_file())

    candidates.update(path for path in root.glob("*.pdf") if path.is_file())

    candidates.update(path for path in root.glob("docs/*.md") if path.is_file())
    readme = root / "README.md"
    if readme.exists():
        candidates.add(readme)

    return tuple(sorted(candidates))


def infer_builtin_source(path: Path, root: Path) -> str:
    if path == root / "README.md" or path.parent == root / "docs":
        return "project_docs"
    return "legacy_project_assets"


def infer_document_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf_snapshot"
    if suffix in {".html", ".htm"}:
        return "html_snapshot"
    if suffix == ".txt":
        return "text_snapshot"
    return "markdown_snapshot"


def load_asset_sidecar(path: Path) -> dict[str, object]:
    direct_sidecar = path.with_suffix(path.suffix + ".meta.json")
    alternate_sidecar = path.with_name(path.stem + ".meta.json")
    for candidate in (direct_sidecar, alternate_sidecar):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def build_asset_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("/", "_").replace("\\", "_").replace(".", "_").lower()


@lru_cache(maxsize=2)
def discover_corpus_assets(
    root_dir: str = str(ROOT_DIR),
    registry_path: str = str(SOURCE_REGISTRY_PATH),
) -> tuple[CorpusAsset, ...]:
    root = Path(root_dir)
    sources = load_corpus_sources(registry_path)
    source_lookup = build_source_lookup(sources)
    assets: list[CorpusAsset] = []

    for path in list_corpus_candidate_paths(root_dir):
        sidecar = load_asset_sidecar(path)
        if path.is_relative_to(CORPUS_RAW_DIR):
            relative = path.relative_to(CORPUS_RAW_DIR)
            if not relative.parts:
                continue
            source_id = relative.parts[0]
        else:
            source_id = infer_builtin_source(path, root)

        source = source_lookup.get(source_id)
        if source is None or not source.enabled:
            continue

        tags = normalize_tags(list(source.tags) + list(normalize_tags(sidecar.get("tags"))))
        assets.append(
            CorpusAsset(
                asset_id=build_asset_id(path, root),
                source_id=source.source_id,
                source_label=source.label,
                source_type=source.source_type,
                trust_tier=source.trust_tier,
                domain=source.domain,
                source_url=str(sidecar.get("source_url", source.base_url)),
                title=str(sidecar.get("title", humanize_label(path.stem))),
                path=str(path),
                file_type=path.suffix.lower().lstrip("."),
                document_kind=str(sidecar.get("document_kind", infer_document_kind(path))),
                tags=tags,
                published_date=str(sidecar["published_date"]) if sidecar.get("published_date") else None,
            )
        )

    return tuple(assets)


@lru_cache(maxsize=2)
def corpus_summary(
    registry_path: str = str(SOURCE_REGISTRY_PATH),
    root_dir: str = str(ROOT_DIR),
) -> dict[str, object]:
    sources = load_corpus_sources(registry_path)
    assets = discover_corpus_assets(root_dir, registry_path)

    active_sources: dict[str, list[CorpusAsset]] = defaultdict(list)
    for asset in assets:
        active_sources[asset.source_id].append(asset)

    source_type_counts = Counter(item.source_type for item in sources if item.enabled)
    trust_counts = Counter(item.trust_tier for item in sources if item.enabled)
    active_trust_counts = Counter(asset.trust_tier for asset in assets)
    active_type_counts = Counter(asset.source_type for asset in assets)

    ranked_sources = []
    lookup = build_source_lookup(sources)
    for source_id, source_assets in active_sources.items():
        source = lookup[source_id]
        ranked_sources.append(
            {
                "source_id": source.source_id,
                "label": source.label,
                "source_type": source.source_type,
                "trust_tier": source.trust_tier,
                "domain": source.domain,
                "base_url": source.base_url,
                "asset_count": len(source_assets),
                "tags": list(source.tags),
            }
        )

    ranked_sources.sort(key=lambda item: (item["asset_count"], item["label"]), reverse=True)

    return {
        "configured_source_count": sum(1 for item in sources if item.source_id not in {"project_docs", "legacy_project_assets"} and item.enabled),
        "active_source_count": len(active_sources),
        "indexed_asset_count": len(assets),
        "official_source_count": sum(1 for item in sources if item.trust_tier in {"official", "regulatory"} and item.enabled),
        "source_type_distribution": sorted(source_type_counts.items()),
        "trust_distribution": sorted(trust_counts.items()),
        "active_source_type_distribution": sorted(active_type_counts.items()),
        "active_trust_distribution": sorted(active_trust_counts.items()),
        "active_sources": ranked_sources,
    }
