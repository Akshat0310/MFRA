# Large Corpus Framework

## Objective

This phase upgrades the project from a small local document set into a managed multi-source corpus framework. The goal is to make the assistant more scalable, more trustworthy, and more ready for real-world RAG expansion.

## What was implemented

### 1. Source registry

A new registry file, `data/corpus/source_registry.json`, now defines trusted external source families such as:

- AMFI
- SEBI investor education
- Investor.gov
- RBI financial education
- multiple AMC source families

Each source carries metadata like:

- source type
- trust tier
- domain
- base URL
- ingestion mode
- tags

### 2. Managed corpus assets

The project now supports a dedicated corpus folder:

- `data/corpus/raw/`

This folder is designed to hold local snapshots from larger knowledge sources. Files can be grouped by source ID and enriched with optional sidecar metadata.

Supported file types include:

- markdown
- text
- HTML
- PDF

### 3. Metadata-aware chunking

The document pipeline now attaches source metadata to every page and every chunk, including:

- source label
- source type
- trust tier
- domain
- source URL
- document kind
- tags

This means retrieval is no longer blind to where a chunk came from.

### 4. Trust-aware vector retrieval

The vector pipeline now applies source-aware ranking signals during retrieval. Higher-trust sources like regulators and official industry bodies are given a ranking advantage over low-context internal notes.

This does not replace semantic similarity, but it improves the quality of the final evidence set.

### 5. Corpus Studio UI

The Streamlit app now includes a `Corpus Studio` view that shows:

- configured source count
- active source count
- indexed assets
- indexed chunks
- trust-tier distribution
- source-type distribution
- active source registry

This makes the large-corpus architecture visible during demos and presentations.

## Why this matters

Without a corpus-management layer, a RAG system quickly becomes messy as more sources are added. This phase makes expansion safer because the project can now distinguish:

- project notes vs external sources
- regulator vs AMC content
- higher-trust vs lower-trust material
- configured sources vs actually indexed assets

## Innovation angle

This project now stands out because it is not only recommending schemes and answering questions. It is also showing deliberate design choices around:

- corpus governance
- source trust
- retrieval quality
- explainability
- growth readiness

That makes it feel closer to a serious financial research assistant than a basic student chatbot.

## Next step

The best next phase after this is evaluation and optimization:

- add more AMC factsheet snapshots
- add scheme-information-document snapshots
- benchmark trust-aware retrieval quality
- measure citation coverage and answer faithfulness
