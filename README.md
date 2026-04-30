# Mutual Fund Recommendation Assistant

An explainable AI project that helps investors discover suitable mutual fund schemes based on profile, goals, risk appetite, and investment horizon.

This repository is being built in phases to match the project guideline:

1. Problem understanding
2. Requirement analysis
3. Data collection
4. Data preprocessing
5. Embedding and vector database
6. LLM and RAG design
7. Application development
8. Evaluation
9. Optimization
10. Documentation and demo

## Current milestone

The current version provides:

- a documented problem statement and requirements baseline
- a scheme-level recommendation engine backed by the local mutual fund dataset
- a conservative master-data enrichment layer that links many schemes to NAV and AUM metadata
- a canonical NAV completion layer that fills missing schemes using real archive backfill first and synthetic peer-based values second
- a Streamlit app that collects investor preferences, explains why specific schemes were shortlisted, and suggests an allocation blueprint
- a PDF extraction and chunking layer that makes local documents searchable for the upcoming RAG stage
- a pure-Python embedding and vector-store layer for semantic and hybrid retrieval over document chunks
- a citation-aware RAG assistant with advisor, auditor, and researcher modes for grounded question answering
- a managed corpus framework with source registry, trust tiers, and multi-site readiness for larger knowledge expansion
- a data inventory showing which CSV and PDF assets can power the next RAG stages

This is the correct foundation for the later AI stages. The project now has document ingestion, a managed corpus, local embeddings, trust-aware vector retrieval, and a grounded RAG assistant in place. Upcoming iterations will deepen the knowledge corpus, strengthen evaluation, and connect the prompt pack to an external LLM when runtime access is available.

## Project structure

```text
MFRA/
├── app.py
├── requirements.txt
├── docs/
│   ├── 01_problem_understanding.md
│   ├── 02_requirements_analysis.md
│   ├── 03_solution_architecture.md
│   ├── 04_data_inventory.md
│   ├── 05_preprocessing_enrichment.md
│   ├── 06_document_ingestion.md
│   ├── 07_nav_completion.md
│   ├── 08_embeddings_vector_store.md
│   ├── 09_rag_llm_design.md
│   └── 10_large_corpus_framework.md
├── data/
│   ├── corpus/
│   ├── raw/
│   ├── processed/
│   └── vector_store/
└── src/
    ├── __init__.py
    ├── corpus_manager.py
    ├── document_pipeline.py
    ├── fund_data.py
    ├── rag_engine.py
    ├── recommendation_engine.py
    └── vector_store.py
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Use an isolated virtual environment. The system Python on this machine currently has a NumPy and pandas compatibility issue, and the project should be run in a clean environment.

The preprocessing pipeline now exports a canonical scheme file at `data/processed/fund_schemes_canonical.csv`. Each row includes a `nav_source` field so real and synthetic NAV values remain distinguishable.
The retrieval pipeline now exports a local vector index at `data/vector_store/document_chunk_index.json`.
The large-corpus pipeline now uses `data/corpus/source_registry.json` plus `data/corpus/raw/` to manage trusted external-source snapshots.

## Product direction

The final target is an explainable recommendation assistant with:

- investor profiling
- category and scheme suitability analysis
- master-data enrichment for fresher scheme context
- complete NAV coverage with source tracking
- searchable document ingestion and retrieval
- local embeddings and vector search for semantic retrieval
- citation-aware RAG over mutual fund factsheets and scheme documents
- source-aware corpus expansion with trust-weighted retrieval
- side-by-side fund comparison
- portfolio health and diversification insights
- goal-based planning with transparent reasoning

## Important note

This project is intended as an educational decision-support system, not a substitute for licensed financial advice.
