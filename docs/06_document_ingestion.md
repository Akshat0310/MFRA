# Document Ingestion And RAG Preparation

## Objective

This phase prepares the project for Retrieval-Augmented Generation by turning local documents into searchable text chunks.

## What was implemented

### 1. Document discovery

The project now scans the local workspace for document sources such as PDFs and project Markdown files and treats them as knowledge inputs.

### 2. Text extraction

Using `pypdf` for PDFs and direct text reads for Markdown files, the project extracts content and normalizes whitespace so the text can be processed consistently.

### 3. Chunking

Each page is split into overlapping word-based chunks. This keeps passages small enough for retrieval while preserving enough context for later RAG responses.

### 4. Searchable retrieval

A lightweight lexical search layer was added so queries can already retrieve relevant chunks. This is not the final vector-search pipeline, but it provides:

- chunk inventory
- query testing
- grounding preview in the Streamlit app
- a clean transition path to embeddings

## Files added or enhanced

- `src/document_pipeline.py`
- `app.py`

## Current scope of the document layer

At the moment, the project can:

- discover local documents
- extract text page by page
- create searchable chunks
- show matching passages inside the app

It does not yet:

- create embeddings
- store chunks in a vector database
- generate LLM answers from retrieved evidence

## Why this step is important

This phase de-risks the later AI work. It proves that the local documents are readable, chunkable, and searchable before we spend effort building the full RAG stack.

## Next step after this

- curate better finance and mutual-fund documents
- add embeddings and vector storage
- connect retrieval output to a grounded answer-generation layer
