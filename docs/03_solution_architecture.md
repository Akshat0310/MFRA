# Solution Architecture

## High-level flow

```text
User Profile
    ->
Suitability Engine
    ->
Category or Scheme Shortlist
    ->
Retriever over factsheets and scheme documents
    ->
LLM explanation layer
    ->
Interactive dashboard and chat assistant
```

## Phase-wise implementation plan

### Phase 1: Foundation

- define problem statement
- finalize scope and assumptions
- create starter app and recommendation logic

### Phase 2: Structured recommendation engine

- maintain a catalog of fund categories and later schemes
- score suitability using risk, horizon, and goal alignment
- provide explainable reasons for each suggestion

### Phase 3: Data pipeline

- collect mutual fund factsheets, key information memorandums, and educational PDFs
- clean extracted text
- chunk and store text for retrieval

### Phase 4: RAG layer

- create embeddings for cleaned document chunks
- store them in a vector database such as Chroma or FAISS
- retrieve relevant evidence for a user query
- generate grounded answers with source-backed explanations

### Phase 5: Evaluation and optimization

- test recommendation consistency
- measure retrieval relevance
- track hallucination and unsupported claims
- optimize prompt quality, retrieval depth, and latency

## Suggested technology stack

- UI: Streamlit
- Core logic: Python
- Data handling: Pandas
- PDF extraction: PyPDF
- Embeddings: sentence-transformers or OpenAI embeddings
- Vector store: Chroma or FAISS
- LLM layer: OpenAI or another supported model

## Evaluation metrics

- suitability accuracy against predefined test profiles
- explanation quality
- retrieval precision
- hallucination rate
- response latency

## Innovation opportunities

- explain why a fund category is recommended and why another is not
- detect profile and portfolio mismatch
- warn users when they are over-focusing on recent returns
- include goal planning and diversification insights

