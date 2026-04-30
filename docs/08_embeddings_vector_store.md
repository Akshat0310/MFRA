# Embeddings And Vector Store

## Objective

This phase adds a real retrieval backbone for the document layer so the project can move from plain keyword search toward RAG-ready semantic retrieval.

## What was implemented

### 1. Local embedding model

The project now creates deterministic local embeddings for each document chunk.

- no external API is required
- no model download is required
- the implementation works fully offline

Because the environment does not currently have a clean local `sentence-transformers` or `chromadb` stack, the embedding layer uses a pure-Python hashed TF-IDF style representation. This is still a true vector representation and is good enough for a reliable project milestone.

### 2. Vector index

Each chunk from the document pipeline is stored with:

- chunk metadata
- text
- embedding vector

The index is exported to:

- `data/vector_store/document_chunk_index.json`

### 3. Retrieval modes

The app now supports:

- `Keyword`: direct lexical overlap
- `Semantic`: vector similarity over local embeddings
- `Hybrid`: combined lexical and vector ranking

## Why this matters

This phase creates the actual retrieval layer needed before adding the LLM answer generator.

It gives us:

- indexed chunk embeddings
- persistent vector storage
- semantic retrieval behavior
- hybrid ranking for better robustness

## Current technical approach

The vector embedding uses:

- tokenization
- domain term expansion such as `AUM -> assets under management`
- unigram and bigram features
- hashed vector projection into a fixed embedding size
- cosine similarity for ranking

## Limitation

This is not yet a transformer-based embedding model. It is a lightweight local vector representation chosen so the project remains runnable in the current environment.

## Next step

The next phase will connect retrieved chunks to an answer-generation layer so the assistant can respond using grounded evidence instead of only showing matching passages.
