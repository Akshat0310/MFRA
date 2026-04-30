# RAG And LLM Design

## Objective

This phase turns the project from a recommendation dashboard into a grounded AI assistant. The goal is not to let the model answer freely, but to make it answer from retrieved evidence and explicit scheme context.

## What was implemented

### 1. Grounded answer engine

A new module, `src/rag_engine.py`, now creates answer bundles that combine:

- investor profile context
- shortlisted scheme metrics
- hybrid-retrieved document chunks
- guardrails and confidence labels
- a prompt pack for future external LLM use

### 2. Three assistant modes

The Streamlit app now includes a `RAG Assistant` tab with three modes:

- `Advisor`: emphasizes suitability and next-step guidance
- `Auditor`: emphasizes risks, blind spots, and data-quality caveats
- `Researcher`: emphasizes citations, retrieval evidence, and methodology

This makes the assistant feel more intentional than a generic chatbot and helps demonstrate clear design thinking in the project.

### 3. Query intent handling

The engine classifies user questions into practical intents such as:

- scheme-fit explanation
- comparison
- risk review
- tax and lock-in questions
- data-quality and synthetic NAV explanation
- allocation strategy

The answer shape changes based on that intent instead of using the same template for every question.

### 4. Citation-aware response structure

Each generated answer now includes:

- a direct answer
- key points
- watch-outs
- evidence citations
- follow-up questions
- a confidence label

This improves explainability and makes the project much stronger for demo and evaluation.

### 5. Prompt-pack design

The app also builds a prompt pack made of:

- `system prompt`
- `user prompt`
- `retrieved context block`

This is important because it means the project is already architected for an external LLM. When an API-backed model is available later, the current pipeline can hand off grounded context instead of starting from scratch.

## Why this phase matters

Without RAG, the app can retrieve and rank, but it cannot answer naturally with grounded reasoning. With this phase, the system now behaves like a proper AI assistant:

- it answers from context
- it cites supporting evidence
- it distinguishes real and synthetic data conditions
- it explains suitability, not just scores

## Current design choice

The current environment does not rely on a live external LLM API, so the answer synthesis is local and rule-guided. That keeps the project runnable in the present setup while still implementing the complete RAG architecture pattern:

1. retrieve context
2. attach structured scheme evidence
3. synthesize grounded response
4. expose prompt pack for future LLM integration

## Innovation angle

What makes this stronger than a basic RAG chatbot is the combination of:

- structured scheme reasoning
- document retrieval
- assistant modes
- data-quality transparency
- prompt-pack export

That gives the project a more product-like feel and makes it easier to explain in demos, viva, or PPT presentations.

## Next step

The next phase should focus on evaluation and optimization:

- test retrieval quality
- test answer faithfulness
- increase the document corpus with real scheme factsheets
- plug the prompt pack into a production LLM when API access is available
