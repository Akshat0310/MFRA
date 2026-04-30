# Canonical NAV Completion

## Objective

The recommendation layer needs a usable NAV value for every scheme. Before this phase, many schemes were missing NAV because the main recommendation dataset and the master dataset did not align perfectly by name.

## What was implemented

### 1. Real NAV hierarchy

The preprocessing layer now tries to resolve NAV in this order:

1. master dataset NAV from `data/raw/mutual_fund_data1.csv`
2. archive NAV backfill from the latest snapshot in `data/raw/archive/DailyNAV/`
3. synthetic peer-based NAV for any still-unresolved scheme

### 2. Synthetic values are clearly marked

Synthetic values are never mixed in silently. Each exported row now carries:

- `nav_source`
- `nav_confidence`
- `nav_is_synthetic`
- `nav_notes`

This keeps the dataset usable for experiments while preserving transparency.

## Why synthetic NAV was added

The project goal for this phase was complete preprocessing coverage. Some schemes still cannot be matched safely to a real NAV source, usually because of naming differences or missing option-level alignment. For those cases, a peer-based estimate is used so every scheme has a NAV-like value.

## How the synthetic NAV is estimated

The estimate is based on comparable schemes, with preference order:

1. same sub-category and same plan style
2. same sub-category
3. same category and risk level
4. same category
5. global fallback

The peer median NAV is then adjusted modestly using:

- fund age
- 3-year and 5-year return profile

The final value is clipped to a realistic peer range so it stays practical-looking.

## Current output

The pipeline now exports:

- `data/processed/fund_schemes_canonical.csv`

This file is the cleanest source to use for downstream filtering, evaluation, and later embedding or RAG work.

## Important caution

Synthetic NAV is useful for preprocessing completeness and experimentation, but it should not be treated as official market data. Any production-style financial workflow should still distinguish real NAV from imputed NAV.
