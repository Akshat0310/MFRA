# Preprocessing And Enrichment

## Why this phase matters

The project already had two strong local datasets:

- `data/raw/mutual_funds_data2.csv` with scheme-level recommendation features
- `data/raw/mutual_fund_data1.csv` with AMFI-style scheme master fields such as scheme code, NAV, AUM, and launch date

These datasets use different naming formats, so a direct join is unreliable. This phase adds a conservative preprocessing layer that improves scheme context without introducing unsafe joins.

## What was implemented

### 1. AMC normalization

AMC names are normalized into canonical forms so that patterns like:

- `Aditya Birla Sun Life Mutual Fund`
- `Aditya Birla Sun Life AMC Limited`

can be treated as the same family.

### 2. Scheme-name normalization

Scheme names are cleaned by:

- lowercasing
- replacing punctuation and separators
- expanding some abbreviations such as `SL -> Sun Life`
- removing option-level words such as `direct`, `growth`, `regular`, and `IDCW`

This creates a stable scheme key for cautious matching.

### 3. Conservative matching strategy

The enrichment layer uses:

1. exact normalized scheme name plus canonical AMC match
2. unique normalized scheme-name fallback when the master side is unambiguous

It intentionally avoids aggressive fuzzy matching by default, because attaching incorrect NAV or AUM data would hurt trust more than leaving some schemes unmatched.

## Outputs of the enrichment phase

When a match is found, the project can now attach:

- scheme code
- latest NAV
- latest NAV date
- average AUM
- scheme type
- master scheme category label
- scheme NAV name
- launch date
- match type diagnostics

## Benefits to the app

- recommendations can show more grounded scheme context
- data coverage is measurable and transparent
- later document ingestion can reference scheme codes and cleaner master metadata
- evaluation becomes easier because data lineage is clearer

## Next preprocessing tasks

- improve category normalization between datasets
- enrich more schemes safely where naming differences are still blocking joins
- use scheme code and launch metadata in evaluation and reporting
- prepare document ingestion inputs for the upcoming embedding and RAG stage
