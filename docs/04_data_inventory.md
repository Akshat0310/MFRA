# Data Inventory

## Current data assets discovered in the workspace

### 1. `mutual_funds_data2.csv`

- Purpose: primary dataset for the current recommendation engine
- Approximate size: 814 schemes
- Key fields:
  - scheme name
  - AMC name
  - category and sub-category
  - risk level
  - rating
  - expense ratio
  - fund size
  - fund age
  - 1-year, 3-year, and 5-year returns
  - minimum SIP and minimum lump sum amount

### 2. `mutual_fund_data1.csv`

- Purpose: AMFI-style scheme master data for later enrichment
- Key fields:
  - scheme code
  - scheme name
  - AMC
  - scheme type and scheme category
  - NAV
  - latest NAV date
  - average AUM
  - launch date

### 3. `archive/DailyNAV/`

- Purpose: historical NAV archive for time-series analysis and performance trend features
- Format: daily CSV snapshots across many dates
- Possible future use:
  - rolling return analysis
  - volatility estimation
  - trend-aware insights
  - scheme consistency scoring

### 4. Available PDFs

- `ammar2026repo.pdf`
- `aqu-vol25-issueIII.pdf`

These PDFs are available for later review. They may become part of the data-collection and RAG stages if they contain relevant mutual fund research, policy, or domain knowledge.

## How these assets map to the project guideline

1. Data Collection:
   - local CSVs and PDFs are already available
2. Data Preprocessing:
   - clean numeric fields, normalize category names, and handle missing return values
3. Embedding and Vector DB:
   - use PDFs and future scheme documents as retrieval sources
4. Evaluation:
   - compare recommendation quality before and after document grounding

## Immediate next data tasks

- clean and normalize scheme and category labels
- merge scheme metrics with master NAV data where feasible
- review PDF relevance
- prepare a small trusted evaluation set of investor profiles and expected recommendation patterns
