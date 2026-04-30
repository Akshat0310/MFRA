# Corpus Raw Folder

Add local snapshots here to expand the assistant's knowledge base.

Recommended structure:

- `data/corpus/raw/amfi/`
- `data/corpus/raw/sebi_investor/`
- `data/corpus/raw/hdfc_mf/`
- `data/corpus/raw/sbi_mf/`

Supported file types:

- `.md`
- `.txt`
- `.html`
- `.htm`
- `.pdf`

Optional sidecar metadata:

- add `<filename>.meta.json`
- supported keys: `title`, `source_url`, `document_kind`, `published_date`, `tags`

This project uses the source folder name to map each document into the managed corpus registry.
