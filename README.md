# InvestIQ — AI-Powered Investment Learning & Research Platform

InvestIQ is a local-first investment learning and company research workspace designed for Indian retail market beginners. It combines source-grounded document question answering (**PageIndex vs. Traditional RAG**) with a research-oriented stock watchlist and company-level annual report analysis.

> **Disclaimer:** InvestIQ is an educational and research tool, not a SEBI-registered investment adviser or broker. All AI-generated explanations cite primary source documents and page numbers for manual verification and do not constitute buy/sell recommendations.

---

## Core Architecture

1. **Module A — AI Learning Assistant:** Answers personal finance and Indian market questions grounded in official SEBI and NCFE publications with page-level citations and explicit abstention when evidence is missing.
2. **Module B — Stock Watchlist & Research Tracker:** Tracks companies by exchange ticker, category (*Studying*, *Long-term*, *Trading*, *Archived*), and timestamped personal research notes.
3. **Module C — Company Research Assistant:** Connects watchlist companies to primary filings (such as annual reports) to answer questions on business segments, financial performance, and risk factors.

### Retrieval Evaluation Pipeline

```text
Official PDFs ─► PyMuPDF Parsing ─► ┌─ PageIndex (Hierarchical Tree Navigation) ─┐
                                    └─ Traditional RAG (Chunking + FAISS)      ─┴─► Benchmark Eval ─► Grounded LLM Citations
```

---

## Tech Stack

* **Backend:** Python, FastAPI, PyMuPDF (`fitz`)
* **Retrieval & Vector Search:** PageIndex, FAISS, Sentence-Transformers
* **Database:** SQLite (watchlists, research notes, document metadata)
* **Frontend:** React, Plotly (planned)
* **LLM:** Gemini API / Local Ollama (planned)

---

## Project Structure

```text
InvestIQ/
├── backend/
│   └── ingest.py           # PyMuPDF parser extracting page text & embedded TOCs
├── data/
│   ├── metadata.json       # Source registry (doc_id, publisher, module, page count)
│   ├── benchmark/          # Evaluation dataset (factual, multi-section, table, unanswerable)
│   ├── raw_pdfs/           # Local source PDFs (gitignored)
│   ├── processed/          # Extracted page-level JSONs (gitignored)
│   └── indexes/            # FAISS vector indexes & SQLite DB (gitignored)
├── docs/                   # Project concept & learning roadmap documentation
└── requirements.txt
```

---

## Setup & Usage

1. **Clone the repository:**
   ```powershell
   git clone [https://github.com/yashwantalli/InvestIQ.git](https://github.com/yashwantalli/InvestIQ.git)
   cd InvestIQ
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   py -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Run the PDF ingestion pipeline:**
   Place the official PDFs into `data/raw_pdfs/` according to `data/metadata.json`, then run:
   ```powershell
   python backend/ingest.py
   ```

---
