# Legal Clause Simplifier

A modular GenAI pipeline that extracts clauses from legal PDF documents,
simplifies them into plain English, detects legal importance, and evaluates
results with NLP metrics.

All heavy NLP tasks use **Llama 3.3 70B via Groq's free API** — far superior
to local BART/PEGASUS models, with no GPU required.

---

## Project Structure

```
legal_clause_simplifier/
├── app.py            # Main pipeline & CLI entry point
├── models.py         # Groq client + sentence-transformer loader
├── processing.py     # Simplify, detect importance, semantic similarity
├── evaluation.py     # BLEU, ROUGE, cosine sim, classification metrics
├── utils.py          # PDF extraction + spaCy clause splitting
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Models

| Task | Model | How |
|---|---|---|
| Simplification | `llama-3.3-70b-versatile` | Groq API (free) |
| Importance Detection | `llama-3.3-70b-versatile` | Groq API (free) |
| Document Summarisation | `llama-3.3-70b-versatile` | Groq API (free) |
| Semantic Similarity | `all-mpnet-base-v2` | Local (CPU) |
| Clause Splitting | `spaCy en_core_web_sm` | Local (CPU) |
| PDF Extraction | `PyPDF2` | Local |

**Groq free tier:** 500,000 tokens/day · 6,000 tokens/min · no credit card needed.

---

## Quick Start

### 1. Get a free Groq API key

1. Go to https://console.groq.com
2. Sign up (no credit card required)
3. Navigate to **API Keys** and create a new key

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set your API key

```bash
# Linux / macOS
export GROQ_API_KEY=your_key_here

# Windows CMD
set GROQ_API_KEY=your_key_here

# Windows PowerShell
$env:GROQ_API_KEY="your_key_here"
```

### 4. Run

```bash
python app.py --pdf contract.pdf --output results.json
```

Or pass the key directly:
```bash
python app.py --pdf contract.pdf --groq-api-key your_key_here
```

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--pdf` | *(required)* | Path to input PDF |
| `--output` | `output.json` | Path for output JSON |
| `--groq-api-key` | env var | Groq API key |
| `--groq-model` | `llama-3.3-70b-versatile` | Groq model ID |
| `--max-clauses` | all | Process only first N clauses |
| `--reference-summary` | none | Gold summary for ROUGE eval |
| `--ground-truth-labels` | none | Comma-separated IMPORTANT/NORMAL labels |
| `--min-clause-length` | `30` | Minimum clause length in characters |

### Alternative Groq models

| Model | Speed | Quality | Best for |
|---|---|---|---|
| `llama-3.3-70b-versatile` | Medium | ⭐⭐⭐⭐⭐ | Best quality (default) |
| `llama-3.1-8b-instant` | Fast | ⭐⭐⭐ | Large docs, tight rate limits |
| `gemma2-9b-it` | Fast | ⭐⭐⭐⭐ | Good balance |

---

## Output Format

```json
{
  "clauses": [
    {
      "original": "The Receiving Party shall not disclose...",
      "simplified": "The receiver must keep the information secret...",
      "importance": "IMPORTANT",
      "semantic_similarity": 0.8712
    }
  ],
  "summary": "This Non-Disclosure Agreement...",
  "evaluation": {
    "bleu": 0.3241,
    "rouge": { "rouge1": 0.45, "rouge2": 0.23, "rougeL": 0.40 },
    "mean_semantic_similarity": 0.85,
    "accuracy": 0.85,
    "precision": 0.88,
    "recall": 0.82,
    "f1": 0.85
  }
}
```