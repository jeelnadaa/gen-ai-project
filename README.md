# Legal Clause Simplifier

A modular GenAI pipeline that extracts clauses from legal PDF documents,
simplifies them into plain English, explains their meaning, detects their
legal importance, and evaluates the results with standard NLP metrics.

---

## Project Structure

```
legal_clause_simplifier/
├── app.py            # Main pipeline & CLI entry point
├── models.py         # Lazy model loading (singletons)
├── processing.py     # Simplify, explain, detect importance, similarity
├── evaluation.py     # BLEU, ROUGE, cosine sim, classification metrics
├── utils.py          # PDF extraction + spaCy clause splitting
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Models Used

| Task | Model |
|---|---|
| Simplification & Explanation | `facebook/bart-large` |
| Importance Detection | `facebook/bart-large-mnli` (zero-shot) |
| Document Summarisation | `facebook/bart-large-cnn` |
| Semantic Similarity | `sentence-transformers/all-MiniLM-L6-v2` |
| Clause Splitting | `spaCy en_core_web_sm` |
| PDF Extraction | `PyPDF2` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **GPU users** — set `--device cuda` to use a CUDA-capable GPU.

### 2. Run the pipeline

```bash
python app.py --pdf contract.pdf --output results.json
```

### 3. Optional flags

| Flag | Default | Description |
|---|---|---|
| `--pdf` | *(required)* | Path to input PDF |
| `--output` | `output.json` | Path for output JSON |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--max-clauses` | all | Process only the first N clauses |
| `--reference-summary` | none | Gold summary for ROUGE eval |
| `--ground-truth-labels` | none | Comma-separated IMPORTANT/NORMAL labels |
| `--min-clause-length` | `30` | Minimum clause length in characters |

### Example with evaluation labels

```bash
python app.py \
  --pdf nda_agreement.pdf \
  --output nda_results.json \
  --device cpu \
  --max-clauses 20 \
  --ground-truth-labels "IMPORTANT,NORMAL,IMPORTANT,NORMAL,IMPORTANT,NORMAL,NORMAL,IMPORTANT,NORMAL,NORMAL,IMPORTANT,NORMAL,NORMAL,NORMAL,IMPORTANT,NORMAL,NORMAL,IMPORTANT,NORMAL,NORMAL"
```

---

## Output Format

```json
{
  "clauses": [
    {
      "original": "The Receiving Party shall not disclose...",
      "simplified": "The receiver must keep the information secret...",
      "explanation": "This means the person who receives the information...",
      "importance": "IMPORTANT",
      "semantic_similarity": 0.8712
    }
  ],
  "summary": "This Non-Disclosure Agreement establishes...",
  "evaluation": {
    "bleu": 0.3241,
    "rouge": {
      "rouge1": 0.4512,
      "rouge2": 0.2341,
      "rougeL": 0.3987
    },
    "mean_semantic_similarity": 0.8541,
    "accuracy": 0.85,
    "precision": 0.88,
    "recall": 0.82,
    "f1": 0.85
  }
}
```

---

## Evaluation Metrics

| Metric | Measures |
|---|---|
| BLEU | Lexical overlap between original and simplified clauses |
| ROUGE-1/2/L | N-gram overlap between generated and reference summary |
| Cosine Similarity | Semantic preservation from original to simplified clause |
| Accuracy / P / R / F1 | Quality of IMPORTANT vs NORMAL classification |

> **Note on BLEU**: Because simplification intentionally changes wording,
> a lower BLEU score can indicate successful paraphrasing rather than poor
> quality. Use it as a relative comparison across system variants.

---

## Module Overview

### `utils.py`
- `extract_text_from_pdf(path)` — reads all pages with PyPDF2, cleans whitespace.
- `split_into_clauses(text)` — uses spaCy `en_core_web_sm` sentence segmentation; filters short sentences and splits overly long ones on semicolons.

### `models.py`
- `load_models(device)` — returns a `ModelBundle` dataclass containing all four models. Subsequent calls return the cached singleton.

### `processing.py`
- `simplify_clause(clause, bundle)` — BART-large generation with a plain-English prompt.
- `explain_clause(clause, bundle)` — BART-large generation with an explanation prompt.
- `detect_importance(clause, bundle)` — zero-shot classification with BART-large-MNLI.
- `compute_semantic_similarity(a, b, bundle)` — MiniLM-L6-v2 embeddings + cosine similarity.
- `summarize_document(text, bundle)` — BART-large-CNN summarisation pipeline.
- `process_clauses(clauses, bundle)` — runs the full per-clause pipeline and returns a list of result dicts.

### `evaluation.py`
- `compute_bleu(originals, simplifications)` — corpus-level BLEU with smoothing.
- `compute_rouge(generated, reference)` — ROUGE-1, ROUGE-2, ROUGE-L F1.
- `aggregate_similarity(clause_results)` — mean cosine similarity.
- `compute_classification_metrics(predicted, ground_truth)` — sklearn accuracy, precision, recall, F1.
- `run_evaluation(...)` — orchestrates all metrics and returns the evaluation dict.

### `app.py`
- CLI entry point; parses arguments, calls the pipeline steps in order, and writes the JSON output.