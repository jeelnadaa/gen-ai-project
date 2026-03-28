# Legal Clause Simplifier

A modular GenAI pipeline that extracts clauses from legal PDF documents,
simplifies them into plain English, detects their legal importance, and
evaluates the results with standard NLP metrics.

---

## Project Structure

```
legal_clause_simplifier/
├── app.py            # Main pipeline & CLI entry point
├── models.py         # Lazy model loading (singletons)
├── processing.py     # Simplify, detect importance, semantic similarity
├── evaluation.py     # BLEU, ROUGE, cosine sim, classification metrics
├── utils.py          # PDF extraction + spaCy clause splitting
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Models Used

| Task | Model | Why Upgraded |
|---|---|---|
| Simplification | `facebook/bart-large-xsum` | Fine-tuned for extreme summarisation → more fluent, concise plain-English output |
| Importance Detection | `cross-encoder/nli-deberta-v3-large` | DeBERTa-v3 NLI; MNLI acc ~91% vs ~89.9% for bart-large-mnli |
| Document Summarisation | `google/pegasus-large` | Pre-trained with GSG objective designed for abstractive summarisation |
| Semantic Similarity | `sentence-transformers/all-mpnet-base-v2` | Higher STS benchmark accuracy than all-MiniLM-L6-v2 |
| Clause Splitting | `spaCy en_core_web_sm` | |
| PDF Extraction | `PyPDF2` | |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

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
  "summary": "This Non-Disclosure Agreement establishes...",
  "evaluation": {
    "bleu": 0.3241,
    "rouge": { "rouge1": 0.4512, "rouge2": 0.2341, "rougeL": 0.3987 },
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
| Mean Cosine Similarity | Average semantic preservation across all clauses |
| Accuracy / P / R / F1 | Quality of IMPORTANT vs NORMAL classification |