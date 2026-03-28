# LexiGen | Legal Clause Simplifier

A premium, high-fidelity legal analysis platform that transforms complex legal PDF documents into plain English. Built with an Apple-inspired minimalist aesthetic and state-of-the-art GenAI models.

---

## Key Features

- **Premium Interface**: A minimalist Black & White dashboard designed for professional readability.
- **Dual-Model Support**: Powered by **Llama 3.3 70B** and **GPT-OSS 120B** via Groq for industry-leading simplification.
- **SOTA Embeddings**: Semantic similarity powered by **BAAI/bge-base-en-v1.5**, a top-tier model for precise legal comparison.
- **Intelligent Risk Analysis**: High-precision importance detection for Liability, Indemnity, Data Security, and Termination clauses.
- **Persistent History**: Browser-side storage with custom deletion controls and active run tracking.
- **Export Ready**: Instant JSON export of processed clauses and executive summaries.

---

## Tech Stack

| Component | Technology | Model / Tool |
|---|---|---|
| **Core LLM** | Groq API | `llama-3.3-70b-versatile` |
| **Alternative LLM** | Groq API | `openai/gpt-oss-120b` |
| **Embeddings** | Sentence-Transformers | `BAAI/bge-base-en-v1.5` |
| **Backend** | Flask | Python 3.10+ |
| **Frontend** | Vanilla JS / CSS | Satoshi Typography |
| **PDF Processing** | PyPDF2 / spaCy | en_core_web_sm |

---

## Quick Start

### 1. Setup Environment
1. Get a free API key at [Groq Console](https://console.groq.com).
2. Create a `.env` file in the root:
   ```bash
   GROQ_API_KEY=gsk_your_key_here
   ```

### 2. Install & Launch
```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Launch the premium web app
python flask_app.py
```

### 3. Access
Open your browser to `http://localhost:5000` to begin your analysis.

---

## Project Structure

```
lexigen/
├── flask_app.py      # Flask Server & SSE updates
├── app.py            # CLI entry point
├── models.py         # Model loading (Groq & BGE)
├── processing.py     # Clause pipeline logic
├── evaluation.py     # BLEU, ROUGE, & Semantic metrics
├── utils.py          # PDF extraction & text cleaning
├── static/           # Premium CSS & JS
└── templates/        # Dashboard HTML
```

---

## CLI Usage (Advanced)

Process documents directly from the terminal:
```bash
python app.py --pdf contract.pdf --output results.json --max-clauses 10
```

---

## Optimization Techniques

- **Clean Tokenization**: Advanced regex-based PDF cleaning to eliminate whitespace glitches.
- **Risk-Aware Prompting**: "High-Precision Legal Risk Analyst" persona for importance classification.
- **Mean Semantic Similarity**: Aggregated embedding distance to measure simplification fidelity.
- **SSE Streaming**: Real-time progress updates for large document processing.
```