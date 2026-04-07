# Legal Clause Simplifier (LexiGen) - Architecture Documentation

## System Overview

**LexiGen** is a premium, high-fidelity legal analysis platform that transforms complex legal PDF documents into plain English using state-of-the-art GenAI models. The system is built with a modular, layered architecture that separates concerns across 10 distinct layers.

---

## Architecture Layers

### **Layer 1: Presentation (Frontend)**

The user-facing interface runs entirely in the browser using vanilla HTML, CSS, and JavaScript.

**Components:**
- **index.html** - Main upload interface with minimalist design
  - File drop zone for PDF uploads
  - Configuration panel (model selection, max clauses, clause length filter)
  - Real-time progress bar with status messages
  - Results dashboard with metrics display
  - History sidebar with persistent browser storage
  - Custom tooltip system and confirmation modals

- **main.js** - Client-side logic (~12KB)
  - File selection and drag-drop handling
  - Form submission to `/process` endpoint
  - Server-Sent Events (SSE) polling via EventSource API
  - Results rendering with clause pagination
  - Local storage management (analysis history)
  - Tooltip and modal interactions

- **style.css** - Professional design (~13KB)
  - Minimalist black-and-white theme
  - Responsive grid layouts
  - Apple-inspired typography (Satoshi font)
  - Component library: buttons, inputs, cards, modals
  - Accessibility-friendly color contrast

**Data Flow:**
1. User uploads PDF via drag-drop or file picker
2. JavaScript creates FormData with PDF + configuration
3. POST request sent to `/process` endpoint
4. Task ID received; SSE EventSource created to `/status/:task_id`
5. Real-time progress updates displayed
6. On completion, results rendered in dashboard

---

### **Layer 2: API Gateway (Flask)**

The backend REST API server handles HTTP requests, task orchestration, and background processing.

**File:** `flask_app.py`

**Endpoints:**

1. **GET / (index)**
   - Returns `index.html` (template rendering)
   - Serves the main application UI

2. **POST /process (async upload)**
   - Accepts multipart/form-data with PDF file
   - Parameters: `groq_model`, `max_clauses`, `min_clause_length`, `reference_summary`
   - Actions:
     - Validates file (16MB max)
     - Generates unique task ID (UUID4)
     - Saves PDF to `.tmp_flask_uploads/`
     - Spawns background thread for processing
     - Returns `{"task_id": "..."}` immediately
   - Status: In-memory task tracking (non-persistent)

3. **GET /status/:task_id (Server-Sent Events)**
   - Bidirectional HTTP streaming protocol
   - Sends JSON updates every 0.5 seconds only if status changes
   - Stream format: `data: {"status": "...", "progress": ..., "message": "...", "result": null}`
   - On completion: Sends final result with `"status": "completed"`
   - On error: Sends error message with `"status": "error"`
   - JavaScript EventSource client closes connection after completion or error

4. **GET /download/:task_id (JSON export)**
   - Returns completed result as JSON file
   - Content-Disposition: attachment
   - Filename: `results_{task_id}.json`
   - Requires task to be in "completed" state

**Background Processing:**
```python
def background_processing(tid, fpath, model, m_clauses, min_len, ref_sum):
    try:
        # Import and call app.process_pdf()
        result = process_pdf(
            fpath,
            groq_model=model,
            max_clauses=m_clauses,
            min_clause_length=min_len,
            reference_summary=ref_sum,
            progress_callback=progress_callback
        )
        tasks[tid]["status"] = "completed"
        tasks[tid]["result"] = result
    except Exception as e:
        tasks[tid]["status"] = "error"
        tasks[tid]["error"] = str(e)
```

**Task State Machine:**
```
"starting" → "processing" (progress: 0-100) → "completed" / "error"
```

---

### **Layer 3: File Management**

Temporary storage and cleanup of uploaded PDFs.

**Directory:** `.tmp_flask_uploads/`

**Features:**
- Named with pattern: `{task_id}_{original_filename}`
- Files persist until server restart (in-memory task cleanup)
- Max file size: 16MB (configurable in `app.config['MAX_CONTENT_LENGTH']`)
- Auto-created on Flask startup if absent

**Considerations:**
- For production: Implement S3/Cloud Storage + scheduled cleanup
- Currently: In-memory tasks; results lost on server restart

---

### **Layer 4: Utilities (PDF & NLP)**

Low-level functions for PDF extraction and text processing.

**File:** `utils.py`

**Functions:**

1. **extract_text_from_pdf(pdf_path: str) → str**
   - Opens PDF file with `PyPDF2.PdfReader`
   - Iterates through all pages
   - Extracts raw text from each page
   - Joins pages with newlines
   - Calls `_clean_text()` for normalization
   - Returns: Full document text
   - Raises: `FileNotFoundError` if PDF missing, `ValueError` if no text extracted

2. **_clean_text(text: str) → str**
   - Regex-based normalization pipeline:
     - Fix "Word\n\n\nWord" (multiple newlines) → "Word Word"
     - Fix "Word\nWord" (PDF columnar split) → "Word Word"
     - Remove extra spaces (2+) → single space
     - Remove form feeds (`\x0c`)
     - Strip leading/trailing whitespace
   - Returns: Clean, normalized text

3. **split_into_clauses(text: str, min_length: int = 30, max_length: int = 1000) → list[str]**
   - Lazy-loads spaCy model `en_core_web_sm` on first call
   - Processes text in 100KB chunks (prevents memory spikes)
   - For each chunk:
     - Parse with spaCy `nlp(chunk)`
     - Extract sentences via `doc.sents`
     - Filter by length (min 30 chars, max 1000 chars)
     - Split long sentences by semicolon if needed
   - Returns: List of clause strings
   - Logs: Clause count extracted

**Dependencies:**
- `PyPDF2` (>=3.0.0) - PDF reading
- `spacy` (>=3.7.0) - Sentence segmentation
- Model: `en_core_web_sm` (must be downloaded separately)

**Performance:**
- PDF extraction: ~1 second per 100 pages
- Clause splitting: ~0.5 seconds for 3000 characters

---

### **Layer 5: Model Management**

Centralized loading and caching of Groq and Sentence-Transformer models.

**File:** `models.py`

**Data Structure:**
```python
@dataclass
class ModelBundle:
    groq_client: Groq = None          # Groq API client
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: SentenceTransformer = None  # BGE embeddings
```

**Functions:**

1. **load_models(groq_model: str = "llama-3.3-70b-versatile") → ModelBundle**
   - Global singleton pattern (`_BUNDLE`)
   - First call: Initialize + cache
   - Subsequent calls: Return cached instance
   - Environment variable: `GROQ_API_KEY` (from `.env`)
   - Raises: `ValueError` if API key missing
   - Steps:
     1. Check `.env` file for `GROQ_API_KEY`
     2. Initialize Groq client with API key
     3. Load Sentence-Transformer `BAAI/bge-large-en-v1.5`
     4. Cache bundle in global `_BUNDLE`
   - Returns: Initialized ModelBundle

**Configuration:**
- API key source: `.env` file only (not environment variables)
- Groq model: Selectable at runtime
- Supported models:
  - `llama-3.3-70b-versatile` (default, best quality)
  - `llama-3.1-8b-instant` (fast, lower quality)
  - `openai/gpt-oss-120b` (alternative)
  - `gemma2-9b-it` (lightweight)

**Groq Setup:**
- Free tier: No credit card, no rate limits
- API key signup: https://console.groq.com
- Cost: $0 (free tier indefinitely)

**Embedding Model:**
- `BAAI/bge-large-en-v1.5` - Runs locally on CPU
- Dimensions: 1024
- Purpose: Semantic similarity computation
- Download: ~500MB (first load only)

---

### **Layer 6: Core Processing**

Main NLP operations on individual clauses and document-level tasks.

**File:** `processing.py`

**Internal Helper:**

**_llm(bundle: ModelBundle, system: str, user: str, max_tokens: int = 300, temperature: float = 0.1) → str**
- Sends chat completion request to Groq API
- Parameters:
  - `system`: System role prompt (expert persona)
  - `user`: User query (clause to process)
  - `max_tokens`: Length limit (300 default)
  - `temperature`: Randomness (0.1 = deterministic)
- Returns: LLM response text (stripped)
- Adds 0.5s delay between calls (rate limiting)

---

**Processing Functions:**

1. **simplify_clause(clause: str, bundle: ModelBundle) → str**
   - System prompt: "Legal expert who specialises in making complex legal language easy to understand"
   - User prompt: "Simplify this legal clause: {clause}"
   - Max tokens: 200
   - Temperature: 0.1 (deterministic)
   - Output: Plain English version of clause
   - Example:
     ```
     Input: "The Licensor hereby indemnifies and holds harmless the Licensee from any third-party claims..."
     Output: "The Licensor will protect the Licensee from lawsuits by third parties."
     ```

2. **detect_importance(clause: str, bundle: ModelBundle) → str**
   - System prompt: "HIGH-PRECISION legal risk analyst"
   - Categories marked as IMPORTANT:
     - Limitation of Liability & Damages
     - Indemnification
     - Data Security & Privacy
     - Termination Rights
     - Intellectual Property
   - Categories marked as NORMAL:
     - Definitions
     - Routine Notices
     - Headings & Fragments
     - Choice of Law (unless unusual)
   - Max tokens: 10
   - Temperature: 0.0 (deterministic)
   - Output: "IMPORTANT" or "NORMAL"
   - Logic: Pattern matching on response + keyword heuristics

3. **compute_semantic_similarity(text_a: str, text_b: str, bundle: ModelBundle) → float**
   - Encodes both texts using BAAI/bge-large-en-v1.5
   - Computes cosine similarity between embedding vectors
   - Returns: Float between 0.0 and 1.0 (clamped)
   - Purpose: Measure how well simplified clause preserves original meaning
   - Example:
     ```
     Original:    "The Licensor hereby indemnifies..."
     Simplified:  "The Licensor will protect..."
     Similarity:  0.85 (high semantic overlap)
     ```

4. **summarize_document(text: str, bundle: ModelBundle) → str**
   - Truncates input to 6000 characters
   - System prompt: "Legal expert. Summarise for a non-lawyer. Cover purpose, obligations, rights, risks."
   - User prompt: "Summarise this legal document: {truncated_text}"
   - Max tokens: 400
   - Temperature: 0.1
   - Output: 3-5 sentence executive summary
   - Logs: Summary length

5. **process_clauses(clauses: list[str], bundle: ModelBundle, progress_callback=None) → list[dict]**
   - Main per-clause pipeline
   - For each clause (indexed 1 to N):
     1. Call `progress_callback(idx, total)` if provided
     2. Call `simplify_clause(clause, bundle)`
     3. Call `detect_importance(clause, bundle)`
     4. Call `compute_semantic_similarity(original, simplified, bundle)`
     5. Append dict: `{"original", "simplified", "importance", "semantic_similarity"}`
   - Returns: List of result dicts
   - Rate limiting: 0.5s delay between Groq API calls

**Performance (typical):**
- Per clause: ~2-3 seconds (2 Groq calls + 1 embedding)
- 25 clauses: ~60 seconds total
- Bottleneck: Groq API latency (not local computation)

---

### **Layer 7: Orchestration**

High-level pipeline coordination and CLI interface.

**File:** `app.py`

**Entry Points:**

1. **main() → None** (CLI)
   - Parses command-line arguments
   - Calls `run_pipeline(args)`
   - Saves output to JSON
   - Prints summary to console

2. **process_pdf(...) → dict** (Python API)
   - Wrapper function for use by Flask backend
   - Accepts Python types (not argparse.Namespace)
   - Returns same dict as CLI
   - Used by `flask_app.py`

**Main Orchestration:**

**run_pipeline(args: argparse.Namespace, progress_callback=None) → dict**

5-step pipeline:

```
STEP 1: Extract PDF Text
  └─→ utils.extract_text_from_pdf(args.pdf)
  └─→ Output: raw_text (str)

STEP 2: Split into Clauses
  └─→ utils.split_into_clauses(raw_text, min_length=args.min_clause_length)
  └─→ Limit to args.max_clauses if specified
  └─→ Output: clauses (list[str])

STEP 3: Load Models
  └─→ models.load_models(groq_model=args.groq_model)
  └─→ Output: bundle (ModelBundle)

STEP 4: Process Clauses
  └─→ processing.process_clauses(clauses, bundle, progress_callback)
  └─→ processing.summarize_document(raw_text, bundle)
  └─→ Output: clause_results (list[dict]), summary (str)

STEP 5: Evaluate Results
  └─→ evaluation.run_evaluation(...)
  └─→ Output: evaluation_metrics (dict)

FINAL OUTPUT:
{
  "clauses": clause_results,
  "summary": summary,
  "evaluation": evaluation_metrics
}
```

**CLI Arguments:**
```
--pdf PATH                          (required) Path to input PDF
--output PATH                       (optional) Output JSON path (default: output.json)
--groq-model MODEL                  (optional) Groq model ID
--max-clauses N                     (optional) Process at most N clauses
--reference-summary TEXT            (optional) Reference text for ROUGE evaluation
--ground-truth-labels LABEL,...     (optional) Comma-separated importance labels
--min-clause-length N               (optional) Minimum clause length (default: 30)
```

**Example CLI Usage:**
```bash
python app.py --pdf contract.pdf --output results.json --max-clauses 10 --groq-model llama-3.3-70b-versatile
```

**Example Python API Usage:**
```python
from app import process_pdf

result = process_pdf(
    "contract.pdf",
    groq_model="llama-3.3-70b-versatile",
    max_clauses=5,
    min_clause_length=30
)
print(result["evaluation"]["mean_semantic_similarity"])
```

**Error Handling:**
- File not found: `FileNotFoundError` → exit code 1
- Missing API key: `ValueError` → exit code 1
- No clauses extracted: `ValueError` → exit code 1
- Unexpected error: `Exception` → logged + exit code 1

---

### **Layer 8: Evaluation Metrics**

Quantitative assessment of simplification, summarization, and classification quality.

**File:** `evaluation.py`

**Metrics Functions:**

1. **compute_bleu(originals: list[str], simplifications: list[str]) → float**
   - BLEU Score (Bilingual Evaluation Understudy)
   - Compares n-gram overlap between original and simplified
   - Range: 0.0 (no overlap) to 1.0 (perfect match)
   - Smoothing: Method 4 (handles short sentences)
   - Interpretation:
     - 0.0-0.3: Poor simplification
     - 0.3-0.6: Moderate simplification
     - 0.6+: High similarity (close paraphrase)
   - Returns: Rounded to 4 decimals
   - Logs: BLEU score

2. **compute_rouge(generated_summary: str, reference_summary: str) → dict**
   - ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)
   - Three variants:
     - `rouge1`: Unigram (single word) overlap
     - `rouge2`: Bigram (two-word sequence) overlap
     - `rougeL`: Longest common subsequence
   - Range: 0.0 to 1.0 for each variant
   - Stemming: Enabled (to handle word variations)
   - Returns: Dict with `{"rouge1": ..., "rouge2": ..., "rougeL": ...}`
   - Logs: All three scores
   - Fallback: If no reference summary, uses first 5 clauses as proxy

3. **aggregate_similarity(clause_results: list[dict]) → float**
   - Computes mean semantic similarity across all clauses
   - Formula: `mean(clause["semantic_similarity"] for clause in clause_results)`
   - Range: 0.0 to 1.0
   - Returns: Rounded to 4 decimals
   - Logs: Mean similarity

4. **compute_classification_metrics(predicted_labels: list[str], ground_truth_labels: list[str]) → dict**
   - Binary classification metrics for importance detection
   - Classes: "IMPORTANT" (1) vs "NORMAL" (0)
   - Metrics:
     - `accuracy`: (TP + TN) / Total
     - `precision`: TP / (TP + FP)
     - `recall`: TP / (TP + FN)
     - `f1`: Harmonic mean of precision and recall
   - Zero division policy: Return 0.0 (avoids NaN)
   - Returns: Dict with all four metrics
   - Logs: Metrics + detailed classification report

**Main Evaluation Function:**

**run_evaluation(clause_results, generated_summary, reference_summary=None, ground_truth_labels=None) → dict**

- Computes all metrics
- Returns combined evaluation dict:
  ```python
  {
      "bleu": 0.0426,
      "rouge": {"rouge1": 0.44, "rouge2": 0.23, "rougeL": 0.26},
      "mean_semantic_similarity": 0.8345,
      "accuracy": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0
  }
  ```
- Classification metrics default to 0.0 if no ground truth provided

**Evaluation Interpretation:**
- **BLEU**: Higher is better; > 0.5 indicates good simplification
- **ROUGE**: Higher is better; > 0.4 indicates good summarization
- **Semantic Similarity**: Higher is better; > 0.8 indicates meaning preserved
- **Classification**: Accuracy + F1 measure importance detection quality

---

### **Layer 9: External Services**

Cloud-based LLM inference and embeddings.

**Groq API (Cloud)**
- Endpoint: `https://api.groq.com/openai/v1/chat/completions`
- Model: `llama-3.3-70b-versatile`
  - Parameters: 70 billion (SOTA open-source)
  - Reasoning: Excellent for legal document analysis
  - Speed: ~50 tokens/second (fast for cloud inference)
  - Cost: Free tier (no credit card)
- Authentication: API key from `.env` file
- Rate limiting: Internal 0.5s delay between calls
- Error handling: Groq SDK handles retries + timeouts

**Sentence-Transformers (Local)**
- Model: `BAAI/bge-large-en-v1.5`
- Type: Dense embeddings (semantic search)
- Dimensions: 1024
- Training: Trained on 1M+ paired text examples
- Speed: ~1000 queries/second (local CPU)
- Download: ~500MB (cached after first run)
- No API key required

---

### **Layer 10: Output**

Final structured result containing all analysis data.

**JSON Schema:**

```json
{
  "clauses": [
    {
      "original": "The Licensor hereby agrees to provide software support services...",
      "simplified": "The Licensor will provide software support services.",
      "importance": "IMPORTANT",
      "semantic_similarity": 0.8345
    },
    {
      "original": "All notices shall be in writing and served via email or courier.",
      "simplified": "All notices must be sent in writing by email or courier.",
      "importance": "NORMAL",
      "semantic_similarity": 0.9102
    }
  ],
  "summary": "This is a Software Service Agreement between ABC Solutions (Service Provider) and XYZ Corp (Client). The Service Provider agrees to design, develop, and maintain software applications as requested by the Client. The agreement covers service terms, payment, intellectual property rights, liability limitations, and termination conditions. Key obligations include quality assurance, confidentiality, and data security.",
  "evaluation": {
    "bleu": 0.0426,
    "rouge": {
      "rouge1": 0.442,
      "rouge2": 0.2346,
      "rougeL": 0.2652
    },
    "mean_semantic_similarity": 0.8345,
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `clauses[].original` | str | Original legal text (max 1000 chars) |
| `clauses[].simplified` | str | Simplified version in plain English |
| `clauses[].importance` | str | "IMPORTANT" or "NORMAL" |
| `clauses[].semantic_similarity` | float | Cosine similarity (0.0-1.0) |
| `summary` | str | Document-level executive summary |
| `evaluation.bleu` | float | BLEU score (0.0-1.0) |
| `evaluation.rouge` | dict | ROUGE-1/2/L scores |
| `evaluation.mean_semantic_similarity` | float | Mean clause similarity |
| `evaluation.accuracy` | float | Classification accuracy (0.0-1.0) |
| `evaluation.precision` | float | Precision for IMPORTANT class |
| `evaluation.recall` | float | Recall for IMPORTANT class |
| `evaluation.f1` | float | F1 score for IMPORTANT class |

**Output Locations:**
- CLI: Saved to `--output` path (default: `output.json`)
- Web UI: Downloadable from `/download/:task_id` endpoint

---

## Data Flow Diagram

```
USER UPLOADS PDF
    ↓
Flask /process endpoint
    ↓
Spawn background thread
    ↓
Extract text (PyPDF2)
    ↓
Split into clauses (spaCy)
    ↓
Load models (Groq + BGE)
    ↓
For each clause:
    ├─ Simplify via Groq LLM
    ├─ Detect importance via Groq LLM
    ├─ Compute similarity via BGE embeddings
    └─ Store result
    ↓
Summarize document via Groq LLM
    ↓
Compute evaluation metrics (BLEU, ROUGE, etc.)
    ↓
Assemble JSON output
    ↓
Store JSON in tasks dict
    ↓
SSE stream notifies browser
    ↓
Browser downloads JSON via /download/:task_id
    ↓
Results displayed in dashboard
```

---

## Performance Characteristics

**Typical Processing Time (25 clauses, ~3500 chars):**
- Step 1 (Extract): 0.1s
- Step 2 (Split): 0.5s
- Step 3 (Load models): 3-5s (first run only; cached after)
- Step 4 (Process): 50-60s (2-3s per clause × 25)
- Step 5 (Evaluate): 1-2s
- **Total: ~55-70 seconds**

**Bottlenecks:**
1. Groq API latency: 1-2 seconds per call (network)
2. Embedding generation: 0.2s per batch (local, but not parallelized)
3. Model loading: 3-5s first run (file I/O)

**Scalability Considerations:**
- Single-threaded Groq calls (sequential)
- No batch API usage (can add for 10-50% speedup)
- Embedding can be vectorized (numpy operations)
- Model caching reduces repeated loads

---

## Deployment Notes

**Development:**
```bash
python flask_app.py  # Runs on http://localhost:5000
```

**Production:**
- Use WSGI server (Gunicorn, uWSGI)
- Configure persistent task storage (Redis, PostgreSQL)
- Implement background job queue (Celery, RQ)
- Set up S3/Cloud Storage for file management
- Add authentication and rate limiting
- Enable HTTPS/SSL

**Example Production Startup:**
```bash
gunicorn --workers 4 --bind 0.0.0.0:5000 flask_app:app
```

---

## Security Considerations

1. **API Key Management**: Read from `.env` only (not environment, not hardcoded)
2. **File Uploads**: 16MB max; sanitized filenames
3. **Path Traversal**: Use `secure_filename()` on uploads
4. **CORS**: Currently unrestricted (add for production)
5. **SQL Injection**: Not applicable (no SQL database)
6. **XSS**: Frontend uses vanilla JS (no templating vulnerabilities)

---

## Future Enhancements

1. **Batch Processing**: Queue multiple PDFs
2. **WebSocket Support**: Replace SSE for bidirectional updates
3. **Multi-Language**: Add language detection + translation
4. **Custom Prompts**: User-defined LLM system prompts
5. **Export Formats**: PDF, Word, CSV in addition to JSON
6. **Advanced Metrics**: Custom evaluation functions
7. **Caching**: Redis for model caching + result deduplication
8. **Analytics**: Track processing stats, user activity

---

## Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | >=2.3.0 | Web server & REST API |
| PyPDF2 | >=3.0.0 | PDF text extraction |
| spacy | >=3.7.0 | Sentence segmentation |
| groq | >=0.9.0 | Cloud LLM API client |
| sentence-transformers | >=2.7.0 | Semantic embeddings |
| torch | >=2.2.0 | Backend for embeddings |
| nltk | >=3.8.1 | BLEU score computation |
| rouge-score | >=0.1.2 | ROUGE metric evaluation |
| scikit-learn | >=1.4.0 | Classification metrics |
| numpy | >=1.26.0 | Mathematical operations |
| python-dotenv | >=1.0.0 | `.env` file parsing |

---

## File Structure

```
lexigen/
├── app.py                    # CLI + Python API (orchestration)
├── models.py                 # Model loading (Groq + BGE)
├── processing.py             # Clause processing pipeline
├── evaluation.py             # Metrics computation
├── utils.py                  # PDF extraction + text cleaning
├── flask_app.py              # Flask REST API server
├── requirements.txt          # Python dependencies
├── .env                      # API key configuration
├── templates/
│   ├── index.html           # Main UI
│   └── result.html          # Results template
├── static/
│   ├── css/
│   │   └── style.css        # Professional styling
│   └── js/
│       └── main.js          # Frontend logic
├── .tmp_flask_uploads/      # Temporary PDF storage
└── ARCHITECTURE.md          # This file
```

---

## Getting Started

**1. Setup:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**2. Configure API Key:**
```bash
# Create .env file
echo "GROQ_API_KEY=gsk_your_key_here" > .env
# Get free key: https://console.groq.com
```

**3. Run Web UI:**
```bash
python flask_app.py
# Open http://localhost:5000 in browser
```

**4. Or CLI:**
```bash
python app.py --pdf contract.pdf --output results.json
```

---

**Document Version:** 1.0
**Last Updated:** April 2026
**Architecture:** Event-Driven REST + Cloud LLM + Local Embeddings
