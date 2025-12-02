# Subcorpus Analysis Workflow

Complete workflow for building, querying, and analyzing subcorpora from large scientific paper collections using semantic search and vector embeddings.

## Overview

This workflow enables you to:

1. **Build subcorpora** from large paper collections using semantic search (RRF-based ranking)
2. **Create vector databases** with sentence and paragraph-level embeddings
3. **Query efficiently** using natural language
4. **Retrieve full text** from compressed archives
5. **Visualize** results in interactive 3D space

The workflow is optimized for the S2ORC corpus but can be adapted to other scientific paper collections.

---

## Table of Contents

- [Subcorpus Analysis Workflow](#subcorpus-analysis-workflow)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Workflow Steps](#workflow-steps)
    - [Step 1: Identify Seed Papers](#step-1-identify-seed-papers)
    - [Step 2: Build Subcorpus with RRF](#step-2-build-subcorpus-with-rrf)
    - [Step 3: Create Milvus Database](#step-3-create-milvus-database)
    - [Step 4: Query the Subcorpus](#step-4-query-the-subcorpus)
    - [Step 5: Retrieve Full Text](#step-5-retrieve-full-text)
    - [Step 6: Visualize Results](#step-6-visualize-results)
  - [System Requirements](#system-requirements)
    - [Suggested hardware](#suggested-hardware)
    - [Software](#software)
    - [Milvus Setup](#milvus-setup)
    - [MongoDB Setup](#mongodb-setup)
  - [Data Pipeline Architecture](#data-pipeline-architecture)
    - [Data Flow Details](#data-flow-details)
  - [Script Reference](#script-reference)
    - [Core Scripts](#core-scripts)
    - [Supporting Scripts](#supporting-scripts)
  - [Advanced Usage](#advanced-usage)
    - [Custom Embedding Models](#custom-embedding-models)
    - [Checkpoint \& Resume](#checkpoint--resume)
    - [Paragraph Size Tuning](#paragraph-size-tuning)
    - [Query Batching](#query-batching)
    - [Context Window Tuning](#context-window-tuning)
  - [Troubleshooting](#troubleshooting)
    - [Milvus Connection Issues](#milvus-connection-issues)
    - [MongoDB Connection Issues](#mongodb-connection-issues)
    - [Memory Issues](#memory-issues)
    - [Missing Papers](#missing-papers)
    - [Index Out of Bounds Errors](#index-out-of-bounds-errors)
  - [Performance Tips](#performance-tips)
  - [Citation](#citation)
  - [License](#license)
  - [Contributing](#contributing)
  - [Contact](#contact)

---

## Quick Start

```bash
# 1. Build subcorpus from seed papers (using query_milvus_rrf.py)
python query_milvus_rrf.py \
    --queries seed_papers.txt \
    --output subcorpus_20251112.pkl \
    --top-k 10000

# 2. Create sentence/paragraph Milvus databases
python build_subcorpus_milvus.py \
    --subcorpus subcorpus_20251112.pkl \
    --s2orc-path /path/to/s2orc/ \
    --db-name compositionality_subcorpus \
    --sentence-collection sentences \
    --paragraph-collection paragraphs

# 3. Query the subcorpus
python query_subcorpus.py \
    --db-name compositionality_subcorpus \
    --collection sentences \
    --queries research_queries.txt \
    --output query_results.json \
    --limit 1000

# 4. Retrieve full text for results
python retrieve_query_texts.py \
    --input query_results.json \
    --output results_with_text.json \
    --s2orc-path /path/to/s2orc/ \
    --include-context \
    --context-mode paragraph

# 5. Visualize (optional)
python visualize_subcorpus.py \
    --subcorpus subcorpus_20251112.pkl \
    --output-dir visualizations/
```

---

## Workflow Steps

### Step 1: Identify Seed Papers

Manually curate or automatically identify seed papers that represent your research topic of interest.

**Input formats:**
- List of S2ORC corpus IDs
- List of paper titles/abstracts
- Bibliography file (BibTeX, RIS)

**Example seed papers file** (`seed_papers.txt`):
```
34474804
3066540
16056957
14559809
10509722
```

The script `s2_api_requests.py` offers functionalities to download the metadata and embeddings from the Semantic Scholar API for further processing. Alternatively, a local database from the `embeddings` dataset can be used.

### Step 2: Build Subcorpus with RRF

Use Reciprocal Rank Fusion (RRF) to identify papers semantically related to your seed papers.

**Script:** `query_milvus_rrf.py`

```bash
python query_milvus_rrf.py \
    --queries seed_papers.txt \
    --output subcorpus_20251112.pkl \
    --top-k 10000 \
    --rrf-k 60
```

**Output:** Pickle file containing:
- `corpus_ids`: List of S2ORC corpus IDs in the subcorpus
- `scores`: RRF scores for each paper
- `embeddings`: Full-paper embeddings (if available)
- `metadata`: Additional information

**Key Parameters:**
- `--top-k`: Number of papers to retrieve per query (default: 10000)
- `--rrf-k`: RRF constant (default: 60, lower = more emphasis on top ranks)
- `--queries`: File with queries (one per line)

### Step 3: Create Milvus Database

Build searchable vector databases at sentence and paragraph level.

**Script:** `build_subcorpus_milvus.py`

```bash
python build_subcorpus_milvus.py \
    --subcorpus subcorpus_20251112.pkl \
    --s2orc-path /path/to/s2orc/corpus/2024-08-06/s2orc-json-standoff/ \
    --db-name compositionality_subcorpus \
    --sentence-collection sentences \
    --paragraph-collection paragraphs \
    --paragraph-size 10 \
    --model multi-qa-MiniLM-L6-cos-v1
```

**What it does:**
1. Loads subcorpus corpus IDs from pickle file
2. Retrieves papers from S2ORC using MongoDB indices and indexed gzip
3. Segments papers into sentences (using spaCy or custom sentencizer)
4. Groups sentences into paragraphs (using annotations or fixed-size chunks)
5. Generates embeddings using sentence-transformers
6. **Stores character indices** (not text) for each sentence/paragraph to save space
7. Inserts into Milvus collections with vector indices

**Database Schema:**

**Sentence Collection:**
```
- id (primary key, auto-generated)
- corpusid (int64)
- sentence_number (int64)
- sentence_indices (array[int64, 2])  # [start_char, end_char]
- vector (float vector, dim=384)
- rrf_score (float, optional)
```

**Paragraph Collection:**
```
- id (primary key, auto-generated)
- corpusid (int64)
- paragraph_number (int64)
- paragraph_indices (array[int64, 2])  # [start_char, end_char]
- sentence_start (int64)  # First sentence number
- sentence_end (int64)    # Last sentence number
- vector (float vector, dim=384)
- rrf_score (float, optional)
```

**Key Features:**
- ✅ **Storage-efficient**: Only stores character indices, not text
- ✅ **Fast retrieval**: Uses indexed gzip for quick paper access
- ✅ **Language filtering**: Optional English-only filtering
- ✅ **Resumable**: Checkpoint support for large subcorpora
- ✅ **Flexible segmentation**: Paragraph annotations or sentence chunks

**Key Parameters:**
- `--subcorpus`: Pickle file from Step 2
- `--s2orc-path`: Path to S2ORC corpus directory
- `--db-name`: Name for Milvus database
- `--sentence-collection`: Collection name for sentences
- `--paragraph-collection`: Collection name for paragraphs
- `--paragraph-size`: Sentences per paragraph if no annotations (default: 10)
- `--model`: Sentence transformer model (default: multi-qa-MiniLM-L6-cos-v1)
- `--index-type`: Vector index type (IVF_PQ, IVF_FLAT, HNSW)
- `--checkpoint-file`: File for resumable processing
- `--no-language-filter`: Disable English-only filtering

### Step 4: Query the Subcorpus

Search the subcorpus using natural language queries.

**Script:** `query_subcorpus.py`

```bash
python query_subcorpus.py \
    --db-name compositionality_subcorpus \
    --collection sentences \
    --queries research_queries.txt \
    --output query_results.json \
    --limit 1000 \
    --metric-type COSINE
```

**Queries file format** (`research_queries.txt`):
```
# Research questions (lines starting with # are comments)
How does compositionality emerge in neural language models?
What computational mechanisms support semantic composition?
Evidence for compositional processing in the human brain

# Hypothesis statements
Neural networks learn compositional representations through hierarchical processing.
```

**Output format** (JSON):
```json
[
  {
    "query_idx": 0,
    "query": "How does compositionality emerge in neural language models?",
    "rank": 1,
    "distance": 0.8543,
    "corpus_id": 12345678,
    "collection": "sentences",
    "sentence_number": 42,
    "sentence_indices": [1523, 1687]
  },
  ...
]
```

**Key Features:**
- ✅ **Batch encoding**: Efficient encoding of multiple queries
- ✅ **Auto-release**: Frees Milvus resources after querying
- ✅ **Multiple formats**: JSON, CSV, Parquet output
- ✅ **Summary statistics**: Automatic result analysis
- ✅ **Custom fields**: Specify which fields to retrieve

**Key Parameters:**
- `--db-name`: Milvus database name (from Step 3)
- `--collection`: Collection to query (sentences or paragraphs)
- `--queries`: Text file with queries (one per line)
- `--output`: Output file path (.json, .csv, or .parquet)
- `--limit`: Max results per query (default: 1000)
- `--model`: Must match model used in Step 3
- `--metric-type`: Distance metric (COSINE, L2, IP)
- `--output-fields`: Fields to retrieve (default: corpusid, sentence_number, sentence_indices)
- `--no-summary`: Skip printing summary statistics

**Recent Updates:**
- ✅ Added automatic collection release to free memory
- ✅ Smart loading: tracks if collection was already loaded
- ✅ Error handling with try-finally blocks

### Step 5: Retrieve Full Text

Extract the actual text using character indices stored in the database.

**Script:** `retrieve_query_texts.py`

```bash
python retrieve_query_texts.py \
    --input query_results.json \
    --output results_with_text.json \
    --s2orc-path /path/to/s2orc/corpus/2024-08-06/s2orc-json-standoff/ \
    --include-context \
    --context-mode chars \
    --context-chars 300
```

**What it does:**
1. Loads query results (with corpus IDs and character indices)
2. Queries MongoDB to get file paths and byte offsets
3. Opens indexed gzip files efficiently
4. Extracts text using character indices
5. Optionally includes surrounding context (character-based or paragraph-based)
6. Adds text columns to results DataFrame

**Context Modes:**
- **`chars`**: Extract a fixed number of characters before/after the match (default)
- **`paragraph`**: Extract the full paragraph containing the match using S2ORC paragraph annotations
  - Automatically falls back to character-based extraction if annotations are unavailable
  - Provides more coherent context for reading and analysis

**Output format** (with `--include-context`):
```json
[
  {
    "query_idx": 0,
    "query": "How does compositionality emerge...",
    "rank": 1,
    "distance": 0.8543,
    "corpus_id": 12345678,
    "sentence_number": 42,
    "sentence_indices": [1523, 1687],
    "text": "Compositional semantics emerges through...",
    "context_before": "Previous sentence context...",
    "context_after": "Following sentence context..."
  },
  ...
]
```

**Key Features:**
- ✅ **Efficient batching**: MongoDB queries batched for performance
- ✅ **Caching**: Papers cached to avoid redundant file access
- ✅ **Flexible context**: Character-based or paragraph-based context extraction
- ✅ **Automatic fallback**: Uses character-based context if paragraph annotations unavailable
- ✅ **Checkpoint/Resume**: Supports resuming interrupted processing with automatic checkpointing
- ✅ **Error handling**: Graceful handling of missing papers
- ✅ **Progress tracking**: tqdm progress bars with file-level grouping

**Key Parameters:**
- `--input`: Query results file (from Step 4)
- `--output`: Output file with text
- `--s2orc-path`: Path to S2ORC corpus
- `--include-context`: Add surrounding text
- `--context-mode`: Context extraction mode - `chars` (default) or `paragraph`
- `--context-chars`: Characters before/after for `chars` mode (default: 200)
- `--batch-size`: MongoDB query batch size (default: 1000)
- `--checkpoint-file`: Path to checkpoint file for resuming (auto-generated if not provided)
- `--save-interval`: Save progress every N papers (default: 100)
- `--mongo-host`, `--mongo-port`: MongoDB connection (default: localhost:27017)
- `--mongo-db`: MongoDB database name (default: papers_db)
- `--mongo-collection`: Collection name (default: papers)

**Recent Updates:**
- ✅ Added paragraph-based context extraction using S2ORC annotations
- ✅ New `--context-mode` parameter (chars/paragraph) for flexible context
- ✅ Automatic fallback from paragraph to character-based context
- ✅ Checkpoint/resume support for large result sets
- ✅ Auto-generated checkpoint files with progress tracking
- ✅ Improved progress display with file-level grouping

### Step 6: Visualize Results

Create interactive 3D visualizations of the subcorpus and query results.

**Script:** `visualize_subcorpus.py`

```bash
python visualize_subcorpus.py \
    --subcorpus subcorpus_20251112.pkl \
    --output-dir visualizations/ \
    --methods umap tsne \
    --n-components 3
```

**Visualizations created:**
- UMAP 3D projection (interactive HTML)
- t-SNE 3D projection (interactive HTML)
- Corpus comparison (seed vs random vs subcorpus)
- RRF score distributions

**Key Features:**
- ✅ **Interactive**: Plotly-based 3D scatter plots
- ✅ **Google Fonts**: Noto Sans for consistent rendering
- ✅ **UI controls**: Buttons, sliders for data filtering
- ✅ **Proper spacing**: No overlapping UI elements
- ✅ **Modern syntax**: Updated Plotly axis title fonts

**Recent Updates:**
- ✅ Fixed font rendering (Google Fonts CDN integration)
- ✅ Repositioned UI elements (buttons at bottom, title at top)
- ✅ Added slider suffix for percentage display
- ✅ Fixed RRF slider visibility logic
- ✅ Updated deprecated Plotly syntax

---

## System Requirements

### Suggested hardware
- **CPU**: Multi-core recommended for parallel processing
- **RAM**: 16GB minimum, 32GB+ recommended for large subcorpora
- **GPU**: Optional but recommended for faster embedding generation
- **Storage**: Large SSD recommended for local S2ORC corpus access

### Software

**Python 3.8+** with packages:
```bash
# Core dependencies
pip install numpy pandas pymilvus pymongo
pip install sentence-transformers transformers
pip install spacy tqdm

# Visualization
pip install plotly umap-learn scikit-learn

# Optional
pip install langdetect  # For language filtering
```

**External Services:**
- **Milvus 2.4**: Vector database (Docker recommended)
- **MongoDB 4.4**: For S2ORC metadata indexing

**S2ORC Corpus:**
- Download from [Semantic Scholar](https://allenai.org/data/s2orc)
- Requires indexed gzip format with MongoDB indices
- Requires the following datasets:
  -  `S2ORC` for full-text, 
  -  `paper` for metadata, 
  -  and `embeddings` for precalculated SPECTER embeddings

### Milvus Setup

```bash
# Using Docker Compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d

# Verify
curl http://localhost:19530/health
```

### MongoDB Setup

```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or install locally
# Import S2ORC metadata using import_papers_mongo.py
```

---

## Data Pipeline Architecture

```
┌─────────────────┐
│  Seed Papers    │ (Manual curation or bibliometrics)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  query_milvus_rrf.py        │ (RRF-based subcorpus selection)
│  - Full corpus search       │
│  - Reciprocal Rank Fusion   │
│  - Top-K paper selection    │
└────────┬────────────────────┘
         │
         ▼ subcorpus_YYYYMMDD.pkl
         │
┌─────────────────────────────┐
│  build_subcorpus_milvus.py  │ (Vector database construction)
│  - Load papers via MongoDB  │
│  - Segment into sent/para   │
│  - Generate embeddings      │
│  - Store indices (not text) │
└────────┬────────────────────┘
         │
         ▼ Milvus DB (sentences, paragraphs)
         │
┌─────────────────────────────┐
│  query_subcorpus.py         │ (Semantic search)
│  - Encode queries           │
│  - Vector similarity search │
│  - Return indices + IDs     │
└────────┬────────────────────┘
         │
         ▼ query_results.json
         │
┌─────────────────────────────┐
│  retrieve_query_texts.py    │ (Text extraction)
│  - MongoDB metadata lookup  │
│  - Indexed gzip access      │
│  - Character index slicing  │
└────────┬────────────────────┘
         │
         ▼ results_with_text.json
         │
┌─────────────────────────────┐
│  visualize_subcorpus.py     │ (Analysis & visualization)
│  - UMAP/t-SNE projections   │
│  - Interactive 3D plots     │
│  - Distribution analysis    │
└─────────────────────────────┘
```

### Data Flow Details

1. **Seed → Subcorpus**
   - Input: ~10-100 seed papers
   - Process: RRF across full corpus (millions of papers)
   - Output: ~1,000-100,000 related papers

2. **Subcorpus → Vector DB**
   - Input: Corpus IDs + RRF scores
   - Process: Segment, encode, index
   - Output: 2 Milvus collections (sentences, paragraphs)
   - Storage: ~1-2GB per 10,000 papers (indices only)

3. **Query → Results**
   - Input: Natural language queries
   - Process: Encode, vector search
   - Output: Top-K matches with indices

4. **Indices → Text**
   - Input: Corpus IDs + character indices
   - Process: Gzip decompression + slicing
   - Output: Full text with optional context

---

## Script Reference

### Core Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `query_milvus_rrf.py` | Build subcorpus using RRF | Seed queries | Pickle file with corpus IDs |
| `build_subcorpus_milvus.py` | Create vector databases | Subcorpus pickle | Milvus collections |
| `query_subcorpus.py` | Query subcorpus | Query text file | Results with indices |
| `retrieve_query_texts.py` | Extract full text | Query results | Results with text |
| `visualize_subcorpus.py` | Visualize subcorpus | Subcorpus pickle | Interactive HTML |

### Supporting Scripts

| Script | Purpose |
|--------|---------|
| `import_papers_mongo.py` | Import S2ORC metadata to MongoDB |
| `sample_random_corpus.py` | Create random paper samples |
| `get_paper_by_line.py` | Retrieve paper by line number |
| `analyze_query_results.py` | Statistical analysis of results |

---

## Advanced Usage

### Custom Embedding Models

```bash
# Use a different sentence transformer
python build_subcorpus_milvus.py \
    --subcorpus subcorpus.pkl \
    --s2orc-path /path/to/s2orc/ \
    --db-name my_subcorpus \
    --model sentence-transformers/all-mpnet-base-v2 \
    --sentence-collection sentences \
    --paragraph-collection paragraphs

# Query with matching model
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection sentences \
    --queries queries.txt \
    --model sentence-transformers/all-mpnet-base-v2 \
    --output results.json
```

### Checkpoint & Resume

```bash
# Build with checkpointing (for large subcorpora)
python build_subcorpus_milvus.py \
    --subcorpus subcorpus_large.pkl \
    --s2orc-path /path/to/s2orc/ \
    --db-name large_subcorpus \
    --sentence-collection sentences \
    --paragraph-collection paragraphs \
    --checkpoint-file checkpoint.pkl \
    --checkpoint-interval 1000

# Resume from checkpoint if interrupted
# (automatically resumes if checkpoint exists)

# Text retrieval with checkpointing (auto-generated checkpoint file)
python retrieve_query_texts.py \
    --input large_results.json \
    --output results_with_text.json \
    --s2orc-path /path/to/s2orc/ \
    --include-context \
    --save-interval 100

# Or specify custom checkpoint file
python retrieve_query_texts.py \
    --input large_results.json \
    --output results_with_text.json \
    --s2orc-path /path/to/s2orc/ \
    --include-context \
    --checkpoint-file my_checkpoint.json \
    --save-interval 50
# If interrupted, simply re-run the same command - it will ask to resume
```

### Paragraph Size Tuning

```bash
# Smaller paragraphs (5 sentences)
python build_subcorpus_milvus.py \
    --subcorpus subcorpus.pkl \
    --s2orc-path /path/to/s2orc/ \
    --db-name subcorpus_small_para \
    --sentence-collection sentences \
    --paragraph-collection paragraphs \
    --paragraph-size 5

# Larger paragraphs (20 sentences)
python build_subcorpus_milvus.py \
    --subcorpus subcorpus.pkl \
    --s2orc-path /path/to/s2orc/ \
    --db-name subcorpus_large_para \
    --sentence-collection sentences \
    --paragraph-collection paragraphs \
    --paragraph-size 20
```

### Query Batching

```bash
# Process large query sets
split -l 100 all_queries.txt queries_batch_
for file in queries_batch_*; do
    python query_subcorpus.py \
        --db-name my_subcorpus \
        --collection sentences \
        --queries "$file" \
        --output "results_${file}.json"
done

# Merge results
python -c "
import pandas as pd
import glob
dfs = [pd.read_json(f) for f in glob.glob('results_queries_batch_*.json')]
pd.concat(dfs).to_json('all_results.json', orient='records')
"
```

### Context Window Tuning

```bash
# Paragraph-based context (uses S2ORC annotations)
python retrieve_query_texts.py \
    --input results.json \
    --output results_paragraph_context.json \
    --s2orc-path /path/to/s2orc/ \
    --include-context \
    --context-mode paragraph

# Large character-based context for reading
python retrieve_query_texts.py \
    --input results.json \
    --output results_large_context.json \
    --s2orc-path /path/to/s2orc/ \
    --include-context \
    --context-mode chars \
    --context-chars 1000

# Minimal context for quick scanning
python retrieve_query_texts.py \
    --input results.json \
    --output results_minimal.json \
    --s2orc-path /path/to/s2orc/ \
    --include-context \
    --context-mode chars \
    --context-chars 50
```

---

## Troubleshooting

### Milvus Connection Issues

```bash
# Check if Milvus is running
curl http://localhost:19530/health

# Check available databases
python -c "
from pymilvus import connections, db
connections.connect(host='localhost', port=19530)
print(db.list_database())
"

# List collections in a database
python -c "
from pymilvus import connections, MilvusClient, db
connections.connect(host='localhost', port=19530)
db.using_database('your_db_name')
client = MilvusClient(uri='http://localhost:19530', db_name='your_db_name')
print(client.list_collections())
"
```

### MongoDB Connection Issues

```bash
# Test MongoDB connection
python -c "
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
print(client.list_database_names())
"

# Check if S2ORC metadata is indexed
python -c "
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['SemanticScholar']
collection = db['gzippedJson']
print(f'Total papers indexed: {collection.count_documents({})}')
"
```

### Memory Issues

```bash
# Reduce batch size for encoding
python build_subcorpus_milvus.py \
    --subcorpus subcorpus.pkl \
    --s2orc-path /path/to/s2orc/ \
    --db-name my_subcorpus \
    --sentence-collection sentences \
    # Add --batch-size 64 (default is 128)

# Use CPU instead of GPU
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection sentences \
    --queries queries.txt \
    --output results.json \
    --no-gpu
```

### Missing Papers

```bash
# Check if corpus ID exists in MongoDB
python -c "
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['SemanticScholar']
collection = db['gzippedJson']
result = collection.find_one({'corpusid': 12345678})
print('Found' if result else 'Not found')
if result:
    print(f\"File: {result['filename']}, Offset: {result['byte_offset']}\")
"

# Verify file exists
python -c "
from pathlib import Path
fos = 'bio'  # from MongoDB
filename = 'example.gz'  # from MongoDB
path = Path('/path/to/s2orc') / fos / filename
print('Exists' if path.exists() else 'Missing')
"
```

### Index Out of Bounds Errors

This can happen if sentence indices exceed paper length. Usually caused by:
- Corrupted standoff annotations
- Mismatch between annotation and text versions

```bash
# Validate indices
python -c "
import json, gzip
from pymongo import MongoClient

corpus_id = 12345678
client = MongoClient('localhost', 27017)
meta = client['SemanticScholar']['gzippedJson'].find_one({'corpusid': corpus_id})

with gzip.open(f\"/path/to/{meta['fos']}/{meta['filename']}\", 'rt') as f:
    f.seek(meta['byte_offset'])
    paper = json.loads(f.readline())
    
text_len = len(paper['content']['text'])
print(f'Text length: {text_len}')

for i, sent in enumerate(paper['content']['annotations']['sentences']):
    if sent['end'] > text_len:
        print(f'Sentence {i}: indices {sent} exceed text length')
"
```

---

## Performance Tips

1. **Use SSD** for S2ORC corpus (10-100x faster than HDD)
2. **Batch processing**: Process queries and papers in batches
3. **GPU encoding**: 10-50x faster for embedding generation
4. **Index properly**: Ensure MongoDB has indices on `corpusid`
5. **Cache strategically**: The text retriever caches papers automatically
6. **Checkpoint frequently**: For large subcorpora, checkpoint every 1000 papers
7. **Choose right collection**: Use sentences for precision, paragraphs for context

---

## Citation

If you use this workflow in your research, please cite:

```bibtex
@software{subcorpus_workflow_2025,
  author = {Your Name},
  title = {Subcorpus Analysis Workflow for Scientific Literature},
  year = {2025},
  url = {https://github.com/yourusername/your-repo}
}
```

---

## License

[Your chosen license - e.g., MIT, Apache 2.0]

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

---

## Contact

[Your contact information or links]

---

**Last Updated:** November 14, 2025

**Version:** 1.1.0
