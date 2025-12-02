# BERTopic Analysis Environment Setup

Reproducible environment setup for topic modeling analysis on Apple Silicon (M4 Pro) with GPU acceleration.

## Quick Start

```bash
# 1. Create the environment
bash setup_environment.sh

# 2. Activate the environment
conda activate bertopic

# 3. Verify installation
python -c "from topic_modeling_analysis import *; print('✅ Success')"

# 4. Run the notebook
jupyter lab topic_modeling_notebook.ipynb
```

## System Requirements

- **OS**: macOS (Apple Silicon - M1/M2/M3/M4)
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: ~5GB for environment + data space
- **Conda**: Miniconda or Anaconda (download from [here](https://docs.conda.io/en/latest/miniconda.html))

## Environment Details

The environment includes:

### Core Dependencies
- **Python**: 3.10
- **PyTorch**: >=2.0.0 with MPS (Metal Performance Shaders) support
- **Transformers**: >=4.30.0
- **Sentence Transformers**: >=2.2.0

### Topic Modeling
- **BERTopic**: >=0.15.0
- **UMAP**: >=0.5.3
- **HDBSCAN**: >=0.8.33
- **Gensim**: >=4.3.0 (coherence metrics)

### NLP Processing
- **spaCy**: >=3.5.0 + en_core_web_sm model
- **NLTK**: >=3.8.0 + punkt, stopwords, wordnet

### Data & Visualization
- **Pandas**: >=2.0.0
- **NumPy**: <2.0.0 (compatibility)
- **Plotly**: >=5.14.0
- **Matplotlib**: >=3.7.0
- **PyMongo**: >=4.3.0

### Development
- **JupyterLab**: >=4.0.0
- **Black, Flake8, Pylint, MyPy**: Code quality tools

## GPU Acceleration on Apple Silicon

### Checking GPU Availability

```python
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using M4 Pro GPU")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU only")
```

### Using GPU in Topic Modeling

The `topic_modeling_analysis.py` module automatically detects and uses GPU when available:

```python
from topic_modeling_analysis import ModelConfig, TopicModeler

# GPU is enabled by default
config = ModelConfig(use_gpu=True)
modeler = TopicModeler(config)

# For CPU-only (if needed)
config = ModelConfig(use_gpu=False)
```

### GPU Acceleration Components

1. **Embeddings** (via sentence-transformers):
   - Automatically uses MPS when available
   - 5-10x speedup on M4 Pro vs CPU

2. **UMAP** (via cuML):
   - Requires cuML (CUDA-only, not available on macOS)
   - Falls back to CPU UMAP automatically

3. **HDBSCAN** (via cuML):
   - Requires cuML (CUDA-only, not available on macOS)
   - Falls back to CPU HDBSCAN automatically

**Note**: UMAP and HDBSCAN GPU acceleration requires NVIDIA CUDA and is not available on Apple Silicon. However, embeddings generation (the most compute-intensive part) will use the M4 Pro GPU through MPS.

## Manual Setup (if script fails)

```bash
# 1. Create environment
conda env create -f environment_bertopic.yml

# 2. Activate environment
conda activate bertopic

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Verify PyTorch MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Updating the Environment

```bash
# Update from environment file
conda env update -n bertopic -f environment_bertopic.yml --prune

# Or update specific packages
conda activate bertopic
conda update --all
```

## Exporting the Environment

For exact reproducibility, export the current environment:

```bash
# Export with exact versions (platform-specific)
conda env export > environment_exact.yml

# Export with minimal specs (cross-platform)
conda env export --from-history > environment_minimal.yml
```

## Troubleshooting

### MPS Not Available

If `torch.backends.mps.is_available()` returns `False`:

1. Check macOS version (requires macOS 12.3+):
   ```bash
   sw_vers
   ```

2. Reinstall PyTorch:
   ```bash
   conda activate bertopic
   conda install pytorch torchvision torchaudio -c pytorch
   ```

3. Verify installation:
   ```python
   import torch
   print(torch.__version__)
   print(torch.backends.mps.is_built())
   ```

### Memory Issues

If encountering memory errors:

1. **Reduce batch size** in embedding generation:
   ```python
   embeddings = embedding_gen.generate(docs, batch_size=64)  # Default is 128
   ```

2. **Process in chunks** for large datasets:
   ```python
   # Split into smaller chunks
   chunk_size = 10000
   for i in range(0, len(docs), chunk_size):
       chunk = docs[i:i+chunk_size]
       # Process chunk
   ```

3. **Monitor memory**:
   ```python
   import psutil
   print(f"RAM usage: {psutil.virtual_memory().percent}%")
   ```

### Package Conflicts

If conda can't resolve dependencies:

```bash
# Create environment with minimal packages
conda create -n bertopic python=3.10
conda activate bertopic

# Install packages in order
conda install pytorch -c pytorch
pip install sentence-transformers bertopic
pip install -r requirements.txt  # If you create one
```

## Performance Optimization

### For M4 Pro GPU

```python
# Enable MPS fallback for unsupported operations
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Set number of threads for CPU operations
import torch
torch.set_num_threads(12)  # M4 Pro has 12 performance cores
```

### Memory Management

```python
from topic_modeling_analysis import EnvironmentSetup

# Clean up memory periodically
EnvironmentSetup.cleanup_memory()
```

## Alternative: Docker (Not Recommended for macOS)

While a Dockerfile is possible, Docker on macOS doesn't provide GPU access for Apple Silicon. If you need Docker for deployment on Linux servers with NVIDIA GPUs, see `Dockerfile.cuda` (create separately for CUDA-based systems).

For macOS with Apple Silicon, **conda is the recommended approach**.

## Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [Sentence Transformers](https://www.sbert.net/)
- [Apple Silicon ML Performance](https://developer.apple.com/metal/pytorch/)

## Version History

- **v2.0.0** (2024-11-14): Modular architecture with dual format support
- **v1.0.0**: Original monolithic script

## License

See main repository LICENSE file.
