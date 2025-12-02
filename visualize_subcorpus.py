"""
Visualize and analyze subcorpus distribution using state-of-the-art techniques.

This script provides comprehensive visualization and analysis of a subcorpus
created by RRF-based querying, including:
- UMAP 2D/3D projections
- t-SNE visualizations
- Density plots
- Hierarchical clustering
- Score distribution analysis
- Interactive plots with Plotly

Usage:
    python visualize_subcorpus.py --subcorpus subcorpus_20251105_164654.pkl --seed-embeddings data/paper_embeddings.pkl
"""

import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# MongoDB for metadata
from pymongo import MongoClient, errors

# Dimensionality reduction
from sklearn.manifold import TSNE
import umap

# Try to import GPU-accelerated t-SNE
try:
    from tsnecuda import TSNE as TSNE_CUDA
    TSNECUDA_AVAILABLE = True
except ImportError:
    TSNECUDA_AVAILABLE = False

# Clustering
from sklearn.cluster import HDBSCAN, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# Statistics
from scipy.stats import gaussian_kde

# Interactive visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Check for available fonts
import matplotlib.font_manager as fm
def check_font_available(font_name: str) -> bool:
    """Check if a font is available on the system."""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    return font_name in available_fonts


def set_text_properties(fontfamily: str):
    """
    Return a dictionary of font properties for matplotlib text elements.
    
    Args:
        fontfamily: Font family name
        
    Returns:
        Dictionary with fontfamily property
    """
    return {'fontfamily': fontfamily}


class SubcorpusVisualizer:
    """Visualize and analyze subcorpus distribution."""
    
    def __init__(self, subcorpus_file: str, seed_embeddings_file: str, output_dir: str = "visualizations", 
                 font_family: str = "Noto Sans"):
        """
        Initialize visualizer.
        
        Args:
            subcorpus_file: Path to pickle file with subcorpus results
            seed_embeddings_file: Path to pickle file with seed paper embeddings
            output_dir: Directory to save visualizations
            font_family: Font family for all plots (default: Noto Sans)
        """
        self.subcorpus_file = subcorpus_file
        self.seed_embeddings_file = seed_embeddings_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set font
        if check_font_available(font_family):
            self.font_family = font_family
            print(f"✓ Using font: {font_family}")
        else:
            print(f"⚠ Font '{font_family}' not available, using default font")
            self.font_family = plt.rcParams['font.family'][0] if isinstance(plt.rcParams['font.family'], list) else plt.rcParams['font.family']
        
        # Set matplotlib font
        plt.rcParams['font.family'] = self.font_family
        
        # Data containers
        self.corpus_ids = None
        self.scores = None
        self.embeddings = None
        self.seed_ids = None
        self.seed_embeddings = None
        self.embedding_dim = None
        self.metadata = {}  # Will store paper metadata (title, year, journal)
        self.metadata_cache_file = self.output_dir / "paper_metadata_cache.json"
        
        # Random sample data for comparison
        self.random_sample_ids = None
        self.random_sample_embeddings = None
        
        # Analysis results
        self.df = None
        self.embeddings_2d = None
        self.embeddings_3d = None
        self.umap_2d = None
        self.umap_3d = None
        
    def load_data(self):
        """Load subcorpus and seed embeddings data."""
        print(f"Loading subcorpus from {self.subcorpus_file}...")
        
        with open(self.subcorpus_file, 'rb') as f:
            subcorpus_data = pickle.load(f)
        
        self.corpus_ids = subcorpus_data['corpus_ids']
        self.scores = subcorpus_data['scores']
        embeddings_dict = subcorpus_data['embeddings']
        self.embedding_dim = subcorpus_data.get('embedding_dim', 768)
        
        # Convert embeddings to numpy array
        self.embeddings = np.array([embeddings_dict[cid] for cid in self.corpus_ids])
        
        print(f"✓ Loaded {len(self.corpus_ids)} papers")
        print(f"  Embedding dimension: {self.embedding_dim}")
        
        # Load seed embeddings
        print(f"\nLoading seed embeddings from {self.seed_embeddings_file}...")
        
        with open(self.seed_embeddings_file, 'rb') as f:
            seed_data = pickle.load(f)
        
        self.seed_ids = set(int(cid) for cid in seed_data.keys())
        self.seed_embeddings = {int(cid): np.array(emb) for cid, emb in seed_data.items()}
        
        print(f"✓ Loaded {len(self.seed_ids)} seed papers")
        
        # Add seed papers that are not in subcorpus to the dataset
        seed_not_in_subcorpus = self.seed_ids - set(self.corpus_ids)
        print(f"  Seed papers not in subcorpus: {len(seed_not_in_subcorpus)}")
        
        if seed_not_in_subcorpus:
            # Add seed papers with score 0 (they weren't ranked in subcorpus)
            additional_corpus_ids = list(seed_not_in_subcorpus)
            additional_scores = [0.0] * len(additional_corpus_ids)
            additional_embeddings = [self.seed_embeddings[cid] for cid in additional_corpus_ids]
            
            # Extend the main lists
            self.corpus_ids = self.corpus_ids + additional_corpus_ids
            self.scores = self.scores + additional_scores
            self.embeddings = np.vstack([self.embeddings, additional_embeddings])
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'corpus_id': self.corpus_ids,
            'rrf_score': self.scores,
            'is_seed': [cid in self.seed_ids for cid in self.corpus_ids],
            'rank': range(1, len(self.corpus_ids) + 1)
        })
        
        print(f"\n✓ Data loaded successfully")
        print(f"  Total papers (including all seeds): {len(self.corpus_ids)}")
        print(f"  Seed papers: {self.df['is_seed'].sum()}")
        print(f"  Subcorpus papers (non-seed): {(~self.df['is_seed']).sum()}")
    
    def load_metadata_cache(self):
        """Load metadata from cache file if it exists."""
        if self.metadata_cache_file.exists():
            print(f"\nLoading metadata from cache: {self.metadata_cache_file}")
            with open(self.metadata_cache_file, 'r') as f:
                self.metadata = json.load(f)
            # Convert string keys back to int
            self.metadata = {int(k): v for k, v in self.metadata.items()}
            print(f"✓ Loaded metadata for {len(self.metadata)} papers from cache")
            return True
        return False
    
    def save_metadata_cache(self):
        """Save metadata to cache file."""
        print(f"\nSaving metadata cache to: {self.metadata_cache_file}")
        # Convert int keys to string for JSON
        metadata_str_keys = {str(k): v for k, v in self.metadata.items()}
        with open(self.metadata_cache_file, 'w') as f:
            json.dump(metadata_str_keys, f, indent=2)
        print(f"✓ Metadata cache saved ({len(self.metadata)} papers)")
    
    def fetch_metadata_from_mongo(self, mongo_host: str = "localhost", mongo_port: int = 27017):
        """
        Fetch paper metadata (title, year, journal) from MongoDB.
        
        Args:
            mongo_host: MongoDB host address
            mongo_port: MongoDB port
        """
        # Try to load from cache first
        if self.load_metadata_cache():
            # Check if all papers are in cache
            missing_ids = set(self.corpus_ids) - set(self.metadata.keys())
            if not missing_ids:
                print("✓ All papers found in cache, no MongoDB query needed")
                return
            else:
                print(f"  {len(missing_ids)} papers not in cache, fetching from MongoDB...")
                corpus_ids_to_fetch = list(missing_ids)
        else:
            print("\nNo metadata cache found, fetching from MongoDB...")
            corpus_ids_to_fetch = self.corpus_ids
        
        # Connect to MongoDB
        print(f"Connecting to MongoDB at {mongo_host}:{mongo_port}...")
        try:
            mongo_client = MongoClient(mongo_host, mongo_port, serverSelectionTimeoutMS=5000)
            # Test connection
            mongo_client.server_info()
            print("✓ Connected to MongoDB")
        except errors.ServerSelectionTimeoutError as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            print("  Continuing without metadata...")
            return
        
        collection = mongo_client.papers_db.papers
        
        # Fetch metadata for each paper
        print(f"Fetching metadata for {len(corpus_ids_to_fetch)} papers...")
        for corpus_id in tqdm(corpus_ids_to_fetch, desc="Fetching metadata"):
            try:
                result = collection.find_one({"corpusid": corpus_id})
                if result:
                    self.metadata[corpus_id] = {
                        "title": result.get("title", "Unknown"),
                        "year": result.get("year", None),
                        "journal": result.get("journal", {}).get("name", "Unknown") if result.get("journal") else "Unknown",
                        "fields": result.get("s2fieldsofstudy", []),
                    }
                else:
                    self.metadata[corpus_id] = {
                        "title": "Unknown",
                        "year": None,
                        "journal": "Unknown"
                    }
            except Exception as e:
                print(f"  Warning: Failed to fetch metadata for {corpus_id}: {e}")
                self.metadata[corpus_id] = {
                    "title": "Unknown",
                    "year": None,
                    "journal": "Unknown"
                }
        
        mongo_client.close()
        print(f"✓ Metadata fetched for {len(self.metadata)} papers")
        
        # Save to cache
        self.save_metadata_cache()
        
        # Add metadata to DataFrame
        self.df['title'] = self.df['corpus_id'].map(lambda cid: self.metadata.get(cid, {}).get('title', 'Unknown'))
        self.df['year'] = self.df['corpus_id'].map(lambda cid: self.metadata.get(cid, {}).get('year', None))
        self.df['journal'] = self.df['corpus_id'].map(lambda cid: self.metadata.get(cid, {}).get('journal', 'Unknown'))
    
    def load_random_sample(self, random_sample_file: str):
        """
        Load random sample from corpus for comparison.
        
        Args:
            random_sample_file: Path to pickle file with random sample from sample_random_corpus.py
        """
        print(f"\nLoading random sample from {random_sample_file}...")
        
        with open(random_sample_file, 'rb') as f:
            sample_data = pickle.load(f)
        
        self.random_sample_ids = sample_data['corpus_ids']
        self.random_sample_embeddings = sample_data['embeddings']
        
        print(f"✓ Loaded random sample")
        print(f"  Sample size: {len(self.random_sample_ids)}")
        print(f"  Embedding dimension: {sample_data.get('embedding_dim', 'Unknown')}")
        
        # Verify embedding dimensions match
        if self.embedding_dim and sample_data.get('embedding_dim'):
            if self.embedding_dim != sample_data['embedding_dim']:
                print(f"  ⚠ Warning: Embedding dimension mismatch!")
                print(f"    Subcorpus: {self.embedding_dim}, Random sample: {sample_data['embedding_dim']}")
    
    def save_plotly_html(self, fig, output_file: Path):
        """
        Save Plotly figure to HTML with Google Fonts link for proper font rendering.
        
        Args:
            fig: Plotly figure object
            output_file: Path to save HTML file
        """
        # Generate HTML
        html_string = fig.to_html(include_plotlyjs='cdn')
        
        # Insert Google Fonts link for Noto Sans into the <head> section
        google_fonts_link = '<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap" rel="stylesheet">'
        
        # Find the </head> tag and insert the font link before it
        html_string = html_string.replace('</head>', f'    {google_fonts_link}\n</head>')
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_string)
    
    def compute_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                     metric: str = 'cosine', random_state: int = 42):
        """
        Compute UMAP projections for 2D and 3D visualization.
        
        Args:
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: Distance metric
            random_state: Random seed
        """
        print("\nComputing UMAP projections...")
        
        # 2D UMAP
        print("  Computing 2D UMAP...")
        umap_2d = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=False
        )
        self.umap_2d = umap_2d.fit_transform(self.embeddings)
        
        self.df['umap_x'] = self.umap_2d[:, 0]
        self.df['umap_y'] = self.umap_2d[:, 1]
        
        # 3D UMAP
        print("  Computing 3D UMAP...")
        umap_3d = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=False
        )
        self.umap_3d = umap_3d.fit_transform(self.embeddings)
        
        self.df['umap_z'] = self.umap_3d[:, 2]
        
        print("✓ UMAP projections computed")
    
    def compute_tsne(self, perplexity: int = 30, max_iter: int = 1000, 
                     random_state: int = 42, n_components: int = 2, use_gpu: bool = True):
        """
        Compute t-SNE projection for 2D and 3D visualization.
        
        Args:
            perplexity: t-SNE perplexity parameter
            max_iter: Maximum number of iterations
            random_state: Random seed
            n_components: Number of dimensions (2 or 3)
            use_gpu: Whether to use GPU acceleration (if available)
        """
        print(f"\nComputing {n_components}D t-SNE projection...")
        
        # Choose t-SNE implementation
        if use_gpu and TSNECUDA_AVAILABLE:
            print("  Using GPU-accelerated t-SNE (tsnecuda)")
            use_cuda = True
        else:
            if use_gpu:
                print("  tsnecuda not available, using CPU t-SNE")
            else:
                print("  Using CPU t-SNE")
            use_cuda = False
        
        # Compute 2D t-SNE
        print("  Computing 2D t-SNE...")
        if use_cuda:
            tsne_2d = TSNE_CUDA(
                n_components=2,
                perplexity=perplexity,
                num_neighbors=min(perplexity * 3, len(self.embeddings) - 1),
                random_seed=random_state
            )
        else:
            tsne_2d = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=max_iter,
                random_state=random_state,
                verbose=0
            )
        
        self.embeddings_2d = tsne_2d.fit_transform(self.embeddings.astype(np.float32) if use_cuda else self.embeddings)
        
        self.df['tsne_x'] = self.embeddings_2d[:, 0]
        self.df['tsne_y'] = self.embeddings_2d[:, 1]
        
        # Compute 3D t-SNE if requested
        if n_components == 3:
            print("  Computing 3D t-SNE...")
            if use_cuda:
                tsne_3d = TSNE_CUDA(
                    n_components=3,
                    perplexity=perplexity,
                    num_neighbors=min(perplexity * 3, len(self.embeddings) - 1),
                    random_seed=random_state
                )
            else:
                tsne_3d = TSNE(
                    n_components=3,
                    perplexity=perplexity,
                    max_iter=max_iter,
                    random_state=random_state,
                    verbose=0
                )
            
            self.tsne_3d = tsne_3d.fit_transform(self.embeddings.astype(np.float32) if use_cuda else self.embeddings)
            
            self.df['tsne_z'] = self.tsne_3d[:, 2]
        
        print("✓ t-SNE projection computed")
    
    def plot_umap_2d(self, save: bool = True):
        """Create 2D UMAP visualization."""
        print("\nCreating 2D UMAP visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Colored by seed status
        ax = axes[0]
        
        # Plot non-seed papers
        non_seed_mask = ~self.df['is_seed']
        ax.scatter(
            self.df.loc[non_seed_mask, 'umap_x'],
            self.df.loc[non_seed_mask, 'umap_y'],
            c='lightblue',
            alpha=0.5,
            s=30,
            label='Subcorpus papers',
            edgecolors='none'
        )
        
        # Plot seed papers
        seed_mask = self.df['is_seed']
        ax.scatter(
            self.df.loc[seed_mask, 'umap_x'],
            self.df.loc[seed_mask, 'umap_y'],
            c='red',
            alpha=0.8,
            s=100,
            marker='*',
            label='Seed papers',
            edgecolors='darkred',
            linewidths=0.5
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontfamily=self.font_family)
        ax.set_ylabel('UMAP 2', fontsize=12, fontfamily=self.font_family)
        ax.set_title('UMAP 2D Projection: Seed vs Subcorpus Papers', fontsize=14, fontweight='bold', fontfamily=self.font_family)
        ax.legend(fontsize=10, prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Colored by RRF score
        ax = axes[1]
        scatter = ax.scatter(
            self.df['umap_x'],
            self.df['umap_y'],
            c=self.df['rrf_score'],
            cmap='viridis',
            alpha=0.6,
            s=50,
            edgecolors='none'
        )
        
        # Overlay seed papers
        ax.scatter(
            self.df.loc[seed_mask, 'umap_x'],
            self.df.loc[seed_mask, 'umap_y'],
            facecolors='none',
            edgecolors='red',
            s=150,
            linewidths=2,
            marker='o',
            label='Seed papers'
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontfamily=self.font_family)
        ax.set_ylabel('UMAP 2', fontsize=12, fontfamily=self.font_family)
        ax.set_title('UMAP 2D Projection: Colored by RRF Score', fontsize=14, fontweight='bold', fontfamily=self.font_family)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('RRF Score', fontsize=10, fontfamily=self.font_family)
        ax.legend(fontsize=10, prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'umap_2d.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {output_file}")
        
        plt.close()
    
    def plot_umap_3d_interactive(self, save: bool = True):
        """Create interactive 3D UMAP visualization with continuous RRF score slider and percentage display."""
        print("\nCreating interactive 3D UMAP visualization with continuous RRF slider...")
        
        # Prepare data
        df_plot = self.df.copy()
        df_plot['type'] = df_plot['is_seed'].map({True: 'Seed', False: 'Subcorpus'})
        
        # Helper function to create hover text
        def make_hover_text(row):
            if self.metadata and row['corpus_id'] in self.metadata:
                meta = self.metadata[row['corpus_id']]
                title = meta.get('title', 'N/A')
                if title and len(title) > 80:
                    title = title[:77] + '...'
                elif not title:
                    title = 'N/A'
                    
                year = meta.get('year', 'N/A')
                if year is None:
                    year = 'N/A'
                    
                journal = meta.get('journal', 'N/A')
                if journal and len(journal) > 40:
                    journal = journal[:37] + '...'
                elif not journal:
                    journal = 'N/A'
                
                return (f"<b>{title}</b><br>"
                       f"Year: {year}<br>"
                       f"Journal: {journal}<br>"
                       f"Corpus ID: {row['corpus_id']}<br>"
                       f"RRF Score: {row['rrf_score']:.4f}<br>"
                       f"Rank: {row['rank']}<br>"
                       f"Type: {row['type']}")
            else:
                return (f"Corpus ID: {row['corpus_id']}<br>"
                       f"RRF Score: {row['rrf_score']:.4f}<br>"
                       f"Rank: {row['rank']}<br>"
                       f"Type: {row['type']}")
        
        df_plot['hover_text'] = df_plot.apply(make_hover_text, axis=1)
        
        # Split data
        df_subcorpus = df_plot[~df_plot['is_seed']].copy()
        df_seed = df_plot[df_plot['is_seed']].copy()
        
        # Get RRF score range for slider
        rrf_min = df_subcorpus['rrf_score'].min()
        rrf_max = df_subcorpus['rrf_score'].max()
        total_subcorpus = len(df_subcorpus)
        
        # Create 100 steps for quasi-continuous slider
        slider_steps = 100
        rrf_thresholds = np.linspace(rrf_min, rrf_max, slider_steps)
        
        # Create figure
        fig = go.Figure()
        
        # Add seed papers (always visible)
        fig.add_trace(go.Scatter3d(
            x=df_seed['umap_x'],
            y=df_seed['umap_y'],
            z=df_seed['umap_z'],
            mode='markers',
            name='Seed Papers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                opacity=0.9,
                line=dict(color='darkred', width=1)
            ),
            text=df_seed['hover_text'],
            hovertemplate='%{text}<extra></extra>',
            visible=True,
            showlegend=True
        ))
        
        # Add subcorpus traces for each threshold (only first one visible initially)
        for i, threshold in enumerate(rrf_thresholds):
            df_filtered = df_subcorpus[df_subcorpus['rrf_score'] >= threshold]
            
            fig.add_trace(go.Scatter3d(
                x=df_filtered['umap_x'],
                y=df_filtered['umap_y'],
                z=df_filtered['umap_z'],
                mode='markers',
                name='Subcorpus',
                marker=dict(
                    size=3,
                    color=df_filtered['rrf_score'],
                    colorscale='Viridis',
                    cmin=rrf_min,
                    cmax=rrf_max,
                    colorbar=dict(
                        title=dict(
                            text="RRF Score",
                            font=dict(family=self.font_family)
                        ),
                        x=1.02,
                        len=0.7,
                        y=0.5,
                        tickfont=dict(family=self.font_family)
                    ),
                    showscale=(i == 0),  # Only show colorbar for first trace
                    opacity=0.6
                ),
                text=df_filtered['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                visible=(i == 0),  # Only first threshold visible initially
                showlegend=(i == 0)  # Only show legend for first trace
            ))
        
        # Calculate the scene range to keep it fixed
        all_x = df_plot['umap_x']
        all_y = df_plot['umap_y']
        all_z = df_plot['umap_z']
        
        x_range = [all_x.min() - 0.5, all_x.max() + 0.5]
        y_range = [all_y.min() - 0.5, all_y.max() + 0.5]
        z_range = [all_z.min() - 0.5, all_z.max() + 0.5]
        
        # Create slider steps with percentage
        steps = []
        for i, threshold in enumerate(rrf_thresholds):
            n_visible = len(df_subcorpus[df_subcorpus['rrf_score'] >= threshold])
            pct_visible = (n_visible / total_subcorpus * 100) if total_subcorpus > 0 else 0
            
            step = dict(
                method="update",
                args=[
                    {"visible": [True] + [j == i for j in range(slider_steps)]},
                    {"title.text": f"Interactive 3D UMAP Projection",
                     "sliders[0].currentvalue.suffix": f" | Showing {n_visible}/{total_subcorpus} ({pct_visible:.1f}%)"}
                ],
                label=f"{threshold:.4f}"
            )
            steps.append(step)
        
        # Create slider with percentage display
        sliders = [dict(
            active=0,
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0.0,
            currentvalue=dict(
                prefix="Min RRF Score: ",
                suffix=f" | Showing {total_subcorpus}/{total_subcorpus} (100.0%)",
                visible=True,
                xanchor="right",
                font=dict(size=14, family=self.font_family)
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            steps=steps,
            font=dict(family=self.font_family)
        )]
        
        # Create toggle buttons
        updatemenus = [dict(
            type="buttons",
            direction="right",
            x=0.5,
            xanchor="center",
            y=-0.25,
            yanchor="top",
            buttons=[
                dict(label="All Papers",
                     method="update",
                     args=[{"visible": [True] + [True] * slider_steps},
                           {"sliders[0].visible": True}]),
                dict(label="Subcorpus Only",
                     method="update",
                     args=[{"visible": [False] + [True] * slider_steps},
                           {"sliders[0].visible": True}]),
                dict(label="Seed Papers Only",
                     method="update",
                     args=[{"visible": [True] + [False] * slider_steps},
                           {"sliders[0].visible": False}]),
            ],
            font=dict(family=self.font_family)
        )]
        
        # Update layout with fixed scene range
        fig.update_layout(
            title=dict(
                text="Interactive 3D UMAP Projection",
                font=dict(size=18, family=self.font_family),
                x=0.5,
                xanchor='center',
                y=0.98
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text="UMAP 1", font=dict(family=self.font_family)),
                    range=x_range,
                    tickfont=dict(family=self.font_family)
                ),
                yaxis=dict(
                    title=dict(text="UMAP 2", font=dict(family=self.font_family)),
                    range=y_range,
                    tickfont=dict(family=self.font_family)
                ),
                zaxis=dict(
                    title=dict(text="UMAP 3", font=dict(family=self.font_family)),
                    range=z_range,
                    tickfont=dict(family=self.font_family)
                ),
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.95,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(family=self.font_family)
            ),
            height=900,
            updatemenus=updatemenus,
            sliders=sliders,
            font=dict(family=self.font_family)
        )
        
        if save:
            output_file = self.output_dir / 'umap_3d_interactive.html'
            self.save_plotly_html(fig, output_file)
            print(f"  Saved to: {output_file}")
            print(f"  RRF score range: {rrf_min:.6f} to {rrf_max:.6f}")
            print(f"  Slider has {slider_steps} quasi-continuous steps")
            print("  Displays percentage of visible datapoints")
        
        return fig
    
    def plot_tsne_3d_interactive(self, save: bool = True):
        """Create interactive 3D t-SNE visualization with continuous RRF score slider and percentage display."""
        if 'tsne_z' not in self.df.columns:
            print("⚠ 3D t-SNE not computed. Run compute_tsne(n_components=3) first.")
            return None
        
        print("\nCreating interactive 3D t-SNE visualization with continuous RRF slider...")
        
        # Prepare data
        df_plot = self.df.copy()
        df_plot['type'] = df_plot['is_seed'].map({True: 'Seed', False: 'Subcorpus'})
        
        # Helper function to create hover text
        def make_hover_text(row):
            if self.metadata and row['corpus_id'] in self.metadata:
                meta = self.metadata[row['corpus_id']]
                title = meta.get('title', 'N/A')
                if title and len(title) > 80:
                    title = title[:77] + '...'
                elif not title:
                    title = 'N/A'
                    
                year = meta.get('year', 'N/A')
                if year is None:
                    year = 'N/A'
                    
                journal = meta.get('journal', 'N/A')
                if journal and len(journal) > 40:
                    journal = journal[:37] + '...'
                elif not journal:
                    journal = 'N/A'
                
                return (f"<b>{title}</b><br>"
                       f"Year: {year}<br>"
                       f"Journal: {journal}<br>"
                       f"Corpus ID: {row['corpus_id']}<br>"
                       f"RRF Score: {row['rrf_score']:.4f}<br>"
                       f"Rank: {row['rank']}<br>"
                       f"Type: {row['type']}")
            else:
                return (f"Corpus ID: {row['corpus_id']}<br>"
                       f"RRF Score: {row['rrf_score']:.4f}<br>"
                       f"Rank: {row['rank']}<br>"
                       f"Type: {row['type']}")
        
        df_plot['hover_text'] = df_plot.apply(make_hover_text, axis=1)
        
        # Split data
        df_subcorpus = df_plot[~df_plot['is_seed']].copy()
        df_seed = df_plot[df_plot['is_seed']].copy()
        
        # Get RRF score range for slider
        rrf_min = df_subcorpus['rrf_score'].min()
        rrf_max = df_subcorpus['rrf_score'].max()
        total_subcorpus = len(df_subcorpus)
        
        # Create 100 steps for quasi-continuous slider
        slider_steps = 100
        rrf_thresholds = np.linspace(rrf_min, rrf_max, slider_steps)
        
        # Create figure
        fig = go.Figure()
        
        # Add seed papers (always visible)
        fig.add_trace(go.Scatter3d(
            x=df_seed['tsne_x'],
            y=df_seed['tsne_y'],
            z=df_seed['tsne_z'],
            mode='markers',
            name='Seed Papers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                opacity=0.9,
                line=dict(color='darkred', width=1)
            ),
            text=df_seed['hover_text'],
            hovertemplate='%{text}<extra></extra>',
            visible=True,
            showlegend=True
        ))
        
        # Add subcorpus traces for each threshold (only first one visible initially)
        for i, threshold in enumerate(rrf_thresholds):
            df_filtered = df_subcorpus[df_subcorpus['rrf_score'] >= threshold]
            
            fig.add_trace(go.Scatter3d(
                x=df_filtered['tsne_x'],
                y=df_filtered['tsne_y'],
                z=df_filtered['tsne_z'],
                mode='markers',
                name='Subcorpus',
                marker=dict(
                    size=3,
                    color=df_filtered['rrf_score'],
                    colorscale='Viridis',
                    cmin=rrf_min,
                    cmax=rrf_max,
                    colorbar=dict(
                        title=dict(
                            text="RRF Score",
                            font=dict(family=self.font_family)
                        ),
                        x=1.02,
                        len=0.7,
                        y=0.5,
                        tickfont=dict(family=self.font_family)
                    ),
                    showscale=(i == 0),  # Only show colorbar for first trace
                    opacity=0.6
                ),
                text=df_filtered['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                visible=(i == 0),  # Only first threshold visible initially
                showlegend=(i == 0)  # Only show legend for first trace
            ))
        
        # Calculate the scene range to keep it fixed
        all_x = df_plot['tsne_x']
        all_y = df_plot['tsne_y']
        all_z = df_plot['tsne_z']
        
        x_range = [all_x.min() - 0.5, all_x.max() + 0.5]
        y_range = [all_y.min() - 0.5, all_y.max() + 0.5]
        z_range = [all_z.min() - 0.5, all_z.max() + 0.5]
        
        # Create slider steps with percentage
        steps = []
        for i, threshold in enumerate(rrf_thresholds):
            n_visible = len(df_subcorpus[df_subcorpus['rrf_score'] >= threshold])
            pct_visible = (n_visible / total_subcorpus * 100) if total_subcorpus > 0 else 0
            
            step = dict(
                method="update",
                args=[
                    {"visible": [True] + [j == i for j in range(slider_steps)]},
                    {"title.text": "Interactive 3D t-SNE Projection",
                     "sliders[0].currentvalue.suffix": f" | Showing {n_visible}/{total_subcorpus} ({pct_visible:.1f}%)"}
                ],
                label=f"{threshold:.4f}"
            )
            steps.append(step)
        
        # Create slider with percentage display
        sliders = [dict(
            active=0,
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0.0,
            currentvalue=dict(
                prefix="Min RRF Score: ",
                suffix=f" | Showing {total_subcorpus}/{total_subcorpus} (100.0%)",
                visible=True,
                xanchor="right",
                font=dict(size=14, family=self.font_family)
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            steps=steps,
            font=dict(family=self.font_family)
        )]
        
        # Create toggle buttons
        updatemenus = [dict(
            type="buttons",
            direction="right",
            x=0.5,
            xanchor="center",
            y=-0.25,
            yanchor="top",
            buttons=[
                dict(label="All Papers",
                     method="update",
                     args=[{"visible": [True] + [True] * slider_steps},
                           {"sliders[0].visible": True}]),
                dict(label="Subcorpus Only",
                     method="update",
                     args=[{"visible": [False] + [True] * slider_steps},
                           {"sliders[0].visible": True}]),
                dict(label="Seed Papers Only",
                     method="update",
                     args=[{"visible": [True] + [False] * slider_steps},
                           {"sliders[0].visible": False}]),
            ],
            font=dict(family=self.font_family)
        )]
        
        # Update layout with fixed scene range
        fig.update_layout(
            title=dict(
                text="Interactive 3D t-SNE Projection",
                font=dict(size=18, family=self.font_family),
                x=0.5,
                xanchor='center',
                y=0.98
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text="t-SNE 1", font=dict(family=self.font_family)),
                    range=x_range,
                    tickfont=dict(family=self.font_family)
                ),
                yaxis=dict(
                    title=dict(text="t-SNE 2", font=dict(family=self.font_family)),
                    range=y_range,
                    tickfont=dict(family=self.font_family)
                ),
                zaxis=dict(
                    title=dict(text="t-SNE 3", font=dict(family=self.font_family)),
                    range=z_range,
                    tickfont=dict(family=self.font_family)
                ),
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.95,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(family=self.font_family)
            ),
            height=900,
            updatemenus=updatemenus,
            sliders=sliders,
            font=dict(family=self.font_family)
        )
        
        if save:
            output_file = self.output_dir / 'tsne_3d_interactive.html'
            self.save_plotly_html(fig, output_file)
            print(f"  Saved to: {output_file}")
            print(f"  RRF score range: {rrf_min:.6f} to {rrf_max:.6f}")
            print(f"  Slider has {slider_steps} quasi-continuous steps")
            print("  Displays percentage of visible datapoints")
        
        return fig
    
    def plot_corpus_comparison_3d(self, save: bool = True, use_tsne: bool = False):
        """
        Create interactive 3D visualization comparing subcorpus with random corpus sample.
        Includes continuous RRF score slider for filtering subcorpus points.
        
        Args:
            save: Whether to save the plot
            use_tsne: If True, use t-SNE coordinates; otherwise use UMAP (default)
            
        Returns:
            Plotly figure object
        """
        if self.random_sample_ids is None or self.random_sample_embeddings is None:
            print("⚠ Random sample not loaded. Call load_random_sample() first.")
            return None
        
        # Determine which coordinates to use
        if use_tsne:
            if 'tsne_z' not in self.df.columns:
                print("⚠ 3D t-SNE not computed. Run compute_tsne(n_components=3) first.")
                return None
            coord_x, coord_y, coord_z = 'tsne_x', 'tsne_y', 'tsne_z'
            method_name = "t-SNE"
        else:
            method_name = "UMAP"
        
        print(f"\nCreating 3D {method_name} comparison visualization with continuous RRF slider...")
        
        # Combine subcorpus and random sample embeddings
        print("  Combining subcorpus and random sample...")
        
        # Get embeddings arrays
        subcorpus_embeddings = self.embeddings
        random_embeddings = np.array([self.random_sample_embeddings[cid] 
                                     for cid in self.random_sample_ids])
        
        # Combine all embeddings
        all_embeddings = np.vstack([subcorpus_embeddings, random_embeddings])
        
        # Compute projection for combined data
        print(f"  Computing {method_name} projection for combined dataset...")
        if use_tsne:
            if TSNECUDA_AVAILABLE:
                print("  Using tsnecuda for GPU acceleration...")
                reducer = TSNE_CUDA(
                    n_components=3,
                    perplexity=30,
                    learning_rate=200,
                    random_seed=42,
                    verbose=0
                )
            else:
                from sklearn.manifold import TSNE
                reducer = TSNE(
                    n_components=3,
                    perplexity=30,
                    max_iter=1000,
                    random_state=42,
                    verbose=0
                )
        else:
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                verbose=False
            )
        
        coords = reducer.fit_transform(all_embeddings)
        
        # Split coordinates back
        n_subcorpus = len(subcorpus_embeddings)
        subcorpus_coords = coords[:n_subcorpus]
        random_coords = coords[n_subcorpus:]
        
        # Calculate the scene range to keep it fixed
        all_x = coords[:, 0]
        all_y = coords[:, 1]
        all_z = coords[:, 2]
        
        x_range = [all_x.min() - 0.5, all_x.max() + 0.5]
        y_range = [all_y.min() - 0.5, all_y.max() + 0.5]
        z_range = [all_z.min() - 0.5, all_z.max() + 0.5]
        
        # Get RRF score range for subcorpus (non-seed) papers
        non_seed_mask = ~self.df['is_seed']
        non_seed_df = self.df[non_seed_mask].copy()
        rrf_min = non_seed_df['rrf_score'].min()
        rrf_max = non_seed_df['rrf_score'].max()
        total_subcorpus = len(non_seed_df)
        
        # Create 100 steps for quasi-continuous slider
        slider_steps = 100
        rrf_thresholds = np.linspace(rrf_min, rrf_max, slider_steps)
        
        # Create figure
        fig = go.Figure()
        
        # Add random sample (always visible, in gray)
        # Helper function for random sample hover text
        def make_random_hover_text(corpus_id):
            if self.metadata and corpus_id in self.metadata:
                meta = self.metadata[corpus_id]
                title = meta.get('title', 'N/A')
                if title and len(title) > 80:
                    title = title[:77] + '...'
                elif not title:
                    title = 'N/A'
                    
                year = meta.get('year', 'N/A')
                if year is None:
                    year = 'N/A'
                    
                journal = meta.get('journal', 'N/A')
                if journal and len(journal) > 40:
                    journal = journal[:37] + '...'
                elif not journal:
                    journal = 'N/A'
                
                return (f"<b>{title}</b><br>"
                       f"Year: {year}<br>"
                       f"Journal: {journal}<br>"
                       f"Corpus ID: {corpus_id}<br>"
                       f"Type: Random Sample")
            else:
                return f"Corpus ID: {corpus_id}<br>Type: Random Sample"
        
        random_hover = [make_random_hover_text(cid) for cid in self.random_sample_ids]
        
        fig.add_trace(go.Scatter3d(
            x=random_coords[:, 0],
            y=random_coords[:, 1],
            z=random_coords[:, 2],
            mode='markers',
            name='Random Corpus Sample',
            marker=dict(
                size=2,
                color='lightgray',
                opacity=0.3
            ),
            text=random_hover,
            hovertemplate='%{text}<extra></extra>',
            visible=True,
            showlegend=True
        ))
        
        # Add seed papers (always visible, red diamonds)
        seed_mask = self.df['is_seed']
        seed_indices = np.where(seed_mask)[0]
        
        if len(seed_indices) > 0:
            seed_coords = subcorpus_coords[seed_indices]
            seed_df = self.df[seed_mask]
            
            # Helper function for seed hover text
            def make_seed_hover_text(row):
                if self.metadata and row['corpus_id'] in self.metadata:
                    meta = self.metadata[row['corpus_id']]
                    title = meta.get('title', 'N/A')
                    if title and len(title) > 80:
                        title = title[:77] + '...'
                    elif not title:
                        title = 'N/A'
                        
                    year = meta.get('year', 'N/A')
                    if year is None:
                        year = 'N/A'
                        
                    journal = meta.get('journal', 'N/A')
                    if journal and len(journal) > 40:
                        journal = journal[:37] + '...'
                    elif not journal:
                        journal = 'N/A'
                    
                    return (f"<b>{title}</b><br>"
                           f"Year: {year}<br>"
                           f"Journal: {journal}<br>"
                           f"Corpus ID: {row['corpus_id']}<br>"
                           f"Type: Seed Paper")
                else:
                    return f"Corpus ID: {row['corpus_id']}<br>Type: Seed Paper"
            
            seed_hover = seed_df.apply(make_seed_hover_text, axis=1).tolist()
            
            fig.add_trace(go.Scatter3d(
                x=seed_coords[:, 0],
                y=seed_coords[:, 1],
                z=seed_coords[:, 2],
                mode='markers',
                name='Seed Papers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='diamond',
                    opacity=0.9,
                    line=dict(color='darkred', width=1)
                ),
                text=seed_hover,
                hovertemplate='%{text}<extra></extra>',
                visible=True,
                showlegend=True
            ))
        
        # Add subcorpus traces for each RRF threshold (only first one visible initially)
        non_seed_indices = np.where(non_seed_mask)[0]
        non_seed_coords = subcorpus_coords[non_seed_indices]
        
        # Helper function for subcorpus hover text
        def make_subcorpus_hover_text(row):
            if self.metadata and row['corpus_id'] in self.metadata:
                meta = self.metadata[row['corpus_id']]
                title = meta.get('title', 'N/A')
                if title and len(title) > 80:
                    title = title[:77] + '...'
                elif not title:
                    title = 'N/A'
                    
                year = meta.get('year', 'N/A')
                if year is None:
                    year = 'N/A'
                    
                journal = meta.get('journal', 'N/A')
                if journal and len(journal) > 40:
                    journal = journal[:37] + '...'
                elif not journal:
                    journal = 'N/A'
                
                return (f"<b>{title}</b><br>"
                       f"Year: {year}<br>"
                       f"Journal: {journal}<br>"
                       f"Corpus ID: {row['corpus_id']}<br>"
                       f"RRF Score: {row['rrf_score']:.4f}<br>"
                       f"Rank: {row['rank']}<br>"
                       f"Type: Subcorpus")
            else:
                return (f"Corpus ID: {row['corpus_id']}<br>"
                       f"RRF Score: {row['rrf_score']:.4f}<br>"
                       f"Rank: {row['rank']}<br>"
                       f"Type: Subcorpus")
        
        non_seed_df['hover_text'] = non_seed_df.apply(make_subcorpus_hover_text, axis=1)
        
        for i, threshold in enumerate(rrf_thresholds):
            # Filter by threshold
            mask_above_threshold = non_seed_df['rrf_score'] >= threshold
            df_filtered = non_seed_df[mask_above_threshold]
            indices_filtered = np.where(mask_above_threshold)[0]
            coords_filtered = non_seed_coords[indices_filtered]
            
            fig.add_trace(go.Scatter3d(
                x=coords_filtered[:, 0],
                y=coords_filtered[:, 1],
                z=coords_filtered[:, 2],
                mode='markers',
                name='Subcorpus',
                marker=dict(
                    size=3,
                    color=df_filtered['rrf_score'],
                    colorscale='Viridis',
                    cmin=rrf_min,
                    cmax=rrf_max,
                    colorbar=dict(
                        title=dict(
                            text="RRF Score",
                            font=dict(family=self.font_family)
                        ),
                        x=1.02,
                        len=0.7,
                        y=0.5,
                        tickfont=dict(family=self.font_family)
                    ),
                    showscale=(i == 0),
                    opacity=0.7
                ),
                text=df_filtered['hover_text'].tolist(),
                hovertemplate='%{text}<extra></extra>',
                visible=(i == 0),
                showlegend=(i == 0)
            ))
        
        # Create slider steps with percentage
        steps = []
        for i, threshold in enumerate(rrf_thresholds):
            n_visible = len(non_seed_df[non_seed_df['rrf_score'] >= threshold])
            pct_visible = (n_visible / total_subcorpus * 100) if total_subcorpus > 0 else 0
            
            # Visibility: random sample (always), seed papers (always), subcorpus (filtered)
            step = dict(
                method="update",
                args=[
                    {"visible": [True, True] + [j == i for j in range(slider_steps)]},
                    {"title.text": f"Subcorpus vs Random Corpus Sample ({method_name} 3D)",
                     "sliders[0].currentvalue.suffix": f" | Showing {n_visible}/{total_subcorpus} ({pct_visible:.1f}%)"}
                ],
                label=f"{threshold:.4f}"
            )
            steps.append(step)
        
        # Create slider
        sliders = [dict(
            active=0,
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0.0,
            currentvalue=dict(
                prefix="Min RRF Score: ",
                suffix=f" | Showing {total_subcorpus}/{total_subcorpus} (100.0%)",
                visible=True,
                xanchor="right",
                font=dict(size=14, family=self.font_family)
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            steps=steps,
            font=dict(family=self.font_family)
        )]
        
        # Add toggle buttons
        fig.update_layout(
            title=dict(
                text=f"Subcorpus vs Random Corpus Sample ({method_name} 3D)",
                font=dict(size=18, family=self.font_family),
                x=0.5,
                xanchor='center',
                y=0.98
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text=f'{method_name} 1', font=dict(family=self.font_family)),
                    range=x_range,
                    tickfont=dict(family=self.font_family)
                ),
                yaxis=dict(
                    title=dict(text=f'{method_name} 2', font=dict(family=self.font_family)),
                    range=y_range,
                    tickfont=dict(family=self.font_family)
                ),
                zaxis=dict(
                    title=dict(text=f'{method_name} 3', font=dict(family=self.font_family)),
                    range=z_range,
                    tickfont=dict(family=self.font_family)
                ),
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.95,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(family=self.font_family)
            ),
            height=900,
            sliders=sliders,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    xanchor="center",
                    y=-0.25,
                    yanchor="top",
                    buttons=[
                        dict(label="All",
                             method="update",
                             args=[{"visible": [True, True] + [True] * slider_steps},
                                   {"sliders[0].visible": True}]),
                        dict(label="Random Sample Only",
                             method="update",
                             args=[{"visible": [True, False] + [False] * slider_steps},
                                   {"sliders[0].visible": False}]),
                        dict(label="Subcorpus Only",
                             method="update",
                             args=[{"visible": [False, False] + [True] * slider_steps},
                                   {"sliders[0].visible": True}]),
                        dict(label="Seed Papers Only",
                             method="update",
                             args=[{"visible": [False, True] + [False] * slider_steps},
                                   {"sliders[0].visible": False}]),
                        dict(label="Random + Subcorpus",
                             method="update",
                             args=[{"visible": [True, False] + [True] * slider_steps},
                                   {"sliders[0].visible": True}]),
                    ],
                    font=dict(family=self.font_family)
                ),
            ],
            font=dict(family=self.font_family)
        )
        
        if save:
            method_suffix = 'tsne' if use_tsne else 'umap'
            output_file = self.output_dir / f'corpus_comparison_3d_{method_suffix}.html'
            self.save_plotly_html(fig, output_file)
            print(f"  Saved to: {output_file}")
            print(f"  RRF score range: {rrf_min:.6f} to {rrf_max:.6f}")
            print(f"  Slider has {slider_steps} quasi-continuous steps")
            print("  Displays percentage of visible datapoints")
        
        return fig
    
    def plot_tsne(self, save: bool = True):
        """Create t-SNE visualization."""
        print("\nCreating t-SNE visualization...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot non-seed papers with RRF score coloring
        non_seed_mask = ~self.df['is_seed']
        scatter = ax.scatter(
            self.df.loc[non_seed_mask, 'tsne_x'],
            self.df.loc[non_seed_mask, 'tsne_y'],
            c=self.df.loc[non_seed_mask, 'rrf_score'],
            cmap='viridis',
            alpha=0.6,
            s=50,
            edgecolors='none'
        )
        
        # Overlay seed papers
        seed_mask = self.df['is_seed']
        ax.scatter(
            self.df.loc[seed_mask, 'tsne_x'],
            self.df.loc[seed_mask, 'tsne_y'],
            c='red',
            alpha=0.9,
            s=200,
            marker='*',
            edgecolors='darkred',
            linewidths=1,
            label='Seed papers'
        )
        
        ax.set_xlabel('t-SNE 1', fontsize=12, fontfamily=self.font_family)
        ax.set_ylabel('t-SNE 2', fontsize=12, fontfamily=self.font_family)
        ax.set_title('t-SNE 2D Projection of Subcorpus', fontsize=14, fontweight='bold', fontfamily=self.font_family)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('RRF Score', fontsize=10, fontfamily=self.font_family)
        ax.legend(fontsize=10, prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'tsne_2d.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {output_file}")
        
        plt.close()
    
    def plot_score_distributions(self, save: bool = True):
        """Plot RRF score distributions and statistics."""
        print("\nCreating score distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Score distribution histogram
        ax = axes[0, 0]
        
        # All papers
        ax.hist(self.df['rrf_score'], bins=50, alpha=0.6, color='blue', 
                label='All papers', edgecolor='black')
        
        # Seed papers
        seed_scores = self.df.loc[self.df['is_seed'], 'rrf_score']
        if len(seed_scores) > 0:
            ax.hist(seed_scores, bins=30, alpha=0.8, color='red', 
                   label='Seed papers', edgecolor='darkred')
        
        ax.set_xlabel('RRF Score', fontsize=11, fontfamily=self.font_family)
        ax.set_ylabel('Frequency', fontsize=11, fontfamily=self.font_family)
        ax.set_title('Distribution of RRF Scores', fontsize=12, fontweight='bold', fontfamily=self.font_family)
        ax.legend(prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative distribution
        ax = axes[0, 1]
        
        sorted_scores = np.sort(self.df['rrf_score'])[::-1]
        cumsum = np.cumsum(sorted_scores)
        cumsum_pct = 100 * cumsum / cumsum[-1]
        
        ax.plot(range(1, len(sorted_scores) + 1), cumsum_pct, 
               linewidth=2, color='navy')
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50%')
        ax.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90%')
        
        ax.set_xlabel('Number of Papers (Ranked)', fontsize=11, fontfamily=self.font_family)
        ax.set_ylabel('Cumulative RRF Score (%)', fontsize=11, fontfamily=self.font_family)
        ax.set_title('Cumulative Distribution of RRF Scores', fontsize=12, fontweight='bold', fontfamily=self.font_family)
        ax.legend(prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Score vs Rank
        ax = axes[1, 0]
        
        # Non-seed papers
        non_seed_df = self.df[~self.df['is_seed']]
        ax.scatter(non_seed_df['rank'], non_seed_df['rrf_score'], 
                  alpha=0.5, s=20, color='blue', label='Subcorpus')
        
        # Seed papers
        seed_df = self.df[self.df['is_seed']]
        ax.scatter(seed_df['rank'], seed_df['rrf_score'], 
                  alpha=0.9, s=100, color='red', marker='*', 
                  edgecolors='darkred', label='Seeds')
        
        ax.set_xlabel('Rank', fontsize=11, fontfamily=self.font_family)
        ax.set_ylabel('RRF Score', fontsize=11, fontfamily=self.font_family)
        ax.set_title('RRF Score vs Rank', fontsize=12, fontweight='bold', fontfamily=self.font_family)
        ax.set_yscale('log')
        ax.legend(prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Box plot comparison
        ax = axes[1, 1]
        
        data_to_plot = [
            self.df.loc[~self.df['is_seed'], 'rrf_score'],
            self.df.loc[self.df['is_seed'], 'rrf_score']
        ]
        
        bp = ax.boxplot(data_to_plot, tick_labels=['Subcorpus', 'Seed Papers'],
                       patch_artist=True, showmeans=True)
        
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('RRF Score', fontsize=11, fontfamily=self.font_family)
        ax.set_title('Score Distribution Comparison', fontsize=12, fontweight='bold', fontfamily=self.font_family)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'score_distributions.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {output_file}")
        
        plt.close()
    
    def plot_density_heatmap(self, save: bool = True):
        """Create 2D density heatmap on UMAP projection."""
        print("\nCreating density heatmap...")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Density for all papers
        ax = axes[0]
        
        # Compute density
        xy = np.vstack([self.df['umap_x'], self.df['umap_y']])
        z = gaussian_kde(xy)(xy)
        
        scatter = ax.scatter(
            self.df['umap_x'],
            self.df['umap_y'],
            c=z,
            cmap='YlOrRd',
            s=30,
            alpha=0.6,
            edgecolors='none'
        )
        
        # Overlay seed papers
        seed_mask = self.df['is_seed']
        ax.scatter(
            self.df.loc[seed_mask, 'umap_x'],
            self.df.loc[seed_mask, 'umap_y'],
            facecolors='none',
            edgecolors='blue',
            s=150,
            linewidths=2,
            marker='o',
            label='Seed papers'
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontfamily=self.font_family)
        ax.set_ylabel('UMAP 2', fontsize=12, fontfamily=self.font_family)
        ax.set_title('Density Heatmap of Subcorpus Papers', fontsize=13, fontweight='bold', fontfamily=self.font_family)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Density', fontsize=10, fontfamily=self.font_family)
        ax.legend(fontsize=10, prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        # Hexbin plot
        ax = axes[1]
        
        hexbin = ax.hexbin(
            self.df['umap_x'],
            self.df['umap_y'],
            gridsize=30,
            cmap='YlOrRd',
            alpha=0.7,
            mincnt=1
        )
        
        # Overlay seed papers
        ax.scatter(
            self.df.loc[seed_mask, 'umap_x'],
            self.df.loc[seed_mask, 'umap_y'],
            c='blue',
            s=100,
            marker='*',
            edgecolors='darkblue',
            linewidths=1,
            label='Seed papers'
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontfamily=self.font_family)
        ax.set_ylabel('UMAP 2', fontsize=12, fontfamily=self.font_family)
        ax.set_title('Hexbin Density Plot', fontsize=13, fontweight='bold', fontfamily=self.font_family)
        
        cbar = plt.colorbar(hexbin, ax=ax)
        cbar.set_label('Count', fontsize=10, fontfamily=self.font_family)
        ax.legend(fontsize=10, prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'density_heatmap.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {output_file}")
        
        plt.close()
    
    def cluster_analysis(self, n_clusters: int = 10, save: bool = True):
        """Perform clustering analysis on embeddings."""
        print(f"\nPerforming clustering analysis ({n_clusters} clusters)...")
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(self.embeddings)
        
        # HDBSCAN clustering (density-based)
        # Note: n_jobs not set to avoid warning when using default parallelism
        hdbscan = HDBSCAN(min_cluster_size=50, min_samples=5, metric='cosine', n_jobs=None)
        self.df['hdbscan_cluster'] = hdbscan.fit_predict(self.embeddings)
        
        # Plot clusters
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # K-Means clusters
        ax = axes[0]
        
        scatter = ax.scatter(
            self.df['umap_x'],
            self.df['umap_y'],
            c=self.df['cluster'],
            cmap='tab10',
            alpha=0.6,
            s=40,
            edgecolors='none'
        )
        
        # Overlay seed papers
        seed_mask = self.df['is_seed']
        ax.scatter(
            self.df.loc[seed_mask, 'umap_x'],
            self.df.loc[seed_mask, 'umap_y'],
            facecolors='none',
            edgecolors='red',
            s=150,
            linewidths=2,
            marker='o',
            label='Seed papers'
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontfamily=self.font_family)
        ax.set_ylabel('UMAP 2', fontsize=12, fontfamily=self.font_family)
        ax.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=13, fontweight='bold', fontfamily=self.font_family)
        
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_clusters))
        cbar.set_label('Cluster', fontsize=10, fontfamily=self.font_family)
        ax.legend(fontsize=10, prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        # HDBSCAN clusters
        ax = axes[1]
        
        scatter = ax.scatter(
            self.df['umap_x'],
            self.df['umap_y'],
            c=self.df['hdbscan_cluster'],
            cmap='tab20',
            alpha=0.6,
            s=40,
            edgecolors='none'
        )
        
        # Overlay seed papers
        ax.scatter(
            self.df.loc[seed_mask, 'umap_x'],
            self.df.loc[seed_mask, 'umap_y'],
            facecolors='none',
            edgecolors='red',
            s=150,
            linewidths=2,
            marker='o',
            label='Seed papers'
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontfamily=self.font_family)
        ax.set_ylabel('UMAP 2', fontsize=12, fontfamily=self.font_family)
        ax.set_title('HDBSCAN Clustering (Density-based)', fontsize=13, fontweight='bold', fontfamily=self.font_family)
        
        n_hdbscan_clusters = len(set(self.df['hdbscan_cluster'])) - (1 if -1 in self.df['hdbscan_cluster'].values else 0)
        ax.text(0.02, 0.98, f'Clusters found: {n_hdbscan_clusters}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontfamily=self.font_family)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster (-1 = noise)', fontsize=10, fontfamily=self.font_family)
        ax.legend(fontsize=10, prop={'family': self.font_family})
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'cluster_analysis.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {output_file}")
        
        plt.close()
        
        # Print cluster statistics
        print("\nCluster Statistics:")
        print(f"  K-Means: {n_clusters} clusters")
        print(f"  HDBSCAN: {n_hdbscan_clusters} clusters found")
        
        for i in range(n_clusters):
            cluster_df = self.df[self.df['cluster'] == i]
            n_seeds = cluster_df['is_seed'].sum()
            print(f"    Cluster {i}: {len(cluster_df)} papers ({n_seeds} seeds)")
    
    def generate_summary_report(self, save: bool = True):
        """Generate comprehensive summary statistics."""
        print("\nGenerating summary report...")
        
        report = []
        report.append("=" * 70)
        report.append("SUBCORPUS ANALYSIS SUMMARY")
        report.append("=" * 70)
        report.append("")
        
        report.append("Dataset Overview:")
        report.append(f"  Total papers in subcorpus: {len(self.corpus_ids)}")
        report.append(f"  Total seed papers: {len(self.seed_ids)}")
        report.append(f"  Seed papers in subcorpus: {self.df['is_seed'].sum()}")
        report.append(f"  Non-seed papers: {(~self.df['is_seed']).sum()}")
        report.append(f"  Embedding dimension: {self.embedding_dim}")
        report.append("")
        
        report.append("RRF Score Statistics:")
        report.append(f"  Overall:")
        report.append(f"    Min score: {self.df['rrf_score'].min():.6f}")
        report.append(f"    Max score: {self.df['rrf_score'].max():.6f}")
        report.append(f"    Mean score: {self.df['rrf_score'].mean():.6f}")
        report.append(f"    Median score: {self.df['rrf_score'].median():.6f}")
        report.append(f"    Std dev: {self.df['rrf_score'].std():.6f}")
        
        if self.df['is_seed'].sum() > 0:
            seed_scores = self.df.loc[self.df['is_seed'], 'rrf_score']
            report.append(f"  Seed papers:")
            report.append(f"    Mean score: {seed_scores.mean():.6f}")
            report.append(f"    Median score: {seed_scores.median():.6f}")
        
        non_seed_scores = self.df.loc[~self.df['is_seed'], 'rrf_score']
        report.append(f"  Non-seed papers:")
        report.append(f"    Mean score: {non_seed_scores.mean():.6f}")
        report.append(f"    Median score: {non_seed_scores.median():.6f}")
        report.append("")
        
        report.append("Top 10 Papers by RRF Score:")
        for idx, row in self.df.head(10).iterrows():
            seed_marker = " [SEED]" if row['is_seed'] else ""
            report.append(f"  {row['rank']}. Corpus ID {row['corpus_id']}: {row['rrf_score']:.6f}{seed_marker}")
        report.append("")
        
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save:
            output_file = self.output_dir / 'summary_report.txt'
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Summary report saved to: {output_file}")
        
        return report_text
    
    def run_full_analysis(self, fetch_metadata: bool = True):
        """
        Run complete analysis pipeline.
        
        Args:
            fetch_metadata: Whether to fetch paper metadata from MongoDB
        """
        print("\n" + "=" * 70)
        print("SUBCORPUS VISUALIZATION AND ANALYSIS")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Fetch metadata if requested
        if fetch_metadata:
            self.fetch_metadata_from_mongo()
        
        # Compute projections
        self.compute_umap()
        self.compute_tsne(n_components=3)  # Compute both 2D and 3D t-SNE
        
        # Generate visualizations
        self.plot_umap_2d()
        self.plot_umap_3d_interactive()
        self.plot_tsne()
        self.plot_tsne_3d_interactive()
        self.plot_score_distributions()
        self.plot_density_heatmap()
        self.cluster_analysis()
        
        # Generate summary
        self.generate_summary_report()
        
        print("\n" + "=" * 70)
        print(f"✓ Analysis complete! All visualizations saved to: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and analyze subcorpus distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--subcorpus',
        type=str,
        required=True,
        help='Path to subcorpus pickle file (output from query_milvus_rrf.py)'
    )
    
    parser.add_argument(
        '--seed-embeddings',
        type=str,
        required=True,
        help='Path to seed embeddings pickle file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations (default: visualizations)'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=10,
        help='Number of clusters for K-Means (default: 10)'
    )
    
    parser.add_argument(
        '--umap-neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter (default: 15)'
    )
    
    parser.add_argument(
        '--tsne-perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter (default: 30)'
    )
    
    parser.add_argument(
        '--fetch-metadata',
        action='store_true',
        default=True,
        help='Fetch paper metadata from MongoDB (default: True)'
    )
    
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip fetching paper metadata from MongoDB'
    )
    
    parser.add_argument(
        '--mongo-host',
        type=str,
        default='localhost',
        help='MongoDB host address (default: localhost)'
    )
    
    parser.add_argument(
        '--mongo-port',
        type=int,
        default=27017,
        help='MongoDB port (default: 27017)'
    )
    
    parser.add_argument(
        '--random-sample',
        type=str,
        help='Path to random sample pickle file (from sample_random_corpus.py) for comparison visualization'
    )
    
    parser.add_argument(
        '--font-family',
        type=str,
        default='Noto Sans',
        help='Font family to use for visualizations (default: Noto Sans)'
    )
    
    args = parser.parse_args()
    
    # Determine whether to fetch metadata
    fetch_metadata = args.fetch_metadata and not args.no_metadata
    
    try:
        # Create visualizer
        visualizer = SubcorpusVisualizer(
            subcorpus_file=args.subcorpus,
            seed_embeddings_file=args.seed_embeddings,
            output_dir=args.output_dir,
            font_family=args.font_family
        )
        
        # Run analysis with custom parameters
        visualizer.load_data()
        
        # Fetch metadata if requested
        if fetch_metadata:
            visualizer.fetch_metadata_from_mongo(
                mongo_host=args.mongo_host,
                mongo_port=args.mongo_port
            )
        
        # Load random sample if provided
        if args.random_sample:
            visualizer.load_random_sample(args.random_sample)
        
        visualizer.compute_umap(n_neighbors=args.umap_neighbors)
        visualizer.compute_tsne(perplexity=args.tsne_perplexity, n_components=3)
        visualizer.plot_umap_2d()
        visualizer.plot_umap_3d_interactive()
        visualizer.plot_tsne()
        visualizer.plot_tsne_3d_interactive()
        
        # Create comparison visualizations if random sample is loaded
        if args.random_sample:
            visualizer.plot_corpus_comparison_3d(use_tsne=False)  # UMAP comparison
            visualizer.plot_corpus_comparison_3d(use_tsne=True)   # t-SNE comparison
        
        visualizer.plot_score_distributions()
        visualizer.plot_density_heatmap()
        visualizer.cluster_analysis(n_clusters=args.n_clusters)
        visualizer.generate_summary_report()
        
        print("\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    from datetime import datetime
    exit_code = main()
    print(f"\n{'='*60}")
    print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    exit(exit_code)
