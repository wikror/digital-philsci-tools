"""
Retrieve full text for query results from query_subcorpus.py output.

This script takes the results from query_subcorpus.py (which contains corpus IDs 
and character indices) and retrieves the actual text from the S2ORC corpus using:
1. MongoDB to get file paths and byte offsets for indexed gzip files
2. Indexed gzip files to efficiently extract the full paper text
3. Character indices to extract the specific sentence/paragraph

The script is designed to work efficiently with large result sets by:
- Batching MongoDB queries
- Caching opened gzip files
- Supporting multiple output formats

Usage:
    # Retrieve text for sentence-level results
    python retrieve_query_texts.py \\
        --input results.json \\
        --output results_with_text.json \\
        --s2orc-path /path/to/s2orc/
    
    # Retrieve with character-based context (200 chars before/after)
    python retrieve_query_texts.py \\
        --input results.json \\
        --output results_with_text.csv \\
        --s2orc-path /path/to/s2orc/ \\
        --include-context \\
        --context-mode chars \\
        --context-chars 200
    
    # Retrieve with full paragraph context (auto-detects from annotations)
    python retrieve_query_texts.py \\
        --input results.json \\
        --output results_with_text.json \\
        --s2orc-path /path/to/s2orc/ \\
        --include-context \\
        --context-mode paragraph
"""

import argparse
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

# Try to import indexed_gzip for faster random access
try:
    import indexed_gzip as igzip
    INDEXED_GZIP_AVAILABLE = True
except ImportError:
    INDEXED_GZIP_AVAILABLE = False
    print("Warning: indexed_gzip not available. Install with 'pip install indexed-gzip' for faster performance.")

# MongoDB configuration
DEFAULT_MONGO_HOST = "localhost"
DEFAULT_MONGO_PORT = 27017
DEFAULT_MONGO_DB = "papers_db"
DEFAULT_MONGO_COLLECTION = "papers"

# S2ORC configuration
DEFAULT_MAX_DIRSIZE = 50000


class TextRetriever:
    """Retrieve text from S2ORC corpus using MongoDB indices and indexed gzip."""
    
    def __init__(self,
                 s2orc_path: str,
                 mongo_host: str = DEFAULT_MONGO_HOST,
                 mongo_port: int = DEFAULT_MONGO_PORT,
                 mongo_db: str = DEFAULT_MONGO_DB,
                 mongo_collection: str = DEFAULT_MONGO_COLLECTION,
                 max_dirsize: int = DEFAULT_MAX_DIRSIZE):
        """
        Initialize text retriever.
        
        Args:
            s2orc_path: Path to S2ORC corpus root directory
            mongo_host: MongoDB server host
            mongo_port: MongoDB server port
            mongo_db: MongoDB database name
            mongo_collection: MongoDB collection name
            max_dirsize: Maximum directory size for S2ORC organization
        """
        self.s2orc_path = Path(s2orc_path)
        self.max_dirsize = max_dirsize
        
        # Connect to MongoDB
        self.mongo_client = MongoClient(host=mongo_host, port=mongo_port)
        self.mongo_db = self.mongo_client[mongo_db]
        self.mongo_collection = self.mongo_db[mongo_collection]
        
        # Cache for file handles and paper data
        self.file_cache = {}
        self.paper_cache = {}
        
        print(f"Connected to MongoDB: {mongo_host}:{mongo_port}/{mongo_db}/{mongo_collection}")
        print(f"S2ORC path: {self.s2orc_path}")
    
    def get_paper_metadata(self, corpusids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get file paths and byte offsets for corpus IDs from MongoDB.
        
        Args:
            corpusids: List of S2ORC corpus IDs
            
        Returns:
            Dictionary mapping corpusid -> {filename, byte_offset}
        """
        # Query MongoDB for all corpus IDs at once
        query = {"corpusid": {"$in": corpusids}}
        fields = {"corpusid": 1, "file_location": 1, "_id": 0}
        
        results = self.mongo_collection.find(query, fields)
        
        metadata = {}
        for doc in results:
            corpusid = int(doc["corpusid"])
            
            # file_location is a list: [filename, byte_offset]
            file_location = doc.get("file_location", [])
            if len(file_location) >= 2:
                metadata[corpusid] = {
                    "filename": file_location[0],
                    "byte_offset": int(file_location[1])
                }
            else:
                print(f"Warning: Invalid file_location for corpus ID {corpusid}")
        
        return metadata
    
    def load_paper(self, corpusid: int, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """
        Load a paper from indexed gzip file.
        
        Args:
            corpusid: S2ORC corpus ID
            metadata: Pre-fetched metadata (optional, will query if not provided)
            
        Returns:
            Paper dictionary or None if not found
        """
        # Check cache first
        if corpusid in self.paper_cache:
            return self.paper_cache[corpusid]
        
        # Get metadata
        if metadata is None:
            metadata_dict = self.get_paper_metadata([corpusid])
            if corpusid not in metadata_dict:
                print(f"Warning: Corpus ID {corpusid} not found in MongoDB")
                return None
            metadata = metadata_dict[corpusid]
        
        # Construct file path
        filename = metadata["filename"]
        byte_offset = metadata["byte_offset"]
        
        # Files are directly in s2orc_path, not organized by fos
        file_path = self.s2orc_path / filename
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            return None
        
        # Open gzipped file and seek to offset
        try:
            # Use indexed_gzip if available for much faster seeks
            if INDEXED_GZIP_AVAILABLE:
                index_file = str(file_path) + '.gzidx'
                
                # Try to load existing index, or create new one that will be saved
                if Path(index_file).exists():
                    # Load pre-built index for faster startup
                    f = igzip.IndexedGzipFile(str(file_path), mode='rb', index_file=index_file)
                else:
                    # Create new index with optimal spacing (will build on-the-fly)
                    f = igzip.IndexedGzipFile(str(file_path), mode='rb', spacing=2**20)
                
                try:
                    f.seek(byte_offset)
                    line = f.readline().decode('utf-8')
                    paper = json.loads(line)
                    
                    # Export index for future use if it doesn't exist yet
                    if not Path(index_file).exists():
                        try:
                            f.export_index(index_file)
                        except Exception as e:
                            # Non-critical error, just log it
                            pass
                finally:
                    f.close()
            else:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    f.seek(byte_offset)
                    line = f.readline()
                    paper = json.loads(line)
            
            # Cache the paper
            self.paper_cache[corpusid] = paper
            
            return paper
        except Exception as e:
            print(f"Error loading paper {corpusid}: {e}")
            return None
    
    def extract_text(self, 
                    paper: Dict, 
                    start_idx: int, 
                    end_idx: int,
                    include_context: bool = False,
                    context_chars: int = 200,
                    context_mode: str = 'chars',
                    debug: bool = False) -> Dict[str, str]:
        """
        Extract text from paper using character indices.
        
        Args:
            paper: Paper dictionary
            start_idx: Start character index
            end_idx: End character index
            include_context: Whether to include surrounding context
            context_chars: Number of characters to include before/after (only for context_mode='chars')
            context_mode: Context extraction mode - 'chars' for character count, 'paragraph' for full paragraph
            debug: Enable debug output
            
        Returns:
            Dictionary with extracted text and optional context
        """
        try:
            if debug:
                print(f"\n  DEBUG extract_text:")
                print(f"    Paper keys: {list(paper.keys())}")
                print(f"    start_idx: {start_idx}, end_idx: {end_idx}")
            
            # Try to find the text field - S2ORC can have different structures
            full_text = None
            
            # Try common S2ORC structures
            if "content" in paper and "text" in paper["content"]:
                full_text = paper["content"]["text"]
                if debug:
                    print(f"    Found text at: paper['content']['text']")
            elif "full_text" in paper:
                full_text = paper["full_text"]
                if debug:
                    print(f"    Found text at: paper['full_text']")
            elif "text" in paper:
                full_text = paper["text"]
                if debug:
                    print(f"    Found text at: paper['text']")
            elif "content" in paper:
                # Maybe content itself is the text?
                if isinstance(paper["content"], str):
                    full_text = paper["content"]
                    if debug:
                        print(f"    Found text at: paper['content'] (direct string)")
                else:
                    if debug:
                        print(f"    ERROR: 'content' exists but structure unknown")
                        print(f"    Content type: {type(paper['content'])}")
                        if isinstance(paper['content'], dict):
                            print(f"    Content keys: {list(paper['content'].keys())}")
            else:
                if debug:
                    print(f"    ERROR: Could not find text field in paper")
                    print(f"    Available keys: {list(paper.keys())}")
                raise KeyError(f"Could not find text in paper. Available keys: {list(paper.keys())}")
            
            if full_text is None:
                raise ValueError("Text field is None after structure search")
            
            if debug:
                print(f"    Full text length: {len(full_text)}")
                print(f"    Extraction range: [{start_idx}:{end_idx}] (length: {end_idx - start_idx})")
            
            # Extract the target text
            target_text = full_text[start_idx:end_idx].strip()
            
            if debug:
                print(f"    Extracted text length: {len(target_text)}")
                print(f"    First 100 chars: {target_text[:100]!r}")
            
            result = {"text": target_text}
            
            if include_context:
                if context_mode == 'paragraph':
                    # Try to find paragraph annotations
                    try:
                        # Look for paragraph annotations in paper
                        paragraphs_annotation = None
                        if "content" in paper and "annotations" in paper["content"]:
                            para_data = paper["content"]["annotations"].get("paragraph")
                            if para_data:
                                # Handle both string and list formats
                                if isinstance(para_data, str):
                                    paragraphs_annotation = json.loads(para_data)
                                else:
                                    paragraphs_annotation = para_data
                        
                        if paragraphs_annotation and len(paragraphs_annotation) > 0:
                            # Find the paragraph containing the sentence
                            containing_para = None
                            for para in paragraphs_annotation:
                                para_start = int(para["start"])
                                para_end = int(para["end"])
                                
                                # Check if sentence is within this paragraph
                                if para_start <= start_idx and end_idx <= para_end:
                                    containing_para = para
                                    break
                            
                            if containing_para:
                                para_start = int(containing_para["start"])
                                para_end = int(containing_para["end"])
                                
                                # Extract context before (from paragraph start to sentence start)
                                context_before = full_text[para_start:start_idx].strip()
                                # Extract context after (from sentence end to paragraph end)
                                context_after = full_text[end_idx:para_end].strip()
                                
                                result["context_before"] = context_before
                                result["context_after"] = context_after
                                
                                if debug:
                                    print(f"    Using paragraph context: [{para_start}:{para_end}]")
                            else:
                                # Sentence not found in any paragraph, fall back to character-based
                                if debug:
                                    print(f"    WARNING: Sentence not found in paragraph annotations, falling back to character-based context")
                                context_start = max(0, start_idx - context_chars)
                                context_before = full_text[context_start:start_idx].strip()
                                context_end = min(len(full_text), end_idx + context_chars)
                                context_after = full_text[end_idx:context_end].strip()
                                result["context_before"] = context_before
                                result["context_after"] = context_after
                        else:
                            # No paragraph annotations, fall back to character-based
                            if debug:
                                print(f"    No paragraph annotations found, falling back to character-based context")
                            context_start = max(0, start_idx - context_chars)
                            context_before = full_text[context_start:start_idx].strip()
                            context_end = min(len(full_text), end_idx + context_chars)
                            context_after = full_text[end_idx:context_end].strip()
                            result["context_before"] = context_before
                            result["context_after"] = context_after
                    
                    except Exception as e:
                        # On any error, fall back to character-based context
                        if debug:
                            print(f"    ERROR processing paragraph context: {e}, falling back to character-based")
                        context_start = max(0, start_idx - context_chars)
                        context_before = full_text[context_start:start_idx].strip()
                        context_end = min(len(full_text), end_idx + context_chars)
                        context_after = full_text[end_idx:context_end].strip()
                        result["context_before"] = context_before
                        result["context_after"] = context_after
                
                else:  # context_mode == 'chars'
                    # Extract context before
                    context_start = max(0, start_idx - context_chars)
                    context_before = full_text[context_start:start_idx].strip()
                    
                    # Extract context after
                    context_end = min(len(full_text), end_idx + context_chars)
                    context_after = full_text[end_idx:context_end].strip()
                    
                    result["context_before"] = context_before
                    result["context_after"] = context_after
            
            return result
        except Exception as e:
            print(f"ERROR extracting text: {e}")
            import traceback
            traceback.print_exc()
            return {"text": "", "error": str(e)}
    
    def process_results(self, 
                       results: pd.DataFrame,
                       include_context: bool = False,
                       context_chars: int = 200,
                       context_mode: str = 'chars',
                       batch_size: int = 1000,
                       checkpoint_file: Optional[str] = None,
                       save_interval: int = 100) -> pd.DataFrame:
        """
        Process query results and add text.
        
        Args:
            results: DataFrame from query_subcorpus.py
            include_context: Whether to include surrounding context
            context_chars: Number of characters for context (only for context_mode='chars')
            context_mode: Context extraction mode - 'chars' for character count, 'paragraph' for full paragraph
            batch_size: Batch size for MongoDB queries
            checkpoint_file: Path to checkpoint file for resuming (optional)
            save_interval: Save progress every N rows (default: 100)
            
        Returns:
            DataFrame with added text columns
        """
        print(f"\nProcessing {len(results)} results...")
        
        # Check for existing checkpoint
        start_idx = 0
        if checkpoint_file and Path(checkpoint_file).exists():
            try:
                checkpoint_data = json.loads(Path(checkpoint_file).read_text())
                checkpoint_idx = checkpoint_data.get('last_processed_idx', 0) + 1
                checkpoint_timestamp = checkpoint_data.get('timestamp', 'unknown time')
                total_rows = checkpoint_data.get('total_rows', 0)
                
                print(f"\n{'='*60}")
                print(f"CHECKPOINT FOUND")
                print(f"{'='*60}")
                print(f"Checkpoint created: {checkpoint_timestamp}")
                print(f"Progress: {checkpoint_idx} / {total_rows} rows")
                print(f"Remaining: {total_rows - checkpoint_idx} rows")
                print(f"{'='*60}")
                
                response = input("Resume from checkpoint? (y/n): ").strip().lower()
                
                if response in ['y', 'yes']:
                    start_idx = checkpoint_idx
                    print(f"Resuming from row {start_idx}")
                else:
                    print("Starting from beginning (checkpoint will be overwritten)")
                    # Delete old checkpoint
                    Path(checkpoint_file).unlink()
                    partial_file = Path(checkpoint_file).with_suffix('.partial.json')
                    if partial_file.exists():
                        partial_file.unlink()
                    start_idx = 0
                
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}. Starting from beginning.")
                start_idx = 0
        
        # Sort results by corpusid to group papers together for efficiency
        print("Sorting results by corpusid for efficient paper loading...")
        results = results.sort_values('corpusid').reset_index(drop=True)
        
        # Get unique corpus IDs
        unique_corpusids = results['corpusid'].unique().tolist()
        
        print(f"Fetching metadata for {len(unique_corpusids)} unique papers...")
        
        # Fetch all metadata in batches
        all_metadata = {}
        for i in range(0, len(unique_corpusids), batch_size):
            batch_ids = unique_corpusids[i:i + batch_size]
            metadata = self.get_paper_metadata(batch_ids)
            all_metadata.update(metadata)
        
        print(f"Retrieved metadata for {len(all_metadata)} papers")
        
        # Add filename to results for grouping
        results['_filename'] = results['corpusid'].map(lambda cid: all_metadata.get(cid, {}).get('filename', ''))
        
        # Group by filename to minimize file reads (multiple papers can be in same file)
        print("Grouping results by filename for optimal file access...")
        results_sorted = results.sort_values(['_filename', 'corpusid']).reset_index(drop=True)
        
        # Initialize result lists
        texts = [''] * len(results_sorted)
        context_before = ([''] * len(results_sorted)) if include_context else None
        context_after = ([''] * len(results_sorted)) if include_context else None
        
        # Group by filename for efficient processing
        grouped_by_file = results_sorted.groupby('_filename', sort=False)
        files_processed = 0
        papers_processed = 0
        papers_in_file = 0
        
        for filename, file_group in tqdm(grouped_by_file, desc="Processing files", total=len(grouped_by_file)):
            if not filename:  # Skip if no filename (missing metadata)
                continue
            
            # Skip if before checkpoint
            if file_group.index[0] < start_idx:
                continue
            
            # Now group by corpusid within this file
            grouped_by_corpusid = file_group.groupby('corpusid', sort=False)
            
            for corpusid, corpus_group in grouped_by_corpusid:
                corpusid = int(corpusid)
                
                # Load paper once for all rows with this corpusid
                if corpusid not in all_metadata:
                    continue
                
                paper = self.load_paper(corpusid, all_metadata[corpusid])
                papers_in_file += 1
                papers_processed += 1
                
                if paper is None:
                    print(f"  WARNING: Failed to load paper {corpusid}")
                    continue
                
                # Debug first paper in first file
                if files_processed == 0 and papers_in_file == 1:
                    print(f"\n  DEBUG: First paper loaded successfully")
                    print(f"    Corpus ID: {corpusid}")
                    print(f"    Paper keys: {list(paper.keys())}")
                    if 'content' in paper:
                        print(f"    Content keys: {list(paper['content'].keys())}")
                        if 'text' in paper['content']:
                            print(f"    Text length: {len(paper['content']['text'])}")
                
                # Process all rows for this paper
                row_count = 0
                for idx, row in corpus_group.iterrows():
                    row_count += 1
                    
                    # Handle both sentence_indices and paragraph_indices
                    if 'sentence_indices' in row:
                        indices = row['sentence_indices']
                        indices_type = 'sentence_indices'
                    elif 'paragraph_indices' in row:
                        indices = row['paragraph_indices']
                        indices_type = 'paragraph_indices'
                    else:
                        print(f"  WARNING: Row {idx} has no sentence_indices or paragraph_indices")
                        continue
                    
                    # Debug first row of first paper
                    if files_processed == 0 and papers_in_file == 1 and row_count == 1:
                        print(f"\n  DEBUG: First row of first paper")
                        print(f"    Row index: {idx}")
                        print(f"    Indices type: {indices_type}")
                        print(f"    Indices value: {indices}")
                        print(f"    Indices type (Python): {type(indices)}")
                    
                    # Parse indices
                    try:
                        if isinstance(indices, str):
                            indices = json.loads(indices)
                        
                        if indices is None or (hasattr(indices, '__len__') and len(indices) < 2):
                            if files_processed == 0 and papers_in_file == 1 and row_count == 1:
                                print(f"    WARNING: Indices invalid (None or too short)")
                            continue
                        
                        if isinstance(indices, (list, tuple)):
                            start_idx_text, end_idx_text = int(indices[0]), int(indices[1])
                        elif isinstance(indices, dict):
                            if 0 in indices and 1 in indices:
                                start_idx_text, end_idx_text = int(indices[0]), int(indices[1])
                            elif 'start' in indices and 'end' in indices:
                                start_idx_text, end_idx_text = int(indices['start']), int(indices['end'])
                            else:
                                if files_processed == 0 and papers_in_file == 1 and row_count == 1:
                                    print(f"    WARNING: Dict indices missing required keys: {list(indices.keys())}")
                                continue
                        else:
                            if files_processed == 0 and papers_in_file == 1 and row_count == 1:
                                print(f"    WARNING: Unexpected indices type: {type(indices)}")
                            continue
                            
                    except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError) as e:
                        if files_processed == 0 and papers_in_file == 1 and row_count == 1:
                            print(f"    ERROR parsing indices: {e}")
                        continue
                    
                    # Extract text (with debug for first row)
                    debug_extraction = (files_processed == 0 and papers_in_file == 1 and row_count == 1)
                    
                    extracted = self.extract_text(
                        paper, start_idx_text, end_idx_text,
                        include_context=include_context,
                        context_chars=context_chars,
                        context_mode=context_mode,
                        debug=debug_extraction
                    )
                    
                    texts[idx] = extracted.get("text", "")
                    if include_context:
                        context_before[idx] = extracted.get("context_before", "")
                        context_after[idx] = extracted.get("context_after", "")
                    
                    if debug_extraction:
                        print(f"    Result stored at index {idx}")
                        print(f"    Text length: {len(texts[idx])}")
                    
                    # Save checkpoint periodically (every N papers)
                    if checkpoint_file and papers_processed % save_interval == 0:
                        self._save_checkpoint(checkpoint_file, idx, results_sorted, texts,
                                             context_before, context_after, include_context)
            
            # Update file counter
            files_processed += 1
            papers_in_file = 0
        
        # Remove temporary filename column
        results_sorted = results_sorted.drop(columns=['_filename'])
        
        # Add text columns to results
        results_sorted['text'] = texts
        
        if include_context:
            results_sorted['context_before'] = context_before
            results_sorted['context_after'] = context_after
        
        print(f"\nExtracted text for {sum(1 for t in texts if t)} / {len(texts)} results")
        print(f"Total files accessed: {files_processed}")
        
        # Clean up checkpoint file if successful
        if checkpoint_file and Path(checkpoint_file).exists():
            Path(checkpoint_file).unlink()
            print(f"Removed checkpoint file: {checkpoint_file}")
        
        return results_sorted
    
    def _save_checkpoint(self, checkpoint_file: str, last_idx: int, 
                        results: pd.DataFrame, texts: List[str],
                        context_before: Optional[List[str]],
                        context_after: Optional[List[str]],
                        include_context: bool):
        """Save progress checkpoint."""
        try:
            # Save checkpoint metadata
            checkpoint_data = {
                'last_processed_idx': int(last_idx),
                'timestamp': datetime.now().isoformat(),
                'total_rows': len(results)
            }
            Path(checkpoint_file).write_text(json.dumps(checkpoint_data, indent=2))
            
            # Save partial results - only rows that have been processed (up to last_idx)
            partial_results = results.iloc[:last_idx + 1].copy()
            partial_results['text'] = texts[:last_idx + 1]
            if include_context:
                partial_results['context_before'] = context_before[:last_idx + 1]
                partial_results['context_after'] = context_after[:last_idx + 1]
            
            partial_output = Path(checkpoint_file).with_suffix('.partial.json')
            partial_results.to_json(partial_output, orient='records', indent=2)
            
            # Checkpoint saved silently
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def close(self):
        """Close connections and clean up."""
        self.mongo_client.close()
        self.file_cache.clear()
        self.paper_cache.clear()
        print("Closed MongoDB connection and cleared caches")


def load_results(filepath: str) -> pd.DataFrame:
    """
    Load query results from file.
    
    Args:
        filepath: Path to results file (JSON, CSV, or Parquet)
        
    Returns:
        DataFrame with query results
    """
    print(f"Loading results from: {filepath}")
    
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        df = pd.read_json(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    print(f"Loaded {len(df)} results")
    
    return df


def save_results(df: pd.DataFrame, output_path: str):
    """
    Save results with text to file.
    
    Args:
        df: Results DataFrame with text
        output_path: Output file path
    """
    print(f"\nSaving results to: {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.json':
        df.to_json(output_path, orient='records', indent=2)
    elif output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    print(f"Results saved ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve full text for query results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic text retrieval
  python retrieve_query_texts.py \\
    --input results.json \\
    --output results_with_text.json \\
    --s2orc-path /path/to/s2orc/
  
  # Include character-based context around matches (200 chars before/after)
  python retrieve_query_texts.py \\
    --input results.json \\
    --output results_with_text.csv \\
    --s2orc-path /path/to/s2orc/ \\
    --include-context \\
    --context-mode chars \\
    --context-chars 200
  
  # Include full paragraph context (uses paragraph annotations if available)
  python retrieve_query_texts.py \\
    --input results.json \\
    --output results_with_text.json \\
    --s2orc-path /path/to/s2orc/ \\
    --include-context \\
    --context-mode paragraph
  
  # Custom MongoDB settings
  python retrieve_query_texts.py \\
    --input results.json \\
    --output results_with_text.json \\
    --s2orc-path /path/to/s2orc/ \\
    --mongo-host remote-server \\
    --mongo-port 27018
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to query results file from query_subcorpus.py'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output file with text'
    )
    
    parser.add_argument(
        '--s2orc-path',
        type=str,
        required=True,
        help='Path to S2ORC corpus root directory'
    )
    
    # Optional arguments
    parser.add_argument(
        '--mongo-host',
        type=str,
        default=DEFAULT_MONGO_HOST,
        help=f'MongoDB server host (default: {DEFAULT_MONGO_HOST})'
    )
    
    parser.add_argument(
        '--mongo-port',
        type=int,
        default=DEFAULT_MONGO_PORT,
        help=f'MongoDB server port (default: {DEFAULT_MONGO_PORT})'
    )
    
    parser.add_argument(
        '--mongo-db',
        type=str,
        default=DEFAULT_MONGO_DB,
        help=f'MongoDB database name (default: {DEFAULT_MONGO_DB})'
    )
    
    parser.add_argument(
        '--mongo-collection',
        type=str,
        default=DEFAULT_MONGO_COLLECTION,
        help=f'MongoDB collection name (default: {DEFAULT_MONGO_COLLECTION})'
    )
    
    parser.add_argument(
        '--include-context',
        action='store_true',
        help='Include surrounding context for each match'
    )
    
    parser.add_argument(
        '--context-mode',
        type=str,
        choices=['chars', 'paragraph'],
        default='chars',
        help="Context extraction mode: 'chars' for character count (default), 'paragraph' for full paragraph (falls back to chars if no annotation)"
    )
    
    parser.add_argument(
        '--context-chars',
        type=int,
        default=200,
        help='Number of characters to include before/after match when using --context-mode=chars (default: 200)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for MongoDB queries (default: 1000)'
    )
    
    parser.add_argument(
        '--checkpoint-file',
        type=str,
        default=None,
        help='Path to checkpoint file for resuming interrupted processing'
    )
    
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='Save checkpoint every N papers (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Create default checkpoint file if not provided
    if args.checkpoint_file is None:
        output_path = Path(args.output)
        checkpoint_file = output_path.parent / f".{output_path.stem}.checkpoint"
        args.checkpoint_file = str(checkpoint_file)
    
    print("=" * 70)
    print("RETRIEVE QUERY RESULT TEXTS")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"S2ORC path: {args.s2orc_path}")
    print(f"Checkpoint file: {args.checkpoint_file}")
    print(f"Save interval: every {args.save_interval} papers")
    print(f"Include context: {args.include_context}")
    if args.include_context:
        print(f"Context mode: {args.context_mode}")
        if args.context_mode == 'chars':
            print(f"Context characters: {args.context_chars}")
        else:
            print(f"Context fallback characters: {args.context_chars}")
    print("=" * 70 + "\n")
    
    try:
        # Load query results
        results = load_results(args.input)
        
        # Initialize retriever
        retriever = TextRetriever(
            s2orc_path=args.s2orc_path,
            mongo_host=args.mongo_host,
            mongo_port=args.mongo_port,
            mongo_db=args.mongo_db,
            mongo_collection=args.mongo_collection
        )
        
        # Process results
        results_with_text = retriever.process_results(
            results,
            include_context=args.include_context,
            context_chars=args.context_chars,
            context_mode=args.context_mode,
            batch_size=args.batch_size,
            checkpoint_file=args.checkpoint_file,
            save_interval=args.save_interval
        )
        
        # Save results
        save_results(results_with_text, args.output)
        
        # Clean up
        retriever.close()
        
        print("\n" + "=" * 70)
        print("Text retrieval completed successfully!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    print(f"\n{'='*60}")
    print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    exit(exit_code)
