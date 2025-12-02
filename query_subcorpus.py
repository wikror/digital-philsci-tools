"""
Query sentence- and paragraph-level Milvus databases created by build_subcorpus_milvus.py

This script provides a flexible interface for querying subcorpus collections with:
- Configurable database and collection names
- Query input from text file (one query per line)
- Support for both sentence and paragraph level search
- Extensible result processing and output formats

Usage:
    # Query sentence collection
    python query_subcorpus.py \\
        --db-name my_subcorpus \\
        --collection sentences \\
        --queries queries.txt \\
        --output results.json
    
    # Query paragraph collection with custom parameters
    python query_subcorpus.py \\
        --db-name my_subcorpus \\
        --collection paragraphs \\
        --queries queries.txt \\
        --limit 100 \\
        --output results.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from pymilvus import connections, MilvusClient, db
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Default configuration
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = 19530
DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"
DEFAULT_LIMIT = 1000
DEFAULT_METRIC_TYPE = "COSINE"


class SubcorpusQueryClient:
    """Client for querying subcorpus Milvus collections."""
    
    def __init__(self,
                 db_name: str,
                 model_name: str = DEFAULT_MODEL,
                 milvus_host: str = DEFAULT_MILVUS_HOST,
                 milvus_port: int = DEFAULT_MILVUS_PORT,
                 use_gpu: bool = True):
        """
        Initialize query client.
        
        Args:
            db_name: Name of the Milvus database
            model_name: Sentence transformer model name
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            use_gpu: Whether to use GPU for encoding
        """
        self.db_name = db_name
        self.model_name = model_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.use_gpu = use_gpu
        
        self.client = None
        self.encoder = None
    
    def connect(self):
        """Establish connection to Milvus."""
        print(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}...")
        
        # Establish connection
        connections.connect(host=self.milvus_host, port=self.milvus_port)
        
        # Enable the database
        db.using_database(self.db_name)
        
        # Setup client
        self.client = MilvusClient(
            uri=f'http://{self.milvus_host}:{self.milvus_port}',
            token='root:Milvus',
            db_name=self.db_name
        )
        
        print(f"Connected to database: {self.db_name}")
    
    def load_encoder(self):
        """Load sentence transformer model."""
        print(f"Loading encoder model: {self.model_name}...")
        
        device = 'cuda' if self.use_gpu else 'cpu'
        self.encoder = SentenceTransformer(self.model_name, device=device)
        
        print(f"Encoder loaded on {device}")
    
    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode queries into embeddings.
        
        Args:
            queries: List of query strings
            batch_size: Batch size for encoding
            
        Returns:
            Array of query embeddings
        """
        print(f"Encoding {len(queries)} queries...")
        
        embeddings = self.encoder.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def query_collection(self,
                        collection_name: str,
                        query_embeddings: np.ndarray,
                        limit: int = DEFAULT_LIMIT,
                        metric_type: str = DEFAULT_METRIC_TYPE,
                        output_fields: Optional[List[str]] = None,
                        release_after: bool = True) -> List[List[Dict]]:
        """
        Query a Milvus collection.
        
        Args:
            collection_name: Name of the collection to query
            query_embeddings: Array of query embeddings
            limit: Maximum number of results per query
            metric_type: Distance metric type (COSINE, L2, IP)
            output_fields: Fields to return in results
            release_after: Whether to release collection after querying (default: True)
            
        Returns:
            List of results for each query
        """
        print(f"Querying collection: {collection_name}")
        
        # Load collection if not already loaded
        load_state = self.client.get_load_state(collection_name=collection_name)
        was_loaded = load_state.get("state") == "Loaded"
        
        if not was_loaded:
            print(f"Loading collection {collection_name}...")
            self.client.load_collection(collection_name=collection_name)
        
        # Default output fields if not specified
        if output_fields is None:
            output_fields = ["corpusid", "sentence_number", "sentence_indices"]
        
        try:
            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                data=query_embeddings.tolist(),
                limit=limit,
                search_params={"metric_type": metric_type, "params": {}},
                output_fields=output_fields
            )
            
            print(f"Retrieved results for {len(results)} queries")
            
            return results
            
        finally:
            # Release collection to free up resources (unless it was already loaded)
            if release_after and not was_loaded:
                print(f"Releasing collection {collection_name} to free up resources...")
                self.client.release_collection(collection_name=collection_name)
                print("Collection released")
    
    def format_results(self,
                      queries: List[str],
                      results: List[List[Dict]],
                      collection_name: str) -> pd.DataFrame:
        """
        Format search results into a DataFrame.
        
        Args:
            queries: Original query strings
            results: Search results from Milvus
            collection_name: Name of the queried collection
            
        Returns:
            DataFrame with formatted results
        """
        print("Formatting results...")
        
        formatted_results = []
        
        for query_idx, (query, query_results) in enumerate(zip(queries, results)):
            for rank, result in enumerate(query_results):
                entity = result.get('entity', {})
                
                # Debug: Print first result to see structure
                if query_idx == 0 and rank == 0:
                    print(f"\nDEBUG: First result structure:")
                    print(f"  Full result keys: {result.keys()}")
                    print(f"  Entity keys: {entity.keys()}")
                    print(f"  Entity content: {entity}")
                    print()
                
                formatted_result = {
                    'query_idx': query_idx,
                    'query': query,
                    'rank': rank + 1,
                    'distance': result.get('distance', 0.0),
                    'corpusid': entity.get('corpusid'),
                    'collection': collection_name,
                }
                
                # Add all entity fields
                # Note: Milvus returns ARRAY fields as protobuf RepeatedScalarContainer objects
                # which don't serialize properly to JSON. Convert to plain Python types.
                for key, value in entity.items():
                    if key not in formatted_result:
                        # Check if it's a protobuf container (has __class__.__module__ starting with 'google')
                        if hasattr(value, '__class__') and 'google' in value.__class__.__module__:
                            # It's a protobuf object - convert by iterating and casting
                            formatted_result[key] = [int(x) for x in value]
                        elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, list)):
                            # Other iterable - convert to list
                            formatted_result[key] = list(value)
                        else:
                            formatted_result[key] = value
                
                # Debug: Check conversion result
                if query_idx == 0 and rank == 0 and 'sentence_indices' in formatted_result:
                    print(f"DEBUG: After list comprehension conversion:")
                    print(f"  sentence_indices value: {formatted_result['sentence_indices']}")
                    print(f"  Type: {type(formatted_result['sentence_indices'])}")
                    print()
                
                formatted_results.append(formatted_result)
        
        df = pd.DataFrame(formatted_results)
        
        print(f"Formatted {len(df)} results")
        
        return df
    
    def close(self):
        """Close connections."""
        if self.client:
            connections.disconnect(alias="default")
            print("Disconnected from Milvus")


def load_queries_from_file(filepath: str) -> List[str]:
    """
    Load queries from text file (one query per line).
    
    Args:
        filepath: Path to queries file
        
    Returns:
        List of query strings
    """
    print(f"Loading queries from: {filepath}")
    
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                queries.append(line)
    
    print(f"Loaded {len(queries)} queries")
    
    return queries


def save_results(df: pd.DataFrame, output_path: str, format: str = 'json'):
    """
    Save results to file.
    
    Args:
        df: Results DataFrame
        output_path: Output file path
        format: Output format (json, csv, or parquet)
    """
    print(f"Saving results to: {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json' or output_path.suffix == '.json':
        # Convert DataFrame to dict and use Python's json module to handle protobuf objects
        records = df.to_dict(orient='records')
        
        # Debug: Check first record before final conversion
        if len(records) > 0 and 'sentence_indices' in records[0]:
            print(f"\nDEBUG: Before final JSON conversion:")
            print(f"  First record sentence_indices: {records[0]['sentence_indices']}")
            print(f"  Type: {type(records[0]['sentence_indices'])}")
            print()
        
        # Ensure any remaining protobuf objects are converted to lists
        for record in records:
            for key, value in list(record.items()):
                # Check if it's a protobuf object
                if hasattr(value, '__class__') and 'google' in value.__class__.__module__:
                    # Convert protobuf container to plain Python list
                    record[key] = [int(x) for x in value]
                elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, list)):
                    record[key] = list(value)
        
        # Debug: Check first record after final conversion
        if len(records) > 0 and 'sentence_indices' in records[0]:
            print(f"DEBUG: After final conversion:")
            print(f"  First record sentence_indices: {records[0]['sentence_indices']}")
            print(f"  Type: {type(records[0]['sentence_indices'])}")
            print()
        
        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2)
    elif format == 'csv' or output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet' or output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved ({len(df)} rows)")


def print_summary(df: pd.DataFrame):
    """Print summary statistics of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"Total results: {len(df)}")
    print(f"Unique queries: {df['query_idx'].nunique()}")
    print(f"Unique documents: {df['corpusid'].nunique()}")
    
    if 'distance' in df.columns:
        print(f"\nDistance statistics:")
        print(f"  Mean: {df['distance'].mean():.4f}")
        print(f"  Std:  {df['distance'].std():.4f}")
        print(f"  Min:  {df['distance'].min():.4f}")
        print(f"  Max:  {df['distance'].max():.4f}")
    
    print("\nTop 5 results by distance:")
    print(df.nlargest(5, 'distance')[['query', 'distance', 'corpusid', 'rank']].to_string(index=False))
    
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Query subcorpus Milvus collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query sentence collection
  python query_subcorpus.py \\
    --db-name my_subcorpus \\
    --collection sentences \\
    --queries queries.txt \\
    --output results.json
  
  # Query paragraph collection with more results
  python query_subcorpus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --queries queries.txt \\
    --limit 500 \\
    --output results.csv
  
  # Query with custom model
  python query_subcorpus.py \\
    --db-name my_subcorpus \\
    --collection sentences \\
    --queries queries.txt \\
    --model sentence-transformers/all-MiniLM-L6-v2 \\
    --output results.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--db-name',
        type=str,
        required=True,
        help='Name of the Milvus database'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        required=True,
        help='Name of the collection to query (e.g., sentences, paragraphs)'
    )
    
    parser.add_argument(
        '--queries',
        type=str,
        required=True,
        help='Path to queries file (one query per line)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output file (json, csv, or parquet)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Sentence transformer model (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=DEFAULT_LIMIT,
        help=f'Maximum results per query (default: {DEFAULT_LIMIT})'
    )
    
    parser.add_argument(
        '--milvus-host',
        type=str,
        default=DEFAULT_MILVUS_HOST,
        help=f'Milvus server host (default: {DEFAULT_MILVUS_HOST})'
    )
    
    parser.add_argument(
        '--milvus-port',
        type=int,
        default=DEFAULT_MILVUS_PORT,
        help=f'Milvus server port (default: {DEFAULT_MILVUS_PORT})'
    )
    
    parser.add_argument(
        '--metric-type',
        type=str,
        default=DEFAULT_METRIC_TYPE,
        choices=['COSINE', 'L2', 'IP'],
        help=f'Distance metric type (default: {DEFAULT_METRIC_TYPE})'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU for encoding'
    )
    
    parser.add_argument(
        '--output-fields',
        type=str,
        nargs='+',
        default=None,
        help='Custom output fields to retrieve (default: corpusid, sentence_number, sentence_indices)'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing results summary'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUERY SUBCORPUS MILVUS COLLECTIONS")
    print("=" * 70)
    print(f"Database: {args.db_name}")
    print(f"Collection: {args.collection}")
    print(f"Queries file: {args.queries}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Results per query: {args.limit}")
    print("=" * 70 + "\n")
    
    try:
        # Load queries
        queries = load_queries_from_file(args.queries)
        
        if not queries:
            print("Error: No queries found in file")
            return 1
        
        # Initialize client
        client = SubcorpusQueryClient(
            db_name=args.db_name,
            model_name=args.model,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            use_gpu=not args.no_gpu
        )
        
        # Connect to Milvus
        client.connect()
        
        # Load encoder
        client.load_encoder()
        
        # Encode queries
        query_embeddings = client.encode_queries(queries)
        
        # Query collection
        results = client.query_collection(
            collection_name=args.collection,
            query_embeddings=query_embeddings,
            limit=args.limit,
            metric_type=args.metric_type,
            output_fields=args.output_fields
        )
        
        # Format results
        df = client.format_results(
            queries=queries,
            results=results,
            collection_name=args.collection
        )
        
        # Save results
        save_results(df, args.output)
        
        # Print summary
        if not args.no_summary:
            print_summary(df)
        
        # Clean up
        client.close()
        
        print("\n" + "=" * 70)
        print("Query completed successfully!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    from datetime import datetime
    exit_code = main()
    print(f"\n{'='*60}")
    print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    exit(exit_code)
