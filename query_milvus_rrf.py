"""
Query Milvus database of full paper embeddings using Reciprocal Rank Fusion (RRF).

This script queries a Milvus database containing paper embeddings using a seed set
of papers. It uses Reciprocal Rank Fusion to aggregate rankings from multiple queries
and returns the top N most relevant papers across all seed queries.

Reciprocal Rank Fusion (RRF) is a state-of-the-art rank aggregation method that:
1. Combines rankings from multiple queries
2. Weights higher-ranked items more heavily
3. Is robust to outliers and varying result set sizes
4. Formula: RRF(d) = Σ(1 / (k + rank_i(d))) where k is typically 60

Usage:
    python query_milvus_rrf.py --embeddings-file data/paper_embeddings.pkl --output-size 1000
"""

import argparse
import pickle
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
from pymilvus import connections, MilvusClient, db, Collection
from tqdm import tqdm

# Configuration
MILVUS_IP = "localhost"
MILVUS_PORT = 19530
DB_NAME = "s2orcFullPaperEmbeddings"
COLLECTION_NAME = "paperEmbeddings"

# RRF parameter (typically 60 in literature)
RRF_K = 60

# Query parameters
TOP_K_PER_QUERY = 100  # How many results to retrieve per seed query
BATCH_SIZE = 10  # Number of queries to process in parallel


class MilvusRRFQuerier:
    """Query Milvus database and aggregate results using Reciprocal Rank Fusion."""
    
    def __init__(self, host: str = MILVUS_IP, port: int = MILVUS_PORT, 
                 db_name: str = DB_NAME, collection_name: str = COLLECTION_NAME,
                 embeddings_file: str = None):
        """
        Initialize connection to Milvus database.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            db_name: Database name
            collection_name: Collection name
            embeddings_file: Path to pickle file containing pre-downloaded embeddings
                           (dict mapping corpus_id -> embedding vector)
        """
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.embeddings_file = embeddings_file
        self.client = None
        self.collection = None
        self._embeddings_cache = None
        
    def connect(self):
        """Connect to Milvus database and load collection into memory."""
        print(f"Connecting to Milvus at {self.host}:{self.port}...")
        
        connections.connect(host=self.host, port=self.port)
        db.using_database(self.db_name)
        
        self.client = MilvusClient(
            uri=f'http://{self.host}:{self.port}',
            token='root:Milvus',
            db_name=self.db_name
        )
        
        # Get Collection object (needed for pymilvus 2.4.4 compatibility)
        self.collection = Collection(self.collection_name)
        
        # Check if collection exists
        if not self.client.has_collection(self.collection_name):
            raise RuntimeError(f"Collection '{self.collection_name}' not found in database '{self.db_name}'")
        
        print(f"✓ Connected to collection '{self.collection_name}'")
        
        # Get collection stats using Collection object
        num_entities = self.collection.num_entities
        print(f"  Collection contains {num_entities:,} papers")
        
        # Load collection into memory
        print(f"Loading collection '{self.collection_name}' into memory...")
        self.collection.load()
        print("✓ Collection loaded into memory")
        
        # Load seed embeddings into memory
        self._load_embeddings_cache()
    
    def _load_embeddings_cache(self):
        """Load embeddings from file into memory cache."""
        if self._embeddings_cache is None and self.embeddings_file:
            print(f"\nLoading embeddings from {self.embeddings_file}...")
            with open(self.embeddings_file, 'rb') as f:
                self._embeddings_cache = pickle.load(f)
            print(f"✓ Loaded {len(self._embeddings_cache)} embeddings into memory")
    
    def get_all_corpus_ids_from_embeddings(self) -> List[int]:
        """
        Get all corpus IDs from the loaded embeddings file.
        
        Returns:
            List of corpus IDs as integers
        """
        if self._embeddings_cache is None:
            raise RuntimeError("Embeddings not loaded! Call connect() first.")
        
        corpus_ids = [int(cid) for cid in self._embeddings_cache.keys()]
        return corpus_ids
    
    def get_embeddings_by_corpus_ids(self, corpus_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Retrieve embeddings for a list of corpus IDs from pre-loaded cache.
        
        Args:
            corpus_ids: List of corpus IDs to retrieve
            
        Returns:
            Dictionary mapping corpus_id -> embedding vector
        """
        if self._embeddings_cache is None:
            raise RuntimeError("Embeddings not loaded! Call connect() first.")
        
        embeddings = {}
        missing_ids = []
        
        print(f"\nRetrieving embeddings for {len(corpus_ids)} seed papers...")
        
        for corpus_id in tqdm(corpus_ids, desc="Loading embeddings"):
            # Try both int and string keys (the file uses string keys)
            corpus_id_str = str(corpus_id)
            
            if corpus_id_str in self._embeddings_cache:
                embedding = self._embeddings_cache[corpus_id_str]
                embeddings[corpus_id] = np.array(embedding, dtype=np.float32)
            elif corpus_id in self._embeddings_cache:
                embedding = self._embeddings_cache[corpus_id]
                embeddings[corpus_id] = np.array(embedding, dtype=np.float32)
            else:
                missing_ids.append(corpus_id)
        
        if missing_ids:
            print(f"  ⚠ Warning: {len(missing_ids)} corpus IDs not found in embeddings file")
            if len(missing_ids) <= 10:
                print(f"    Missing IDs: {missing_ids}")
        
        print(f"✓ Retrieved {len(embeddings)}/{len(corpus_ids)} embeddings")
        return embeddings
    
    def search_similar_papers(self, query_embedding: np.ndarray, top_k: int = TOP_K_PER_QUERY,
                              exclude_ids: Set[int] = None) -> List[Tuple[int, float]]:
        """
        Search for similar papers using a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            exclude_ids: Set of corpus IDs to exclude from results (e.g., seed papers)
            
        Returns:
            List of (corpus_id, similarity_score) tuples, ranked by similarity
        """
        try:
            # Ensure embedding is the right shape and type
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Milvus search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding.tolist()],
                anns_field="vector",
                limit=top_k + (len(exclude_ids) if exclude_ids else 0),  # Get extra to account for exclusions
                output_fields=["corpusid"]
            )
            
            # Extract corpus IDs and scores
            ranked_results = []
            for hit in results[0]:
                corpus_id = hit.get('entity', {}).get('corpusid') or hit.get('id')
                
                # Skip excluded IDs (seed papers)
                if exclude_ids and corpus_id in exclude_ids:
                    continue
                
                distance = hit.get('distance', 0.0)
                ranked_results.append((int(corpus_id), float(distance)))
                
                if len(ranked_results) >= top_k:
                    break
            
            return ranked_results
            
        except Exception as e:
            print(f"  ✗ Error during search: {e}")
            return []
    
    def reciprocal_rank_fusion(self, rankings: List[List[Tuple[int, float]]], 
                               k: int = RRF_K) -> List[Tuple[int, float]]:
        """
        Aggregate multiple rankings using Reciprocal Rank Fusion.
        
        RRF formula: RRF_score(d) = Σ(1 / (k + rank_i(d)))
        where k is a constant (typically 60) and rank_i(d) is the rank of document d in ranking i.
        
        Args:
            rankings: List of ranked result lists, each containing (corpus_id, score) tuples
            k: RRF constant (default 60)
            
        Returns:
            Aggregated ranking as list of (corpus_id, rrf_score) tuples, sorted by score descending
        """
        rrf_scores = defaultdict(float)
        
        for ranking in rankings:
            for rank, (corpus_id, original_score) in enumerate(ranking, start=1):
                # RRF formula: 1 / (k + rank)
                rrf_scores[corpus_id] += 1.0 / (k + rank)
        
        # Sort by RRF score (descending)
        aggregated_ranking = sorted(
            [(corpus_id, score) for corpus_id, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return aggregated_ranking
    
    def query_with_seed_set(self, seed_corpus_ids: List[int], output_size: int = 1000,
                       top_k_per_query: int = TOP_K_PER_QUERY) -> Tuple[List[Tuple[int, float]], Dict[int, np.ndarray]]:
        """
        Query database using multiple seed papers and aggregate with RRF.
        
        Args:
            seed_corpus_ids: List of corpus IDs to use as seed queries
            output_size: Number of papers to return in final subcorpus
            top_k_per_query: Number of results to retrieve per seed query
            
        Returns:
            Tuple of:
            - List of (corpus_id, rrf_score) tuples for the top N papers
            - Dictionary mapping corpus_id -> embedding vector for all returned papers
        """
        # Get embeddings for seed papers
        seed_embeddings = self.get_embeddings_by_corpus_ids(seed_corpus_ids)
        
        if not seed_embeddings:
            raise RuntimeError("No embeddings found for seed papers!")
        
        seed_ids_set = set(seed_embeddings.keys())
        
        # Query database with each seed paper
        print(f"\nQuerying database with {len(seed_embeddings)} seed papers...")
        all_rankings = []
        
        for corpus_id, embedding in tqdm(seed_embeddings.items(), desc="Running queries"):
            ranking = self.search_similar_papers(
                embedding, 
                top_k=top_k_per_query,
                exclude_ids=seed_ids_set  # Exclude seed papers from results
            )
            
            if ranking:
                all_rankings.append(ranking)
        
        print(f"✓ Completed {len(all_rankings)} queries")
        
        # Aggregate rankings using RRF
        print(f"\nAggregating results with Reciprocal Rank Fusion (k={RRF_K})...")
        aggregated_ranking = self.reciprocal_rank_fusion(all_rankings, k=RRF_K)
        
        # Return top N results
        final_results = aggregated_ranking[:output_size]
        final_corpus_ids = [corpus_id for corpus_id, score in final_results]
        
        print(f"✓ Selected top {len(final_results)} papers from {len(aggregated_ranking)} unique candidates")
        
        # Retrieve embeddings for the subcorpus from Milvus
        print(f"\nRetrieving embeddings for subcorpus papers from Milvus...")
        result_embeddings = self.get_embeddings_from_milvus(final_corpus_ids)
        
        return final_results, result_embeddings

    def get_embeddings_from_milvus(self, corpus_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Retrieve embeddings from Milvus for a list of corpus IDs.
        
        Args:
            corpus_ids: List of corpus IDs to retrieve
            
        Returns:
            Dictionary mapping corpus_id -> embedding vector
        """
        embeddings = {}
        missing_ids = []
        
        # Query Milvus in batches
        batch_size = 100
        
        for i in tqdm(range(0, len(corpus_ids), batch_size), desc="Fetching embeddings from Milvus"):
            batch = corpus_ids[i:i + batch_size]
            
            # Query by corpus IDs
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f"corpusid in {batch}",
                output_fields=["corpusid", "vector"]
            )
            
            for result in results:
                corpus_id = result.get('corpusid')
                vector = result.get('vector')
                
                if corpus_id and vector:
                    embeddings[int(corpus_id)] = np.array(vector, dtype=np.float32)
        
        # Check for missing embeddings
        found_ids = set(embeddings.keys())
        missing_ids = [cid for cid in corpus_ids if cid not in found_ids]
        
        if missing_ids:
            print(f"  ⚠ Warning: {len(missing_ids)} corpus IDs not found in Milvus")
            if len(missing_ids) <= 10:
                print(f"    Missing IDs: {missing_ids}")
        
        print(f"✓ Retrieved {len(embeddings)}/{len(corpus_ids)} embeddings from Milvus")
        return embeddings
    
    def close(self):
        """Release collection from memory and close connection to Milvus."""
        if self.collection:
            print("\nReleasing collection from memory...")
            self.collection.release()
            print("✓ Collection released")
        if self.client:
            connections.disconnect(alias="default")
            print("✓ Disconnected from Milvus")


def load_seed_corpus_ids(filepath: str) -> List[int]:
    """
    Load seed corpus IDs from a file.
    
    Supports:
    - Plain text file (one corpus ID per line)
    - JSON file (list of corpus IDs or dict with 'corpus_ids' key)
    - Pickle file (list or set of corpus IDs)
    
    Args:
        filepath: Path to file containing seed corpus IDs
        
    Returns:
        List of corpus IDs as integers
    """
    print(f"Loading seed corpus IDs from {filepath}...")
    
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                corpus_ids = [int(cid) for cid in data]
            elif isinstance(data, dict) and 'corpus_ids' in data:
                corpus_ids = [int(cid) for cid in data['corpus_ids']]
            else:
                raise ValueError("JSON file must contain a list or dict with 'corpus_ids' key")
    
    elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, (list, set)):
                corpus_ids = [int(cid) for cid in data]
            else:
                raise ValueError("Pickle file must contain a list or set of corpus IDs")
    
    else:
        # Plain text file
        with open(filepath, 'r') as f:
            corpus_ids = [int(line.strip()) for line in f if line.strip()]
    
    print(f"✓ Loaded {len(corpus_ids)} seed corpus IDs")
    return corpus_ids


def save_results(corpus_ids: List[int], scores: List[float], embeddings: Dict[int, np.ndarray], 
                output_prefix: str):
    """
    Save results in multiple formats.
    
    Args:
        corpus_ids: List of corpus IDs
        scores: List of RRF scores
        embeddings: Dictionary mapping corpus_id -> embedding vector
        output_prefix: Prefix for output files
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as plain text (corpus IDs only)
    txt_file = f"{output_prefix}_{timestamp}.txt"
    with open(txt_file, 'w') as f:
        for corpus_id in corpus_ids:
            f.write(f"{corpus_id}\n")
    print(f"  Saved corpus IDs to: {txt_file}")
    
    # Save as JSON with scores
    json_file = f"{output_prefix}_{timestamp}.json"
    results = [
        {"corpus_id": int(cid), "rrf_score": float(score)}
        for cid, score in zip(corpus_ids, scores)
    ]
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved detailed results to: {json_file}")
    
    # Save as pickle with embeddings for visualization
    pickle_file = f"{output_prefix}_{timestamp}.pkl"
    pickle_data = {
        'corpus_ids': corpus_ids,
        'scores': scores,
        'embeddings': {int(cid): embedding.tolist() for cid, embedding in embeddings.items()},
        'embedding_dim': len(next(iter(embeddings.values()))) if embeddings else 0
    }
    with open(pickle_file, 'wb') as f:
        pickle.dump(pickle_data, f)
    print(f"  Saved pickle with embeddings to: {pickle_file}")
    print(f"    Embedding dimension: {pickle_data['embedding_dim']}")
    print(f"    Number of embeddings: {len(embeddings)}")


def main():
    parser = argparse.ArgumentParser(
        description="Query Milvus database using Reciprocal Rank Fusion for seed set expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use all papers from embeddings file as seeds
    python query_milvus_rrf.py --embeddings-file data/paper_embeddings.pkl --output-size 1000
    
    # Use specific seed file (optional)
    python query_milvus_rrf.py --seed-file seed_corpus_ids.txt --embeddings-file data/paper_embeddings.pkl --output-size 1000
    
    # Use more results per query for better coverage
    python query_milvus_rrf.py --embeddings-file data/paper_embeddings.pkl --output-size 5000 --top-k 200
    
    # Adjust RRF parameter
    python query_milvus_rrf.py --embeddings-file data/paper_embeddings.pkl --rrf-k 100
        """
    )
    
    parser.add_argument(
        '--seed-file',
        type=str,
        required=False,
        help='Path to file containing seed corpus IDs (txt, json, or pickle). If not provided, all corpus IDs from embeddings file will be used as seeds.'
    )
    
    parser.add_argument(
        '--output-size',
        type=int,
        default=1000,
        help='Number of papers to include in output subcorpus (default: 1000)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='subcorpus',
        help='Prefix for output files (default: subcorpus)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=TOP_K_PER_QUERY,
        help=f'Number of results to retrieve per seed query (default: {TOP_K_PER_QUERY})'
    )
    
    parser.add_argument(
        '--rrf-k',
        type=int,
        default=RRF_K,
        help=f'RRF constant parameter (default: {RRF_K})'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=MILVUS_IP,
        help=f'Milvus server host (default: {MILVUS_IP})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=MILVUS_PORT,
        help=f'Milvus server port (default: {MILVUS_PORT})'
    )
    
    parser.add_argument(
        '--embeddings-file',
        type=str,
        required=True,
        help='Path to pickle file containing pre-downloaded embeddings (corpus_id -> vector mapping)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Milvus Query with Reciprocal Rank Fusion")
    print("=" * 70)
    
    try:
        # Initialize querier
        querier = MilvusRRFQuerier(
            host=args.host,
            port=args.port,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            embeddings_file=args.embeddings_file
        )
        
        # Connect to database (this also loads embeddings into memory)
        querier.connect()
        
        # Load seed corpus IDs
        if args.seed_file:
            seed_corpus_ids = load_seed_corpus_ids(args.seed_file)
        else:
            print("\nNo seed file provided - using all corpus IDs from embeddings file as seeds")
            seed_corpus_ids = querier.get_all_corpus_ids_from_embeddings()
            print(f"✓ Using {len(seed_corpus_ids)} corpus IDs from embeddings file")
        
        # Query with seed set (now returns embeddings too)
        results, result_embeddings = querier.query_with_seed_set(
            seed_corpus_ids=seed_corpus_ids,
            output_size=args.output_size,
            top_k_per_query=args.top_k
        )
        
        # Extract corpus IDs and scores
        corpus_ids = [corpus_id for corpus_id, score in results]
        scores = [score for corpus_id, score in results]
        
        # Save results with embeddings
        print(f"\nSaving results...")
        save_results(corpus_ids, scores, result_embeddings, args.output_prefix)
        
        # Print statistics
        print(f"\n{'=' * 70}")
        print("Summary Statistics:")
        print(f"  Seed papers: {len(seed_corpus_ids)}")
        print(f"  Output subcorpus size: {len(corpus_ids)}")
        print(f"  Papers with embeddings: {len(result_embeddings)}")
        print(f"  Top-K per query: {args.top_k}")
        print(f"  RRF parameter k: {args.rrf_k}")
        print(f"  RRF score range: {min(scores):.6f} to {max(scores):.6f}")
        print(f"  Mean RRF score: {np.mean(scores):.6f}")
        print(f"  Median RRF score: {np.median(scores):.6f}")
        print(f"{'=' * 70}")
        
        # Close connection
        querier.close()
        
        print("\n✓ Query completed successfully!")
        
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
