"""
Sample random papers from Milvus corpus for comparison with subcorpus.

This script draws a random sample of papers from the full corpus in Milvus
and saves their corpus IDs and embeddings to a pickle file for later use
in visualizations.

Usage:
    python sample_random_corpus.py --sample-size 5000 --output random_sample.pkl
"""

import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from pymilvus import connections, Collection, MilvusClient
from pymilvus import db as milvus_db


# Default configuration
MILVUS_IP = "localhost"
MILVUS_PORT = 19530
DEFAULT_DB = "s2orcFullPaperEmbeddings"
DEFAULT_COLLECTION = "paperEmbeddings"


class RandomCorpusSampler:
    """Sample random papers from Milvus corpus."""
    
    def __init__(self, 
                 db_name: str = DEFAULT_DB,
                 collection_name: str = DEFAULT_COLLECTION,
                 host: str = MILVUS_IP,
                 port: int = MILVUS_PORT):
        """
        Initialize the sampler.
        
        Args:
            db_name: Name of the Milvus database
            collection_name: Name of the collection
            host: Milvus server host
            port: Milvus server port
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None
        self.client = None
        
    def connect(self):
        """Connect to Milvus and load collection."""
        print(f"\nConnecting to Milvus at {self.host}:{self.port}...")
        
        # Connect to Milvus
        connections.connect(host=self.host, port=self.port)
        
        # Use the database
        milvus_db.using_database(self.db_name)
        
        # Setup client
        self.client = MilvusClient(
            uri=f'http://{self.host}:{self.port}',
            token='root:Milvus',
            db_name=self.db_name
        )
        
        # Create Collection object
        self.collection = Collection(self.collection_name)
        
        print(f"✓ Connected to database '{self.db_name}'")
        print(f"  Collection: {self.collection_name}")
        
        # Load collection into memory
        print("  Loading collection into memory...")
        self.collection.load()
        print("✓ Collection loaded")
        
        # Get collection stats
        num_entities = self.collection.num_entities
        print(f"  Total entities in collection: {num_entities:,}")
        
        return num_entities
    
    def get_random_sample_ids(self, sample_size: int, total_entities: int) -> List[int]:
        """
        Generate random sample of entity IDs using iterator-based approach.
        
        Args:
            sample_size: Number of papers to sample
            total_entities: Total number of entities in collection
            
        Returns:
            List of random corpus IDs
        """
        print(f"\nGenerating random sample of {sample_size:,} papers...")
        print(f"  Total entities in collection: {total_entities:,}")
        
        # Strategy: Use query iterator to scan through collection and sample probabilistically
        # Calculate sampling probability
        sampling_prob = min(1.0, sample_size * 3 / total_entities)  # 3x oversampling
        
        print(f"  Sampling probability: {sampling_prob:.4f}")
        print("  Scanning collection with query iterator...")
        
        sampled_ids = []
        batch_size = 1000
        
        # Use iterator to scan through collection
        try:
            # Create iterator
            iterator = self.collection.query_iterator(
                expr="corpusid >= 0",
                output_fields=["corpusid"],
                batch_size=batch_size
            )
            
            total_scanned = 0
            with tqdm(desc="Scanning corpus") as pbar:
                while True:
                    # Get next batch
                    batch = iterator.next()
                    if not batch:
                        break
                    
                    total_scanned += len(batch)
                    
                    # Probabilistic sampling
                    for result in batch:
                        if random.random() < sampling_prob:
                            sampled_ids.append(result['corpusid'])
                    
                    pbar.update(len(batch))
                    pbar.set_postfix({'sampled': len(sampled_ids)})
                    
                    # Stop if we have enough samples
                    if len(sampled_ids) >= sample_size * 2:
                        break
            
            # Close iterator
            iterator.close()
            
            print(f"  Scanned {total_scanned:,} entities")
            print(f"  Collected {len(sampled_ids):,} samples")
            
        except AttributeError:
            # Fallback: query_iterator not available in this pymilvus version
            print("  ⚠ query_iterator not available, using batch query approach...")
            return self._get_random_sample_ids_fallback(sample_size, total_entities)
        
        # Randomly select final sample size
        if len(sampled_ids) > sample_size:
            print(f"  Randomly selecting {sample_size:,} from {len(sampled_ids):,} samples...")
            corpus_ids = random.sample(sampled_ids, sample_size)
        else:
            corpus_ids = sampled_ids
            if len(corpus_ids) < sample_size:
                print(f"  ⚠ Warning: Only {len(corpus_ids):,} samples collected (requested {sample_size:,})")
        
        print(f"✓ Generated {len(corpus_ids):,} unique corpus IDs")
        return corpus_ids
    
    def _get_random_sample_ids_fallback(self, sample_size: int, total_entities: int) -> List[int]:
        """
        Fallback method using batch queries within Milvus limits.
        
        Args:
            sample_size: Number of papers to sample
            total_entities: Total number of entities in collection
            
        Returns:
            List of random corpus IDs
        """
        # Milvus has a limit: offset + limit <= 16384
        max_query_window = 16384
        batch_size = 1000
        
        all_corpus_ids = []
        offset = 0
        
        # Fetch as many IDs as possible within Milvus limits
        with tqdm(total=min(total_entities, max_query_window), desc="Fetching IDs") as pbar:
            while offset < min(total_entities, max_query_window):
                # Calculate batch size to stay within limit
                remaining_in_window = max_query_window - offset
                current_batch = min(batch_size, remaining_in_window)
                
                if current_batch <= 0:
                    break
                
                try:
                    # Query batch
                    results = self.collection.query(
                        expr="corpusid >= 0",
                        output_fields=["corpusid"],
                        limit=current_batch,
                        offset=offset
                    )
                    
                    batch_ids = [r['corpusid'] for r in results]
                    all_corpus_ids.extend(batch_ids)
                    
                    offset += len(batch_ids)
                    pbar.update(len(batch_ids))
                    
                    # Break if we got fewer results than requested
                    if len(batch_ids) < current_batch:
                        break
                        
                except Exception as e:
                    print(f"\n  Error at offset {offset}: {e}")
                    break
        
        print(f"  Fetched {len(all_corpus_ids):,} corpus IDs from database")
        
        # If we have more IDs than needed, randomly sample
        if len(all_corpus_ids) > sample_size:
            print(f"  Randomly selecting {sample_size:,} from {len(all_corpus_ids):,} IDs...")
            corpus_ids = random.sample(all_corpus_ids, sample_size)
        else:
            corpus_ids = all_corpus_ids
            if len(corpus_ids) < sample_size:
                print(f"  ⚠ Warning: Only {len(corpus_ids):,} IDs available (requested {sample_size:,})")
                print(f"  ⚠ Due to Milvus query window limit, max sample size is ~16,000")
        
        return corpus_ids
    
    def get_embeddings(self, corpus_ids: List[int], batch_size: int = 100) -> Dict[int, np.ndarray]:
        """
        Retrieve embeddings for given corpus IDs.
        
        Args:
            corpus_ids: List of corpus IDs
            batch_size: Batch size for retrieval
            
        Returns:
            Dictionary mapping corpus_id to embedding vector
        """
        print(f"\nRetrieving embeddings for {len(corpus_ids):,} papers...")
        
        embeddings = {}
        
        for i in tqdm(range(0, len(corpus_ids), batch_size), desc="Fetching embeddings"):
            batch_ids = corpus_ids[i:i + batch_size]
            
            # Create expression for batch
            expr = f"corpusid in {batch_ids}"
            
            # Query for embeddings
            results = self.collection.query(
                expr=expr,
                output_fields=["corpusid", "vector"]
            )
            
            # Store embeddings
            for result in results:
                corpus_id = result['corpusid']
                vector = np.array(result['vector'])
                embeddings[corpus_id] = vector
        
        print(f"✓ Retrieved {len(embeddings):,} embeddings")
        
        # Get embedding dimension
        if embeddings:
            first_embedding = next(iter(embeddings.values()))
            embedding_dim = len(first_embedding)
            print(f"  Embedding dimension: {embedding_dim}")
        
        return embeddings
    
    def save_sample(self, corpus_ids: List[int], embeddings: Dict[int, np.ndarray], output_file: str):
        """
        Save sample to pickle file.
        
        Args:
            corpus_ids: List of corpus IDs
            embeddings: Dictionary of embeddings
            output_file: Output file path
        """
        print(f"\nSaving sample to {output_file}...")
        
        # Get embedding dimension
        embedding_dim = len(next(iter(embeddings.values()))) if embeddings else 0
        
        # Create data structure
        data = {
            'corpus_ids': corpus_ids,
            'embeddings': embeddings,
            'embedding_dim': embedding_dim,
            'sample_size': len(corpus_ids),
            'source': f'{self.db_name}/{self.collection_name}'
        }
        
        # Save to pickle
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Sample saved successfully")
        print(f"  Corpus IDs: {len(corpus_ids):,}")
        print(f"  Embeddings: {len(embeddings):,}")
        print(f"  Embedding dimension: {embedding_dim}")
    
    def close(self):
        """Release collection and close connection."""
        if self.collection:
            print("\nReleasing collection from memory...")
            self.collection.release()
            print("✓ Collection released")
        if self.client:
            connections.disconnect(alias="default")
            print("✓ Disconnected from Milvus")


def main():
    parser = argparse.ArgumentParser(
        description="Sample random papers from Milvus corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        required=True,
        help='Number of papers to sample'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='random_sample.pkl',
        help='Output pickle file (default: random_sample.pkl)'
    )
    
    parser.add_argument(
        '--db-name',
        type=str,
        default=DEFAULT_DB,
        help=f'Milvus database name (default: {DEFAULT_DB})'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        default=DEFAULT_COLLECTION,
        help=f'Milvus collection name (default: {DEFAULT_COLLECTION})'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=MILVUS_IP,
        help=f'Milvus host (default: {MILVUS_IP})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=MILVUS_PORT,
        help=f'Milvus port (default: {MILVUS_PORT})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (optional)'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    print("=" * 70)
    print("RANDOM CORPUS SAMPLING")
    print("=" * 70)
    
    try:
        # Create sampler
        sampler = RandomCorpusSampler(
            db_name=args.db_name,
            collection_name=args.collection,
            host=args.host,
            port=args.port
        )
        
        # Connect and get total entities
        total_entities = sampler.connect()
        
        # Validate sample size
        if args.sample_size > total_entities:
            print(f"\n⚠ Warning: Requested sample size ({args.sample_size:,}) exceeds total entities ({total_entities:,})")
            print(f"  Adjusting to maximum: {total_entities:,}")
            args.sample_size = total_entities
        
        # Get random sample IDs
        corpus_ids = sampler.get_random_sample_ids(args.sample_size, total_entities)
        
        # Get embeddings
        embeddings = sampler.get_embeddings(corpus_ids)
        
        # Save sample
        sampler.save_sample(corpus_ids, embeddings, args.output)
        
        # Close connection
        sampler.close()
        
        print("\n" + "=" * 70)
        print("✓ Sampling completed successfully!")
        print("=" * 70)
        
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
