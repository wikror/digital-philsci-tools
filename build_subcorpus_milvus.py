"""
Build Milvus databases for subcorpus with sentence-level and paragraph-level embeddings.

This script creates two Milvus collections in a separate database:
1. Sentence-level embeddings
2. Paragraph-level embeddings (using paragraph annotations or 10-sentence chunks)

Uses indexed gzip for fast access to S2ORC papers via MongoDB metadata (filename + byte offset).

The script is modular and designed for extensibility.

Usage:
    python build_subcorpus_milvus.py --subcorpus subcorpus_20251105_164654.pkl --s2orc-path /path/to/s2orc/
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np

from pymilvus import connections, MilvusClient, db, DataType
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# For language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("langdetect not available, skipping language filtering.")
    LANGDETECT_AVAILABLE = False

# For custom sentence segmentation
try:
    from sentencizer import sentencizer as sts
    from spacy.lang.en import English
    CUSTOM_SENTENCIZER_AVAILABLE = True
except ImportError:
    print("Custom sentencizer not available, using standard spaCy model.")
    CUSTOM_SENTENCIZER_AVAILABLE = False

# Configuration
MILVUS_IP = "localhost"
MILVUS_PORT = 19530
MONGO_IP = "localhost"
MONGO_PORT = 27017

# Default embedding model
DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_DIMENSION = 384  # for multi-qa-MiniLM-L6-cos-v1

# Paragraph settings
DEFAULT_PARAGRAPH_SIZE = 10  # sentences per paragraph if no annotation


class SubcorpusSelector:
    """Base class for subcorpus selection strategies."""
    
    def get_corpus_ids(self) -> List[int]:
        """Return list of corpus IDs to include in subcorpus."""
        raise NotImplementedError


class PickleSubcorpusSelector(SubcorpusSelector):
    """Select subcorpus from query_milvus_rrf.py pickle output."""
    
    def __init__(self, pickle_path: str):
        """
        Initialize selector from pickle file.
        
        Args:
            pickle_path: Path to pickle file from query_milvus_rrf.py
        """
        self.pickle_path = pickle_path
        self.metadata = {}
    
    def get_corpus_ids(self) -> List[int]:
        """Load corpus IDs from pickle file."""
        with open(self.pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'corpus_ids' in data:
            corpus_ids = [int(cid) for cid in data['corpus_ids']]
            
            # Extract metadata
            if 'scores' in data:
                self.metadata['scores'] = {corpus_ids[i]: float(data['scores'][i]) 
                                          for i in range(len(corpus_ids))}
            
            if 'embeddings' in data:
                self.metadata['embeddings'] = {int(k): v for k, v in data['embeddings'].items()}
            
            if 'embedding_dim' in data:
                self.metadata['embedding_dim'] = data['embedding_dim']
        else:
            raise ValueError("Invalid pickle format - expected dict with 'corpus_ids' key")
        
        return corpus_ids


class ListSubcorpusSelector(SubcorpusSelector):
    """Select subcorpus from a simple list of IDs."""
    
    def __init__(self, corpus_ids: List[int]):
        """
        Initialize selector from list of corpus IDs.
        
        Args:
            corpus_ids: List of corpus IDs
        """
        self.corpus_ids = corpus_ids
        self.metadata = {}
    
    def get_corpus_ids(self) -> List[int]:
        """Return the corpus IDs."""
        return self.corpus_ids


class MilvusSubcorpusBuilder:
    """Modular builder for subcorpus Milvus databases using indexed gzip access."""
    
    def __init__(self, 
                 db_name: str,
                 s2orc_path: str,
                 selector: SubcorpusSelector,
                 model_name: str = DEFAULT_MODEL,
                 embedding_dim: int = EMBEDDING_DIMENSION,
                 milvus_host: str = MILVUS_IP,
                 milvus_port: int = MILVUS_PORT,
                 mongo_host: str = MONGO_IP,
                 mongo_port: int = MONGO_PORT,
                 use_gpu: bool = True,
                 filter_language: bool = True):
        """
        Initialize the builder.
        
        Args:
            db_name: Name of the Milvus database (will be created if doesn't exist)
            s2orc_path: Path to S2ORC gzipped JSONL files
            selector: SubcorpusSelector instance for choosing papers
            model_name: Name of the sentence transformer model
            embedding_dim: Dimension of embeddings
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            mongo_host: MongoDB server host
            mongo_port: MongoDB server port
            use_gpu: Whether to use GPU for embeddings
            filter_language: Whether to filter out non-English papers
        """
        self.db_name = db_name
        self.s2orc_path = Path(s2orc_path)
        self.selector = selector
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.mongo_host = mongo_host
        self.mongo_port = mongo_port
        self.use_gpu = use_gpu
        self.filter_language = filter_language
        
        self.milvus_client = None
        self.mongo_client = None
        self.encoder = None
        self.nlp = None
    
    def connect(self):
        """Connect to Milvus, MongoDB, and load models."""
        print(f"\n{'='*70}")
        print("INITIALIZING CONNECTIONS")
        print(f"{'='*70}")
        
        # Connect to Milvus
        print(f"\nConnecting to Milvus at {self.milvus_host}:{self.milvus_port}...")
        connections.connect(host=self.milvus_host, port=self.milvus_port)
        
        # Create database if it doesn't exist
        existing_dbs = db.list_database()
        if self.db_name not in existing_dbs:
            print(f"  Creating database '{self.db_name}'...")
            db.create_database(self.db_name)
            print(f"  ✓ Database '{self.db_name}' created")
        else:
            print(f"  ✓ Database '{self.db_name}' already exists")
        
        # Use the database
        db.using_database(self.db_name)
        
        # Setup Milvus client
        self.milvus_client = MilvusClient(
            uri=f'http://{self.milvus_host}:{self.milvus_port}',
            token='root:Milvus',
            db_name=self.db_name
        )
        print(f"✓ Connected to Milvus database '{self.db_name}'")
        
        # Connect to MongoDB
        print(f"\nConnecting to MongoDB at {self.mongo_host}:{self.mongo_port}...")
        self.mongo_client = MongoClient(self.mongo_host, self.mongo_port)
        # Test connection
        try:
            self.mongo_client.admin.command('ping')
            print("✓ Connected to MongoDB")
        except Exception as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            raise
        
        # Load encoder model
        print(f"\nLoading encoder model '{self.model_name}'...")
        if self.use_gpu:
            self.encoder = SentenceTransformer(self.model_name).cuda()
            print("✓ Model loaded on GPU")
        else:
            self.encoder = SentenceTransformer(self.model_name)
            print("✓ Model loaded on CPU")
        
        # Auto-detect and update embedding dimension from the actual model
        actual_dim = self.encoder.get_sentence_embedding_dimension()
        if self.embedding_dim != actual_dim:
            print(f"  ⚠️  Embedding dimension mismatch detected!")
            print(f"     Configured: {self.embedding_dim}, Actual model output: {actual_dim}")
            print(f"     Updating to use actual dimension: {actual_dim}")
            self.embedding_dim = actual_dim
        
        # Load spaCy sentence segmenter
        print("\nLoading sentence segmenter...")
        if CUSTOM_SENTENCIZER_AVAILABLE:
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")
            self.nlp = sts.add_custom_sentencizer(self.nlp)
            print("✓ Custom sentencizer loaded")
        else:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ Standard spaCy model loaded")
        
        print(f"\n{'='*70}")
        print("INITIALIZATION COMPLETE")
        print(f"{'='*70}\n")
    
    def get_paper_metadata_from_mongo(self, corpus_id: int) -> Optional[Dict]:
        """
        Get paper metadata from MongoDB including filename and byte offset.
        
        Args:
            corpus_id: Corpus ID of the paper
            
        Returns:
            Metadata dict with 'filename' and 'offset' fields, or None if not found
        """
        collection = self.mongo_client.papers_db.papers
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = collection.find_one({"corpusid": corpus_id})
                if result is None:
                    return None
                
                # Extract relevant metadata
                metadata = {
                    'corpusid': corpus_id,
                    'file_location': result.get('file_location'),  # [filename, offset]
                    'title': result.get('title'),
                    'year': result.get('year'),
                    'authors': result.get('authors'),
                    'journal': result.get('journal', {}).get('name') if result.get('journal') else None,
                    's2fieldsofstudy': result.get('s2fieldsofstudy', [])
                }
                
                return metadata
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)
                else:
                    print(f"  ✗ Error fetching metadata for {corpus_id}: {e}")
                    return None
    
    def load_paper_from_gzip(self, filename: str, offset: int) -> Optional[Dict]:
        """
        Load a paper from gzipped JSONL using indexed access.
        
        Args:
            filename: Name of the gzipped JSONL file
            offset: Byte offset in the file
            
        Returns:
            Paper dictionary or None if not found
        """
        filepath = self.s2orc_path / filename
        
        if not filepath.exists():
            return None
        
        try:
            import indexed_gzip as igzip
            
            # Open with indexed_gzip for fast random access
            with igzip.IndexedGzipFile(str(filepath)) as f:
                f.seek(offset)
                line = f.readline()
                paper = json.loads(line)
                return paper
        except ImportError:
            # Fallback to standard gzip if indexed_gzip not available
            import gzip
            with gzip.open(filepath, 'rt') as f:
                f.seek(offset)
                line = f.readline()
                paper = json.loads(line)
                return paper
        except Exception as e:
            print(f"  ✗ Error loading paper from {filename} at offset {offset}: {e}")
            return None
    
    def load_paper(self, corpus_id: int) -> Optional[Dict]:
        """
        Load a paper using MongoDB metadata for indexed gzip access.
        Filters out non-English papers if langdetect is available and filter_language is True.
        
        Args:
            corpus_id: Corpus ID of the paper
            
        Returns:
            Paper dictionary or None if not found or non-English
        """
        # Get metadata from MongoDB
        metadata = self.get_paper_metadata_from_mongo(corpus_id)
        
        if metadata is None:
            return None
        
        file_location = metadata.get('file_location')
        filename = file_location[0] if file_location else None
        offset = file_location[1] if file_location else None
        
        if filename is None or offset is None:
            print(f"  ✗ Missing filename or offset for corpus ID {corpus_id}")
            return None
        
        # Load paper from gzipped file
        paper = self.load_paper_from_gzip(filename, offset)
        
        if paper is None:
            return None
        
        # Language detection (filter out non-English papers)
        if self.filter_language and LANGDETECT_AVAILABLE:
            try:
                text = paper.get("content", {}).get("text", "")
                if text and detect(text) != "en":
                    return None  # Skip non-English papers
            except LangDetectException:
                # If langdetect fails, skip the paper (likely not useful text)
                return None
            except Exception:
                # If any other error, continue processing (don't skip)
                pass
        
        # Add metadata to paper
        paper['metadata'] = metadata
        
        return paper
    
    def segment_sentences(self, paper: Dict) -> List[Dict[str, int]]:
        """
        Segment paper text into sentences using spaCy, creating standoff annotations.
        Based on the v0_to_v1_spacy_standoff.py preprocessing logic.
        
        Args:
            paper: Paper dictionary with content.text and content.annotations
            
        Returns:
            List of sentence annotations with 'start' and 'end' offsets
        """
        sentences = []
        text = paper["content"]["text"]
        annotations = paper["content"]["annotations"]
        
        # First, segment abstract if it exists
        try:
            abstract_dict = annotations.get("abstract")
            if abstract_dict:
                if isinstance(abstract_dict, str):
                    abstract_dict = json.loads(abstract_dict)[0]
                elif isinstance(abstract_dict, list):
                    abstract_dict = abstract_dict[0]
                
                abstract_text = text[int(abstract_dict["start"]):int(abstract_dict["end"])]
                first_sent_pos = int(abstract_dict["start"])
                
                # Segment abstract
                doc = self.nlp(abstract_text)
                
                sent_start = first_sent_pos
                first_sent = True
                
                for sent in doc.sents:
                    if not first_sent:
                        sent_end = first_sent_pos + sent[0].idx
                        sentences.append({"start": int(sent_start), "end": int(sent_end)})
                    sent_start = first_sent_pos + sent[0].idx
                    first_sent = False
                
                # Last sentence of abstract
                sent_end = int(abstract_dict["end"])
                sentences.append({"start": int(sent_start), "end": int(sent_end)})
                
        except (TypeError, KeyError, IndexError):
            abstract_dict = None
        
        # Now segment paragraphs if they exist
        paragraphs = None
        try:
            paragraph_annotation = annotations.get("paragraph")
            if paragraph_annotation:
                if isinstance(paragraph_annotation, str):
                    paragraphs = json.loads(paragraph_annotation)
                else:
                    paragraphs = paragraph_annotation
                
                for paragraph in paragraphs:
                    para_text = text[int(paragraph["start"]):int(paragraph["end"])]
                    first_sent_pos = int(paragraph["start"])
                    
                    doc = self.nlp(para_text)
                    
                    sent_start = first_sent_pos
                    first_sent = True
                    
                    for sent in doc.sents:
                        if not first_sent:
                            sent_end = first_sent_pos + sent[0].idx
                            sentences.append({"start": int(sent_start), "end": int(sent_end)})
                        sent_start = first_sent_pos + sent[0].idx
                        first_sent = False
                    
                    # Last sentence of paragraph
                    sent_end = int(paragraph["end"])
                    sentences.append({"start": int(sent_start), "end": int(sent_end)})
        
        except (TypeError, KeyError):
            paragraphs = None
        
        # If no paragraphs, segment the remaining text (after abstract or from beginning)
        if paragraphs is None:
            if abstract_dict is not None:
                remaining_text = text[int(abstract_dict["end"]):]
                first_sent_pos = int(abstract_dict["end"])
            else:
                remaining_text = text
                first_sent_pos = 0
            
            if remaining_text:
                doc = self.nlp(remaining_text)
                
                sent_start = first_sent_pos
                first_sent = True
                
                for sent in doc.sents:
                    if not first_sent:
                        sent_end = first_sent_pos + sent[0].idx
                        sentences.append({"start": int(sent_start), "end": int(sent_end)})
                    sent_start = first_sent_pos + sent[0].idx
                    first_sent = False
                
                # Last sentence
                sent_end = first_sent_pos + len(remaining_text)
                sentences.append({"start": int(sent_start), "end": int(sent_end)})
        
        return sentences
    
    def create_collection(self,
                         collection_name: str,
                         level: str = "sentence",
                         index_type: str = "IVF_PQ",
                         metric_type: str = "COSINE",
                         index_params: Optional[Dict] = None,
                         include_rrf_score: bool = False) -> bool:
        """
        Create a Milvus collection for the subcorpus.
        
        Args:
            collection_name: Name of the collection
            level: Either "sentence" or "paragraph"
            index_type: Type of vector index (IVF_PQ, IVF_FLAT, HNSW)
            metric_type: Distance metric (COSINE, L2, IP)
            index_params: Custom index parameters
            include_rrf_score: Whether to include RRF score field (from query_milvus_rrf.py)
            
        Returns:
            True if collection was created, False if it already exists
        """
        print(f"\nCreating collection '{collection_name}'...")
        
        # Check if collection exists
        if self.milvus_client.has_collection(collection_name):
            print(f"  Collection '{collection_name}' already exists")
            return False
        
        # Create schema
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="corpusid", datatype=DataType.INT64)
        
        # Add RRF score if available from selector metadata
        if include_rrf_score and hasattr(self.selector, 'metadata') and 'scores' in self.selector.metadata:
            schema.add_field(field_name="rrf_score", datatype=DataType.FLOAT)
            print("  ✓ Including RRF score field")
        
        if level == "sentence":
            schema.add_field(field_name="sentence_number", datatype=DataType.INT64)
            schema.add_field(field_name="sentence_indices", datatype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=2)
        elif level == "paragraph":
            schema.add_field(field_name="paragraph_number", datatype=DataType.INT64)
            schema.add_field(field_name="paragraph_indices", datatype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=2)
            schema.add_field(field_name="sentence_start", datatype=DataType.INT64)
            schema.add_field(field_name="sentence_end", datatype=DataType.INT64)
        
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        
        # Setup index
        idx_params = self.milvus_client.prepare_index_params()
        
        # Index on corpusid for fast lookup
        idx_params.add_index(
            field_name="corpusid",
            index_type="INVERTED"
        )
        
        # Vector index
        if index_params is None:
            if index_type == "IVF_PQ":
                index_params = {"nlist": 1024, "m": 4, "nbits": 8}
            elif index_type == "IVF_FLAT":
                index_params = {"nlist": 1024}
            elif index_type == "HNSW":
                index_params = {"M": 16, "efConstruction": 200}
            else:
                index_params = {}
        
        idx_params.add_index(
            field_name="vector",
            index_type=index_type,
            params=index_params,
            metric_type=metric_type
        )
        
        # Create collection
        self.milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=idx_params
        )
        
        print(f"✓ Collection '{collection_name}' created successfully")
        return True
    
    def extract_sentences(self, paper: Dict) -> List[Tuple[int, Tuple[int, int]]]:
        """
        Extract sentences from a paper by segmenting the text.
        
        Args:
            paper: Paper dictionary with content.text and content.annotations
            
        Returns:
            List of (sentence_number, (start_idx, end_idx)) tuples
        """
        try:
            # Check if sentences are already annotated
            if "sentences" in paper["content"]["annotations"]:
                sentence_annotations = paper["content"]["annotations"]["sentences"]
            else:
                # Segment sentences using spaCy
                sentence_annotations = self.segment_sentences(paper)
            
            text = paper["content"]["text"]
            
            result = []
            seen = set()
            
            for sent_num, sent in enumerate(sentence_annotations):
                start_idx = int(sent["start"])
                end_idx = int(sent["end"])
                sent_text = text[start_idx:end_idx].strip()
                
                # Skip empty or duplicate sentences
                if sent_text and sent_text not in seen:
                    result.append((sent_num, (start_idx, end_idx)))
                    seen.add(sent_text)
            
            return result
        except (KeyError, TypeError) as e:
            print(f"  ✗ Error extracting sentences: {e}")
            return []
    
    def extract_paragraphs(self, 
                          paper: Dict, 
                          paragraph_size: int = DEFAULT_PARAGRAPH_SIZE) -> List[Tuple[int, int, int, Tuple[int, int]]]:
        """
        Extract paragraphs from a paper using standoff annotations or sentence chunking.
        
        Args:
            paper: Paper dictionary
            paragraph_size: Number of sentences per paragraph if no annotation
            
        Returns:
            List of (paragraph_number, sentence_start, sentence_end, (para_start_idx, para_end_idx)) tuples
        """
        try:
            sentences = self.extract_sentences(paper)
            text = paper["content"]["text"]
            
            # Check if paper has paragraph annotations
            try:
                if isinstance(paper["content"]["annotations"].get("paragraph"), str):
                    paragraphs_annotation = json.loads(paper["content"]["annotations"]["paragraph"])
                else:
                    paragraphs_annotation = paper["content"]["annotations"].get("paragraph", [])
            except (KeyError, TypeError):   
                paragraphs_annotation = []

            if paragraphs_annotation and len(paragraphs_annotation) > 0:
                # Use paragraph annotations
                result = []
                
                for para_num, para in enumerate(paragraphs_annotation):
                    para_start_idx = int(para["start"])
                    para_end_idx = int(para["end"])
                    para_text = text[para_start_idx:para_end_idx].strip()
                    
                    if not para_text:
                        continue
                    
                    # Find sentence range for this paragraph
                    sent_start = None
                    sent_end = None
                    
                    for sent_num, (start_idx, end_idx) in sentences:
                        # Check if sentence is within paragraph bounds
                        if start_idx >= para_start_idx and end_idx <= para_end_idx:
                            if sent_start is None:
                                sent_start = sent_num
                            sent_end = sent_num
                    
                    if sent_start is not None:
                        result.append((para_num, sent_start, sent_end, (para_start_idx, para_end_idx)))
                
                return result
            else:
                # Create paragraphs from sentence chunks
                result = []
                
                for para_num in range(0, len(sentences), paragraph_size):
                    chunk = sentences[para_num:para_num + paragraph_size]
                    
                    sent_start = chunk[0][0]
                    sent_end = chunk[-1][0]
                    # Paragraph spans from start of first sentence to end of last sentence
                    para_start_idx = chunk[0][1][0]
                    para_end_idx = chunk[-1][1][1]
                    
                    result.append((para_num // paragraph_size, sent_start, sent_end, (para_start_idx, para_end_idx)))
                
                return result
        except Exception as e:
            print(f"  ✗ Error extracting paragraphs: {e}")
            return []
    
    def encode_batch(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """
        Encode a batch of texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        return self.encoder.encode(texts, batch_size=batch_size, show_progress_bar=False)
    
    def insert_sentences(self, 
                        corpus_id: int,
                        sentences: List[Tuple[int, Tuple[int, int]]],
                        collection_name: str,
                        batch_size: int = 100,
                        paper: Optional[Dict] = None) -> int:
        """
        Insert sentence embeddings into collection.
        
        Args:
            corpus_id: Corpus ID of the paper
            sentences: List of (sentence_number, (start_idx, end_idx)) tuples
            collection_name: Name of the collection
            batch_size: Batch size for insertion
            paper: Paper dictionary (needed to extract text)
            
        Returns:
            Number of sentences inserted
        """
        if not sentences or not paper:
            return 0
        
        text = paper["content"]["text"]
        
        # Extract texts and encode
        sent_texts = [text[start_idx:end_idx].strip() for _, (start_idx, end_idx) in sentences]
        embeddings = self.encode_batch(sent_texts)
        
        # Get RRF score if available
        rrf_score = None
        if hasattr(self.selector, 'metadata') and 'scores' in self.selector.metadata:
            rrf_score = self.selector.metadata['scores'].get(corpus_id)
        
        # Prepare data
        data = []
        for (sent_num, (start_idx, end_idx)), embedding in zip(sentences, embeddings):
            row = {
                "corpusid": corpus_id,
                "sentence_number": sent_num,
                "sentence_indices": [start_idx, end_idx],
                "vector": embedding.tolist()
            }
            if rrf_score is not None:
                row["rrf_score"] = rrf_score
            data.append(row)
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                self.milvus_client.insert(
                    collection_name=collection_name,
                    data=batch
                )
                total_inserted += len(batch)
            except Exception as e:
                print(f"  ✗ Error inserting batch for corpus ID {corpus_id}: {e}")
        
        return total_inserted
    
    def insert_paragraphs(self,
                         corpus_id: int,
                         paragraphs: List[Tuple[int, int, int, Tuple[int, int]]],
                         collection_name: str,
                         batch_size: int = 100,
                         paper: Optional[Dict] = None) -> int:
        """
        Insert paragraph embeddings into collection.
        
        Args:
            corpus_id: Corpus ID of the paper
            paragraphs: List of (para_num, sent_start, sent_end, (para_start_idx, para_end_idx)) tuples
            collection_name: Name of the collection
            batch_size: Batch size for insertion
            paper: Paper dictionary (needed to extract text)
            
        Returns:
            Number of paragraphs inserted
        """
        if not paragraphs or not paper:
            return 0
        
        text = paper["content"]["text"]
        
        # Extract texts and encode
        para_texts = [text[start_idx:end_idx].strip() for _, _, _, (start_idx, end_idx) in paragraphs]
        embeddings = self.encode_batch(para_texts)
        
        # Get RRF score if available
        rrf_score = None
        if hasattr(self.selector, 'metadata') and 'scores' in self.selector.metadata:
            rrf_score = self.selector.metadata['scores'].get(corpus_id)
        
        # Prepare data
        data = []
        for (para_num, sent_start, sent_end, (para_start_idx, para_end_idx)), embedding in zip(paragraphs, embeddings):
            row = {
                "corpusid": corpus_id,
                "paragraph_number": para_num,
                "paragraph_indices": [para_start_idx, para_end_idx],
                "sentence_start": sent_start,
                "sentence_end": sent_end,
                "vector": embedding.tolist()
            }
            if rrf_score is not None:
                row["rrf_score"] = rrf_score
            data.append(row)
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                self.milvus_client.insert(
                    collection_name=collection_name,
                    data=batch
                )
                total_inserted += len(batch)
            except Exception as e:
                print(f"  ✗ Error inserting batch for corpus ID {corpus_id}: {e}")
        
        return total_inserted
    
    def process_paper(self,
                     corpus_id: int,
                     sentence_collection: Optional[str] = None,
                     paragraph_collection: Optional[str] = None,
                     paragraph_size: int = DEFAULT_PARAGRAPH_SIZE) -> Dict[str, int]:
        """
        Process a single paper and insert into collections.
        
        Args:
            corpus_id: Corpus ID of the paper
            sentence_collection: Name of sentence collection (optional)
            paragraph_collection: Name of paragraph collection (optional)
            paragraph_size: Sentences per paragraph if no annotation
            
        Returns:
            Dictionary with insertion counts and status
        """
        result = {"sentences": 0, "paragraphs": 0, "success": False, "filtered": False}
        
        # Load paper using indexed gzip
        paper = self.load_paper(corpus_id)
        if paper is None:
            # Could be filtered (non-English) or missing
            result["filtered"] = True
            return result
        
        # Process sentences
        if sentence_collection:
            sentences = self.extract_sentences(paper)
            result["sentences"] = self.insert_sentences(
                corpus_id, sentences, sentence_collection, paper=paper
            )
        
        # Process paragraphs
        if paragraph_collection:
            paragraphs = self.extract_paragraphs(paper, paragraph_size)
            result["paragraphs"] = self.insert_paragraphs(
                corpus_id, paragraphs, paragraph_collection, paper=paper
            )
        
        result["success"] = True
        return result
    
    def build_from_subcorpus(self,
                           sentence_collection: str,
                           paragraph_collection: str,
                           paragraph_size: int = DEFAULT_PARAGRAPH_SIZE,
                           checkpoint_file: Optional[str] = None,
                           checkpoint_interval: int = 100,
                           resume_from_checkpoint: bool = True) -> Dict:
        """
        Build collections from a subcorpus using the selector.
        
        Args:
            sentence_collection: Name of sentence collection
            paragraph_collection: Name of paragraph collection
            paragraph_size: Sentences per paragraph if no annotation
            checkpoint_file: Path to checkpoint file for resuming
            checkpoint_interval: Save checkpoint every N papers
            resume_from_checkpoint: Whether to resume from checkpoint if it exists
            
        Returns:
            Statistics dictionary
        """
        # Get corpus IDs from selector
        subcorpus_ids = self.selector.get_corpus_ids()
        
        # Load checkpoint if exists and resume is enabled
        processed_ids = set()
        if resume_from_checkpoint and checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                processed_ids = pickle.load(f)
            print(f"\nResuming from checkpoint: {len(processed_ids)} papers already processed")
        
        # Filter to unprocessed papers
        remaining_ids = [cid for cid in subcorpus_ids if cid not in processed_ids]
        
        print(f"\n{'='*70}")
        print(f"Processing {len(remaining_ids)} papers...")
        print(f"  Sentence collection: {sentence_collection}")
        print(f"  Paragraph collection: {paragraph_collection}")
        print(f"{'='*70}\n")
        
        stats = {
            "total_papers": len(remaining_ids),
            "processed_papers": 0,
            "failed_papers": 0,
            "filtered_papers": 0,  # Papers filtered out (e.g., non-English)
            "total_sentences": 0,
            "total_paragraphs": 0
        }
        
        for i, corpus_id in enumerate(tqdm(remaining_ids, desc="Processing papers")):
            # Process paper (no FOS needed - using MongoDB metadata)
            result = self.process_paper(
                corpus_id=corpus_id,
                sentence_collection=sentence_collection,
                paragraph_collection=paragraph_collection,
                paragraph_size=paragraph_size
            )
            
            if result["success"]:
                stats["processed_papers"] += 1
                stats["total_sentences"] += result["sentences"]
                stats["total_paragraphs"] += result["paragraphs"]
            elif result.get("filtered", False):
                stats["filtered_papers"] += 1
            else:
                stats["failed_papers"] += 1
            
            # Add to processed
            processed_ids.add(corpus_id)
            
            # Save checkpoint
            if checkpoint_file and (i + 1) % checkpoint_interval == 0:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(processed_ids, f)
        
        # Final checkpoint save
        if checkpoint_file:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(processed_ids, f)
        
        return stats
    
    def drop_collections(self, collection_names: List[str]):
        """Drop specified collections if they exist."""
        try:
            for collection_name in collection_names:
                if self.milvus_client.has_collection(collection_name):
                    print(f"  Dropping collection '{collection_name}'...")
                    self.milvus_client.drop_collection(collection_name)
                    print(f"  ✓ Collection '{collection_name}' dropped successfully")
                else:
                    print(f"  Collection '{collection_name}' does not exist (nothing to drop)")
        except Exception as e:
            print(f"  ✗ Error dropping collections: {e}")
            raise
    
    def close(self):
        """Close connections."""
        if self.mongo_client:
            self.mongo_client.close()
        if self.milvus_client:
            connections.disconnect(alias="default")
        print("\n✓ All connections closed")


def main():
    parser = argparse.ArgumentParser(
        description="Build Milvus databases for subcorpus with sentence and paragraph embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from query_milvus_rrf.py output (recommended):
  python build_subcorpus_milvus.py \\
    --subcorpus subcorpus_20251105_164654.pkl \\
    --s2orc-path /path/to/s2orc/ \\
    --db-name my_subcorpus
  
  # Customize collections and parameters:
  python build_subcorpus_milvus.py \\
    --subcorpus subcorpus.pkl \\
    --s2orc-path /path/to/s2orc/ \\
    --db-name my_subcorpus \\
    --sentence-collection sentences \\
    --paragraph-collection paragraphs \\
    --paragraph-size 10 \\
    --no-gpu
        """
    )
    
    parser.add_argument(
        '--subcorpus',
        type=str,
        required=True,
        help='Path to subcorpus file (pickle from query_milvus_rrf.py recommended)'
    )
    
    parser.add_argument(
        '--s2orc-path',
        type=str,
        required=True,
        help='Path to S2ORC gzipped JSONL files directory'
    )
    
    parser.add_argument(
        '--db-name',
        type=str,
        default='subcorpus',
        help='Name of the Milvus database to create (default: subcorpus)'
    )
    
    parser.add_argument(
        '--sentence-collection',
        type=str,
        default='sentences',
        help='Name of sentence collection (default: sentences)'
    )
    
    parser.add_argument(
        '--paragraph-collection',
        type=str,
        default='paragraphs',
        help='Name of paragraph collection (default: paragraphs)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Sentence transformer model (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=None,
        help='Embedding dimension (default: auto-detect from model or subcorpus file)'
    )
    
    parser.add_argument(
        '--paragraph-size',
        type=int,
        default=DEFAULT_PARAGRAPH_SIZE,
        help=f'Sentences per paragraph if no annotation (default: {DEFAULT_PARAGRAPH_SIZE})'
    )
    
    parser.add_argument(
        '--checkpoint-file',
        type=str,
        default='subcorpus_checkpoint.pkl',
        help='Checkpoint file for resuming (default: subcorpus_checkpoint.pkl)'
    )
    
    parser.add_argument(
        '--milvus-host',
        type=str,
        default=MILVUS_IP,
        help=f'Milvus server host (default: {MILVUS_IP})'
    )
    
    parser.add_argument(
        '--milvus-port',
        type=int,
        default=MILVUS_PORT,
        help=f'Milvus server port (default: {MILVUS_PORT})'
    )
    
    parser.add_argument(
        '--mongo-host',
        type=str,
        default=MONGO_IP,
        help=f'MongoDB server host (default: {MONGO_IP})'
    )
    
    parser.add_argument(
        '--mongo-port',
        type=int,
        default=MONGO_PORT,
        help=f'MongoDB server port (default: {MONGO_PORT})'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU for encoding'
    )
    
    parser.add_argument(
        '--no-language-filter',
        action='store_true',
        help='Disable language filtering (allow non-English papers)'
    )
    
    parser.add_argument(
        '--index-type',
        type=str,
        default='IVF_PQ',
        choices=['IVF_PQ', 'IVF_FLAT', 'HNSW'],
        help='Vector index type (default: IVF_PQ)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BUILD MILVUS DATABASES FOR SUBCORPUS")
    print("Using indexed gzip access via MongoDB metadata")
    print("=" * 70)
    
    # Check for existing checkpoint
    resume_from_checkpoint = False
    if os.path.exists(args.checkpoint_file):
        print(f"\n⚠️  Checkpoint file detected: {args.checkpoint_file}")
        while True:
            response = input("Do you want to (r)esume from checkpoint or start from (s)cratch? [r/s]: ").strip().lower()
            if response == 'r':
                resume_from_checkpoint = True
                print("  → Resuming from checkpoint")
                break
            elif response == 's':
                print("  → Starting from scratch")
                while True:
                    confirm = input(f"    This will DROP the collections '{args.sentence_collection}' and '{args.paragraph_collection}' and delete the checkpoint file. Are you sure? [y/n]: ").strip().lower()
                    if confirm == 'y':
                        print("  → User confirmed: will drop collections and start fresh")
                        break
                    elif confirm == 'n':
                        print("  → Cancelled. Exiting.")
                        return 0
                    else:
                        print("    Invalid input. Please enter 'y' or 'n'.")
                break
            else:
                print("  Invalid input. Please enter 'r' to resume or 's' to start from scratch.")
    
    try:
        # Create selector from subcorpus file
        print(f"\nLoading subcorpus from {args.subcorpus}...")
        selector = PickleSubcorpusSelector(args.subcorpus)
        corpus_ids = selector.get_corpus_ids()
        print(f"✓ Loaded {len(corpus_ids)} corpus IDs")
        if 'scores' in selector.metadata:
            print("  ✓ RRF scores available")
        if 'embeddings' in selector.metadata:
            print(f"  ✓ Full-article embeddings available ({len(selector.metadata['embeddings'])})")
            if 'embedding_dim' in selector.metadata:
                print(f"    (Full-article embedding dim: {selector.metadata['embedding_dim']})")
            print(f"    Note: Full-article embeddings are NOT used by this script.")
            print(f"          This script generates NEW sentence/paragraph embeddings from scratch.")
        
        # Determine embedding dimension for sentence/paragraph embeddings
        # This is independent of any embeddings in the subcorpus file
        embedding_dim = args.embedding_dim
        if embedding_dim is None:
            # Default to the model's expected dimension
            embedding_dim = EMBEDDING_DIMENSION
            print(f"\n  Sentence/paragraph embedding dimension: {embedding_dim} (default)")
            print(f"  (Will be auto-detected from model '{args.model}' during initialization)")
        else:
            print(f"\n  Sentence/paragraph embedding dimension: {embedding_dim} (user-specified)")
            print(f"  (Will be validated against model '{args.model}' during initialization)")
        
        # Initialize builder
        builder = MilvusSubcorpusBuilder(
            db_name=args.db_name,
            s2orc_path=args.s2orc_path,
            selector=selector,
            model_name=args.model,
            embedding_dim=embedding_dim,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            mongo_host=args.mongo_host,
            mongo_port=args.mongo_port,
            use_gpu=not args.no_gpu,
            filter_language=not args.no_language_filter
        )
        
        # Connect to all services
        builder.connect()
        
        # Print final embedding dimension after model loading
        print(f"\n  Final embedding dimension: {builder.embedding_dim}")
        
        # Handle starting from scratch if user chose not to resume
        if os.path.exists(args.checkpoint_file) and not resume_from_checkpoint:
            # Drop the collections
            print("\nDropping existing collections and checkpoint...")
            builder.drop_collections([args.sentence_collection, args.paragraph_collection])
            
            # Delete checkpoint file
            try:
                os.remove(args.checkpoint_file)
                print(f"  ✓ Checkpoint file '{args.checkpoint_file}' deleted")
            except Exception as e:
                print(f"  ⚠️  Could not delete checkpoint file: {e}")
        
        # Print language filtering status
        if builder.filter_language and LANGDETECT_AVAILABLE:
            print("\n  Language filtering: ENABLED (English only)")
        elif builder.filter_language and not LANGDETECT_AVAILABLE:
            print("\n  Language filtering: DISABLED (langdetect not available)")
        else:
            print("\n  Language filtering: DISABLED (by user)")
        
        # Determine if we should include RRF scores
        include_rrf = 'scores' in selector.metadata
        
        # Create collections
        builder.create_collection(
            collection_name=args.sentence_collection,
            level="sentence",
            index_type=args.index_type,
            include_rrf_score=include_rrf
        )
        
        builder.create_collection(
            collection_name=args.paragraph_collection,
            level="paragraph",
            index_type=args.index_type,
            include_rrf_score=include_rrf
        )
        
        # Build databases
        stats = builder.build_from_subcorpus(
            sentence_collection=args.sentence_collection,
            paragraph_collection=args.paragraph_collection,
            paragraph_size=args.paragraph_size,
            checkpoint_file=args.checkpoint_file,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Print statistics
        print("\n" + "=" * 70)
        print("BUILD STATISTICS")
        print("=" * 70)
        print(f"Total papers: {stats['total_papers']}")
        print(f"Processed papers: {stats['processed_papers']}")
        print(f"Failed papers: {stats['failed_papers']}")
        print(f"Filtered papers (non-English/invalid): {stats['filtered_papers']}")
        print(f"Total sentences inserted: {stats['total_sentences']}")
        print(f"Total paragraphs inserted: {stats['total_paragraphs']}")
        print("=" * 70)
        
        # Close connections
        builder.close()
        
        print("\n✓ Build completed successfully!")
        
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
