import requests
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import pickle
from typing import List, Dict, Optional, Tuple

DATA_PATH = '/home/wikror/gdrive/corpus-study/compositionality-data'

# ============================================================================
# CORE INDIVIDUAL QUERY FUNCTION
# ============================================================================

def get_paper(paper_id: str, headers: Dict, fields: List[str], sleep_time=1, max_retries=3):
    """
    Generic function to fetch a single paper with retry logic.
    
    Args:
        paper_id: Paper identifier (already formatted, e.g., 'CorpusId:123', 'DOI:10.1234/...')
        headers: API headers including API key
        fields: List of fields to retrieve (e.g., ['corpusId', 'title', 'embedding.specter_v2'])
        sleep_time: Initial sleep time for exponential backoff
        max_retries: Maximum number of retries for transient errors
    
    Returns:
        Tuple of (paper data dictionary or None, error_code or None)
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    
    params = {
        'fields': ','.join(fields)
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 404:
                # Paper not found - not a retry case
                return None, 404
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = sleep_time * (2 ** attempt)
                print(f"    Rate limit hit for {paper_id}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            elif response.status_code in [500, 502, 503, 504]:
                # Server errors - retry
                wait_time = sleep_time * (2 ** attempt)
                print(f"    Server error {response.status_code} for {paper_id}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                # Other errors - don't retry
                print(f"    Error fetching {paper_id}: Status {response.status_code}")
                return None, response.status_code
                
        except requests.exceptions.Timeout:
            wait_time = sleep_time * (2 ** attempt)
            print(f"    Timeout for {paper_id}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            print(f"    Error fetching {paper_id}: {e}")
            return None, 'exception'
    
    # All retries exhausted
    print(f"    ✗ Failed after {max_retries} attempts: {paper_id}")
    return None, 'max_retries_exceeded'

# ============================================================================
# HELPER FUNCTIONS FOR LOADING DATA AND CONFIGURATION
# ============================================================================

def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_headers() -> Dict:
    """Load API headers from .env file."""
    load_dotenv()
    headers_str = os.getenv('HEADERS')

    try:
        headers = json.loads(headers_str) if headers_str else {}
    except json.JSONDecodeError:
        print("Error: HEADERS in .env file is not valid JSON")
        headers = {}

    if not headers:
        print("Warning: HEADERS not found in .env file")
        print("The script will use the public API with rate limits")
    
    return headers

def load_corpus_ids(filepath: str) -> List[str]:
    """Load corpus IDs from a text file (one per line)."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        corpus_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(corpus_ids)} corpus IDs from {filepath}")
    return corpus_ids

def load_dois(filepath: str) -> List[str]:
    """Load DOIs from a text file (one per line), removing duplicates."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        dois = [line.strip() for line in f if line.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dois = []
    for doi in dois:
        if doi not in seen:
            seen.add(doi)
            unique_dois.append(doi)
    
    print(f"Loaded {len(unique_dois)} unique DOIs from {filepath}")
    return unique_dois

def format_corpus_id(corpus_id: str) -> str:
    """Format corpus ID with CorpusId: prefix."""
    return f"CorpusId:{corpus_id}"

def format_doi(doi: str) -> str:
    """Format DOI with DOI: prefix."""
    return f"DOI:{doi}"

def save_failed_items(failed_items: List[Tuple[str, str]], prefix: str, timestamp: str):
    """
    Save failed items to a timestamped JSON file.
    
    Args:
        failed_items: List of (identifier, error_code) tuples
        prefix: Prefix for filename (e.g., 'corpus_ids', 'dois')
        timestamp: Timestamp string for filename
    """
    if not failed_items:
        return None
    
    failed_file = f'{DATA_PATH}/failed_{prefix}_{timestamp}.json'
    
    # Convert to list of dicts for better JSON structure
    failed_data = [
        {
            'identifier': identifier,
            'error_code': str(error_code)
        }
        for identifier, error_code in failed_items
    ]
    
    with open(failed_file, 'w') as f:
        json.dump(failed_data, f, indent=2)
    
    return failed_file

def retry_failed_requests(failed_items: List[Tuple[str, str]], headers: Dict, fields: List[str], 
                         format_func, rate_limit_delay=1, max_retry_rounds=3):
    """
    Retry failed requests with progressive backoff.
    
    Args:
        failed_items: List of (identifier, error_code) tuples
        headers: API headers
        fields: Fields to retrieve
        format_func: Function to format the identifier (format_corpus_id or format_doi)
        rate_limit_delay: Base delay between requests
        max_retry_rounds: Maximum number of retry rounds
    
    Returns:
        Tuple of (successful_results, still_failed_items)
    """
    successful = []
    still_failed = failed_items.copy()
    
    for round_num in range(1, max_retry_rounds + 1):
        if not still_failed:
            break
            
        print(f"\n{'='*60}")
        print(f"Retry Round {round_num}/{max_retry_rounds}")
        print(f"Attempting to retry {len(still_failed)} failed requests...")
        print(f"{'='*60}\n")
        
        round_failed = []
        wait_time = rate_limit_delay * (2 ** (round_num - 1))  # Exponential backoff between rounds
        
        for identifier, error_code in still_failed:
            paper_id = format_func(identifier)
            paper, new_error = get_paper(paper_id, headers, fields, sleep_time=wait_time)
            
            if paper:
                successful.append((identifier, paper))
                print(f"  ✓ Retry successful: {identifier}")
            else:
                round_failed.append((identifier, new_error))
            
            time.sleep(wait_time)
        
        print(f"\nRound {round_num} results: {len(successful)} recovered, {len(round_failed)} still failed")
        still_failed = round_failed
    
    return successful, still_failed

# ============================================================================
# INTERFACE FUNCTIONS FOR SPECIFIC USE CASES
# ============================================================================

def fetch_paper_embeddings(rate_limit_delay=1):
    """
    Fetch paper embeddings for corpus IDs individually with automatic retry.
    Saves results as a pickle file with corpus_id -> embedding vector mapping.
    
    Args:
        rate_limit_delay: Delay between requests in seconds
    """
    timestamp = get_timestamp()
    headers = load_headers()
    corpus_ids = load_corpus_ids(f'{DATA_PATH}/corpus_ids.txt')
    
    if not corpus_ids:
        return
    
    print(f"Processing {len(corpus_ids)} papers individually...\n")
    
    # Define fields to retrieve
    fields = ['corpusId', 'embedding.specter_v2']
    
    embeddings = {}
    failed_items = []
    no_embedding_count = 0
    not_found_count = 0
    
    # First pass: Process each paper individually
    for i, corpus_id in enumerate(corpus_ids, 1):
        paper_id = format_corpus_id(corpus_id)
        paper, error_code = get_paper(paper_id, headers, fields)
        # print(f"### DEBUG: {paper}")
        
        if paper:
            corpus_id_str = str(paper.get('corpusId'))
            
            # Extract embedding if available
            embedding_data = paper.get('embedding')
            # print(f"### DEBUG: {paper.get('embedding')}")
            # print(f"### DEBUG: {embedding_data}")
            if embedding_data and embedding_data.get('model') == 'specter_v2' and 'vector' in embedding_data:
                embeddings[corpus_id_str] = embedding_data['vector']
                if i % 100 == 0:
                    print(f"  [{i}/{len(corpus_ids)}] ✓ {corpus_id}")
            else:
                no_embedding_count += 1
                if i % 100 == 0 or no_embedding_count <= 10:
                    print(f"  [{i}/{len(corpus_ids)}] ⚠ No embedding for {corpus_id}")
        elif error_code == 404:
            not_found_count += 1
            if i % 100 == 0 or not_found_count <= 10:
                print(f"  [{i}/{len(corpus_ids)}] ✗ Not found: {corpus_id}")
        else:
            # Retriable error
            failed_items.append((corpus_id, error_code))
            print(f"  [{i}/{len(corpus_ids)}] ⚠ Failed (will retry): {corpus_id} [Error: {error_code}]")
        
        time.sleep(rate_limit_delay)
    
    # Retry failed requests
    if failed_items:
        print(f"\n{'='*60}")
        print(f"First pass complete. Retrying {len(failed_items)} failed requests...")
        print(f"{'='*60}")
        
        successful_retries, still_failed = retry_failed_requests(
            failed_items, headers, fields, format_corpus_id, rate_limit_delay
        )
        
        # Add successful retries to embeddings
        for corpus_id, paper in successful_retries:
            corpus_id_str = str(paper.get('corpusId'))
            embedding_data = paper.get('embedding')
            if embedding_data and 'specter_v2' in embedding_data:
                embeddings[corpus_id_str] = embedding_data['specter_v2']
            else:
                no_embedding_count += 1
        
        # Save failed corpus IDs with timestamp
        if still_failed:
            failed_file = save_failed_items(still_failed, 'corpus_ids_embeddings', timestamp)
            print(f"\n⚠ {len(still_failed)} corpus IDs still failed after retries")
            print(f"  Saved to: {failed_file}")
    
    # Save embeddings to pickle file
    output_file = f'{DATA_PATH}/paper_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Embeddings retrieved: {len(embeddings)}/{len(corpus_ids)}")
    print(f"  No embedding available: {no_embedding_count}")
    print(f"  Not found (404): {not_found_count}")
    if failed_items:
        print(f"  Failed after retries: {len(still_failed)}")
    print(f"\nOutput:")
    print(f"  {output_file}")
    print(f"{'='*60}")

def fetch_papers_metadata(rate_limit_delay=1):
    """
    Fetch paper metadata for corpus IDs individually with automatic retry.
    Saves results as a JSON file with full paper metadata.
    
    Args:
        rate_limit_delay: Delay between requests in seconds
    """
    timestamp = get_timestamp()
    headers = load_headers()
    corpus_ids = load_corpus_ids(f'{DATA_PATH}/corpus_ids.txt')
    
    if not corpus_ids:
        return
    
    print(f"Processing {len(corpus_ids)} papers individually...\n")
    
    fields = [
        'corpusId', 'paperId', 'title', 'year', 'authors',
        'citationCount', 'abstract', 'venue', 'publicationDate',
        'referenceCount', 'influentialCitationCount', 'fieldsOfStudy',
        'publicationTypes', 'journal', 'isOpenAccess'
    ]
    
    paper_metadata = []
    failed_items = []
    not_found_count = 0
    
    # First pass
    for i, corpus_id in enumerate(corpus_ids, 1):
        paper_id = format_corpus_id(corpus_id)
        paper, error_code = get_paper(paper_id, headers, fields)
        
        if paper:
            paper_metadata.append(paper)
            if i % 100 == 0:
                print(f"  [{i}/{len(corpus_ids)}] Retrieved metadata")
        elif error_code == 404:
            not_found_count += 1
        else:
            failed_items.append((corpus_id, error_code))
            print(f"  [{i}/{len(corpus_ids)}] ⚠ Failed (will retry): {corpus_id}")
        
        time.sleep(rate_limit_delay)
    
    # Retry failed requests
    if failed_items:
        successful_retries, still_failed = retry_failed_requests(
            failed_items, headers, fields, format_corpus_id, rate_limit_delay
        )
        
        for corpus_id, paper in successful_retries:
            paper_metadata.append(paper)
        
        if still_failed:
            failed_file = save_failed_items(still_failed, 'corpus_ids_metadata', timestamp)
            print(f"\n⚠ {len(still_failed)} corpus IDs still failed after retries")
            print(f"  Saved to: {failed_file}")
    
    # Save to JSON
    output_file = f'{DATA_PATH}/paper_metadata.json'
    with open(output_file, 'w') as f:
        json.dump(paper_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Metadata retrieved: {len(paper_metadata)}/{len(corpus_ids)} papers")
    print(f"  Not found (404): {not_found_count}")
    if failed_items:
        print(f"  Failed after retries: {len(still_failed)}")
    print(f"\nOutput:")
    print(f"  {output_file}")
    print(f"{'='*60}")

def fetch_corpus_ids_from_dois(rate_limit_delay=1):
    """
    Fetch corpus IDs from DOIs individually with automatic retry.
    Saves results as JSON (with full mapping) and text file (corpus IDs only).
    
    Args:
        rate_limit_delay: Delay between requests in seconds
    """
    timestamp = get_timestamp()
    headers = load_headers()
    dois = load_dois('data/seed-paper-list.txt')
    
    if not dois:
        return
    
    print(f"Processing {len(dois)} DOIs individually...\n")
    
    fields = ['corpusId', 'paperId', 'title']
    
    results = []
    failed_items = []
    not_found_dois = []
    
    # First pass
    for i, doi in enumerate(dois, 1):
        paper_id = format_doi(doi)
        paper, error_code = get_paper(paper_id, headers, fields)
        
        if paper and paper.get('corpusId'):
            result = {
                'doi': doi,
                'corpusId': paper.get('corpusId'),
                'paperId': paper.get('paperId'),
                'title': paper.get('title')
            }
            results.append(result)
            print(f"  [{i}/{len(dois)}] ✓ {doi[:50]}... → Corpus ID: {result['corpusId']}")
        elif error_code == 404:
            not_found_dois.append(doi)
            print(f"  [{i}/{len(dois)}] ✗ {doi[:50]}... → Not found")
        else:
            failed_items.append((doi, error_code))
            print(f"  [{i}/{len(dois)}] ⚠ Failed (will retry): {doi[:50]}...")
        
        time.sleep(rate_limit_delay)
    
    # Retry failed requests
    if failed_items:
        successful_retries, still_failed = retry_failed_requests(
            failed_items, headers, fields, format_doi, rate_limit_delay
        )
        
        for doi, paper in successful_retries:
            if paper.get('corpusId'):
                result = {
                    'doi': doi,
                    'corpusId': paper.get('corpusId'),
                    'paperId': paper.get('paperId'),
                    'title': paper.get('title')
                }
                results.append(result)
        
        # Remaining failures
        for doi, error_code in still_failed:
            not_found_dois.append(doi)
    
    # Save results
    json_file = f'{DATA_PATH}/corpus_ids.json'
    txt_file = f'{DATA_PATH}/corpus_ids.txt'
    
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(txt_file, 'w') as f:
        for result in results:
            f.write(f"{result['corpusId']}\n")
    
    # Save failed DOIs with timestamp
    if not_found_dois or (failed_items and still_failed):
        all_failed = [(doi, 404) for doi in not_found_dois] + (still_failed if failed_items else [])
        failed_file = save_failed_items(all_failed, 'dois', timestamp)
        print(f"\n⚠ {len(all_failed)} DOIs failed")
        print(f"  Saved to: {failed_file}")
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Success: {len(results)}/{len(dois)} corpus IDs retrieved")
    print(f"  Not found/Failed: {len(not_found_dois) + (len(still_failed) if failed_items else 0)} DOIs")
    print(f"\nOutput:")
    print(f"  {json_file}")
    print(f"  {txt_file}")
    print(f"{'='*60}")

def fetch_embeddings_and_metadata(rate_limit_delay=1):
    """
    Fetch both embeddings and metadata for corpus IDs individually with automatic retry.
    Saves embeddings as pickle and metadata as JSON.
    
    Args:
        rate_limit_delay: Delay between requests in seconds
    """
    timestamp = get_timestamp()
    headers = load_headers()
    corpus_ids = load_corpus_ids(f'{DATA_PATH}/corpus_ids.txt')
    
    if not corpus_ids:
        return
    
    print(f"Processing {len(corpus_ids)} papers individually...\n")
    
    fields = [
        'corpusId', 'paperId', 'title', 'year', 'authors',
        'citationCount', 'abstract', 'embedding.specter_v2'
    ]
    
    embeddings = {}
    paper_metadata = []
    failed_items = []
    
    # First pass
    for i, corpus_id in enumerate(corpus_ids, 1):
        paper_id = format_corpus_id(corpus_id)
        paper, error_code = get_paper(paper_id, headers, fields)
        
        if paper:
            corpus_id_str = str(paper.get('corpusId'))
            
            # Extract embedding
            embedding_data = paper.get('embedding')
            if embedding_data and 'specter_v2' in embedding_data:
                embeddings[corpus_id_str] = embedding_data['specter_v2']
            
            # Store metadata
            metadata = {k: v for k, v in paper.items() if k != 'embedding'}
            metadata['has_embedding'] = corpus_id_str in embeddings
            paper_metadata.append(metadata)
            
            if i % 100 == 0:
                print(f"  [{i}/{len(corpus_ids)}] Processed")
        elif error_code != 404:
            failed_items.append((corpus_id, error_code))
        
        time.sleep(rate_limit_delay)
    
    # Retry failed requests
    if failed_items:
        successful_retries, still_failed = retry_failed_requests(
            failed_items, headers, fields, format_corpus_id, rate_limit_delay
        )
        
        for corpus_id, paper in successful_retries:
            corpus_id_str = str(paper.get('corpusId'))
            
            embedding_data = paper.get('embedding')
            if embedding_data and 'specter_v2' in embedding_data:
                embeddings[corpus_id_str] = embedding_data['specter_v2']
            
            metadata = {k: v for k, v in paper.items() if k != 'embedding'}
            metadata['has_embedding'] = corpus_id_str in embeddings
            paper_metadata.append(metadata)
        
        if still_failed:
            failed_file = save_failed_items(still_failed, 'corpus_ids_combined', timestamp)
            print(f"\n⚠ {len(still_failed)} corpus IDs still failed after retries")
            print(f"  Saved to: {failed_file}")
    
    # Save embeddings
    embeddings_file = f'{DATA_PATH}/paper_embeddings.pkl'
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save metadata
    metadata_file = f'{DATA_PATH}/paper_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(paper_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Embeddings: {len(embeddings)}/{len(corpus_ids)}")
    print(f"  Metadata: {len(paper_metadata)}/{len(corpus_ids)}")
    print(f"\nOutput:")
    print(f"  {embeddings_file}")
    print(f"  {metadata_file}")
    print(f"{'='*60}")

# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    print("=" * 60)
    print("Semantic Scholar API - Individual Query Interface")
    print("=" * 60)
    print("\nChoose operation:")
    print("1. Fetch corpus IDs from DOIs")
    print("2. Fetch embeddings only")
    print("3. Fetch metadata only")
    print("4. Fetch embeddings + metadata")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    print("\n" + "=" * 60)
    
    if choice == "1":
        fetch_corpus_ids_from_dois()
    elif choice == "2":
        fetch_paper_embeddings()
    elif choice == "3":
        fetch_papers_metadata()
    elif choice == "4":
        fetch_embeddings_and_metadata()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    from datetime import datetime
    main()
    print(f"\n{'='*60}")
    print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")