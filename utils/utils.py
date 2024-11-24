def process_duplicates(strings):
    """
    Remove duplicates from a list of strings and return the duplicate items.
    
    Args:
        strings (list): List of strings to process
    
    Returns:
        tuple: (list of unique strings, list of duplicate strings)
    """
    seen = set()
    duplicates = set()
    
    # Find duplicates while preserving order
    result = []
    for item in strings:
        if item in seen:
            duplicates.add(item)
        else:
            result.append(item)
            seen.add(item)
    
    return result, sorted(list(duplicates))

from difflib import SequenceMatcher
import json
from typing import List, Dict, Tuple
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
import time
from tqdm import tqdm
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter




class chunk_repo:
    def __init__(self, file_path, chunk_size=528, chunk_overlap=128):
        print(file_path)
        self.text = self._get_text(file_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = self._chunk()
    
    
    def _chunk(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(self.text)
        return chunks
    
    def _get_text(self , file_path):
        # Read the content of the file
        with open(file_path, 'r') as file:
            return file.read()



class ChunkURLMapper:
    def __init__(self, clean_chunks: List[str], raw_content: str, 
                 similarity_threshold: float = 0.7,
                 max_workers: int = None,
                 log_level: int = logging.INFO):
        """
        Initialize the mapper with clean chunks and raw content containing URLs.
        
        Args:
            clean_chunks: List of pre-processed text chunks without URLs
            raw_content: Raw string content with URLs in format: url\n\ncontent\n\n
            similarity_threshold: Minimum similarity score to consider chunks matching
            max_workers: Maximum number of parallel processes (None = CPU count)
            log_level: Logging level (default: logging.INFO)
        """
        self.clean_chunks = clean_chunks
        self.similarity_threshold = similarity_threshold
        self.max_workers = max_workers
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Parse content and log basic statistics
        self.logger.info("Initializing ChunkURLMapper...")
        start_time = time.time()
        self.url_content_pairs = self._parse_raw_content(raw_content)
        self.logger.info(f"Found {len(self.url_content_pairs)} URL-content pairs")
        self.logger.info(f"Working with {len(clean_chunks)} clean chunks")
        self.logger.info(f"Initialization took {time.time() - start_time:.2f} seconds")

    def _setup_logging(self, log_level: int) -> None:
        """Configure logging with both file and console handlers."""
        self.logger = logging.getLogger('ChunkURLMapper')
        self.logger.setLevel(log_level)
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # File handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'logs/chunk_mapper_{timestamp}.log')
        fh.setLevel(log_level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _parse_raw_content(self, raw_content: str) -> List[Tuple[str, str]]:
        """Parse raw content into list of (url, content) tuples."""
        self.logger.debug("Parsing raw content...")
        entries = raw_content.strip().split('\n\n')
        pairs = []
        
        for i in range(0, len(entries), 2):
            if i + 1 < len(entries):
                url = entries[i].strip()
                content = entries[i + 1].strip()
                pairs.append((url, content))
        
        return pairs

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity ratio between two text strings."""
        return SequenceMatcher(None, text1, text2).ratio()

    def _process_chunk_batch(self, args: Tuple[str, List[str], List[str]]) -> Tuple[str, List[str]]:
        """Process a batch of chunks for a single URL."""
        url, clean_chunks_batch, raw_chunks = args
        matching_chunks = []
        
        for clean_chunk in clean_chunks_batch:
            best_similarity = 0
            for raw_chunk in raw_chunks:
                similarity = self._calculate_similarity(clean_chunk, raw_chunk)
                if similarity > best_similarity:
                    best_similarity = similarity
            
            if best_similarity >= self.similarity_threshold:
                matching_chunks.append(clean_chunk)
        
        return url, matching_chunks

    def _chunk_raw_content(self, content: str) -> List[str]:
        """Chunk raw content using the same parameters as the clean chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=528,
            chunk_overlap=128
        )
        return text_splitter.split_text(content)

    def create_mapping(self) -> Dict[str, List[str]]:
        """
        Create mapping between URLs and matching clean chunks using parallel processing.
        
        Returns:
            Dict with URLs as keys and lists of matching clean chunks as values
        """
        self.logger.info("Starting mapping creation...")
        start_time = time.time()
        mapping = {}
        
        # Prepare tasks for parallel processing
        tasks = []
        batch_size = max(1, len(self.clean_chunks) // (os.cpu_count() or 1))
        
        for url, raw_content in self.url_content_pairs:
            raw_chunks = self._chunk_raw_content(raw_content)
            self.logger.debug(f"Processing URL: {url} with {len(raw_chunks)} raw chunks")
            
            # Create batches of clean chunks for parallel processing
            for i in range(0, len(self.clean_chunks), batch_size):
                batch = self.clean_chunks[i:i + batch_size]
                tasks.append((url, batch, raw_chunks))
        
        # Process chunks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_chunk_batch, task) for task in tasks]
            
            # Use tqdm for progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                results.append(future.result())
        
        # Combine results
        for url, chunks in results:
            if url not in mapping:
                mapping[url] = []
            mapping[url].extend(chunks)
        
        # Remove duplicates while preserving order
        for url in mapping:
            mapping[url] = list(dict.fromkeys(mapping[url]))
        
        processing_time = time.time() - start_time
        self.logger.info(f"Mapping creation completed in {processing_time:.2f} seconds")
        self.logger.info(f"Found matches for {len(mapping)} URLs")
        
        return mapping

    def export_to_json(self, output_path: str) -> None:
        """
        Create mapping and save to JSON file.
        
        Args:
            output_path: Path where JSON file should be saved
        """
        self.logger.info(f"Exporting mapping to {output_path}")
        start_time = time.time()
        
        mapping = self.create_mapping()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Export completed in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Results saved to {output_path}")


import re
from typing import Dict, List
import numpy as np
from collections import defaultdict
import xxhash
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

@dataclass
class MinHashParams:
    num_perm: int = 128  # Number of permutations
    ngram_size: int = 3  # Size of character n-grams
    hash_bits: int = 64  # Hash size in bits
    bucket_size: int = 4  # LSH bucket size

from typing import List, Dict, Tuple, Set
import json
import logging
from datetime import datetime
import time
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from collections import defaultdict
import xxhash
from dataclasses import dataclass
import re

@dataclass
class MinHashParams:
    num_perm: int = 128  # Number of permutations
    ngram_size: int = 3  # Size of character n-grams
    hash_bits: int = 64  # Hash size in bits
    bucket_size: int = 4  # LSH bucket size
    
class FastChunkURLMapper:
    def __init__(
        self, 
        clean_chunks: List[str], 
        raw_content: str,
        similarity_threshold: float = 0.7,
        minhash_params: MinHashParams = None,
        max_workers: int = None,
        log_level: int = logging.INFO
    ):
        self.clean_chunks = clean_chunks
        self.similarity_threshold = similarity_threshold
        self.max_workers = max_workers or os.cpu_count()
        self.minhash_params = minhash_params or MinHashParams()
        
        self._setup_logging(log_level)
        self.timers = defaultdict(float)
        
        start = time.time()
        self.logger.info("Initializing FastChunkURLMapper...")
        self.url_content_pairs = self._parse_raw_content(raw_content)
        self.logger.info(f"Found {len(self.url_content_pairs)} URL-content pairs")
        self.logger.info(f"Working with {len(clean_chunks)} clean chunks")
        
        self.clean_chunk_signatures = self._compute_minhash_signatures(clean_chunks)
        self.timers['init'] = time.time() - start
        self.logger.info(f"Initialization took {self.timers['init']:.2f} seconds")

    def _setup_logging(self, log_level: int) -> None:
        self.logger = logging.getLogger('FastChunkURLMapper')
        self.logger.setLevel(log_level)
        
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'logs/chunk_mapper_{timestamp}.log')
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        for handler in [fh, ch]:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            self.logger.addHandler(handler)

    def _create_ngrams(self, text: str) -> Set[str]:
        text = re.sub(r'\s+', ' ', text.lower())
        return {text[i:i+self.minhash_params.ngram_size] 
                for i in range(len(text) - self.minhash_params.ngram_size + 1)}

    def _compute_minhash_signature(self, text: str) -> np.ndarray:
        ngrams = self._create_ngrams(text)
        signature = np.full(self.minhash_params.num_perm, np.inf)
        
        for ngram in ngrams:
            hash_val = xxhash.xxh64(ngram).intdigest()
            for i in range(self.minhash_params.num_perm):
                h = xxhash.xxh64(ngram + str(i)).intdigest()
                signature[i] = min(signature[i], h)
        return signature

    def _compute_minhash_signatures(self, texts: List[str]) -> np.ndarray:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            signatures = list(tqdm(
                executor.map(self._compute_minhash_signature, texts),
                total=len(texts),
                desc="Computing signatures"
            ))
        return np.array(signatures)

    def _estimate_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        return np.mean(sig1 == sig2)

    def _create_lsh_buckets(self, signatures: np.ndarray) -> Dict[tuple, List[int]]:
        buckets = defaultdict(list)
        num_bands = self.minhash_params.num_perm // self.minhash_params.bucket_size
        
        for idx, sig in enumerate(signatures):
            for band in range(num_bands):
                start = band * self.minhash_params.bucket_size
                end = start + self.minhash_params.bucket_size
                band_sig = tuple(sig[start:end])
                buckets[band_sig].append(idx)
        return buckets

    def _find_similar_chunks(
        self, 
        query_signature: np.ndarray, 
        reference_signatures: np.ndarray,
        lsh_buckets: Dict[tuple, List[int]]
    ) -> List[int]:
        candidates = set()
        num_bands = self.minhash_params.num_perm // self.minhash_params.bucket_size
        
        for band in range(num_bands):
            start = band * self.minhash_params.bucket_size
            end = start + self.minhash_params.bucket_size
            band_sig = tuple(query_signature[start:end])
            candidates.update(lsh_buckets.get(band_sig, []))
        
        similar_chunks = []
        for candidate_idx in candidates:
            similarity = self._estimate_similarity(
                query_signature, 
                reference_signatures[candidate_idx]
            )
            if similarity >= self.similarity_threshold:
                similar_chunks.append(candidate_idx)
        return similar_chunks

    def _parse_raw_content(self, raw_content: str) -> List[Tuple[str, str]]:
        url_content_pairs = []
        lines = raw_content.splitlines()
        for i in range(0, len(lines), 3): # Increment by 3 to skip "---"
            url = lines[i].strip()
            content = lines[i+1].strip() if i + 1 < len(lines) else "" # Handle potential IndexError
            if url and content:
                url_content_pairs.append((url, content))
        return url_content_pairs


    def _process_url_content(
        self, 
        url_content: Tuple[str, str], 
        clean_signatures: np.ndarray,
        lsh_buckets: Dict[tuple, List[int]]
    ) -> Tuple[str, List[str]]:
        url, content = url_content
        content_signature = self._compute_minhash_signature(content)
        similar_indices = self._find_similar_chunks(
            content_signature, 
            clean_signatures,
            lsh_buckets
        )
        return url, [self.clean_chunks[i] for i in similar_indices]

    def create_mapping(self) -> Dict[str, List[str]]:
        start = time.time()
        self.logger.info("Creating LSH buckets...")
        lsh_buckets = self._create_lsh_buckets(self.clean_chunk_signatures)
        
        self.logger.info("Processing URL-content pairs...")
        mapping = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._process_url_content, 
                    pair, 
                    self.clean_chunk_signatures,
                    lsh_buckets
                )
                for pair in self.url_content_pairs
            ]
            
            for future in tqdm(futures, desc="Mapping chunks"):
                url, chunks = future.result()
                if chunks:
                    mapping[url] = chunks
        
        self.timers['mapping'] = time.time() - start
        self.logger.info(
            f"Mapping completed in {self.timers['mapping']:.2f} seconds. "
            f"Found matches for {len(mapping)} URLs"
        )
        return mapping

    def export_to_json(self, output_path: str) -> None:
        start = time.time()
        self.logger.info(f"Exporting mapping to {output_path}")
        mapping = self.create_mapping()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        self.timers['export'] = time.time() - start
        self.logger.info(
            f"Export completed in {self.timers['export']:.2f} seconds. "
            f"Total processing time: {sum(self.timers.values()):.2f} seconds"
        )