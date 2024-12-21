import re
import json
import logging
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Set

import numpy as np
import xxhash
from difflib import SequenceMatcher
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_duplicates(strings: List[str]) -> Tuple[List[str], List[str]]:
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


@dataclass
class MinHashParams:
    num_perm: int = 128  # Number of permutations
    ngram_size: int = 3  # Size of character n-grams
    hash_bits: int = 64  # Hash size in bits
    bucket_size: int = 4  # LSH bucket size


class ChunkRepo:
    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the ChunkRepo object by reading the text from the file
        and preparing the chunks based on the specified chunk size and overlap.

        Args:
            file_path (str): Path to the text file to be chunked.
            chunk_size (int): Number of characters in each chunk (default: 1000).
            chunk_overlap (int): Number of overlapping characters between consecutive chunks (default: 200).
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text = self._get_text()
        self.chunks = self._chunk()

    def _get_text(self) -> str:
        """
        Read and return the text from the specified file with robust error handling.

        Returns:
            str: Content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If there are permission issues accessing the file.
            UnicodeDecodeError: If the file cannot be decoded with the specified encoding.
            Exception: For any other exceptions that may occur.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                logging.info(f"Successfully opened file: {self.file_path}")
                content = file.read()
                if not content:
                    logging.warning(f"The file {self.file_path} is empty.")
                return content
        except FileNotFoundError:
            logging.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except PermissionError:
            logging.error(f"Permission denied: {self.file_path}")
            raise PermissionError(f"Permission denied: {self.file_path}")
        except UnicodeDecodeError:
            logging.error(f"Failed to decode file with utf-8 encoding: {self.file_path}")
            raise UnicodeDecodeError(f"Failed to decode file with utf-8 encoding: {self.file_path}")
        except Exception as e:
            logging.error(f"Error reading file {self.file_path}: {str(e)}")
            raise Exception(f"Error reading file {self.file_path}: {str(e)}")

    def _chunk(self) -> List[str]:
        """
        Split the text into chunks with specified size and overlap.

        Returns:
            List[str]: List of text chunks.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_text(self.text)
            logging.info(f"Successfully split text into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logging.error(f"Error during text chunking: {str(e)}")
            raise Exception(f"Error during text chunking: {str(e)}")


class ChunkURLMapper:
    def __init__(self, clean_chunks: List[str], raw_content: str, 
                 similarity_threshold: float = 0.7,
                 max_workers: int = None,
                 log_level: int = logging.INFO):
        """
        Initialize the mapper with clean chunks and raw content containing URLs.

        Args:
            clean_chunks (List[str]): List of pre-processed text chunks without URLs.
            raw_content (str): Raw string content with URLs and associated content separated by '---'.
            similarity_threshold (float): Minimum similarity score to consider chunks matching.
            max_workers (int, optional): Maximum number of parallel processes (None = CPU count).
            log_level (int, optional): Logging level (default: logging.INFO).
        """
        self.clean_chunks = clean_chunks
        self.similarity_threshold = similarity_threshold
        self.max_workers = max_workers or os.cpu_count()
        
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
        """
        Configure logging with both file and console handlers.

        Args:
            log_level (int): Logging level.
        """
        self.logger = logging.getLogger('ChunkURLMapper')
        self.logger.setLevel(log_level)
        
        # Avoid adding multiple handlers if already added
        if not self.logger.handlers:
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
        """
        Parse raw content into list of (url, content) tuples.

        Args:
            raw_content (str): Raw string content with URLs and associated content separated by '---'.

        Returns:
            List[Tuple[str, str]]: List of tuples containing URL and its associated content.
        """
        self.logger.debug("Parsing raw content...")
        # Split entries by '---'
        entries = [entry.strip() for entry in raw_content.split('---') if entry.strip()]
        pairs = []
        
        url_pattern = re.compile(r'^URL:\s*(https?://\S+)', re.MULTILINE)
        
        for entry in entries:
            url_match = url_pattern.search(entry)
            if url_match:
                url = url_match.group(1).strip()
                # Remove the URL line to get the content
                content = url_pattern.sub('', entry).strip()
                if content:
                    pairs.append((url, content))
                    self.logger.debug(f"Parsed URL: {url} with content length: {len(content)}")
                else:
                    self.logger.warning(f"No content found for URL: {url}")
            else:
                self.logger.warning(f"No URL found in entry: {entry[:50]}...")
        
        return pairs

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity ratio between two text strings.

        Args:
            text1 (str): First text string.
            text2 (str): Second text string.

        Returns:
            float: Similarity ratio.
        """
        return SequenceMatcher(None, text1, text2).ratio()

    def _process_chunk_batch(self, args: Tuple[str, List[str], List[str]]) -> Tuple[str, List[str]]:
        """
        Process a batch of chunks for a single URL.

        Args:
            args (Tuple[str, List[str], List[str]]): Tuple containing URL, list of clean chunks batch, and list of raw chunks.

        Returns:
            Tuple[str, List[str]]: URL and list of matching clean chunks.
        """
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
        """
        Chunk raw content using the specified chunk size and overlap.

        Args:
            content (str): Raw content associated with a URL.

        Returns:
            List[str]: List of raw content chunks.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            raw_chunks = text_splitter.split_text(content)
            self.logger.debug(f"Split raw content into {len(raw_chunks)} chunks.")
            return raw_chunks
        except Exception as e:
            self.logger.error(f"Error during raw content chunking: {str(e)}")
            return []

    def create_mapping(self) -> Dict[str, List[str]]:
        """
        Create mapping between URLs and matching clean chunks using parallel processing.

        Returns:
            Dict[str, List[str]]: Dictionary with URLs as keys and lists of matching clean chunks as values.
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
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing a chunk batch: {str(e)}")
        
        # Combine results
        for url, chunks in results:
            if url not in mapping:
                mapping[url] = []
            mapping[url].extend(chunks)
        
        # Remove duplicates while preserving order
        for url in mapping:
            mapping[url], duplicates = process_duplicates(mapping[url])
            if duplicates:
                self.logger.info(f"Removed {len(duplicates)} duplicate chunks for URL: {url}")
        
        processing_time = time.time() - start_time
        self.logger.info(f"Mapping creation completed in {processing_time:.2f} seconds")
        self.logger.info(f"Found matches for {len(mapping)} URLs")
        
        return mapping

    def export_to_json(self, mapping: Dict[str, List[str]], output_path: str) -> None:
        """
        Export the mapping to a JSON file.

        Args:
            mapping (Dict[str, List[str]]): The URL to chunks mapping dictionary.
            output_path (str): Path where JSON file should be saved.
        """
        self.logger.info(f"Exporting mapping to {output_path}")
        start_time = time.time()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Export completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export mapping to JSON: {str(e)}")


# Main Execution Code
def main():
    # File paths
    cleaned_file_path = 'Cleaned_MOHAP.txt'
    raw_file_path = 'crawled_content.txt'
    output_json_path = 'url_chunk_mapping_1000.json'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Check if input files exist
    if not os.path.exists(cleaned_file_path):
        logging.error(f"Cleaned file not found: {cleaned_file_path}")
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_file_path}")
    if not os.path.exists(raw_file_path):
        logging.error(f"Raw crawled content file not found: {raw_file_path}")
        raise FileNotFoundError(f"Raw crawled content file not found: {raw_file_path}")
    
    # Initialize ChunkRepo with the cleaned text file
    chunk_repo = ChunkRepo(
        file_path=cleaned_file_path,
        chunk_size=1000,
        chunk_overlap=200
    )
    print(f"length of chunks before duplicates({len(chunk_repo.chunks)})")
    
    # Optionally, remove duplicate chunks if needed
    unique_chunks, duplicate_chunks = process_duplicates(chunk_repo.chunks)
    if duplicate_chunks:
        logging.info(f"Removed {len(duplicate_chunks)} duplicate chunks.")
    else:
        logging.info("No duplicate chunks found.")
    
    # Read raw crawled content from the file
    try:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        logging.info(f"Successfully read raw crawled content from {raw_file_path}")
    except Exception as e:
        logging.error(f"Error reading raw crawled content file: {str(e)}")
        raise Exception(f"Error reading raw crawled content file: {str(e)}")
    
    # Initialize ChunkURLMapper with the unique chunks and raw content
    mapper = ChunkURLMapper(
        clean_chunks=unique_chunks,
        raw_content=raw_content,
        similarity_threshold=0.7,  # Adjust based on your needs
        max_workers=4,             # Adjust based on your CPU
        log_level=logging.INFO
    )
    
    # Create the mapping
    # mapping = mapper.create_mapping()
    
    # Export the mapping to a JSON file
    # mapper.export_to_json(mapping, output_json_path)
    
    print(f"Mapping completed and saved to {output_json_path}")


if __name__ == "__main__":
    main()
