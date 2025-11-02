from tools.dynamo import db_client, s3_client, lambda_client
import numpy as np
import google.generativeai as genai
from fastapi import HTTPException
from typing import List, Dict
import tempfile
import json
import os
import logging
import faiss
from pydantic import BaseModel
class RetrievalResult(BaseModel):
    content: str
    summary: str
    section_title: str
    section_index: int
    score: float
    source_file: str
    original_file: str
    s3_key: str
    project: str
bucket_name = "ragtestsigmoyd"


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper functions
def get_embedding_from_query(query: str) -> np.ndarray:
    """Convert query to embedding vector using Google Gemini"""
    try:
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="RETRIEVAL_QUERY"  # Use RETRIEVAL_QUERY for queries
        )["embedding"]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        # logger.error(f"Error generating query embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {str(e)}")
    



async def list_project_files(user_id: str, project: str) -> List[str]:
    """List all files in a user's project directory"""
    try:
        prefix = f"{user_id}/{project}/"
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
       
        if 'Contents' not in response:
            return []
       
        # Extract unique file base names (without _metadata.json or _index.faiss suffixes)
        file_bases = set()
        for obj in response['Contents']:
            key = obj['Key']
            filename = key.split('/')[-1]  # Get just the filename
           
            if filename.endswith('_metadata.json'):
                base_name = filename.replace('_metadata.json', '')
                file_bases.add(base_name)
            elif filename.endswith('_index.faiss'):
                base_name = filename.replace('_index.faiss', '')
                file_bases.add(base_name)
       
        return list(file_bases)
   
    except Exception as e:
        logger.error(f"Error listing project files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing project files: {str(e)}")

async def load_file_index_and_metadata(user_id: str, project: str, file_base: str) -> tuple:
    """Load FAISS index and metadata for a specific file"""
    try:
        # S3 keys for the file's index and metadata
        index_key = f"{user_id}/{project}/{file_base}_index.faiss"
        metadata_key = f"{user_id}/{project}/{file_base}_metadata.json"
       
        # Download files to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp_index:
            s3_client.download_fileobj(bucket_name, index_key, tmp_index)
            tmp_index_path = tmp_index.name
       
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_metadata:
            s3_client.download_fileobj(bucket_name, metadata_key, tmp_metadata)
            tmp_metadata_path = tmp_metadata.name
       
        # Load FAISS index
        index = faiss.read_index(tmp_index_path)
       
        # Load metadata
        with open(tmp_metadata_path, 'r') as f:
            metadata = json.load(f)
       
        # Cleanup temp files
        os.unlink(tmp_index_path)
        os.unlink(tmp_metadata_path)
       
        return index, metadata, file_base
   
    except Exception as e:
        logger.error(f"Error loading index/metadata for {file_base}: {str(e)}")
        return None, None, file_base

async def search_single_file(index, metadata: Dict, file_base: str, query_embedding: np.ndarray,
                           top_k: int, project: str) -> List[RetrievalResult]:
    """Search a single FAISS index and return top-k results"""
    try:
        if index is None or metadata is None:
            return []
       
        # Perform similarity search
        scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
       
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                print(f"Invalid index {idx} for file {file_base}, skipping")
                continue
               
            # Get metadata for this chunk
            chunk_metadata = metadata[idx]
           
            result = RetrievalResult(
                content=chunk_metadata.get('full_content', ''),
                summary=chunk_metadata.get('summary', ''),
                section_title=chunk_metadata.get('section_title', ''),
                section_index=chunk_metadata.get('section_index', idx),
                score=float(score),
                source_file=file_base,
                original_file=chunk_metadata.get('original_file', file_base),
                s3_key=chunk_metadata.get('s3_key', ''),
                project=project
            )
            results.append(result)
       
        return results
   
    except Exception as e:
        logger.error(f"Error searching file {file_base}: {str(e)}")
        return []

def rerank_results(all_results: List[RetrievalResult], final_top_k: int, query: str = None) -> List[RetrievalResult]:
    """Rerank and filter results across all files with enhanced scoring"""
    if not all_results:
        return []
   
    # For Gemini embeddings, higher cosine similarity = better match
    # Sort by similarity score (higher is better)
    sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)
   
    # Optional: Add additional ranking factors
    if query:
        # You could add text-based relevance scoring here
        # For example, boost results where section_title matches query terms
        query_lower = query.lower()
        for result in sorted_results:
            title_boost = 0.1 if any(word in result.section_title.lower() for word in query_lower.split()) else 0
            result.score += title_boost
       
        # Re-sort after boosting
        sorted_results.sort(key=lambda x: x.score, reverse=True)
    print(f"reranked: {len(sorted_results)}, first : {sorted_results[0].score if sorted_results else 'N/A'}")
    return sorted_results[:final_top_k]