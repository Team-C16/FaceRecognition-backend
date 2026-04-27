"""
Face Database - stores and searches face embeddings using Qdrant.

Storage format (Qdrant Point):
- id: UUID (deterministically generated from person's name)
- vector: 512-d averaged embedding
- payload: 
    {
        "name": "Ali Yilmaz",
        "num_samples": 8,
        "enrolled_at": "2026-04-15T18:00:00",
        "updated_at": "2026-04-15T18:00:00"
    }
"""

import os
import uuid
import numpy as np
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models

try:
    from config import (
        QDRANT_HOST,
        QDRANT_PORT,
        QDRANT_COLLECTION,
        QDRANT_VECTOR_SIZE,
        ENROLLED_FACES_DIR,
        RECOGNITION_THRESHOLD,
    )
except ImportError:
    from .config import (
        QDRANT_HOST,
        QDRANT_PORT,
        QDRANT_COLLECTION,
        QDRANT_VECTOR_SIZE,
        ENROLLED_FACES_DIR,
        RECOGNITION_THRESHOLD,
    )


class FaceDatabase:
    """
    Manages face embeddings storage and search using Qdrant Vector DB.
    Provides fast and scalable similarity search.
    """

    def __init__(self):
        self._ensure_dirs()
        self.collection_name = QDRANT_COLLECTION
        
        # Connect to Qdrant
        print(f"[FaceDB] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
            self._ensure_collection()
            
            # Count existing points
            count_result = self.client.count(collection_name=self.collection_name)
            print(f"[FaceDB] Connected. Collection '{self.collection_name}' has {count_result.count} enrolled people.")
        except Exception as e:
            print(f"[FaceDB] ERROR: Could not connect to Qdrant: {e}")
            print(f"[FaceDB] Ensure Qdrant is running on {QDRANT_HOST}:{QDRANT_PORT}")
            raise

    def _ensure_dirs(self):
        """Create data directories if they don't exist."""
        os.makedirs(ENROLLED_FACES_DIR, exist_ok=True)

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"[FaceDB] Creating new collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=QDRANT_VECTOR_SIZE,
                    distance=models.Distance.COSINE
                ),
            )
            
            # Create a payload index on the 'name' field for faster filtering if needed
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="name",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _get_id_from_name(self, name):
        """Generate a deterministic UUID from the person's name."""
        return str(uuid.uuid5(uuid.NAMESPACE_OID, name))

    def enroll(self, name, embeddings):
        """
        Enroll a person with their face embeddings.

        Args:
            name: Person's name (unique identifier)
            embeddings: list of numpy arrays (512-d each)

        Returns:
            dict with enrollment result
        """
        if len(embeddings) < 1:
            raise ValueError("At least 1 embedding required")

        # Average all embeddings for robust representation
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        now = datetime.now().isoformat()
        point_id = self._get_id_from_name(name)

        payload = {
            "name": name,
            "num_samples": len(embeddings),
            "enrolled_at": now,
            "updated_at": now,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=avg_embedding.tolist(),
                    payload=payload
                )
            ]
        )
        
        print(f"[FaceDB] Enrolled '{name}' with {len(embeddings)} samples (Qdrant ID: {point_id[:8]}...)")

        return {
            "name": name,
            "num_samples": len(embeddings),
            "enrolled_at": now,
        }

    def update(self, name, new_embeddings):
        """
        Update an existing person's embeddings by adding new samples.
        Recomputes the average embedding.

        Args:
            name: Person's name
            new_embeddings: list of new numpy arrays to add
        """
        point_id = self._get_id_from_name(name)
        
        # Retrieve existing point
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=True
            )
        except Exception as e:
            raise ValueError(f"Failed to query Qdrant: {e}")

        if not records:
            raise ValueError(f"Person '{name}' not found in Qdrant")

        record = records[0]
        old_embedding = np.array(record.vector)
        payload = record.payload
        old_count = payload.get("num_samples", 1)

        # Weighted average: keep old samples' contribution
        all_embeddings = [old_embedding * old_count] + list(new_embeddings)
        total_count = old_count + len(new_embeddings)

        avg_embedding = np.sum(all_embeddings, axis=0) / total_count
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        payload["num_samples"] = total_count
        payload["updated_at"] = datetime.now().isoformat()

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=avg_embedding.tolist(),
                    payload=payload
                )
            ]
        )
        
        print(f"[FaceDB] Updated '{name}': {total_count} total samples")

    def recognize(self, query_embedding, threshold=None):
        """
        Find the best matching person for a query embedding using Qdrant vector search.

        Args:
            query_embedding: numpy array (512-d)
            threshold: minimum similarity score (default from config)

        Returns:
            dict: {"name": str, "score": float, "matched": bool}
        """
        if threshold is None:
            threshold = RECOGNITION_THRESHOLD

        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=1,
                with_payload=True
            )
        except Exception as e:
            print(f"[FaceDB] Search error: {e}")
            return {"name": "error", "score": 0.0, "matched": False}

        if not search_result:
            return {"name": "unknown", "score": 0.0, "matched": False}

        best_hit = search_result[0]
        best_name = best_hit.payload.get("name", "unknown")
        best_score = best_hit.score  # Cosine similarity score
        
        matched = best_score >= threshold

        return {
            "name": best_name if matched else "unknown",
            "score": round(best_score, 4),
            "matched": matched,
        }

    def search_top_k(self, query_embedding, k=5):
        """
        Find top-k most similar people.

        Args:
            query_embedding: numpy array (512-d)
            k: number of results to return

        Returns:
            list of dicts: [{"name": str, "score": float}, ...]
        """
        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
            with_payload=True
        )

        scores = []
        for hit in search_result:
            scores.append({
                "name": hit.payload.get("name", "unknown"),
                "score": round(hit.score, 4)
            })

        return scores

    def get_all_people(self):
        """
        Get list of all enrolled people.

        Returns:
            list of dicts with person info (without embeddings)
        """
        # Using scroll API to get all points (suitable for thousands of records)
        people = []
        offset = None
        
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            for record in records:
                p = record.payload
                people.append({
                    "name": p.get("name", "unknown"),
                    "num_samples": p.get("num_samples", 0),
                    "enrolled_at": p.get("enrolled_at", ""),
                    "updated_at": p.get("updated_at", ""),
                })
                
            if offset is None:
                break
                
        return people

    def remove_person(self, name):
        """
        Remove a person from the database.

        Args:
            name: Person's name

        Returns:
            bool: True if removed, False if not found
        """
        point_id = self._get_id_from_name(name)
        
        # Check if exists first to return correct boolean
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=False
            )
            
            if not records:
                return False
                
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id],
                ),
            )
            print(f"[FaceDB] Removed '{name}'")
            return True
            
        except Exception as e:
            print(f"[FaceDB] Error removing person: {e}")
            return False

    def clear(self):
        """Remove all enrolled people by recreating the collection."""
        print("[FaceDB] Cleared all entries (Deleting and recreating collection)")
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
