"""
Face Recognizer using InsightFace (ArcFace model).
Runs on GPU for fast, high-accuracy face embedding extraction.

Model: buffalo_l (ArcFace-based)
- 99.8% accuracy on LFW benchmark
- 512-dimensional embedding vectors
- GPU-accelerated via ONNX Runtime
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
try:
    from config import INSIGHTFACE_MODEL, GPU_DEVICE_ID, DETECTION_SIZE
except ImportError:
    from .config import INSIGHTFACE_MODEL, GPU_DEVICE_ID, DETECTION_SIZE


class FaceRecognizer:
    """
    GPU-accelerated face recognition using InsightFace ArcFace.
    Extracts 512-dimensional face embeddings for comparison.
    """

    def __init__(self):
        print(f"[FaceRecognizer] Loading model: {INSIGHTFACE_MODEL}")
        print(f"[FaceRecognizer] GPU Device: {GPU_DEVICE_ID}")

        # Determine execution providers
        if GPU_DEVICE_ID >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            providers=providers,
        )
        self.app.prepare(ctx_id=GPU_DEVICE_ID, det_size=DETECTION_SIZE)

        print("[FaceRecognizer] Ready")

    def get_embedding(self, face_image):
        """
        Extract face embedding from an image.

        The image can be either:
        - A cropped face (will run detection first to align)
        - A full frame (will detect + align the largest face)

        Args:
            face_image: BGR image (numpy array)

        Returns:
            numpy array of shape (512,) or None if no face found
        """
        if face_image is None or face_image.size == 0:
            return None

        # Resize if very small (InsightFace needs reasonable size)
        h, w = face_image.shape[:2]
        if h < 112 or w < 112:
            scale = max(112 / h, 112 / w)
            face_image = cv2.resize(
                face_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

        faces = self.app.get(face_image)

        if len(faces) == 0:
            return None

        # Return embedding of the largest face
        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        embedding = faces[0].embedding
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def get_all_embeddings(self, image):
        """
        Extract embeddings for ALL faces in an image.

        Args:
            image: BGR image (numpy array)

        Returns:
            list of dicts: [{"embedding": np.array, "bbox": [x1,y1,x2,y2]}, ...]
        """
        if image is None or image.size == 0:
            return []

        faces = self.app.get(image)
        results = []

        for face in faces:
            emb = face.embedding / np.linalg.norm(face.embedding)
            results.append({
                "embedding": emb,
                "bbox": face.bbox.tolist(),
            })

        return results

    @staticmethod
    def compute_similarity(embedding1, embedding2):
        """
        Compute cosine similarity between two face embeddings.

        Args:
            embedding1: numpy array (512,)
            embedding2: numpy array (512,)

        Returns:
            float: cosine similarity score (0.0 to 1.0)
        """
        return float(np.dot(embedding1, embedding2))
