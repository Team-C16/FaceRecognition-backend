"""
Backend Server Configuration
"""

import os

# ---- Server ----
HOST = "0.0.0.0"
PORT = 8000

# ---- Face Recognition ----
# InsightFace model pack: "buffalo_l" (high accuracy), "buffalo_s" (faster, lower accuracy)
INSIGHTFACE_MODEL = "buffalo_l"

# GPU device ID (0 = first GPU, -1 = CPU)
GPU_DEVICE_ID = -1  # -1 = CPU (for laptop testing), 0 = first GPU

# Detection size for InsightFace (used during enrollment)
DETECTION_SIZE = (640, 640)

# ---- Recognition Thresholds ----
# Cosine similarity threshold for positive match (higher = stricter)
RECOGNITION_THRESHOLD = 0.4

# ---- Face Database ----
# Qdrant Vector Database Configuration
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "faces"
QDRANT_VECTOR_SIZE = 512

# ---- Enrollment ----
MIN_ENROLLMENT_IMAGES = 3      # Minimum images needed to enroll a person
MAX_ENROLLMENT_IMAGES = 15     # Maximum images per enrollment

# ---- Liveness / Security ----
LIVENESS_STRICT_THRESHOLD   = 0.90   # MiniFASNet confidence cutoff (must be >= this to be "real")
PROXIMITY_RATIO_LIMIT       = 0.45   # Max face_height / frame_height before proximity block
TEMPORAL_CONSISTENCY_FRAMES = 5      # Consecutive real frames required before door unlocks
