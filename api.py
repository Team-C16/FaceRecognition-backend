"""
FastAPI Backend Server for Face Recognition.

Provides:
- REST API endpoints for enrollment and recognition
- WebSocket endpoint for real-time face recognition from Raspberry Pi
- People management (list, delete)

Usage:
    python api.py
    # or
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import io
import cv2
import os

# Bugfix for Windows OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import base64
import numpy as np
import uvicorn
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from config import HOST, PORT, MIN_ENROLLMENT_IMAGES, MAX_ENROLLMENT_IMAGES
from face_recognizer import FaceRecognizer
from face_database import FaceDatabase
from audit_logger import AuditLogger
from liveness_checker import LivenessChecker

# ── Initialize ──────────────────────────────────────────────
app = FastAPI(
    title="Face Recognition API",
    description="Real-time face recognition backend for Raspberry Pi camera system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face recognizer (GPU), database, and audit logger
recognizer = FaceRecognizer()
face_db = FaceDatabase()
audit_log = AuditLogger()

print("[API] Security pipeline: LivenessChecker (MiniFASNet) will be initialized per WebSocket connection.")


# ── Helper Functions ────────────────────────────────────────

def decode_base64_image(b64_string):
    """Decode a base64 string to OpenCV BGR image."""
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def decode_upload_file(file_bytes):
    """Decode uploaded file bytes to OpenCV BGR image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ── REST API Endpoints ──────────────────────────────────────

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Face Recognition API",
        "status": "running",
        "enrolled_people": len(face_db.get_all_people()),
    }


@app.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """
    Enroll a new person by uploading multiple face images.

    - Upload 3-15 face images of the same person
    - Images should show different angles and lighting
    - Returns enrollment confirmation with sample count

    Args:
        name: Person's name (unique identifier)
        images: List of face image files (JPEG/PNG)
    """
    if len(images) < MIN_ENROLLMENT_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_ENROLLMENT_IMAGES} images, got {len(images)}"
        )

    if len(images) > MAX_ENROLLMENT_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_ENROLLMENT_IMAGES} images allowed, got {len(images)}"
        )

    # Extract embeddings from each image
    embeddings = []
    failed = 0

    for img_file in images:
        contents = await img_file.read()
        img = decode_upload_file(contents)

        if img is None:
            failed += 1
            continue

        embedding = recognizer.get_embedding(img)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            failed += 1

    if len(embeddings) < MIN_ENROLLMENT_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Could only extract {len(embeddings)} valid face embeddings. "
                   f"Need at least {MIN_ENROLLMENT_IMAGES}. "
                   f"Ensure images contain clear, visible faces."
        )

    # Enroll in database
    result = face_db.enroll(name, embeddings)
    result["failed_images"] = failed

    return result


@app.post("/recognize")
async def recognize_face(
    image: UploadFile = File(...),
):
    """
    Recognize a face from an uploaded image.

    Args:
        image: Face image file (JPEG/PNG)

    Returns:
        Recognition result with name and confidence score
    """
    contents = await image.read()
    img = decode_upload_file(contents)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    embedding = recognizer.get_embedding(img)

    if embedding is None:
        return {
            "name": "no_face_detected",
            "score": 0.0,
            "matched": False,
        }

    result = face_db.recognize(embedding)
    return result


@app.get("/people")
async def list_people():
    """List all enrolled people."""
    return {"people": face_db.get_all_people()}


@app.delete("/people/{name}")
async def delete_person(name: str):
    """Remove an enrolled person."""
    if face_db.remove_person(name):
        return {"message": f"Removed '{name}'"}
    raise HTTPException(status_code=404, detail=f"Person '{name}' not found")


# ── WebSocket Endpoint ──────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Real-time face recognition via WebSocket — with full security pipeline.

    Security rules enforced (per connection, stateful):
      1. Proximity Block         — face_height_ratio > 0.45 → reject
      2. Temporal Consistency    — must pass 5 consecutive real frames
      3. Strict Liveness         — MiniFASNet confidence >= 0.90
      4. Ghost Blink Fix         — counter resets on type=no_face
      5. Anti-Tailgating Lockdown — face_count > 1 → lockdown

    Pi sends one of two message types:

      Face detected:
        {
          "type": "recognize",
          "face": "<base64 JPEG cropped face>",
          "box": [x1, y1, x2, y2],
          "face_height_ratio": 0.35,
          "face_count": 1,
          "timestamp": 1234567890.0
        }

      No face in frame (triggers ghost blink reset):
        { "type": "no_face", "timestamp": 1234567890.0 }

    Backend responds:
        {
          "name": "Ali Yilmaz",
          "score": 0.87,
          "matched": true,
          "is_validated": true,
          "liveness_conf": 0.95,
          "consecutive_frames": 5,
          "lockdown": false,
          "too_close": false,
          "label": "REAL (0.95)",
          "timestamp": 1234567890.0
        }
    """
    await ws.accept()
    client_host = ws.client.host if ws.client else "unknown"
    print(f"[WebSocket] Client connected: {client_host}")

    # Each connection gets its own independent liveness state
    liveness = LivenessChecker()

    try:
        while True:
            data = await ws.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")
            timestamp = message.get("timestamp", 0)

            # ── RULE 4: Ghost Blink Fix ───────────────────────────────────────
            # Pi reports no face → wipe the frame counter completely.
            if msg_type == "no_face":
                liveness.reset()
                await ws.send_json({
                    "label": "NO_FACE",
                    "is_validated": False,
                    "matched": False,
                    "timestamp": timestamp,
                })
                continue

            if msg_type != "recognize":
                continue  # Ignore unknown message types

            # ── Decode cropped face ───────────────────────────────────────────
            face_img = decode_base64_image(message.get("face", ""))
            if face_img is None:
                await ws.send_json({
                    "error": "Failed to decode face image",
                    "is_validated": False,
                    "matched": False,
                    "timestamp": timestamp,
                })
                continue

            face_height_ratio = float(message.get("face_height_ratio", 0.0))
            face_count        = int(message.get("face_count", 1))

            # ── Run all 5 security rules ──────────────────────────────────────
            result = liveness.check(face_img, face_height_ratio, face_count)

            # Build base response (always sent, even before validation)
            response = {
                "is_validated": result.is_fully_validated,
                "liveness_conf": round(result.liveness_conf, 4),
                "consecutive_frames": result.consecutive_frames,
                "lockdown": result.is_multi_face_lockdown,
                "too_close": result.too_close,
                "label": result.label,
                "timestamp": timestamp,
            }

            # ── Recognition only fires after full validation ───────────────────
            if not result.is_fully_validated:
                response["matched"] = False
                response["name"] = "unknown"
                response["score"] = 0.0
                await ws.send_json(response)
                continue

            # ── Extract embedding and match against database ───────────────────
            embedding = recognizer.get_embedding(face_img)
            if embedding is None:
                response["matched"] = False
                response["name"] = "no_face_detected"
                response["score"] = 0.0
                await ws.send_json(response)
                continue

            match = face_db.recognize(embedding)
            response["name"]    = match["name"]
            response["score"]   = match["score"]
            response["matched"] = match["matched"]

            # ── Audit log — only on the exact 5th frame (no spam) ─────────────
            if result.consecutive_frames == 5:
                if match["matched"]:
                    audit_log.log_event(
                        "DOOR_UNLOCK_SUCCESS",
                        match["name"],
                        match["score"],
                        f"Person passed liveness and matched (via WebSocket from {client_host})",
                    )
                    print(f"\033[96m[AUDIT]\033[0m Door unlocked for '{match['name']}' | score={match['score']:.2f}")
                else:
                    audit_log.log_event(
                        "DOOR_UNLOCK_FAILED",
                        "UNKNOWN",
                        match["score"],
                        f"Person passed liveness but no database match (via WebSocket from {client_host})",
                    )
                    print(f"\033[91m[AUDIT]\033[0m Unknown person attempted unlock | score={match['score']:.2f}")

            await ws.send_json(response)

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected: {client_host}")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")



# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Face Recognition API Server")
    print(f"  Listening on http://{HOST}:{PORT}")
    print(f"  Docs: http://{HOST}:{PORT}/docs")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=HOST, port=PORT)
