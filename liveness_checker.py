"""
LivenessChecker - Per-connection security state machine.

Encapsulates all 5 security rules from test_recognition_only.py:
  1. Proximity Block         - face too close to camera → reject
  2. Temporal Consistency    - must pass N consecutive frames → unlock
  3. Strict Liveness         - MiniFASNet confidence must be >= threshold
  4. Ghost Blink Fix         - frame counter resets when no face is present
  5. Anti-Tailgating Lockdown - more than 1 face in frame → full lockdown

One LivenessChecker instance per WebSocket connection.
"""

import numpy as np
from dataclasses import dataclass
from uniface.spoofing import MiniFASNet

try:
    from config import (
        LIVENESS_STRICT_THRESHOLD,
        PROXIMITY_RATIO_LIMIT,
        TEMPORAL_CONSISTENCY_FRAMES,
    )
except ImportError:
    from .config import (
        LIVENESS_STRICT_THRESHOLD,
        PROXIMITY_RATIO_LIMIT,
        TEMPORAL_CONSISTENCY_FRAMES,
    )


@dataclass
class LivenessResult:
    """Result returned by LivenessChecker.check()."""
    is_real: bool
    is_fully_validated: bool     # True when consecutive_real_frames >= threshold
    is_multi_face_lockdown: bool
    too_close: bool
    liveness_conf: float
    consecutive_frames: int
    label: str                   # Human-readable status for the Pi / display


# Module-level shared MiniFASNet instance (heavy to load, reuse across connections)
_spoofer = None

def _get_spoofer():
    global _spoofer
    if _spoofer is None:
        print("[LivenessChecker] Loading MiniFASNet anti-spoofing model...")
        _spoofer = MiniFASNet()
        print("[LivenessChecker] MiniFASNet ready.")
    return _spoofer


class LivenessChecker:
    """
    Stateful per-connection liveness and security checker.

    Usage:
        checker = LivenessChecker()

        # When no face is in frame (Pi sends type=no_face):
        checker.reset()

        # When a face is detected:
        result = checker.check(face_img, face_height_ratio, face_count)
    """

    def __init__(self):
        self.spoofer = _get_spoofer()
        self.consecutive_real_frames = 0

    def reset(self):
        """
        SECURITY RULE 4 – Ghost Blink Fix.
        Call this when the Pi reports no face in the frame.
        Wipes the consecutive frame counter so an attacker cannot swap in a photo
        right after a real person steps away.
        """
        self.consecutive_real_frames = 0

    def check(
        self,
        face_img: np.ndarray,
        face_height_ratio: float,
        face_count: int,
    ) -> LivenessResult:
        """
        Run all 5 security rules against the current face image.

        Args:
            face_img:          Cropped face BGR numpy array.
            face_height_ratio: face_height / frame_height (computed by Pi).
            face_count:        Total faces detected by Pi this frame.

        Returns:
            LivenessResult dataclass.
        """
        # ── RULE 5: Anti-Tailgating Lockdown ─────────────────────────────────
        is_multi_face_lockdown = face_count > 1

        # ── RULE 1: Proximity Block ───────────────────────────────────────────
        too_close = face_height_ratio > PROXIMITY_RATIO_LIMIT

        # ── RULE 3: MiniFASNet Liveness Detection ─────────────────────────────
        # Build a synthetic bbox that covers the entire cropped image.
        # MiniFASNet.predict(frame, bbox) crops internally — passing the full
        # cropped face as the "frame" with a full-image bbox is equivalent.
        h, w = face_img.shape[:2]
        synthetic_bbox = [0, 0, w, h]

        spoof_result = self.spoofer.predict(face_img, synthetic_bbox)
        is_real = spoof_result.is_real
        liveness_conf = float(spoof_result.confidence)

        # ── RULE 3 (strict override) ──────────────────────────────────────────
        # Even if MiniFASNet says "real", reject if confidence is below threshold.
        if is_real and liveness_conf < LIVENESS_STRICT_THRESHOLD:
            is_real = False

        # ── Combined failure reset ────────────────────────────────────────────
        # Any blocking condition resets the temporal counter.
        if too_close or not is_real or is_multi_face_lockdown:
            self.consecutive_real_frames = 0
            is_real = False

        # ── RULE 2: Temporal Consistency ─────────────────────────────────────
        if is_real and not is_multi_face_lockdown:
            self.consecutive_real_frames += 1
        # (counter already reset in the block above for failure cases)

        is_fully_validated = self.consecutive_real_frames >= TEMPORAL_CONSISTENCY_FRAMES

        # ── Human-readable label ──────────────────────────────────────────────
        if is_multi_face_lockdown:
            label = "ANTI-TAILGATE: ONE PERSON ONLY"
        elif too_close:
            label = "MOVE FURTHER AWAY"
        elif is_fully_validated:
            label = f"REAL ({liveness_conf:.2f})"
        elif is_real:
            label = f"HOLD STILL... ({self.consecutive_real_frames}/{TEMPORAL_CONSISTENCY_FRAMES})"
        else:
            label = f"SPOOF/FAKE ({liveness_conf:.2f})"

        return LivenessResult(
            is_real=is_real,
            is_fully_validated=is_fully_validated,
            is_multi_face_lockdown=is_multi_face_lockdown,
            too_close=too_close,
            liveness_conf=liveness_conf,
            consecutive_frames=self.consecutive_real_frames,
            label=label,
        )
