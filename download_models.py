"""
Download and verify InsightFace models.

This script downloads the required ArcFace model pack
used for face recognition on the backend server.

Usage:
    python download_models.py
"""

import os
import sys

# Bugfix for Windows OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def download_models():
    """Download InsightFace models."""
    print("=" * 50)
    print("  InsightFace Model Downloader")
    print("=" * 50)

    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        print("\nERROR: insightface not installed.")
        print("Run: pip install insightface onnxruntime-gpu")
        sys.exit(1)

    model_name = "buffalo_l"
    print(f"\nDownloading model pack: {model_name}")
    print("This may take a few minutes on first run...\n")

    try:
        # This triggers automatic model download
        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=0, det_size=(640, 640))

        print("\n[OK] Models downloaded and verified successfully!")
        print(f"  Model: {model_name}")

        # Check available providers
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"\n  Available ONNX Runtime providers:")
        for p in providers:
            marker = "*" if p == "CUDAExecutionProvider" else " "
            print(f"    [{marker}] {p}")

        if "CUDAExecutionProvider" in providers:
            print("\n  [OK] GPU (CUDA) support detected!")
        else:
            print("\n  [WARN] GPU not detected. Running on CPU (slower).")
            print("  To enable GPU: pip install onnxruntime-gpu")

    except Exception as e:
        print(f"\nERROR: Failed to download models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_models()
