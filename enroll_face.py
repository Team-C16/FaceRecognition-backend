"""
Face Enrollment Script - Scan a person's face using a webcam.

Opens the webcam, lets you capture multiple photos of a person,
and sends them to the backend for enrollment.

Usage:
    python enroll_face.py --name "Ali Yilmaz"
    python enroll_face.py --name "Ali Yilmaz" --captures 10
    python enroll_face.py --name "Ali Yilmaz" --server http://192.168.1.100:8000

Controls:
    SPACE  - Capture a photo
    Q      - Finish and enroll
    ESC    - Cancel
"""

import cv2
import sys
import argparse
import requests
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Enroll a face for recognition")
    parser.add_argument("--name", required=True, help="Person's name")
    parser.add_argument("--captures", type=int, default=8, help="Number of photos to capture (default: 8)")
    parser.add_argument("--server", default="http://localhost:8000", help="Backend server URL")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    return parser.parse_args()


def capture_faces(name, num_captures, camera_index):
    """
    Open webcam and capture face photos interactively.

    Instructions displayed on screen guide the user to move their head
    for different angles, improving recognition accuracy.

    Returns:
        list of JPEG-encoded face images (bytes)
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured = []
    instructions = [
        "Look straight at the camera",
        "Slightly turn head LEFT",
        "Slightly turn head RIGHT",
        "Tilt head slightly UP",
        "Tilt head slightly DOWN",
        "Smile naturally",
        "Neutral expression",
        "Slightly different distance",
    ]

    print(f"\n{'='*50}")
    print(f"  Face Enrollment: {name}")
    print(f"  Captures needed: {num_captures}")
    print(f"{'='*50}")
    print(f"\n  SPACE = Capture | Q = Finish | ESC = Cancel\n")

    while len(captured) < num_captures:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw header
        cv2.rectangle(display, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(
            display, f"Enrolling: {name}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            display, f"Captured: {len(captured)}/{num_captures}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        # Draw instruction
        instruction_idx = min(len(captured), len(instructions) - 1)
        instruction = instructions[instruction_idx]
        cv2.putText(
            display, f">> {instruction}",
            (w // 2 - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        # Draw guide rectangle (center frame)
        cx, cy = w // 2, h // 2
        guide_size = 150
        cv2.rectangle(
            display,
            (cx - guide_size, cy - guide_size - 30),
            (cx + guide_size, cy + guide_size + 30),
            (255, 255, 0), 2,
        )

        cv2.imshow("Face Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # SPACE to capture
            # Encode as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            captured.append(buffer.tobytes())
            print(f"  ✓ Captured {len(captured)}/{num_captures}: {instruction}")

            # Brief flash effect
            flash = frame.copy()
            flash[:] = (255, 255, 255)
            cv2.imshow("Face Enrollment", flash)
            cv2.waitKey(100)

        elif key == ord("q"):  # Q to finish early
            if len(captured) >= 3:
                print(f"\n  Finishing with {len(captured)} captures")
                break
            else:
                print(f"  Need at least 3 captures (have {len(captured)})")

        elif key == 27:  # ESC to cancel
            print("\n  Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    return captured


def send_enrollment(name, face_images, server_url):
    """
    Send captured face images to backend for enrollment.

    Args:
        name: Person's name
        face_images: List of JPEG bytes
        server_url: Backend server URL
    """
    print(f"\n  Sending {len(face_images)} images to server...")

    files = []
    for i, img_bytes in enumerate(face_images):
        files.append(("images", (f"face_{i}.jpg", img_bytes, "image/jpeg")))

    try:
        response = requests.post(
            f"{server_url}/enroll",
            data={"name": name},
            files=files,
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n  ✓ Successfully enrolled '{name}'!")
            print(f"    Samples used: {result.get('num_samples', 'N/A')}")
            print(f"    Failed images: {result.get('failed_images', 0)}")
        else:
            print(f"\n  ✗ Enrollment failed: {response.status_code}")
            print(f"    {response.json().get('detail', response.text)}")

    except requests.exceptions.ConnectionError:
        print(f"\n  ✗ Cannot connect to server: {server_url}")
        print("    Make sure the backend is running!")
    except Exception as e:
        print(f"\n  ✗ Error: {e}")


if __name__ == "__main__":
    args = parse_args()
    face_images = capture_faces(args.name, args.captures, args.camera)
    if face_images:
        send_enrollment(args.name, face_images, args.server)
