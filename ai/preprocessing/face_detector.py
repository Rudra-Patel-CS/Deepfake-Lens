import logging
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from ai.utils.config import (
    FAKE_FACES,
    FAKE_FRAMES,
    IMG_SIZE,
    LOG_DIR,
    REAL_FACES,
    REAL_FRAMES,
)


_RETINAFACE_MODEL = None


def _get_retinaface_model():
    from retinaface import RetinaFace
    global _RETINAFACE_MODEL
    if _RETINAFACE_MODEL is None:
        _RETINAFACE_MODEL = RetinaFace.build_model()
    return _RETINAFACE_MODEL


def _largest_face(image_path):
    """Crop the largest RetinaFace detection from a path or a BGR array."""
    from retinaface import RetinaFace
    image = cv2.imread(str(image_path)) if isinstance(image_path, (str, Path)) else image_path
    if image is None:
        return None
    detections = RetinaFace.detect_faces(
        image, model=_get_retinaface_model()
    )
    if not isinstance(detections, dict):
        return None

    candidates = []
    for detection in detections.values():
        area = detection.get("facial_area")
        if area and len(area) == 4:
            x1, y1, x2, y2 = area
            candidates.append((max(0, x2 - x1) * max(0, y2 - y1), area))

    if not candidates:
        return None

    _, (x1, y1, x2, y2) = max(candidates, key=lambda item: item[0])
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(image[y1:y2, x1:x2], IMG_SIZE, interpolation=cv2.INTER_AREA)


def get_face_cropper(detector="retinaface"):
    """Create a reusable cropper; OpenCV does not import TensorFlow."""
    if detector == "retinaface":
        return _largest_face
    if detector != "opencv":
        raise ValueError(f"Unknown face detector: {detector}")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV face detector data is missing")

    def crop_face(source):
        frame = cv2.imread(str(source)) if isinstance(source, (str, Path)) else source
        if frame is None:
            return None
        boxes = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                         scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(boxes) == 0:
            return None
        x, y, width, height = max(boxes, key=lambda box: box[2] * box[3])
        return cv2.resize(frame[y:y + height, x:x + width], IMG_SIZE, interpolation=cv2.INTER_AREA)
    return crop_face


def process_frames_to_faces(
    frames_dir: Path, faces_dir: Path, limit: int | None = None, detector="retinaface"
):
    crop_face = get_face_cropper(detector)
    logger = logging.getLogger("face_detector")
    summary = {"frames_processed": 0, "faces_found": 0, "failures": 0}
    if not frames_dir.exists():
        print(f"Directory not found: {frames_dir}")
        return summary

    video_dirs = sorted(path for path in frames_dir.iterdir() if path.is_dir())
    if limit is not None:
        video_dirs = video_dirs[:limit]

    print(
        f"\nProcessing face detection on frames from {frames_dir.name} "
        f"({len(video_dirs)} videos)"
    )

    for video_dir in tqdm(video_dirs):
        output_video_dir = faces_dir / video_dir.name
        output_video_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(video_dir.glob("*.jpg")):
            summary["frames_processed"] += 1
            output_face_path = output_video_dir / f"{img_path.stem}_face.jpg"
            if output_face_path.exists():
                summary["faces_found"] += 1
                continue

            try:
                face = crop_face(img_path)
                if face is None:
                    logger.warning("No face found: %s", img_path)
                    continue
                if not cv2.imwrite(str(output_face_path), face):
                    raise OSError(f"Could not write {output_face_path}")
                summary["faces_found"] += 1
            except Exception as error:
                summary["failures"] += 1
                logger.exception("Face detection failed for %s: %s", img_path, error)
    return summary


def detect_faces(limit: int | None = None, detector="retinaface"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "face_detector.log",
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("\n" + "=" * 60)
    print("FACE DETECTION")
    print("=" * 60)

    real_summary = process_frames_to_faces(REAL_FRAMES, REAL_FACES, limit=limit, detector=detector)
    fake_summary = process_frames_to_faces(FAKE_FRAMES, FAKE_FACES, limit=limit, detector=detector)
    summary = {
        key: real_summary[key] + fake_summary[key]
        for key in real_summary
    }
    print(
        "Face summary: "
        f"frames processed={summary['frames_processed']}, "
        f"faces found={summary['faces_found']}, "
        f"failures={summary['failures']}"
    )
    print("\nFace detection completed.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and crop faces from frames.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum videos per class.")
    detect_faces(limit=parser.parse_args().limit)
