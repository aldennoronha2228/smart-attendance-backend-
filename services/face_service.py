import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Support a local SDK checkout at ai-service/sdk when available.
_SDK_ROOT = Path(__file__).resolve().parents[1] / "sdk"
if _SDK_ROOT.exists():
    sdk_path = str(_SDK_ROOT)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

try:
    previous_cwd = Path.cwd()
    if _SDK_ROOT.exists():
        os.chdir(_SDK_ROOT)

    from face_detect.detect_imgs import get_face_boundingbox
    from face_feature.GetFeature import get_face_feature
    from face_landmark.GetLandmark import get_face_landmark

    SDK_READY = True
except Exception as sdk_import_error:
    SDK_READY = False
    logger.warning("Face SDK import failed: %s", sdk_import_error)
finally:
    if _SDK_ROOT.exists():
        os.chdir(previous_cwd)

_HAAR_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)


@dataclass
class FaceDetection:
    box_xyxy: List[float]
    area: float
    score: float
    raw_box: Any


def _to_numpy_box(raw_box: Any) -> np.ndarray:
    if hasattr(raw_box, "detach"):
        return raw_box.detach().cpu().numpy()
    if hasattr(raw_box, "data"):
        return raw_box.data.cpu().numpy()
    return np.asarray(raw_box)


def detect_faces(image: np.ndarray) -> List[FaceDetection]:
    if not SDK_READY:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = _HAAR_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        detections: List[FaceDetection] = []
        for (x, y, w, h) in boxes:
            detections.append(
                FaceDetection(
                    box_xyxy=[float(x), float(y), float(x + w), float(y + h)],
                    area=float(w * h),
                    score=1.0,
                    raw_box=np.array([x, y, x + w, y + h], dtype=np.float32),
                )
            )
        logger.info("Detected %d face(s) with OpenCV fallback", len(detections))
        return detections

    boxes, scores = get_face_boundingbox(image)
    detections: List[FaceDetection] = []

    for raw_box, raw_score in zip(boxes, scores):
        box = _to_numpy_box(raw_box).astype(float).reshape(-1)
        if box.size < 4:
            continue

        x1, y1, x2, y2 = box[:4].tolist()
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        score = float(raw_score.item() if hasattr(raw_score, "item") else raw_score)
        detections.append(
            FaceDetection(
                box_xyxy=[x1, y1, x2, y2],
                area=area,
                score=score,
                raw_box=raw_box,
            )
        )

    logger.info("Detected %d face(s)", len(detections))
    return detections


def get_largest_face(faces: List[FaceDetection]) -> Optional[FaceDetection]:
    if not faces:
        return None
    return max(faces, key=lambda face: face.area)


def get_embedding(image: np.ndarray, face: FaceDetection) -> np.ndarray:
    if not SDK_READY:
        # Fallback embedding for local/dev runs when SDK is unavailable.
        x1, y1, x2, y2 = [int(v) for v in face.box_xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise RuntimeError("Cannot build fallback embedding from empty face crop")

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_crop, (32, 32), interpolation=cv2.INTER_AREA)
        vector = resized.astype(np.float32).flatten()
        vector = vector - np.mean(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SDK expects landmark input from the detector box representation.
    landmarks = get_face_landmark(gray, face.raw_box)
    if hasattr(landmarks, "detach"):
        landmarks_np = landmarks.detach().cpu().numpy()
    elif hasattr(landmarks, "data"):
        landmarks_np = landmarks.data.cpu().numpy()
    else:
        landmarks_np = np.asarray(landmarks)

    _, feature = get_face_feature(image, landmarks_np)
    embedding = np.asarray(feature, dtype=np.float32).flatten()
    return embedding
