import logging
from typing import Dict, List, Tuple

import numpy as np

from services.database import load_embeddings
from services.face_service import detect_faces, get_embedding

logger = logging.getLogger(__name__)


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def _similarity_percent(embedding_1: np.ndarray, embedding_2: np.ndarray) -> float:
    # Same metric used by the original SDK example: maps cosine [-1, 1] -> [0, 100]
    score = (float(np.dot(embedding_1, embedding_2)) + 1.0) * 50.0
    return max(0.0, min(100.0, score))


class RecognitionService:
    def __init__(self, threshold: float = 75.0) -> None:
        self.threshold = threshold

    def _load_known_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        payload = load_embeddings()
        known: List[Tuple[str, np.ndarray]] = []

        for name, info in payload.items():
            if not isinstance(name, str) or not isinstance(info, dict):
                continue

            raw_embedding = info.get("embedding")
            if not isinstance(raw_embedding, list) or not raw_embedding:
                continue

            try:
                vector = _normalize_embedding(np.array(raw_embedding, dtype=np.float32))
            except Exception:
                continue

            if vector.size == 0:
                continue
            known.append((name, vector))

        return known

    def recognize(self, image: np.ndarray) -> Dict:
        detections = detect_faces(image)
        faces_detected = len(detections)

        if faces_detected == 0:
            return {"faces_detected": 0, "recognized": [], "unknown_count": 0}

        known_embeddings = self._load_known_embeddings()
        if not known_embeddings:
            return {
                "faces_detected": faces_detected,
                "recognized": [],
                "unknown_count": faces_detected,
            }

        recognized: List[Dict] = []
        unknown_count = 0

        for detection in detections:
            try:
                candidate = _normalize_embedding(get_embedding(image, detection))
            except Exception as embedding_error:
                logger.exception("Failed to create embedding for one detected face")
                unknown_count += 1
                continue

            best_name = ""
            best_score = -1.0

            for known_name, known_vector in known_embeddings:
                if known_vector.shape[0] != candidate.shape[0]:
                    continue

                score = _similarity_percent(candidate, known_vector)
                if score > best_score:
                    best_score = score
                    best_name = known_name

            x1, y1, x2, y2 = detection.box_xyxy
            box = [
                max(0.0, x1),
                max(0.0, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
            ]

            if best_score >= self.threshold and best_name:
                recognized.append(
                    {
                        "name": best_name,
                        "confidence": round(best_score, 2),
                        "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    }
                )
            else:
                unknown_count += 1

        return {
            "faces_detected": faces_detected,
            "recognized": recognized,
            "unknown_count": unknown_count,
        }
