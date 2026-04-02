import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from services.face_service import (
    FaceDetection,
    detect_faces,
    get_embedding,
    get_largest_face,
)

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(
        self,
        min_face_area: int = 1600,
        min_face_ratio: float = 0.02,
        blur_threshold: float = 75.0,
    ) -> None:
        self.min_face_area = min_face_area
        self.min_face_ratio = min_face_ratio
        self.blur_threshold = blur_threshold

    def _is_blurry(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def validate_face_quality(self, image: np.ndarray, face: FaceDetection) -> Tuple[bool, str]:
        image_area = float(image.shape[0] * image.shape[1])
        if image_area <= 0:
            return False, "invalid image shape"

        if face.area < float(self.min_face_area):
            return False, "face too small"

        if (face.area / image_area) < self.min_face_ratio:
            return False, "face occupies too little of image"

        x1, y1, x2, y2 = [int(v) for v in face.box_xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return False, "invalid face bounding box"

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return False, "empty face crop"

        if self._is_blurry(face_crop):
            return False, "face is blurry"

        return True, "ok"

    def process_images(self, images: List[np.ndarray]) -> Dict:
        embeddings: List[np.ndarray] = []
        valid_images = 0
        skipped_images = 0
        skipped_reasons: List[str] = []
        sample_image: np.ndarray | None = None

        for index, image in enumerate(images, start=1):
            try:
                faces = detect_faces(image)
                if not faces:
                    skipped_images += 1
                    skipped_reasons.append(f"image_{index}: no face detected")
                    logger.info("Skipped image %d: no face detected", index)
                    continue

                face = get_largest_face(faces)
                if face is None:
                    skipped_images += 1
                    skipped_reasons.append(f"image_{index}: unable to pick a face")
                    logger.info("Skipped image %d: unable to pick a face", index)
                    continue

                is_valid, reason = self.validate_face_quality(image, face)
                if not is_valid:
                    skipped_images += 1
                    skipped_reasons.append(f"image_{index}: {reason}")
                    logger.info("Skipped image %d: %s", index, reason)
                    continue

                embedding = get_embedding(image, face)
                if embedding.size == 0:
                    skipped_images += 1
                    skipped_reasons.append(f"image_{index}: empty embedding")
                    logger.info("Skipped image %d: empty embedding", index)
                    continue

                embeddings.append(embedding)
                valid_images += 1
                if sample_image is None:
                    sample_image = image.copy()

                logger.info("Processed image %d successfully", index)
            except Exception as processing_error:
                skipped_images += 1
                skipped_reasons.append(f"image_{index}: {processing_error}")
                logger.exception("Skipped image %d due to error", index)

        return {
            "embeddings": embeddings,
            "valid_images": valid_images,
            "skipped_images": skipped_images,
            "skipped_reasons": skipped_reasons,
            "sample_image": sample_image,
        }

    def compute_average_embedding(self, embeddings: List[np.ndarray]) -> np.ndarray:
        if not embeddings:
            raise ValueError("No embeddings available to average")

        matrix = np.vstack(embeddings).astype(np.float32)
        avg_embedding = np.mean(matrix, axis=0)

        # Normalize for stable cosine similarity comparisons later.
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding
