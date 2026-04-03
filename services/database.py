import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"
SAMPLES_DIR = DATA_DIR / "samples"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return safe or "student"


def load_embeddings() -> Dict:
    if not EMBEDDINGS_FILE.exists():
        return {}

    try:
        with EMBEDDINGS_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        logger.warning("embeddings.json is invalid JSON. Starting with empty store.")
        return {}


def _save_sample_image(name: str, sample_image: np.ndarray) -> str:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # Keep previous sample images instead of overwriting.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_name = f"{_safe_name(name)}_{timestamp}.jpg"
    sample_path = SAMPLES_DIR / file_name
    cv2.imwrite(str(sample_path), sample_image)
    return str(sample_path)


def get_student_embedding(name: str) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
    """Return (embedding_vector, samples_used, sample_image_path) for a student if present."""
    database = load_embeddings()
    payload = database.get(name)
    if not isinstance(payload, dict):
        return None, 0, None

    samples_used = int(payload.get("samples_used", 0) or 0)
    sample_image = payload.get("sample_image")
    sample_image_path = sample_image if isinstance(sample_image, str) else None

    raw = payload.get("embedding")
    if not isinstance(raw, list) or not raw:
        return None, samples_used, sample_image_path

    try:
        vector = np.asarray(raw, dtype=np.float32).flatten()
    except Exception:
        return None, samples_used, sample_image_path

    if vector.size == 0:
        return None, samples_used, sample_image_path

    return vector, samples_used, sample_image_path


def save_embedding(
    name: str,
    embedding: np.ndarray,
    samples_used: int,
    sample_image: Optional[np.ndarray] = None,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    database = load_embeddings()

    payload = {
        "embedding": [float(value) for value in embedding.tolist()],
        "samples_used": int(samples_used),
        "updated_at": _utc_now(),
    }

    if sample_image is not None:
        payload["sample_image"] = _save_sample_image(name, sample_image)

    # Overwrite behavior for existing students.
    database[name] = payload

    with EMBEDDINGS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(database, handle, indent=2)

    logger.info("Saved embedding for student '%s' with %d samples", name, samples_used)


def list_students() -> List[Dict]:
    database = load_embeddings()
    students: List[Dict] = []

    for name, payload in database.items():
        if not isinstance(payload, dict):
            payload = {}

        students.append(
            {
                "name": name,
                "samples_used": int(payload.get("samples_used", 0)),
                "updated_at": payload.get("updated_at"),
                "sample_image": payload.get("sample_image"),
            }
        )

    def _sort_key(student: Dict) -> str:
        updated = student.get("updated_at")
        return updated if isinstance(updated, str) else ""

    students.sort(key=_sort_key, reverse=True)
    return students
