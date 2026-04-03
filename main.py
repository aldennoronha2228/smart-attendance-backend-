from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import Dict, List
import logging
import os
import cv2
import numpy as np

# Import the student counter
from student_counter import StudentCounter
from services.database import get_student_embedding, list_students, save_embedding
from services.recognition_service import RecognitionService
from services.training_service import TrainingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Attendance AI Service",
    description="AI-powered student counting service for classroom verification",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize student counter
counter = StudentCounter()
training_service = TrainingService()
recognition_threshold = float(os.getenv("RECOGNITION_THRESHOLD", "55"))
recognition_service = RecognitionService(threshold=recognition_threshold)

@app.on_event("startup")
async def startup_event():
    """Initialize the AI model on startup"""
    logger.info("Starting up AI service...")
    logger.info("AI service ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Smart Attendance AI",
        "version": "1.0.0",
        "model": "YOLOv8 Person Detection"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": counter.model is not None,
        "ready": True
    }


@app.get("/students")
async def get_students() -> Dict:
    """Return previously enrolled students from embeddings storage."""
    students = list_students()
    return {
        "count": len(students),
        "students": students,
    }

@app.post("/count-students")
async def count_students(file: UploadFile = File(...)) -> Dict:
    """
    Count students in a classroom photo
    
    Args:
        file: Uploaded image file
        
    Returns:
        Dictionary with count, min_count, max_count
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Count students
        result = counter.count_students(image)
        
        logger.info(f"Detection complete: {result}")
        
        return {
            "count": result["count"],
            "min_count": result["min_count"],
            "max_count": result["max_count"],
            "confidence": result.get("confidence", 0.5),
            "detections": result.get("detections", []),
            "message": f"Detected {result['count']} students (range: {result['min_count']}-{result['max_count']})"
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/enroll")
async def enroll_student(
    name: str = Form(...),
    images: List[UploadFile] = File(...),
) -> Dict:
    """
    Enroll a student using multiple face images and store average embedding.

    - Skips invalid/low-quality images
    - Uses largest face when multiple faces are present
    - Stores one averaged embedding vector per student
    """
    student_name = name.strip()
    if not student_name:
        raise HTTPException(status_code=400, detail="Student name is required")

    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    if len(images) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per training request",
        )

    logger.info("Starting enrollment for '%s' with %d image(s)", student_name, len(images))

    decoded_images: List[np.ndarray] = []
    skipped_before_processing = 0
    pre_skip_reasons: List[str] = []

    for file_index, image_file in enumerate(images, start=1):
        if not image_file.content_type or not image_file.content_type.startswith("image/"):
            skipped_before_processing += 1
            pre_skip_reasons.append(f"image_{file_index}: invalid file type")
            logger.info("Skipping file %s due to invalid content type", image_file.filename)
            continue

        content = await image_file.read()
        if not content:
            skipped_before_processing += 1
            pre_skip_reasons.append(f"image_{file_index}: empty file")
            logger.info("Skipping file %s because it is empty", image_file.filename)
            continue

        np_buffer = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if image is None:
            skipped_before_processing += 1
            pre_skip_reasons.append(f"image_{file_index}: cannot decode image")
            logger.info("Skipping file %s because OpenCV decode failed", image_file.filename)
            continue

        decoded_images.append(image)

    if not decoded_images:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No decodable images were uploaded",
                "valid_images": 0,
                "skipped_images": skipped_before_processing,
                "skipped_reasons": pre_skip_reasons,
            },
        )

    training_result = training_service.process_images(decoded_images)
    valid_images = training_result["valid_images"]
    skipped_images = skipped_before_processing + training_result["skipped_images"]
    skipped_reasons = pre_skip_reasons + training_result["skipped_reasons"]

    if valid_images == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Enrollment failed: no valid faces found",
                "valid_images": 0,
                "skipped_images": skipped_images,
                "skipped_reasons": skipped_reasons,
            },
        )

    new_avg_embedding = training_service.compute_average_embedding(training_result["embeddings"])

    existing_embedding, existing_samples, _ = get_student_embedding(student_name)
    combined_samples = int(existing_samples) + int(valid_images)

    if existing_embedding is not None and existing_samples > 0 and existing_embedding.shape == new_avg_embedding.shape:
        # Merge embeddings without deleting prior samples (running weighted average).
        merged = (existing_embedding * float(existing_samples)) + (new_avg_embedding * float(valid_images))
        merged = merged / float(combined_samples)

        # Normalize for stable cosine similarity comparisons.
        norm = float(np.linalg.norm(merged))
        if norm > 0:
            merged = merged / norm

        final_embedding = merged
    else:
        final_embedding = new_avg_embedding

    save_embedding(
        name=student_name,
        embedding=final_embedding,
        samples_used=combined_samples,
        sample_image=training_result["sample_image"],
    )

    logger.info(
        "Enrollment successful for '%s': valid=%d skipped=%d",
        student_name,
        valid_images,
        skipped_images,
    )

    return {
        "message": "Enrollment successful",
        "valid_images": valid_images,
        "skipped_images": skipped_images,
        "total_samples": combined_samples,
    }


@app.post("/recognize")
async def recognize_faces(image: UploadFile = File(...)) -> Dict:
    """Recognize faces in a class image using enrolled student embeddings."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    np_buffer = np.frombuffer(content, dtype=np.uint8)
    decoded_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if decoded_image is None:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded image")

    try:
        result = recognition_service.recognize(decoded_image)
        return result
    except Exception as recognize_error:
        logger.exception("Recognition failed")
        raise HTTPException(status_code=500, detail="Recognition failed") from recognize_error

@app.post("/test-count")
async def test_count():
    """Test endpoint that returns mock data without requiring an image"""
    import random
    base_count = random.randint(15, 45)
    return {
        "count": base_count,
        "min_count": base_count - 2,
        "max_count": base_count + 2,
        "confidence": 0.85,
        "detections": [],
        "message": f"Test mode: Detected {base_count} students",
        "test_mode": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
