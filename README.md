# Smart Attendance Backend

FastAPI backend for face recognition and student enrollment used by the Smart Attendance frontend.

## Features

- Face recognition endpoint for attendance
- Student enrollment endpoint using multiple images
- Trained students listing endpoint
- Local embeddings/data support

## Tech Stack

- Python
- FastAPI
- Uvicorn
- PyTorch-based face SDK components

## Project Structure

- `main.py`: FastAPI app entrypoint
- `services/`: backend services for recognition/training/database operations
- `sdk/`: face detection and face feature extraction code
- `data/`: embeddings and sample data
- `requirements.txt`: Python dependencies
- `Dockerfile`: container build file

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. Open API docs (if enabled):

```text
http://localhost:8000/docs
```

## Expected Endpoints

- `GET /students`
- `POST /enroll`
- `POST /recognize`

## Render Deployment

Use a Render Web Service with:

- Runtime: `Python`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

After deploy, use the Render URL as frontend `BACKEND_API_BASE_URL` in Vercel.

## Notes

- Keep secrets in environment variables, not in source files.
- `yolov8n.pt` and SDK model files are included in this repo; deployment will use them at runtime.
