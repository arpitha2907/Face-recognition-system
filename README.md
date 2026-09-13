# Face Recognition Identification System

A production-oriented face recognition and identification system using pretrained **RetinaFace** and **ArcFace** models from InsightFace. The system supports face enrollment, face embedding generation, cosine-similarity identification, UNKNOWN rejection, image identification, and real-time webcam recognition through a FastAPI backend and web interface.

## Features

- RetinaFace face detection and alignment
- ArcFace face embedding generation
- L2-normalized face embeddings
- Cosine similarity matching
- Configurable recognition threshold
- UNKNOWN face rejection
- NO_FACE and MULTIPLE_FACES handling
- Multi-image identity enrollment
- SQLite embedding storage
- FastAPI REST API
- Real-time webcam identification
- Image upload identification
- Recognition pipeline visualization
- Evaluation script
- Pytest unit tests
- Responsive web interface

## System Architecture

```text
                  Image / Webcam Frame
                           |
                           v
                    +-------------+
                    |  RetinaFace |
                    |   Detection |
                    +-------------+
                           |
                           v
                    Face Alignment
                           |
                           v
                    +-------------+
                    |   ArcFace   |
                    |  Embedding  |
                    +-------------+
                           |
                           v
                   Face Embedding
                           |
                           v
                   L2 Normalization
                           |
                           v
                  Cosine Similarity
                           |
                           v
                    Best Match Score
                       /         \\
                      /           \\
                >= 0.45          < 0.45
                   |                 |
                   v                 v
                 MATCH            UNKNOWN
```

## Technology Stack

| Layer | Technologies |
|---|---|
| AI / Computer Vision | InsightFace, RetinaFace, ArcFace, ONNX Runtime |
| Image Processing | OpenCV, NumPy |
| Backend | Python, FastAPI, Uvicorn |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Camera | Browser MediaDevices API |
| Testing | Pytest |

## Face Recognition Pipeline

### Face Detection

RetinaFace detects faces in the input image or webcam frame. The identification pipeline requires a single detected face and explicitly handles no-face and multiple-face cases.

### Face Embedding Generation

The pretrained ArcFace recognition model converts the detected face into a numerical embedding vector representing facial features.

```text
Face Image
    |
    v
RetinaFace
    |
    v
Detected + Aligned Face
    |
    v
ArcFace
    |
    v
Face Embedding Vector
```

The embedding is converted to `float32` and L2-normalized before matching.

### Enrollment

```text
Person Images -> RetinaFace -> ArcFace -> Face Embeddings -> SQLite Gallery
```

Multiple valid images can be enrolled for the same identity. Images containing no face, multiple faces, or invalid embeddings are rejected.

### Identification

```text
Query Image / Camera Frame
          |
          v
      Face Detection
          |
          v
    ArcFace Embedding
          |
          v
   Compare with Gallery
          |
          v
   Cosine Similarity
          |
          v
      Best Match
```

## Matching Threshold

Current matching threshold:

```text
0.45
```

Decision rule:

```text
similarity >= 0.45  ->  MATCH
similarity <  0.45  ->  UNKNOWN
```

The threshold is configurable and should ideally be calibrated using a larger validation dataset for production deployment.

## UNKNOWN Rejection

The system does not force every face to match an enrolled identity.

Example:

```text
Query Face
    |
    v
Best similarity = 0.105
    |
    v
Threshold = 0.45
    |
    v
UNKNOWN
```

## Real-Time Camera Identification

The browser captures webcam frames and periodically sends JPEG frames to the FastAPI camera endpoint.

```text
Webcam -> Frame -> FastAPI -> RetinaFace -> ArcFace -> Cosine Similarity
```

The UI displays the status of **RETINAFACE -> ARCFACE -> COSINE MATCH** and supports starting and stopping the camera.

## Failure Cases

| Condition | System Response |
|---|---|
| No face detected | `NO_FACE` |
| Multiple faces detected | `MULTIPLE_FACES` |
| Similarity below threshold | `UNKNOWN` |
| Similarity above threshold | `MATCH` |
| Invalid image / API failure | `ERROR` |

## API Endpoints

### Health Check

```http
GET /api/health
```

### Image Identification

```http
POST /api/identify
```

Example response:

```json
{
  "status": "MATCH",
  "identity": "person_01",
  "similarity": 0.68,
  "threshold": 0.45,
  "reason": null
}
```

### Camera Identification

```http
POST /api/camera/identify
```

### Identity Enrollment

```http
POST /api/enroll
```

## Project Structure

```text
face-recognition-identification-system/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── pyproject.toml
├── main.py
├── api/
│   └── main.py
├── src/
│   └── face_id/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── database.py
│       ├── matcher.py
│       ├── engine.py
│       └── cli.py
├── scripts/
│   └── evaluate.py
├── tests/
│   └── test_matcher.py
├── web/
│   ├── app.js
│   ├── index.html
│   └── styles.css
└── data/
    ├── enrolled/
    └── evaluation/
```

## Installation

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

InsightFace downloads the required pretrained model when initialized.

## Running the Application

```bash
python main.py
```

Then use the web interface to enroll identities, upload images, or start the webcam.

## Configuration

| Parameter | Current Value |
|---|---|
| Model | `buffalo_l` |
| Detection threshold | `0.25` |
| Matching threshold | `0.45` |
| Detection size | `640 x 640` |
| Execution provider | CPU |

## Evaluation Results

Run:

```bash
python scripts/evaluate.py
```

Current results:

| Metric | Result |
|---|---:|
| Samples evaluated | 6 |
| Accuracy | **100%** |
| Macro Precision | **100%** |
| Macro Recall | **100%** |
| Macro F1 | **100%** |
| False Accept Rate | **0%** |
| False Reject Rate | **0%** |
| Mean inference time | **0.379 seconds** |
| Matching threshold | **0.45** |
| Failures | **0** |

> **Evaluation note:** The 100% result was obtained on only 6 evaluation samples. It should not be interpreted as 100% real-world recognition accuracy. A larger and more diverse validation/test dataset is required for a reliable estimate of real-world performance.

## Model Training

This project does **not** train or fine-tune RetinaFace or ArcFace. Both are pretrained InsightFace models.

Therefore, conventional **training accuracy is not applicable (N/A)**.

The project-specific engineering work covers enrollment, embedding extraction and storage, similarity matching, threshold-based rejection, API integration, real-time camera processing, evaluation, and UI.

## Testing

Run:

```bash
pytest -q
```

Current result:

```text
3 passed in 0.14s
```

## Performance

The current evaluation environment uses CPU inference.

```text
~0.379 seconds per sample
```

Performance varies with hardware, image resolution, detection size, number of faces, and execution provider.

## Limitations

1. The evaluation dataset is small.
2. The matching threshold has not been calibrated on a large validation dataset.
3. The current pipeline is optimized for single-face identification.
4. CPU inference introduces latency.
5. Recognition performance can vary with lighting, pose, blur, and occlusion.
6. The system does not currently implement liveness detection.
7. Additional security and privacy controls are required for high-stakes deployment.

## Future Improvements

- Larger and more diverse evaluation datasets
- FAR/FRR-based threshold calibration
- ROC and Precision-Recall curves
- GPU acceleration
- Batch embedding processing
- Liveness / anti-spoofing detection
- Face quality assessment
- Authentication and authorization
- Encryption for stored embeddings
- Audit logging
- Vector database integration for large galleries
- Multi-face tracking
- Docker and cloud deployment
- Monitoring and performance metrics

## Security and Privacy

Face embeddings are biometric representations and should be treated as sensitive data.

A production deployment should consider explicit consent, secure storage, encryption at rest, access control, data retention policies, secure API authentication, and audit logging.

## Key Engineering Concepts Demonstrated

- Computer vision
- Face detection and alignment
- Deep face embeddings
- Pretrained deep learning models
- Vector similarity search
- Threshold-based classification
- Unknown-class rejection
- REST API development
- Database persistence
- Real-time webcam processing
- Automated testing
- Model evaluation
- Frontend/backend integration
- Error handling

## Conclusion

This project implements an end-to-end face identification pipeline using RetinaFace for detection, ArcFace for face embeddings, and cosine similarity for identity matching.

It provides identity enrollment, persistent embedding storage, UNKNOWN rejection, REST APIs, live webcam recognition, evaluation, automated tests, and a professional web interface.

The system achieved **100% accuracy on the available 6-sample evaluation set**, while explicitly documenting the limitations of the small test set and the need for broader validation before production deployment.
