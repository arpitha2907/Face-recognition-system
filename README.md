# Face Recognition Identification System

A production-oriented face recognition and identification system using pretrained RetinaFace and ArcFace models from InsightFace. It supports face enrollment, face embedding generation, cosine-similarity identification, UNKNOWN rejection, image identification, and real-time webcam recognition through FastAPI and a web interface.

## Features

- RetinaFace face detection
- ArcFace face embeddings
- L2-normalized embeddings
- Cosine similarity matching
- Configurable recognition threshold
- UNKNOWN rejection
- NO_FACE and MULTIPLE_FACES handling
- Multi-image enrollment
- SQLite embedding storage
- FastAPI REST API
- Real-time webcam identification
- Image upload identification
- Pipeline visualization
- Evaluation script
- Pytest unit tests
- Responsive web UI

## Architecture

Image / Webcam Frame
        |
        v
   RetinaFace
Face Detection + Alignment
        |
        v
      ArcFace
 Face Embedding Vector
        |
        v
   L2 Normalization
        |
        v
 Cosine Similarity
        |
        v
 Best Match Score
      /       \
 >= 0.45     < 0.45
    |            |
    v            v
  MATCH       UNKNOWN

## Technology Stack

**AI / Computer Vision:** InsightFace, RetinaFace, ArcFace, ONNX Runtime, OpenCV, NumPy

**Backend:** Python, FastAPI, SQLite, Uvicorn

**Frontend:** HTML, CSS, JavaScript, Browser MediaDevices API

**Testing:** Pytest

## Face Embeddings

The system uses the pretrained ArcFace recognition model provided by InsightFace to convert a detected face into a numerical embedding vector.

During enrollment, valid embeddings are stored in the SQLite gallery. During identification, the query embedding is compared against stored embeddings using cosine similarity.

Face -> RetinaFace -> ArcFace -> Embedding -> Cosine Similarity -> Decision

## Enrollment

Multiple images can be enrolled for an identity. Each image is processed through face detection and ArcFace embedding extraction. Images with no face, multiple faces, or invalid embeddings are rejected.

## Identification

The highest cosine similarity score is compared with the configured threshold.

Current threshold:0.45

If the score is at least 0.45, the identity is accepted. Otherwise the result is `UNKNOWN`.

## Real-Time Camera

The browser captures webcam frames and periodically sends JPEG frames to the FastAPI camera endpoint.

Webcam -> Frame -> FastAPI -> RetinaFace -> ArcFace -> Cosine Match

The UI displays the status of the RetinaFace, ArcFace, and Cosine Match stages.

## Failure Handling

| Condition | Response |

| No face detected | `NO_FACE` |
| Multiple faces detected | `MULTIPLE_FACES` |
| Similarity below threshold | `UNKNOWN` |
| Similarity above threshold | `MATCH` |
| API / invalid input error | `ERROR` |

## API Endpoints

### Health

```http
GET /api/health
```
Returns model, threshold, identity count, and embedding count.

### Image Identification

```http
POST /api/identify
```

Example response:
{
  "status": "MATCH",
  "identity": "person_01",
  "similarity": 0.68,
  "threshold": 0.45,
  "reason": null
}

### Camera Identification

```http
POST /api/camera/identify
```

Processes a webcam frame.

### Enrollment

```http
POST /api/enroll
```

Accepts an identity name and one or more images.

## Project Structure

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
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── data/
    ├── enrolled/
    └── evaluation/

## Installation

Python 3.11 is recommended.

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

InsightFace downloads the required pretrained model when initialized.

## Running

python main.py

Then use the web interface to enroll identities, upload images, or start the webcam.

## Configuration

Model: buffalo_l
Detection threshold: 0.25
Matching threshold: 0.45
Detection size: 640 x 640
Execution provider: CPU

## Evaluation

Run: python scripts/evaluate.py
Current evaluation:

| Metric | Result | :
| Samples evaluated | 6 |
| Accuracy | 100% |
| Macro Precision | 100% |
| Macro Recall | 100% |
| Macro F1 | 100% |
| False Accept Rate | 0% |
| False Reject Rate | 0% |
| Mean inference time | 0.379 seconds |
| Threshold | 0.45 |
| Failures | 0 |

**Important:** The 100% result is from only 6 evaluation samples and should not be interpreted as 100% real-world accuracy. A larger and more diverse test set is required for reliable performance estimation.

## Model Training

The project does **not** train or fine-tune RetinaFace or ArcFace. Both are pretrained InsightFace models.

Therefore, conventional training accuracy is **N/A**.

The project-specific engineering work covers enrollment, embedding extraction and storage, similarity matching, threshold-based rejection, API integration, real-time camera processing, evaluation, and UI.

## Testing

Run: pytest -q

Current result: 3 passed in 0.14s

## Performance

Current measured mean inference time is approximately: 0.379 seconds per sample

The current environment uses CPU inference. Performance varies with hardware, image resolution, detection size, and execution provider.

## Limitations

- Small evaluation dataset
- Threshold has not been calibrated on a large validation dataset
- Optimized for single-face identification
- CPU inference introduces latency
- Recognition can vary with lighting, pose, blur, and occlusion
- No liveness / anti-spoofing detection
- Additional security and privacy controls are required for high-stakes deployment

## Future Improvements

- Larger and more diverse evaluation datasets
- FAR/FRR-based threshold calibration
- ROC and Precision-Recall curves
- GPU acceleration
- Liveness detection
- Face quality assessment
- Authentication and authorization
- Encryption for stored embeddings
- Audit logging
- Vector database for large galleries
- Multi-face tracking
- Docker and cloud deployment
- Monitoring and performance metrics

## Security and Privacy

Face embeddings are biometric representations and should be treated as sensitive data.

A production deployment should use appropriate consent, access control, encryption, retention policies, secure APIs, and audit logging.

## Key Concepts Demonstrated

- Computer vision
- Face detection and alignment
- Deep face embeddings
- Pretrained deep learning models
- Vector similarity
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

This project implements an end-to-end face identification pipeline using RetinaFace for detection, ArcFace for face embeddings, and cosine similarity for identity matching. It provides enrollment, persistent embedding storage, UNKNOWN rejection, REST APIs, live webcam recognition, evaluation, and a professional web interface.
