# Face Recognition Identification System

Production-oriented face identification pipeline using RetinaFace for detection and ArcFace embeddings, with cosine-similarity matching and explicit UNKNOWN rejection.

## Architecture
Image -> RetinaFace -> single-face validation -> ArcFace embedding -> L2 normalization -> cosine similarity -> threshold -> MATCH / UNKNOWN

## Stack
- Python 3.10+
- InsightFace runtime
- RetinaFace detection
- ArcFace embeddings
- SQLite persistence
- scikit-learn evaluation
- pytest tests

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the configured InsightFace model pack.

## Enrollment
Put consented images here:
```text
data/enrolled/
  person_001/
    01.jpg
    02.jpg
  person_002/
    01.jpg
```

Then:
```bash
PYTHONPATH=src python -m face_id.cli enroll --root data/enrolled
```

Each image must contain exactly one face. Images with zero or multiple faces are rejected.

## Identification
```bash
PYTHONPATH=src python -m face_id.cli identify --image path/to/query.jpg
```

A successful result looks like:
```json
{"status":"MATCH","identity":"person_001","similarity":0.81,"threshold":0.45,"reason":null}
```

An unknown:
```json
{"status":"UNKNOWN","identity":null,"similarity":0.31,"threshold":0.45,"reason":"BELOW_THRESHOLD"}
```

## Configuration
`.env`:
```env
FACE_MODEL=buffalo_s
MATCH_THRESHOLD=0.45
DETECTION_THRESHOLD=0.50
DB_PATH=data/face_id.sqlite3
CTX_ID=-1
```

### Threshold
0.45 is only a starting operating point. A real deployment should select the threshold using a representative validation set of genuine and impostor comparisons, based on an acceptable false-accept rate (FAR) and false-reject rate (FRR). Do not tune on the final test set.

## Evaluation
Use:
```text
data/evaluation/
  known/
    person_001/
      01.jpg
    person_002/
      01.jpg
  unknown/
    stranger_01.jpg
    stranger_02.jpg
```

Run:
```bash
PYTHONPATH=src python scripts/evaluate.py --root data/evaluation
```

Reports accuracy, macro precision/recall/F1, FAR, FRR and mean inference time.

For meaningful results, evaluation images should differ from enrollment images in pose, lighting, camera distance, expression and/or capture device. Randomly splitting near-duplicate images can give misleadingly high scores.

## Failure cases
- `NO_FACE`: no detectable face
- `MULTIPLE_FACES`: more than one face detected
- `UNKNOWN`: best similarity is below threshold
- blur / small faces / extreme pose / occlusion: can reduce detection or similarity
- look-alike faces: can cause false accepts
- domain shift: can cause false rejects

## Production improvements
1. Calibrate threshold against a held-out validation set and target FAR.
2. Store multiple embeddings per identity for varied capture conditions.
3. Add face-quality checks for blur, pose, illumination and minimum face size.
4. Add liveness / presentation-attack detection for access-control use.
5. Encrypt biometric data at rest and in transit.
6. Add authentication, authorization, rate limiting and audit logging.
7. Avoid storing raw images unless necessary; define retention/deletion policies.
8. Expose the engine through an authenticated FastAPI service.
9. Benchmark latency and memory on deployment hardware.
10. Monitor FAR/FRR after deployment and recalibrate when the camera/domain changes.

## Model
RetinaFace is used for detection/alignment and ArcFace for discriminative face embeddings through InsightFace. `buffalo_s` is used as the default model pack for a lighter CPU/cloud deployment footprint. Verify model-weight licensing before commercial deployment.


## Web UI

The project includes a responsive high-end dark interface with a restrained 3D-inspired biometric scanner visual, enrollment workflow, live system status, similarity meter and explicit MATCH/UNKNOWN states.

Start the API and UI:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The UI talks to the same recognition engine through FastAPI endpoints:

- `GET /api/health`
- `POST /api/enroll`
- `POST /api/identify`

The visual treatment deliberately avoids generic dashboard/card-template styling: it uses a cinematic dark surface, restrained typography, depth via layered gradients/blur, a rotating biometric orb, and compact technical metadata.

## Privacy
Face embeddings are biometric identifiers. Use only with appropriate consent/legal basis, access controls, encryption, retention and deletion policies. This repository intentionally contains no real people's images.
