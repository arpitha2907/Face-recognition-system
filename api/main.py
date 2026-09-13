
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.face_id.config import settings
from src.face_id.database import EmbeddingStore
from src.face_id.engine import FaceEngine
from src.face_id.matcher import CosineMatcher

app = FastAPI(title="Face Identification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

def get_engine():
    global engine
    if engine is None:
        engine = FaceEngine(
            settings.model_name,
            settings.detection_threshold,
            settings.ctx_id,
        )
    return engine

@app.get("/api/health")
def health():
    store = EmbeddingStore(settings.db_path)
    gallery = store.all_embeddings()
    store.close()
    return {
        "status": "online",
        "model": settings.model_name,
        "threshold": settings.match_threshold,
        "identities": len(set(x[0] for x in gallery)),
        "embeddings": len(gallery),
    }

@app.post("/api/identify")
async def identify(file: UploadFile = File(...)):
    suffix = Path(file.filename or ".jpg").suffix or ".jpg"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    store = EmbeddingStore(settings.db_path)
    try:
        emb = get_engine().embedding_from_file(tmp_path)
        result = CosineMatcher(settings.match_threshold).match(
            emb, store.all_embeddings()
        )
        return result.__dict__
    except ValueError as exc:
        reason = str(exc)
        if reason in {"NO_FACE", "MULTIPLE_FACES"}:
            return {
                "status": reason,
                "identity": None,
                "similarity": None,
                "threshold": settings.match_threshold,
                "reason": reason,
            }
        raise HTTPException(status_code=400, detail=reason)
    finally:
        store.close()
        Path(tmp_path).unlink(missing_ok=True)
@app.post("/api/camera/identify")
async def camera_identify(file: UploadFile = File(...)):
    """
    Identify a face from a live camera frame.
    Designed for repeated browser webcam requests.
    """
    data = await file.read()

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    store = EmbeddingStore(settings.db_path)

    try:
        emb = get_engine().embedding_from_file(tmp_path)

        result = CosineMatcher(
            settings.match_threshold
        ).match(
            emb,
            store.all_embeddings()
        )

        return result.__dict__

    except ValueError as exc:
        reason = str(exc)

        if reason in {"NO_FACE", "MULTIPLE_FACES"}:
            return {
                "status": reason,
                "identity": None,
                "similarity": None,
                "threshold": settings.match_threshold,
                "reason": reason,
            }

        raise HTTPException(
            status_code=400,
            detail=reason
        )

    finally:
        store.close()
        Path(tmp_path).unlink(missing_ok=True)
@app.post("/api/enroll")
async def enroll(
    identity: str = Form(...),
    files: list[UploadFile] = File(...),
):
    identity = identity.strip()
    if not identity:
        raise HTTPException(status_code=400, detail="Identity is required.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    vectors = []
    rejected = []
    store = EmbeddingStore(settings.db_path)

    try:
        for file in files:
            suffix = Path(file.filename or ".jpg").suffix or ".jpg"
            data = await file.read()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                vectors.append(get_engine().embedding_from_file(tmp_path))
            except Exception as exc:
                rejected.append({"file": file.filename, "reason": str(exc)})
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        if not vectors:
            raise HTTPException(
                status_code=400,
                detail={"message": "No valid face images could be enrolled.", "rejected": rejected},
            )

        store.replace_embeddings(identity, vectors)
        return {
            "status": "ENROLLED",
            "identity": identity,
            "enrolled_images": len(vectors),
            "rejected_images": rejected,
        }
    finally:
        store.close()
