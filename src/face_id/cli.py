import argparse
import json
import sys
from pathlib import Path

from .config import settings
from .database import EmbeddingStore
from .engine import FaceEngine
from .matcher import CosineMatcher

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def image_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def cmd_enroll(args):
    root = Path(args.root)
    engine = FaceEngine(settings.model_name, settings.detection_threshold, settings.ctx_id)
    store = EmbeddingStore(settings.db_path)

    identities = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not identities:
        raise SystemExit(f"No identity directories found under {root}")

    for person_dir in identities:
        vectors, failures = [], []
        for image_path in image_files(person_dir):
            try:
                vectors.append(engine.embedding_from_file(image_path))
            except Exception as exc:
                failures.append({"image": str(image_path), "error": str(exc)})
        if vectors:
            store.replace_embeddings(person_dir.name, vectors)
        print(json.dumps({
            "identity": person_dir.name,
            "enrolled_images": len(vectors),
            "rejected_images": failures,
            "status": "ENROLLED" if vectors else "NOT_ENROLLED"
        }))
    store.close()

def cmd_identify(args):
    engine = FaceEngine(settings.model_name, settings.detection_threshold, settings.ctx_id)
    store = EmbeddingStore(settings.db_path)
    matcher = CosineMatcher(settings.match_threshold)
    try:
        query = engine.embedding_from_file(args.image)
        result = matcher.match(query, store.all_embeddings())
        print(json.dumps(result.__dict__, indent=2))
    except ValueError as exc:
        reason = str(exc)
        print(json.dumps({
            "status": reason if reason in {"NO_FACE", "MULTIPLE_FACES"} else "ERROR",
            "identity": None,
            "similarity": None,
            "threshold": settings.match_threshold,
            "reason": reason
        }, indent=2))
        sys.exit(2)
    finally:
        store.close()

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Identification System")
    sub = parser.add_subparsers(required=True)

    enroll = sub.add_parser("enroll")
    enroll.add_argument("--root", default="data/enrolled")
    enroll.set_defaults(func=cmd_enroll)

    identify = sub.add_parser("identify")
    identify.add_argument("--image", required=True)
    identify.set_defaults(func=cmd_identify)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
