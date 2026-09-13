import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_id.config import settings
from face_id.database import EmbeddingStore
from face_id.engine import FaceEngine
from face_id.matcher import CosineMatcher

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def image_files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/evaluation")
    args = parser.parse_args()

    root = Path(args.root)
    engine = FaceEngine(settings.model_name, settings.detection_threshold, settings.ctx_id)
    store = EmbeddingStore(settings.db_path)
    matcher = CosineMatcher(settings.match_threshold)
    gallery = store.all_embeddings()

    y_true, y_pred, timings, failures = [], [], [], []

    known_root = root / "known"
    unknown_root = root / "unknown"

    for person_dir in sorted(p for p in known_root.iterdir() if p.is_dir()):
        for image in image_files(person_dir):
            start = time.perf_counter()
            try:
                result = matcher.match(engine.embedding_from_file(image), gallery)
                y_true.append(person_dir.name)
                y_pred.append(result.identity if result.status == "MATCH" else "UNKNOWN")
            except Exception as exc:
                failures.append({"image": str(image), "error": str(exc)})
            timings.append(time.perf_counter() - start)

    for image in image_files(unknown_root):
        start = time.perf_counter()
        try:
            result = matcher.match(engine.embedding_from_file(image), gallery)
            y_true.append("UNKNOWN")
            y_pred.append(result.identity if result.status == "MATCH" else "UNKNOWN")
        except Exception as exc:
            failures.append({"image": str(image), "error": str(exc)})
        timings.append(time.perf_counter() - start)

    labels = sorted(set(y_true) | set(y_pred))
    known_n = sum(x != "UNKNOWN" for x in y_true)
    unknown_n = sum(x == "UNKNOWN" for x in y_true)

    metrics = {
        "samples_evaluated": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else None,
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if y_true else None,
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if y_true else None,
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if y_true else None,
        "false_accept_rate": sum(t == "UNKNOWN" and p != "UNKNOWN" for t,p in zip(y_true,y_pred)) / max(1, unknown_n),
        "false_reject_rate": sum(t != "UNKNOWN" and p == "UNKNOWN" for t,p in zip(y_true,y_pred)) / max(1, known_n),
        "mean_inference_seconds": statistics.mean(timings) if timings else None,
        "threshold": settings.match_threshold,
        "failures": failures
    }
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
