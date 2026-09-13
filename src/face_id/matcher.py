import numpy as np
from .schemas import RecognitionResult

def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Embedding has zero norm")
    return vector / norm

class CosineMatcher:
    def __init__(self, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def match(self, query: np.ndarray, gallery: list[tuple[str, np.ndarray]]) -> RecognitionResult:
        if not gallery:
            return RecognitionResult("UNKNOWN", None, None, self.threshold, "EMPTY_GALLERY")

        q = normalize(query)
        best_identity, best_score = None, -1.0
        for identity, embedding in gallery:
            score = float(np.dot(q, normalize(embedding)))
            if score > best_score:
                best_identity, best_score = identity, score

        if best_score >= self.threshold:
            return RecognitionResult("MATCH", best_identity, best_score, self.threshold)

        return RecognitionResult("UNKNOWN", None, best_score, self.threshold, "BELOW_THRESHOLD")
