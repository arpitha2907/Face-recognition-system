from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, model_name="buffalo_s", detection_threshold=0.50, ctx_id=-1):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_thresh=detection_threshold)

    def read_image(self, path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        return image

    def extract_single_embedding(self, image: np.ndarray) -> np.ndarray:
        faces = self.app.get(image)
        if len(faces) == 0:
            raise ValueError("NO_FACE")
        if len(faces) > 1:
            raise ValueError("MULTIPLE_FACES")
        embedding = faces[0].embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("INVALID_EMBEDDING")
        return embedding / norm

    def embedding_from_file(self, path: str | Path) -> np.ndarray:
        return self.extract_single_embedding(self.read_image(path))
