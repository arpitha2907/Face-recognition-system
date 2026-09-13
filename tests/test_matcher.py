import numpy as np
from src.face_id.matcher import CosineMatcher

def test_known_face_matches():
    result = CosineMatcher(0.8).match(
        np.array([1.0, 0.0]), [("alice", np.array([1.0, 0.0]))]
    )
    assert result.status == "MATCH"
    assert result.identity == "alice"

def test_unknown_face_rejected():
    result = CosineMatcher(0.8).match(
        np.array([0.0, 1.0]), [("alice", np.array([1.0, 0.0]))]
    )
    assert result.status == "UNKNOWN"
    assert result.identity is None

def test_empty_gallery_is_unknown():
    result = CosineMatcher(0.8).match(np.array([1.0, 0.0]), [])
    assert result.status == "UNKNOWN"
    assert result.reason == "EMPTY_GALLERY"
