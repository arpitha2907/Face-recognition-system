from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RecognitionResult:
    status: str
    identity: Optional[str]
    similarity: Optional[float]
    threshold: float
    reason: Optional[str] = None
