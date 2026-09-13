from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("FACE_MODEL", "buffalo_s")
    match_threshold: float = float(os.getenv("MATCH_THRESHOLD", "0.45"))
    detection_threshold: float = float(os.getenv("DETECTION_THRESHOLD", "0.50"))
    db_path: str = os.getenv("DB_PATH", "data/face_id.sqlite3")
    ctx_id: int = int(os.getenv("CTX_ID", "-1"))

settings = Settings()
