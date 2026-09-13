import sqlite3
from pathlib import Path
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(identity_id) REFERENCES identities(id) ON DELETE CASCADE
);
"""

class EmbeddingStore:
    def __init__(self, db_path: str):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_identity(self, identity: str) -> int:
        self.conn.execute("INSERT OR IGNORE INTO identities(identity) VALUES (?)", (identity,))
        row = self.conn.execute("SELECT id FROM identities WHERE identity=?", (identity,)).fetchone()
        self.conn.commit()
        return int(row[0])

    def replace_embeddings(self, identity: str, vectors: list[np.ndarray]) -> None:
        identity_id = self.upsert_identity(identity)
        self.conn.execute("DELETE FROM embeddings WHERE identity_id=?", (identity_id,))
        for vector in vectors:
            v = np.asarray(vector, dtype=np.float32)
            self.conn.execute(
                "INSERT INTO embeddings(identity_id, embedding, dimension) VALUES (?, ?, ?)",
                (identity_id, v.tobytes(), int(v.size)),
            )
        self.conn.commit()

    def all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        rows = self.conn.execute(
            "SELECT i.identity, e.embedding, e.dimension FROM identities i "
            "JOIN embeddings e ON i.id=e.identity_id ORDER BY i.identity, e.id"
        ).fetchall()
        return [
            (identity, np.frombuffer(blob, dtype=np.float32, count=dim).copy())
            for identity, blob, dim in rows
        ]

    def close(self):
        self.conn.close()
