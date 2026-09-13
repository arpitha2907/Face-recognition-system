
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.main import app

app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path("web/index.html"))
