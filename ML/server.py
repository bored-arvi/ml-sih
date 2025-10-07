from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
import uvicorn

# Local imports
from .verify_adapter import (
    TRIAL_FILENAME,
    TRIAL_HEATMAP,
    run_verify_pipeline,
    build_trial_response,
)

BASE_DIR = Path(__file__).resolve().parent
# Allow overriding uploads directory via env (useful for serverless: set to /tmp/uploads)
uploads_root = os.environ.get("UPLOAD_DIR")
UPLOAD_DIR = Path(uploads_root) if uploads_root else (BASE_DIR / "uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ML Verification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/verify")
async def api_verify(
    file: UploadFile | None = File(default=None),
    filename: str = Form(default=TRIAL_FILENAME),
    certificate_id: str = Form(default="CERT-001"),
):
    saved_path: Path | None = None

    # Save uploaded file if present
    if file is not None and file.filename:
        safe_name = Path(file.filename).name
        saved_path = UPLOAD_DIR / safe_name
        try:
            with saved_path.open("wb") as out_f:
                shutil.copyfileobj(file.file, out_f)
            filename = safe_name
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    # Trial response path (mock)
    if filename == TRIAL_FILENAME:
        return JSONResponse(content=build_trial_response())

    # For non-trial, require a file to have been provided or exist in uploads
    candidate_path = saved_path or (UPLOAD_DIR / filename)
    if not candidate_path.exists():
        raise HTTPException(status_code=400, detail="File not found. Upload a file or provide a valid filename in uploads.")

    try:
        result = run_verify_pipeline(
            image_path=str(candidate_path),
            certificate_id=certificate_id,
            output_dir=str(UPLOAD_DIR),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}")

    return JSONResponse(content=result)


@app.get("/heatmap/{filename}")
def serve_heatmap(filename: str):
    # Only serve files that live under uploads directory
    safe_name = Path(filename).name
    heatmap_path = UPLOAD_DIR / safe_name
    if heatmap_path.exists() and heatmap_path.is_file():
        return FileResponse(str(heatmap_path), media_type="image/png")
    # Backward-compat: also check project root for precomputed assets
    legacy_path = BASE_DIR / safe_name
    if legacy_path.exists() and legacy_path.is_file():
        return FileResponse(str(legacy_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    # For local development
    uvicorn.run("ML.server:app", host="0.0.0.0", port=8000, reload=True)


