## ML Verification API

Production-ready FastAPI server that exposes the certificate verification pipeline.

### Endpoints
- `GET /health`: service health
- `POST /api/verify`: form-data fields `file` (UploadFile, optional), `filename` (str), `certificate_id` (str)
- `GET /heatmap/{filename}`: serve generated heatmaps from `ML/uploads`

If `filename` equals `trial color mismatch.png`, a mock response is returned.

### Run without Docker
Windows (PowerShell):
```powershell
cd ML
./run.ps1 -Port 8000
```

Linux/macOS:
```bash
cd ML
chmod +x run.sh
./run.sh 8000
```

### Docker
```bash
docker build -t ml-verify-api -f ML/Dockerfile .
docker run --rm -p 8000:8000 -v %cd%/ML/uploads:/app/ML/uploads ml-verify-api
# On Linux/Mac adjust volume mount accordingly
```

### Deploy to Render (no Docker)
Prereqs: A GitHub repo with this project.

1. Commit and push your code to GitHub.
2. Ensure these files exist at repo root: `render.yaml`, `requirements.txt` (already added).
3. In Render, create a New Web Service and choose your repo.
4. Render will auto-detect `render.yaml`. Confirm settings:
   - Environment: Python
   - Build command: `pip install -r ML/requirements.txt && pip install fastapi uvicorn[standard] python-multipart`
   - Start command: `uvicorn ML.server:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
5. Click Create Web Service. Wait for deploy to finish.
6. Test: open `https://<your-service>.onrender.com/health` then POST to `/api/verify`.

Note: If your pipeline requires assets (e.g., `.pkl`, `.faiss`, reference images), commit them or store in a cloud bucket your code can read at runtime.

### verify.py integration
Place your pipeline at `ML/verify.py` with the following functions:
```text
extract_text(path)
extract_fields_regex(text)
extract_fields_ner(text)
combine_db_fields(regex_fields, ner_fields)
extract_non_db_text(extracted_text, db_fields_text)
color_embedding(path)
compare_against_stored(certificate_id, new_non_db_text, new_color_emb, new_db_fields_text, new_image_path)
deltaE_heatmap(ref_image_path, new_image_path, output_path)
```

### Notes
- Uploads and heatmaps are stored under `ML/uploads`.
- For production, front a process manager (e.g. Docker or systemd) and consider HTTPS termination via a reverse proxy.


