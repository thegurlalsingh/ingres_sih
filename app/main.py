# app/main.py
import os
from fastapi import FastAPI, Depends, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import FileResponse
from .database import engine, Base, get_db
from . import etl, chatbot, i18n, visualization
from .schemas import IngestResponse, QueryRequest, ChatResponse
from pathlib import Path

app = FastAPI(title="INGRES AI Backend")

# Create DB tables on startup (only if necessary)
@app.on_event("startup")
def startup_event():
    etl.init_db()
    # optionally build index - heavy; skip for first run
    # from .embeddings import build_index_from_db
    # build_index_from_db()

@app.post("/ingest_folder", response_model=IngestResponse)
def ingest_folder(folder_path: str):
    # expects server-local path where CSVs placed; for production use proper upload endpoint
    count = etl.bulk_ingest(folder_path)
    return {"success": True, "inserted": count}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    # upload single CSV and ingest rows
    tmp = Path("tmp_uploads")
    tmp.mkdir(exist_ok=True)
    filepath = tmp / file.filename
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)
    db = next(get_db())
    inserted = etl.ingest_csv_file(db, str(filepath))
    return {"success": True, "inserted": inserted}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: QueryRequest):
    res = chatbot.handle_query(req.query, lang=req.lang, top_k=req.top_k, visualize=req.return_visual)
    return {"answer": res["answer"], "sources": res["sources"], "image_url": res["image_url"]}

@app.get("/i18n/{lang}")
def get_translation(lang: str):
    return i18n.load_translation(lang)

@app.get("/static_image/{fname}")
def get_image(fname: str):
    path = Path("outputs") / fname
    if path.exists():
        return FileResponse(path)
    else:
        return {"error": "file not found"}

@app.get("/timeseries")
def timeseries(state: str, district: str, metric: str):
    data = visualization.get_timeseries_data(state, district, metric)
    return data
