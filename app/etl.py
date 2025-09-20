# app/etl.py
import os
import pandas as pd
from sqlalchemy.orm import Session
from .models import DistrictData, EmbeddingStore
from .database import SessionLocal, engine
from .utils import detect_language
from tqdm import tqdm

# call this to create tables
def init_db():
    from .database import Base
    Base.metadata.create_all(bind=engine)

def canonicalize_row(row: pd.Series) -> dict:
    # create canonical metrics dict from raw csv row; adapt column names to your csvs
    metrics = {}
    for col in row.index:
        val = row[col]
        # try numeric conversion where appropriate
        try:
            if pd.isnull(val):
                metrics[col] = None
            else:
                metrics[col] = float(val) if isinstance(val, (int, float, str)) and str(val).replace('.', '', 1).isdigit() else val
        except:
            metrics[col] = val
    return metrics

def make_doc_text(state, district, year, metrics):
    # pick high-value fields - adapt
    parts = [f"State: {state}", f"District: {district}", f"Year: {year}"]
    # include a subset of important metrics
    for k in ["Rainfall (mm)", "Ground Water Recharge (ham)", "Total Geographical Area (ha)"]:
        if k in metrics and metrics[k] is not None:
            parts.append(f"{k}: {metrics[k]}")
    # include stage if exists
    if "Stage of Groundwater Extraction" in metrics:
        parts.append(f"Stage: {metrics['Stage of Groundwater Extraction']}")
    # also include small JSON snippet
    return " | ".join(parts)

def ingest_csv_file(db: Session, filepath: str):
    df = pd.read_csv(filepath)
    inserted = 0
    for _, row in df.iterrows():
        state = row.get("State", row.get("state", "Unknown"))
        district = row.get("District", row.get("district", "Unknown"))
        year = int(row.get("Year", row.get("year", 0) or 0))
        metrics = canonicalize_row(row)
        doc_text = make_doc_text(state, district, year, metrics)
        category = metrics.get("Stage of Groundwater Extraction", None)
        dd = DistrictData(state=state, district=district, year=year, unit=None, metrics=metrics, doc_text=doc_text, category=category)
        db.add(dd)
        inserted += 1
    db.commit()
    return inserted

def bulk_ingest(folder_path: str):
    db = SessionLocal()
    total = 0
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            path = os.path.join(folder_path, fname)
            print("Ingest:", path)
            n = ingest_csv_file(db, path)
            total += n
    db.close()
    return total
