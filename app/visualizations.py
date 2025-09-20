# app/visualization.py
import os
import io
import matplotlib.pyplot as plt
from fastapi import UploadFile
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import DistrictData

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_static_chart_recharge_vs_extraction(state: str, district: str, years: list):
    db = SessionLocal()
    q = db.query(DistrictData).filter(DistrictData.state == state, DistrictData.district == district, DistrictData.year.in_(years))
    rows = q.all()
    db.close()
    years_sorted = sorted(rows, key=lambda r: r.year)
    x = [r.year for r in years_sorted]
    recharge = [r.metrics.get("Ground Water Recharge (ham)", 0) or 0 for r in years_sorted]
    extraction = [r.metrics.get("Total Extraction (ham)", 0) or 0 for r in years_sorted]

    plt.figure(figsize=(8,4))
    plt.plot(x, recharge, marker='o', label='Recharge (ham)')
    plt.plot(x, extraction, marker='o', label='Extraction (ham)')
    plt.title(f"{district}, {state} — Recharge vs Extraction")
    plt.xlabel("Year")
    plt.ylabel("Volume (ham)")
    plt.legend()
    fname = os.path.join(OUTPUT_DIR, f"{state}_{district}_recharge_extraction.png".replace(" ", "_"))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname

def get_timeseries_data(state: str, district: str, metric_key: str):
    db = SessionLocal()
    q = db.query(DistrictData).filter(DistrictData.state == state, DistrictData.district == district).order_by(DistrictData.year)
    rows = q.all()
    db.close()
    return {
        "years": [r.year for r in rows],
        "values": [r.metrics.get(metric_key, None) for r in rows],
        "metric": metric_key,
        "state": state,
        "district": district
    }
# app/visualization.py
import os
import io
import matplotlib.pyplot as plt
from fastapi import UploadFile
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import DistrictData

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_static_chart_recharge_vs_extraction(state: str, district: str, years: list):
    db = SessionLocal()
    q = db.query(DistrictData).filter(DistrictData.state == state, DistrictData.district == district, DistrictData.year.in_(years))
    rows = q.all()
    db.close()
    years_sorted = sorted(rows, key=lambda r: r.year)
    x = [r.year for r in years_sorted]
    recharge = [r.metrics.get("Ground Water Recharge (ham)", 0) or 0 for r in years_sorted]
    extraction = [r.metrics.get("Total Extraction (ham)", 0) or 0 for r in years_sorted]

    plt.figure(figsize=(8,4))
    plt.plot(x, recharge, marker='o', label='Recharge (ham)')
    plt.plot(x, extraction, marker='o', label='Extraction (ham)')
    plt.title(f"{district}, {state} — Recharge vs Extraction")
    plt.xlabel("Year")
    plt.ylabel("Volume (ham)")
    plt.legend()
    fname = os.path.join(OUTPUT_DIR, f"{state}_{district}_recharge_extraction.png".replace(" ", "_"))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname

def get_timeseries_data(state: str, district: str, metric_key: str):
    db = SessionLocal()
    q = db.query(DistrictData).filter(DistrictData.state == state, DistrictData.district == district).order_by(DistrictData.year)
    rows = q.all()
    db.close()
    return {
        "years": [r.year for r in rows],
        "values": [r.metrics.get(metric_key, None) for r in rows],
        "metric": metric_key,
        "state": state,
        "district": district
    }
