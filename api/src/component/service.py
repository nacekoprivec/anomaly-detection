import os
import numpy as np
from api.src.component.schemas import AvailableConfigs
import main

import argparse
import tempfile
import json

import main

from .models import *
from ..database import get_db
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import time, timedelta
from typing import Optional
import pandas as pd
import time
from typing import Dict, Any, Optional

# depracated later
CONFIG_DIR = os.path.abspath("configuration")
DATA_DIR = os.path.abspath("data")

def load_config(conf_name: str) -> Dict[str, Any]:
    config_file = os.path.join(CONFIG_DIR, conf_name)
    with open(config_file, "r") as f:
        return json.load(f)

def handle_configuration(body: dict) -> str:
    with open("C:\\Users\\nacek\\OneDrive\\Desktop\\siht\\DataPoint-detection\\configuration", "r") as f:
        required_config = json.load(f)

    for key, default_value in required_config.items():
        body.setdefault(key, default_value)

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".json") as tmp:
        json.dump(body, tmp)
        config_path = tmp.name

    args = argparse.Namespace(**body)
    args.config = config_path

    main.start_consumer(args)

    return config_path

def scrape_data(n: int):
    url = "http://hmljn.arso.gov.si/vode/podatki/stanje_voda_samodejne.html"
    tables = pd.read_html(url)

    df = tables[2]

    vodostaj = df["Vodostaj", "cm"].head(n)  

    timestamp = float(time.time())

    datapoints = []
    for idx, value in enumerate(vodostaj):
        datapoints.append({
            "place_id": n-idx,     
            "timestamp": timestamp,
            "vodostaj": float(value) if pd.notna(value) else None
        })
    return datapoints

# Calculate confusion matrix

def confusion_matrix(tp, fp, fn, tn) -> None:
        """Confusion matrix for anomaly detection"""
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return [precision, recall, f1]

# CREATE anomaly detectors/logs/datapoints
def create_anomaly_detector(db: Session, name: str, description: str = None, config_name: str = "border_check.json") -> AnomalyDetector:
    try:
        config_data = load_config(config_name)

        detector = AnomalyDetector(
            name=name,
            description=description,
            updated_at=datetime.now(timezone.utc),
            status="active"
        )
        db.add(detector)
        db.flush() 

        log_entry = Log(
            detector_id=detector.id,
            config=json.dumps(config_data),
            config_name=config_name,
        )
        db.add(log_entry)
        db.commit()
        db.refresh(detector)

        return detector

    except FileNotFoundError:
        db.rollback()
        raise HTTPException(
            status_code=404,
            detail=f"Config file '{config_name.value}' not found."
        )
    except json.JSONDecodeError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Config file '{config_name.value}' contains invalid JSON."
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {e}"
        )

# deprecated
def create_log(db: Session, config):
    log_entry = Log(
        config=json.dumps(config),
    )

    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    return log_entry

# READ logs/datapoints/anomaly detectors

def get_logs(db: Session, skip: int = 0, limit: int = 10):
    return db.query(Log).offset(skip).limit(limit).all()

def get_log(db: Session, log_id: int):
    return db.query(Log).filter(Log.id == log_id).first()

def get_anomaly_detector(db: Session, detector_id: int):
    return db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()

def get_anomaly_detectors(db: Session, skip: int = 0, limit: int = 50):
    return db.query(AnomalyDetector).offset(skip).limit(limit).all()

# UPDATE anomaly detector/log

def update_anomaly_detector(
    db: Session,
    detector_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
    if not detector:
        return None
    if name is not None:
        detector.name = name
    if description is not None:
        detector.description = description
    detector.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(detector)
    return detector

# Delete logs/datapoints/anomaly detectors

def delete_anomaly_detector(detector_id: int, db: Session):
    try:
        detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
        log = db.query(Log).filter(Log.detector_id == detector_id).first()
        if detector:
            log.end_at = datetime.now(timezone.utc)
            log.duration_seconds = int((log.end_at - log.start_at).total_seconds())
            db.add(log)
            detector.status = "inactive"
            db.add(detector)
            db.commit()
        return detector
    except Exception as e:
        db.rollback()
        print(f"Error stopping anomaly detector: {e}")
        return None

def delete_log(log_id: int, db: Session):
    try:
        log = db.query(Log).filter(Log.id == log_id).first()
        if log:
            db.delete(log)
            db.commit()
        return log
    except Exception as e:
        db.rollback()
        print(f"Error deleting log: {e}")
        return None
    
def delete_all_logs(db: Session):
    try:
        num_deleted = db.query(Log).delete()
        db.commit()
        return num_deleted
    except Exception as e:
        db.rollback()
        print(f"Error deleting all logs: {e}")
        return 0

def delete_all_detectors(db: Session):
    try:
        num_deleted = db.query(AnomalyDetector).delete()
        db.commit()
        return num_deleted
    except Exception as e:
        db.rollback()
        print(f"Error deleting all detectors: {e}")
        return 0
    
# set anomaly detector active/inactive
def set_detector_status(detector_id: int, status: str, db: Session):
    try:
        detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
        if detector:
            detector.status = status
            db.add(detector)
            db.commit()
            db.refresh(detector)
        return detector
    except Exception as e:
        db.rollback()
        print(f"Error updating detector status: {e}")
        return None 
    

# Format HH:MM:SS
def format_seconds(seconds: float) -> str:
    if seconds is None:
        return "N/A"
    seconds = round(seconds)
    return seconds

# Print Statements

def detect_datapoints():
    print("Detecting datapoints...")

    return 0

def configuration():
    print("Configuring...")

    return 0

def create_config():
    print("Creating configuration...")

    return 0