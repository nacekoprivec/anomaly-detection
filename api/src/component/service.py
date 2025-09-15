import os
import numpy as np
from api.src.component.schemas import AvailableConfigs, DetectorCreateRequest
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
from enum import Enum

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


def create_available_configs_enum():
    files = [
        f for f in os.listdir(CONFIG_DIR)
        if os.path.isfile(os.path.join(CONFIG_DIR, f)) and f.endswith(".json")
    ]

    # Format enum member names: remove extension, replace spaces with underscore, capitalize
    enum_members = {}
    for f in files:
        name = os.path.splitext(f)[0]  # remove .json
        name = "".join(word.capitalize() for word in name.split("_"))  # CamelCase
        enum_members[name] = f

    return Enum("AvailableConfigs", enum_members)

# Create json config
def create_json_config(body: dict, name: str) -> str:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_name = f"detector_{name}.json"
    config_path = os.path.join(CONFIG_DIR, config_name)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            existing_config = json.load(f)
        existing_config.update(body)
        body = existing_config
    except FileNotFoundError:
        pass

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)

    return config_name

# CREATE anomaly detectors/logs/datapoints
def create_anomaly_detector(request: DetectorCreateRequest, db: Session) -> AnomalyDetector:
    try:
        if request.config_name:
            config_data = load_config(request.config_name)
        else:
            if not request.anomaly_detection_alg or not request.anomaly_detection_conf:
                raise ValueError("Either config_name or anomaly_detection_alg + anomaly_detection_conf must be provided")
            config_data = {
                "anomaly_detection_alg": request.anomaly_detection_alg,
                "anomaly_detection_conf": request.anomaly_detection_conf
            }
            request.config_name = create_json_config(config_data, request.name)

        detector = AnomalyDetector(
            name=request.name,
            description=request.description,
            updated_at=datetime.now(timezone.utc),
            status="inactive",
            config_name=request.config_name,
            config=json.dumps(config_data)
        )
        db.add(detector)
        db.commit()        
        db.refresh(detector)
  
        return detector

    except FileNotFoundError:
        db.rollback()
        raise HTTPException(
            status_code=404,
            detail=f"Config file not found."
        )
    except json.JSONDecodeError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Config file contains invalid JSON."
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
) -> Optional[AnomalyDetector]:
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
        if detector:
            db.delete(detector)
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