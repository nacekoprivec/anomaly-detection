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

CONFIG_DIR = os.path.abspath("configuration")
DATA_DIR = os.path.abspath("data")

def load_config(conf_name: str) -> Dict[str, Any]:
    config_file = os.path.join(CONFIG_DIR, conf_name)
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Config file '{conf_name}' not found."
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail=f"Config file '{conf_name}' contains invalid JSON."
        )

def create_available_configs_enum():
    """Returns config names as Enum"""
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

def create_json_config(body: dict, name: str) -> str:
    # Save the configuration to a file named detector_{name}.json 
    # return the filename
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_name = f"detector_{name}.json"
    config_path = os.path.join(CONFIG_DIR, config_name)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error creating config file {config_path}: {e}")
        raise

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
    
        detector_conf_name = create_json_config(config_data, request.name)

        detector = AnomalyDetector(
            name=request.name,
            description=request.description,
            updated_at=datetime.now(timezone.utc),
            status="inactive",
            config_name=detector_conf_name,
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


# READ datapoints/anomaly detectors

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
            if detector.config_name and detector.config_name.startswith("detector_"):
                config_path = os.path.join(CONFIG_DIR, detector.config_name)
                if os.path.exists(config_path):
                    os.remove(config_path)
            db.delete(detector)
            db.commit()
        return detector
    except Exception as e:
        db.rollback()
        print(f"Error stopping anomaly detector: {e}")
        return None

def delete_all_detectors(db: Session):
    try:
        detectors = db.query(AnomalyDetector).all()
        for detector in detectors:
            if detector.config_name and detector.config_name.startswith("detector_"):
                config_path = os.path.join(CONFIG_DIR, detector.config_name)
                if os.path.exists(config_path):
                    os.remove(config_path)
            db.delete(detector)
        db.commit()
        return len(detectors)
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

#deprecated
# def delete_all_logs(db: Session):
#     try:
#         num_deleted = db.query(Log).delete()
#         db.commit()
#         return num_deleted
#     except Exception as e:
#         db.rollback()
#         print(f"Error deleting all logs: {e}")
#         return 0

# def delete_log(log_id: int, db: Session):
#     try:
#         log = db.query(Log).filter(Log.id == log_id).first()
#         if log:
#             db.delete(log)
#             db.commit()
#         return log
#     except Exception as e:
#         db.rollback()
#         print(f"Error deleting log: {e}")
#         return None
    
# def get_logs(db: Session, skip: int = 0, limit: int = 10):
#     return db.query(Log).offset(skip).limit(limit).all()

# def get_log(db: Session, log_id: int):
#     return db.query(Log).filter(Log.id == log_id).first()