import asyncio
import os
import json
import tempfile
import argparse
from typing import Dict, Any, Optional
import traceback

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy import Float

import main
from .service import *


import pandas as pd

from datetime import datetime

from .models import Log, AnomalyDetector 
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db

from .schemas import *
from .exceptions import *

CONFIG_DIR = os.path.abspath("configuration")
DATA_DIR = os.path.abspath("data")

router = APIRouter()

@router.get("/configuration/{config_name}")
async def detect_with_custom_config(config_name: str):
    """
    Endpoint for loading and returning a configuration by name, for name you provided.
    """
    try:
        print("Loading config:", config_name)
        config = load_config(config_name)
        if not config:
            raise ConfigFileException(config_name, "Config file is empty or missing required fields.")
        return JSONResponse(content=config)
    except FileNotFoundError:
        raise ConfigFileException(config_name, "Config file not found.")
    except ValueError as ve: 
        raise JSONDecodeException(config_name, str(ve))
    except ConfigFileException:
        raise 
    except Exception as e:
        raise InternalServerException(str(e))

@router.post("/configuration/{config_name}")
async def override_config(config_name: str, request: Request, db: Session = Depends(get_db)):
    """
    Endpoint for loading a configuration by name, merging it with overrides from the request body,
    and saving it as detector_{detector_id}.json for the specified detector.
    """
    try:
        detector_id = request.detector_id 
        detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
        if not detector:
            raise DetectorNotFoundException(detector_id)
        
        file_path = os.path.join(CONFIG_DIR, f"detector_{detector.name}.json") # Path to the detector config file
        overrides = {}

        default_config = load_config(config_name)
        overrides = await request.json()

    
        merged_config = {**default_config, **overrides}

        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(file_path, "w") as detector_config_file:
            json.dump(merged_config, detector_config_file)

        return JSONResponse(content={"status": "OK", "used_config": config_name})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"{e} - {overrides}"}
        )

### Anomaly Detectors Crud operations

@router.get("/detectors/{detector_id}/parameters")
async def get_detector_parameters(detector_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the anomaly detection configuration parameters for a specific detector.
    """
    detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
    if not detector:
        raise DetectorNotFoundException(detector_id)

    try:
        if detector.config is None:
            raise ConfigFileException(str(detector_id), "Detector config is empty.")
        config = json.loads(detector.config)
    except json.JSONDecodeError as e:
        raise JSONDecodeException(str(detector_id), f"Invalid JSON in detector config: {e}")

    if "anomaly_detection_conf" not in config:
        raise ConfigFileException(str(detector_id), "Missing 'anomaly_detection_conf' section.")

    return  config["anomaly_detection_conf"]

@router.post("/detectors/{detector_id}/{timestamp}&{ftr_vector}")
async def is_anomaly(detector_id: int, timestamp: str, ftr_vector: float, db: Session = Depends(get_db)):
    detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
    
    if not detector:
        raise DetectorNotFoundException(detector_id)  
    if detector.status != "active":
        raise DetectorNotActiveException(detector_id)

    data = {
        "timestamp": float(timestamp),
        "ftr_vector": [ftr_vector]
    }

    args = argparse.Namespace(
        config=detector.config_name,
        data_file=False,
        data_both=False,
        watchdog=False,
        test=True,
        param_tunning=False,
        data=data
    )

    try:
        loop = asyncio.get_running_loop()
        test_instance = await loop.run_in_executor(None, lambda: main.start_consumer(args))
    except Exception as e:
        print("Exception inside start_consumer:", traceback.format_exc())
        raise ProcessingException(f"An error occurred in start_consumer: {e}")

    return test_instance.pred_is_anomaly
    
@router.post("/detectors/")
def create_detector_db(request: DetectorCreateRequest, db: Session = Depends(get_db)):
    """
    Create a new anomaly detector in the database and set its initial status to 'inactive'.
    Args:
        request (DetectorCreateRequest): 
            The detector creation request containing metadata (name, description) and configuration details. 
            Must include either:
                - `config_name`: The name of an existing configuration to load, OR
                - `anomaly_detection_alg` and `anomaly_detection_conf`: To build a new configuration.
    """
    detector = create_anomaly_detector(request, db)
    return {
        "detector": detector
    }

@router.get("/detectors")
def get_detectors(db: Session = Depends(get_db)):
    try:
        detectors = db.query(AnomalyDetector).all()
        if not detectors:
            raise DetectorNotFoundException
    except Exception as e:
        raise InternalServerException(f"Database error while fetching detectors: {e}")
    return [
        {
            "id": detector.id,
            "name": detector.name,
            "description": detector.description,
            "created_at": detector.created_at,
            "updated_at": detector.updated_at,
            "status": detector.status,
            "config_name": detector.config_name,
            "config" : detector.config
        }
        for detector in detectors
    ]

@router.get("/detectors/{detector_id}")
def get_detector(detector_id: int, db: Session = Depends(get_db)):
    try:
        detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
        if not detector:
            raise DetectorNotFoundException
    except Exception as e:
        raise InternalServerException(f"Database error while fetching detectors: {e}")
    return {
        "id": detector.id,
        "name": detector.name,
        "description": detector.description,
        "created_at": detector.created_at,
            "updated_at": detector.updated_at,
            "status": detector.status,
            "config_name": detector.config_name,
            "config" : detector.config
        }

@router.put("/detectors/{detector_id}/{status}")
def set_detector_status_db(detector_id: int, status: str, db: Session = Depends(get_db)):
    """Update the status of a detector (e.g., 'active', 'inactive')."""
    detector = set_detector_status(detector_id, status, db)
    if not detector:
        raise DetectorNotFoundException(detector_id)

    return detector

@router.put("/detectors/{detector_id}")
def update_anomaly_detector_db(detector_id: int, request: DetectorUpdateRequest, db: Session = Depends(get_db)):
    """Update the name and/or description of an existing anomaly detector."""
    detector = update_anomaly_detector(db, detector_id, name=request.name, description=request.description)
    if not detector:
        raise DetectorNotFoundException
    return detector

@router.delete("/detectors/{detector_id}")
def delete_detector_db(detector_id: int, db: Session = Depends(get_db)):
    """Delete a specific anomaly detector and its associated config file."""
    detector = delete_anomaly_detector(detector_id, db)
    return detector

@router.delete("/detectors")
def delete_all_detectors_db(db: Session = Depends(get_db)):
    """Delete all anomaly detectors and their associated config files."""
    try:
        detectors = db.query(AnomalyDetector).all()
        if not detectors:
            raise DetectorNotFoundException
        delete_all_detectors(db)
    except Exception as e:
        raise InternalServerException(f"Database error while deleting detectors: {e}")
    return JSONResponse(content={"status": "OK"})

@router.get("/available_configs")
async def get_available_configs():
    """
    Returns all available configuration filenames as:
    [{"name": config.name, "filename": config.value}, ...]
    Name isn't used for now.
    """
    try:
        AvailableConfigs = create_available_configs_enum()
        return [{"name": member.name, "filename": member.value} for member in AvailableConfigs]
    except InternalServerException:
        raise
    except Exception as e:
        raise InternalServerException(f"Failed to list available configs: {e}")

### Logs Crud operations deprecateed
# @router.get("/logs")
# async def get_logs_db(db: Session = Depends(get_db)):
#     logs = db.query(Log).all()
#     return [
#         {
#             "id": log.id,
#             "start_at": log.start_at,
#             "end_at": log.end_at,
#             "config": json.loads(log.config),
#             "duration_formated": format_seconds(log.duration_seconds)
#         }
#         for log in logs
#     ]

# @router.delete("/logs/{log_id}")
# def delete_log_db(log_id: int, db: Session = Depends(get_db)):
#     delete_log(log_id, db)
#     return JSONResponse(content={"status": "OK"})

# @router.delete("/logs")
# def delete_logs_db(db: Session = Depends(get_db)):
#     delete_all_logs(db)
#     return JSONResponse(content={"status": "OK"})
