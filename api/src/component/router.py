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

import Test

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
    try:
        config = load_config(config_name)
        return JSONResponse(content=config)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/configuration/{config_name}")
async def detect_with_custom_config(config_name: str, request: Request):
    tmp_file_path = os.path.join(CONFIG_DIR, "tmp.json")
    overrides = {}
    try:
        default_config = load_config(config_name)
        overrides = await request.json()
        merged_config = {**default_config, **overrides}

        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(tmp_file_path, "w") as tmp_file:
            json.dump(merged_config, tmp_file)

        return JSONResponse(content={"status": "OK", "used_config": config_name})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"{e} - {overrides}"}
        )

### Anomaly Detectors Crud operations

@router.get("/detectors/{detector_id}/parameters")
async def get_detector_parameters(detector_id: int, db: Session = Depends(get_db)):
    detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
    config = json.loads(detector.config) if detector and detector.config else {}
    if not detector:
        raise HTTPException(status_code=404, detail="config not found")
    return config["anomaly_detection_conf"]

@router.post("/detectors/{detector_id}/{timestamp}&{ftr_vector}")
async def is_anomaly(
    detector_id: int,
    timestamp: str,
    ftr_vector: float,
    db: Session = Depends(get_db)
):
    """
    Check if the given vodostaj is an anomaly.
    Data comes both from the URL and JSON body.
    """
    try:
        detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
        if detector.status != "active":
            raise HTTPException(status_code=400, detail="Detector is not active")
        
        if not detector:
            raise HTTPException(status_code=404, detail="Detector not found")
        data = {
                    "timestamp": float(timestamp),
                    "ftr_vector": [ftr_vector]  
        }
        print(detector.config_name)
        args = argparse.Namespace(
                            config=detector.config_name,
                            data_file=False,
                            data_both=False,
                            watchdog=False,
                            test=True,
                            param_tunning=False,
                            data=data
                        )

        test_instance = main.start_consumer(args)
        return test_instance.pred_is_anomaly

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "type": e.__class__.__name__,
                "traceback": traceback.format_exc()
            }
        )
    
@router.post("/detectors/")
def create_detector_db(request: DetectorCreateRequest, db: Session = Depends(get_db)):
    detector = create_anomaly_detector(request, db)
    return {
        "detector": detector
    }

@router.get("/detectors")
def get_detectors(db: Session = Depends(get_db)):
    detectors = db.query(AnomalyDetector).all()
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
    detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
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

# detector set status
@router.put("/detectors/{detector_id}/{status}")
def set_detector_status_db(detector_id: int, status: str, db: Session = Depends(get_db)):
    print("Setting status to:", status)
    detector = set_detector_status(detector_id, status, db)
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    return {
        "id": detector.id,
        "name": detector.name,
        "description": detector.description,
        "created_at": detector.created_at,
        "updated_at": detector.updated_at,
        "status": detector.status
    }

@router.put("/detectors/{detector_id}")
def update_anomaly_detector_db(detector_id: int, request: DetectorUpdateRequest, db: Session = Depends(get_db)):
    detector = update_anomaly_detector(
        db,
        detector_id,
        name=request.name,
        description=request.description
    )
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    return {
        "id": detector.id,
        "name": detector.name,
        "description": detector.description,
        "created_at": detector.created_at,
        "updated_at": detector.updated_at,
        "status": detector.status
    }

@router.delete("/detectors/{detector_id}")
def delete_detector_db(detector_id: int, db: Session = Depends(get_db)):
    detector = delete_anomaly_detector(detector_id, db)
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

@router.delete("/detectors")
def delete_all_detectors_db(db: Session = Depends(get_db)):
    delete_all_detectors(db)
    return JSONResponse(content={"status": "OK"})

### Logs Crud operations
@router.get("/logs")
async def get_logs_db(db: Session = Depends(get_db)):
    logs = db.query(Log).all()
    return [
        {
            "id": log.id,
            "start_at": log.start_at,
            "end_at": log.end_at,
            "config": json.loads(log.config),
            "duration_formated": format_seconds(log.duration_seconds)
        }
        for log in logs
    ]

@router.delete("/logs/{log_id}")
def delete_log_db(log_id: int, db: Session = Depends(get_db)):
    delete_log(log_id, db)
    return JSONResponse(content={"status": "OK"})

@router.delete("/logs")
def delete_logs_db(db: Session = Depends(get_db)):
    delete_all_logs(db)
    return JSONResponse(content={"status": "OK"})

@router.get("/available_configs")
async def get_available_configs():
    AvailableConfigs = create_available_configs_enum()
    return [
        {"name": config.name, "value": config.value}
        for config in AvailableConfigs
    ]