import os
import json
import tempfile
import argparse
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy import Float

import main
from .service import *

import Test

import pandas as pd

from datetime import datetime

from .models import Log, DataPoint, AnomalyDetector 
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db

from .schemas import *
from .exceptions import *


CONFIG_DIR = os.path.abspath("configuration")
DATA_DIR = os.path.abspath("data")

router = APIRouter()

def load_config(conf_name: str) -> Dict[str, Any]:
    config_file = os.path.join(CONFIG_DIR, conf_name)
    with open(config_file, "r") as f:
        return json.load(f)

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

@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    config_name: Optional[str] = Form(None)
):
    tmp_config_path = None
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        data_file_path = os.path.join(DATA_DIR, "tmp.csv")
        with open(data_file_path, "wb") as f:
            f.write(await file.read())

        config_to_load = config_name or "tmp.json"
        base_config = load_config(config_to_load)
        base_config["file_name"] = "data/tmp.csv"

        os.makedirs(CONFIG_DIR, exist_ok=True)
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False, dir=CONFIG_DIR) as tmp_file:
            json.dump(base_config, tmp_file)
            tmp_config_path = tmp_file.name

        return JSONResponse(content={
            "status": "OK",
            "used_config": os.path.basename(tmp_config_path),
            "used_data_file": "tmp.csv"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
    
@router.post("/detectors/")
def create_detector(request: DetectorCreateRequest, db: Session = Depends(get_db)):
    # Create the detector
    detector = AnomalyDetector(
        name=request.name,
        description=request.description,
        updated_at=datetime.now(timezone.utc)
    )
    db.add(detector)
    db.commit()
    db.refresh(detector)

    # Load config and create log
    config_dict = load_config(request.config_name)
    log = Log(
        detector_id=detector.id,
        start_at=datetime.now(timezone.utc),
        config=json.dumps(config_dict)
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    # Return combined info
    return {
        "detector": {
            "id": detector.id,
            "name": detector.name,
            "description": detector.description,
            "created_at": detector.created_at,
            "updated_at": detector.updated_at
        },
        "log": {
            "id": log.id,
            "start_at": log.start_at,
            "config": config_dict
        }
    }

@router.get("/detectors/")
def get_detectors(db: Session = Depends(get_db)):
    detectors = db.query(AnomalyDetector).all()
    return [
        {
            "id": detector.id,
            "name": detector.name,
            "description": detector.description,
            "created_at": detector.created_at,
            "updated_at": detector.updated_at
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
        "updated_at": detector.updated_at
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
        "updated_at": detector.updated_at
    }

@router.delete("/detectors/{detector_id}")
def delete_detector_db(detector_id: int, db: Session = Depends(get_db)):
    delete_anomaly_detector(detector_id, db)
    return JSONResponse(content={"status": "OK"})

@router.get("/logs")
async def get_logs_db(db: Session = Depends(get_db)):
    logs = db.query(Log).all()
    return [
        {
            "id": log.id,
            "start_at": log.start_at,
            "end_at": log.end_at,
            "config": json.loads(log.config),
            "duration_formated": format_seconds(log.duration_seconds),
            "precision": log.precision,
            "recall": log.recall,
            "f1": log.f1,
            "datapoints": [
                {"id": a.id, "timestamp": a.timestamp, "ftr_vector": a.ftr_vector}
                for a in log.datapoints
            ]
        }
        for log in logs
    ]



@router.delete("/logs/{log_id}")
def delete_log_db(log_id: int, db: Session = Depends(get_db)):
    delete_log(log_id, db)
    return JSONResponse(content={"status": "OK"})

@router.post("/run/{name}")
async def run(name: str = "border_check.json", db: Session = Depends(get_db)):
    tmp_file_path = os.path.join(CONFIG_DIR, "tmp.json")
    data_file_path = os.path.join(DATA_DIR, "tmp.csv")

    try:
        # Pick which config to use
        if os.path.exists(tmp_file_path):
            config_to_use = os.path.basename(tmp_file_path)
        else:
            config_to_use = name

        # Build args (as if from argparse)
        args = argparse.Namespace(
            config=config_to_use,
            data_file=False,
            data_both=False,
            watchdog=False,
            test=True,
            param_tunning=False,
            id=100
        )

        test_instance = main.start_consumer(args)

        # for log in logs:
        #     config_dict = json.loads(log.config) 
        #     print(f"Log ID: {log.id}")
        #     print(f"Start Time: {log.start_at}")
        #     print(f"End Time:   {log.end_timedate}")
        #     print(json.dumps(config_dict, indent=4))
        #     print(f"Duration: {log.duration_seconds} seconds")
        #     print(f"Precision: {log.precision}, Recall: {log.recall}, F1 Score: {log.f1}")
        #     print("-" * 60)


    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
    return JSONResponse(content={
    "name": config_to_use,
    "result": "TP: {}, TN: {},FP: {}, FN: {},Precision: {}, Recall: {}, F1 Score: {}".format(
                test_instance.tp, test_instance.tn, test_instance.fp, test_instance.fn,
                test_instance.precision, test_instance.recall, test_instance.f1
            )
        })




@router.get("/available_configs")
async def get_available_configs():
    return [
        {"name": config.name, "value": config.value}
        for config in AvailableConfigs
    ]