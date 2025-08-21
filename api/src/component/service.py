import numpy as np
import main

import argparse
import tempfile
import json

import main

from .models import *
from ..database import get_db
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

def handle_configuration(body: dict) -> str:
    with open("C:\\Users\\nacek\\OneDrive\\Desktop\\siht\\anomaly-detection\\configuration", "r") as f:
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


# CRUD
# Create logs

def create_log(db: Session, start_time, end_time, config, anomalies: dict = None):
    log_entry = Log(
        start_timedate=start_time,
        end_timedate=end_time,
        config=json.dumps(config)
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    # If anomalies provided, save them
    if anomalies:
        for ts, ftr in anomalies.items():
            anomaly = Anomaly(
                timestamp=ts,
                ftr_vector=ftr,
                log_id=log_entry.id
            )
            db.add(anomaly)
        db.commit()

    return log_entry


# Read logs/anomalies

def get_logs(db: Session, skip: int = 0, limit: int = 10):
    return db.query(Log).offset(skip).limit(limit).all()


def get_log(db: Session, log_id: int):
    return db.query(Log).filter(Log.id == log_id).first()

def get_anomalies(db: Session, log_id: int):
    return db.query(Anomaly).filter(Anomaly.log_id == log_id).all()

def get_anomaly(db: Session, anomaly_id: int):
    return db.query(Anomaly).filter(Anomaly.id == anomaly_id).first()

# Delete logs/anomalies

def delete_log(db: Session, log_id: int):
    log = db.query(Log).filter(Log.id == log_id).first()
    if log:
        db.delete(log)
        db.commit()
    return log

def delete_anomaly(db: Session, anomaly_id: int):
    anomaly = db.query(Anomaly).filter(Anomaly.id == anomaly_id).first()
    if anomaly:
        db.delete(anomaly)
        db.commit()
    return anomaly

# Print Statements

def detect_anomalies():
    print("Detecting anomalies...")

    return 0

def configuration():
    print("Configuring...")

    return 0

def create_config():
    print("Creating configuration...")

    return 0