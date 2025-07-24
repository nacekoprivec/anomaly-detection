from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# from app.auth import get_current_user
# from app.schemas import AlgorithmConfig, DetectionResult
import logging

import subprocess
import json
import tempfile

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # adds the parent directory to sys.path
import main


app = FastAPI()
logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

@app.post("/")
def read_root(ready: str):
    return {"msg": "Anomaly Detection API {ready}"}

# @app.post("/upload/")
# async def upload_data(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
#     content = await file.read()
#     logger.info(f"Received file from user {user['username']}: {file.filename}")
#     return {"filename": file.filename, "status": "received"}

