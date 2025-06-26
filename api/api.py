from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import numpy as np

from src.consumer import ConsumerAbstract, ConsumerFile, ConsumerKafka


from src.algorithms.border_check import BorderCheck
from src.algorithms.cumulative import Cumulative
from src.algorithms.ema import EMA

app = FastAPI()

ALGORITHM_REGISTRY = {
    "border_check": "border_check.json",
    "cumulative": "cumulative.json",
    "ema": "ema.json",
}

class AnomalyRequest(BaseModel):
    algorithm: Literal["border_check", "cumulative", "ema"]  
    #config: Literal["border_check.json", "cumulative.json", "ema.json"]  
    #data: List[float]

@app.post("/detect")
def detect_anomaly(req: AnomalyRequest):
    algo_name = req.algorithm

    if algo_name not in ALGORITHM_REGISTRY:
        raise HTTPException(status_code=400, detail="Unsupported algorithm")
    filename = ALGORITHM_REGISTRY[algo_name]
    consumer = ConsumerFile(configuration_location=filename)
    consumer.read()
    
    
@app.post("/input_mode/")
async def set_input_mode(mode: str = Body(..., embed=True)):
    """
    Set the input mode: 'file' or 'datastreaming'.
    """
    if mode not in ["file", "datastreaming"]:
        return {"error": "Invalid mode. Choose 'file' or 'datastreaming'"}
    return {"message": f"Input mode set to '{mode}'"}


# --- UPLOAD FILE ---

@app.post("/input_mode/file/upload")
async def upload_data_file(file: UploadFile = File(...)):
    """
    Upload a data file.
    """
    file_location = f"data/{file.filename}"
    with open(file_location, "wb+") as f:
        f.write(await file.read())
    return {"filename": file.filename, "status": "uploaded"}


# --- ALGORITHM CONFIGURATION ---

@app.post("/algorithm/config")
async def configure_algorithm(config: AlgorithmConfig):
    """
    Set algorithm and configuration. Uses default values if not provided.
    """
    return {
        "message": "Algorithm configured successfully.",
        "configuration": config.dict()
    }


# --- RUN ALGORITHM ---

@app.post("/algorithm/run")
async def run_algorithm():
    """
    Start anomaly detection with current config/input.
    """
    return {"message": "Anomaly detection algorithm started."}


# --- CONSOLE OUTPUT ---

@app.get("/console")
async def get_console_log():
    """
    Show currently running algorithms/logs.
    """
    return {"log": "Running: RRCF_trees on ads-1.csv"}


@app.get("/console/{algorithm_name}")
async def get_algorithm_log(algorithm_name: str):
    """
    Show detailed log for specific algorithm.
    """
    return {"algorithm": algorithm_name, "log": f"{algorithm_name} is currently executing."}


@app.get("/console/{algorithm_name}/results")
async def get_algorithm_results(algorithm_name: str):
    """
    Show results (anomalies + confusion matrix).
    """
    return {
        "algorithm": algorithm_name,
        "anomalies_detected": 12,
        "confusion_matrix": {
            "TP": 10,
            "FP": 2,
            "FN": 3,
            "TN": 85
        }
    }





