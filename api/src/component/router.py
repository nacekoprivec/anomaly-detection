import os
import json
import tempfile
import argparse
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

import main
from .service import *

CONFIG_DIR = os.path.abspath("configuration")
DATA_DIR = os.path.abspath("data")

router = APIRouter()

def load_config(name: str) -> Dict[str, Any]:
    config_file = os.path.join(CONFIG_DIR, name)
    with open(config_file, "r") as f:
        return json.load(f)

@router.get("/detect")
async def detect():
    args = argparse.Namespace(
        config="border_check.json",
        data_file=True,
        data_both=False,
        watchdog=False,
        test=False,
        param_tunning=False
    )
    main.start_consumer(args)
    return {"status": "OK"}

@router.post("/configuration/{name}")
async def detect_with_custom_config(name: str, request: Request):
    tmp_file_path = os.path.join(CONFIG_DIR, "tmp.json")
    overrides = {}
    try:
        default_config = load_config(name)
        overrides = await request.json()
        merged_config = {**default_config, **overrides}

        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(tmp_file_path, "w") as tmp_file:
            json.dump(merged_config, tmp_file)

        return JSONResponse(content={"status": "OK", "used_config": name})

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

@router.post("/run/{name}")
async def run(name: str):
    tmp_file_path = os.path.join(CONFIG_DIR, "tmp.json")
    data_file_path = os.path.join(DATA_DIR, "tmp.csv")

    try:
        if os.path.exists(tmp_file_path):
            config_to_use = os.path.basename(tmp_file_path)
        else:
            config_to_use = name

        args = argparse.Namespace(
            config=config_to_use,
            data_file=True,
            data_both=False,
            watchdog=False,
            test=False,
            param_tunning=False
        )
        main.start_consumer(args)

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
