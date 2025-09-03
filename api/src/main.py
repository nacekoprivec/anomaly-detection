from fastapi import Depends, FastAPI
import argparse
from .component.router import router
from fastapi.middleware.cors import CORSMiddleware
from .database import *
from .component.models import AnomalyDetector, Log, DataPoint
from .component.service import scrape_data
import main
import asyncio
import pandas as pd
from sqlalchemy.orm import Session
from datetime import *
from .component.service import confusion_matrix as calculate_confusion_matrix


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

async def detector_loop(data):
     while True:
         db = SessionLocal()
         try:
             detectors = db.query(AnomalyDetector).all()

             for detector in detectors:
                log = db.query(Log).filter(Log.detector_id == detector.id).order_by(Log.start_at.desc()).first()

                entry = (data[detector.id]['timestamp'], data[detector.id]['vodostaj'])


                print("DEBUG detector.id:", detector.id)
                print("DEBUG data entry:", data[detector.id])
                print("DEBUG entry:", entry)

                args = argparse.Namespace(
                    config="ema.json",
                    data_file=False,
                    data_both=False,
                    watchdog=False,
                    test=True,
                    param_tunning=False,
                    entry=entry
                )

                test_instance = main.start_consumer(args)
                log.tp += test_instance.tp
                log.fp += test_instance.fp
                log.tn += test_instance.tn
                log.fn += test_instance.fn
                res = calculate_confusion_matrix(log.tp, log.fp, log.fn, log.tn)

                log.precision = res['precision']
                log.recall = res['recall']
                log.f1 = res['f1']

                # Save entry
                # entry = DataPoint(
                #     timestamp=entry[0],
                #     ftr_vector=entry[1],
                #     is_anomaly=test_instance.is_anomaly,
                #     log_id=log.id
                # )
                # db.add(entry)
                
                db.commit()
         except Exception as e:
             db.rollback()
             # Log the exception or handle it
             print(f"Exception in detector_loop: {e}")
         finally:
             db.close()
         await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    data_list = scrape_data(2)
    data = {item['place_id']: item for item in data_list}
    asyncio.create_task(detector_loop(data))
