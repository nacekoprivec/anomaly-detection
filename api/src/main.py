from fastapi import Depends, FastAPI
import argparse
from .component.router import router
from fastapi.middleware.cors import CORSMiddleware
from .database import *
from .component.models import AnomalyDetector, Log
from .component.service import scrape_data
import main
import asyncio
import pandas as pd
from sqlalchemy.orm import Session
from datetime import *
from .component.service import confusion_matrix as calculate_confusion_matrix
import traceback


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

async def detector_loop():
    while True:
        db = SessionLocal()
        try:
            detectors = db.query(AnomalyDetector).all()

            data_list = scrape_data(len(detectors))
            for detector in detectors:
                log = db.query(Log).filter(Log.detector_id == detector.id).order_by(Log.start_at.desc()).first()

                timestamp = data_list[detector.id-1]['timestamp']
                ftr_vector = data_list[detector.id-1]['vodostaj']

                data = {
                    "timestamp": timestamp,
                    "ftr_vector": [ftr_vector]  
                }

                args = argparse.Namespace(
                    config="ema.json",
                    data_file=False,
                    data_both=False,
                    watchdog=False,
                    test=True,
                    param_tunning=False,
                    data=data
                )

                test_instance = main.start_consumer(args)

                # log.tp += test_instance.tp
                # log.fp += test_instance.fp
                # log.tn += test_instance.tn
                # log.fn += test_instance.fn
                # res = calculate_confusion_matrix(log.tp, log.fp, log.fn, log.tn)

                # log.precision = res['precision']
                # log.recall = res['recall']
                # log.f1 = res['f1']


                db.add(log)
                db.commit()
        except Exception as e:
            db.rollback()
            # Print full traceback
            print("Exception in detector_loop:")
            traceback.print_exc()
            # Or log it as a string
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            print(tb_str)
        finally:
            db.close()
        await asyncio.sleep(60)


