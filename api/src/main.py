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



