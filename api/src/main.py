from fastapi import FastAPI
import argparse
from .component.router import router

app = FastAPI()

app.include_router(router)
    
    
@app.get("/")
def read_root():
    return {"msg": "Anomaly Detection API"}
