from fastapi import FastAPI
import argparse
from .component.router import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
    
@app.get("/")
def read_root():
    return {"msg": "Anomaly Detection API"}
