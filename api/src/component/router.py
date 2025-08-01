import argparse
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

import main

router = APIRouter()

@router.get("/detect")
async def say_hello():
    args = argparse.Namespace(
        config="border_check.json",  
        data_file=True,             
        data_both=False,           
        watchdog=False,              
        test=False,        
        param_tunning=False         
    )

    return {"status": "OK"}