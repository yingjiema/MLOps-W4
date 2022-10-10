from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import requests
import json

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Face FastAPI')

#Call your get function for a health Check
#to receive both (face-bokeh and face-emotion)

@app.get("/", tags=["Health Check"])
async def root():
    return {
        "face-bokeh": json.loads(requests.get("http://bokeh:8001/").text).get("message", ""),
        "face-emotion": json.loads(requests.get("http://emotion:8002/").text).get("message", ""),
    } 


# At least 20G
# Start fresh EC2 instance
# Reference: https://stackoverflow.com/questions/70418419/docker-containers-not-communicating-between-each-others-using-fastapi-and-reques
