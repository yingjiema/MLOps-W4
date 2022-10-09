from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import requests

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Face FastAPI')

#Call your get function for a health Check
#to receive both (face-bokeh and face-emotion)

@app.get("/", tags=["Health Check"])
async def root():
    return {requests.get("http://bokeh:8001/"), requests.get("http://emotion:8002/")}


# At least 20G
# Start fresh EC2 instance
# Reference: https://stackoverflow.com/questions/70418419/docker-containers-not-communicating-between-each-others-using-fastapi-and-reques
