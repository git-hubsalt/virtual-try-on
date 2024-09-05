from fastapi import FastAPI
from pydantic import BaseModel
import os
from get_vton import get_vton

app = FastAPI()

class VtonRequest(BaseModel):
    person_image_path: str
    upper_cloth_path: str
    lower_cloth_path: str
    mask_image_path: str
    cloth_type: str
    username: str

@app.get('/')
def home():
    return 'Virtual Try On'

@app.post('/virtual_try_on')
async def virtual_try_on(request: VtonRequest):
    get_vton(
        person_image_path=request.person_image_path,
        upper_cloth_path=request.upper_cloth_path,
        lower_cloth_path=request.lower_cloth_path,
        mask_image_path=request.mask_image_path,
        cloth_type=request.cloth_type,
        username=request.username
    )

    return {'message': 'VTON started successfully'}