from fastapi import FastAPI
from pydantic import BaseModel
import os
from get_vton import get_vton
import uvicorn

app = FastAPI()


class VtonRequest(BaseModel):
    person_image_url: str
    upper_cloth_url: str
    lower_cloth_url: str
    mask_image_url: str
    cloth_type: str
    userId: str
    timestamp: str


@app.get("/")
def home():
    return "Virtual Try On"

@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.post("/invocations")
async def virtual_try_on(request: VtonRequest):
    get_vton(
        person_image_url=request.person_image_url,
        upper_cloth_url=request.upper_cloth_url,
        lower_cloth_url=request.lower_cloth_url,
        mask_image_url=request.mask_image_url,
        cloth_type=request.cloth_type,
        username=request.userId,
        timestamp=request.timestamp,
    )

    return {"message": "VTON run successfully"}


if __name__ == "__main__":
    uvicorn.run("vton_api:app", host="0.0.0.0", port=8080, reload=False)
