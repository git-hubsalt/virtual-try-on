FROM python:3.9-slim

WORKDIR /app

# RUN apt-get install -y \
#     mesa-libGL \
#     mesa-libGL-devel \
#     libXtst \
#     libXrender \
#     libXext

COPY . /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "vton_api.py"]
