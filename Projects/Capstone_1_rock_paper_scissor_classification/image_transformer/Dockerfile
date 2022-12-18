FROM python:3.9.15-slim

WORKDIR /app

COPY "image_transformer.py" .

RUN pip install kserve==0.9.0 pillow==9.2.0 numpy==1.21.6

ENTRYPOINT ["python", "image_transformer.py"]

