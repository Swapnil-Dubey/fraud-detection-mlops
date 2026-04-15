FROM python:3.8-slim

WORKDIR /app

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy from src folder
COPY src/serve.py /app
COPY src/preprocess.py /app

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]