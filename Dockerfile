FROM python:3.11-slim

WORKDIR /app

# Install API dependencies
COPY requirements.api.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.api.txt

# Copy source code + model artifact
COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
