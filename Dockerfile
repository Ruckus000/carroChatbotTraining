FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the NLU model and code
COPY api.py .
COPY inference.py .
COPY trained_nlu_model/ ./trained_nlu_model/

# Expose the port the app runs on
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 