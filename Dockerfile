# Use lightweight Python base
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifact
COPY main.py .
COPY best_insurance_model.pkl .

# Expose port for the FastAPI server
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
