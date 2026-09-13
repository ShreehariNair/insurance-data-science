# Use Python 3.11 base image for scikit-learn 1.9 compatibility
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create internal data folder
RUN mkdir -p /app/data

# Copy application code and model artifact from data/ directory
COPY main.py .
COPY data/best_insurance_model.pkl ./data/best_insurance_model.pkl

# Expose port for FastAPI server
EXPOSE 8000

# Start Uvicorn application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]