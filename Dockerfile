# Use official Python slim image
FROM python:3.11-slim-buster

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libatlas-base-dev \
        libopenblas-dev \
        liblapack3 \
        libgfortran5 \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files into container
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
