# Base image with Python - CHANGED TO 3.11-slim for PyTorch 2.4.1 compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install PyTorch with specific CUDA 11.8 support.
# This must be done separately using the index-url.
# IMPORTANT: Ensure torch, torchvision, and torchaudio are REMOVED from requirements.txt
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install the remaining packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
