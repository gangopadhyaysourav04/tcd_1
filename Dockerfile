FROM python:3.11

WORKDIR /app

# RESOLUTION: Switched from 'python:3.11-slim' to 'python:3.11' to force a stable Python 
# environment (avoiding the Python 3.13 conflicts) and ensure necessary build 
# tools for scientific libraries are present.

# Update and install minimal required system libs (only need gfortran for scipy/numpy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

# Using stable versions known to work together for Python 3.11
RUN pip install --no-cache-dir \
    streamlit==1.38.0 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scipy==1.11.4 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    scikit-learn==1.2.2

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
