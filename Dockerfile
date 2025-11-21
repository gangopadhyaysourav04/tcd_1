FROM python:3.11-buster

WORKDIR /app

# Step 1: Install system dependencies for scientific libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libblas3 \
        liblapack3 \
        libgfortran5 \
        libatlas-base-dev \
        libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Step 2: FORCED RESOLUTION - Manually create the requirements.txt file inside 
# the container to override any external/hidden requirements file that the 
# deployment environment might be reading, which is causing the version conflicts.
RUN echo "streamlit==1.38.0" > requirements.txt && \
    echo "pandas==2.2.3" >> requirements.txt && \
    echo "numpy==1.26.4" >> requirements.txt && \
    echo "scipy==1.11.4" >> requirements.txt && \
    echo "matplotlib==3.9.2" >> requirements.txt && \
    echo "seaborn==0.13.2" >> requirements.txt && \
    echo "scikit-learn==1.2.2" >> requirements.txt

# Step 3: Install dependencies from the newly created requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
