FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libblas3 \
        liblapack3 \
        libgfortran5 \
        libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit==1.38.0 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scipy==1.11.4 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    scikit-learn==1.3.0

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
