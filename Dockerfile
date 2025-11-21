
FROM python:3.11-slim


WORKDIR /app


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt


COPY . .


EXPOSE 8501


CMD ["streamlit", "run", "main/gangopadhyaysourav04/tcd_1/main/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

