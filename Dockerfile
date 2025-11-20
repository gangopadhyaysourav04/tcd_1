FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 

--index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
