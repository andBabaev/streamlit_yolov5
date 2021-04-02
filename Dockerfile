FROM python:3.8-slim-buster

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libquadmath0 python3-dev build-essential gcc -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt --no-cache-dir
EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]