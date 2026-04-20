# Game Playback Analytics — Streamlit
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sql/ sql/
COPY src/ src/
COPY app/ app/
COPY .streamlit/ .streamlit/

# DB and raw JSON are mounted at runtime or baked in for demos — see README
RUN mkdir -p data/raw data/processed

EXPOSE 8501

# Cloud platforms set PORT; default 8501 for local Docker
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
