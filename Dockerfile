FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + trained model
COPY app.py .
COPY model.pkl .
COPY templates/ ./templates/

# Hugging Face Spaces / Render expect $PORT
ENV PORT=7860
EXPOSE 7860

CMD gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 60 app:app
