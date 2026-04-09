FROM python:3.10-slim

# Keep logs unbuffered (nice for Docker logs)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .

EXPOSE 5000

# Serve with waitress (production-ish)
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]