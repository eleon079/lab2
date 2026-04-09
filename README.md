# HuggingFace Model Service (Flask + Waitress + Docker)

## Build
docker build -t eleon079/hf-waitress-service:latest .

## Run
docker run --rm -p 5000:5000 eleon079/hf-waitress-service:latest

## Test
### Health
curl http://localhost:5000/health

### Predict
curl -X POST http://localhost:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"I love this class!\"}"



## Docker Hub Link
https://hub.docker.com/r/eleon079/hf-waitress-service