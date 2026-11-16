# SuperKart Sales Prediction - Flask API + Streamlit (Docker)

Run inside a GitHub Codespace opened on this repository:

```bash
cd backend && docker build -t superkart-backend .
cd ../frontend && docker build -t superkart-frontend .
docker network create superkart-network
docker run -d --name backend --network superkart-network -p 7860:7860 superkart-backend
docker run -d --name frontend --network superkart-network -p 8501:8501 superkart-frontend
```

Set ports 7860 and 8501 to Public in the Ports tab. The API is on 7860 and the app on 8501.

API: `GET /health`, `POST /v1/predict` (JSON, ten fields), `POST /v1/predictbatch` (CSV in form field `file`).
