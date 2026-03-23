# Deployment Guide

This project now supports single-service deployment:
- Flask API serves both backend routes and frontend static files.
- Frontend API calls automatically use same-domain routes in production.

## Option 1: Deploy to Render (Recommended)

### 1. Push this project to GitHub
- Commit your latest changes.
- Push to your repository.

### 2. Create Render Web Service
- Open Render Dashboard.
- Click New + and choose Blueprint.
- Select your repository.
- Render will detect render.yaml and create the service.

### 3. Verify startup command
Render will run:
- Build: pip install -r requirements.txt
- Start: gunicorn --chdir backend backend:app --workers 1 --threads 4 --timeout 180

### 4. Open deployed URL
- Once build is green, open the generated Render URL.
- Frontend is served at /. 
- API health check is at /health.

## Option 2: Deploy on any VM/VPS

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run production server
gunicorn --chdir backend backend:app --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 180

### 3. Optional reverse proxy
Put Nginx/Caddy in front and route domain traffic to port 5000.

## Important Notes
- Large model files increase deploy time and memory use.
- If the KIT dataset CSV is not present on server, dataset comparison endpoints will return an error while core prediction still works.
- Keep FLASK_DEBUG=0 in production.
