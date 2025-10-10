# Frikkie 3 — WhatsApp “nearest used cars” bot

This service receives WhatsApp messages, asks for location if needed, and returns the nearest businesses from Firestore (exact category match: “used cars”, then fallback). Deployed to existing Cloud Run service frikkie-2-service.

Key choices
- Backend: Python (FastAPI)
- NLU: Vertex AI (model configurable; default gemini-1.5-flash)
- Data: Firestore (collection: businesses)
- Geospatial: geohash prefix queries + Haversine filtering
- Ranking: distance (km). Default radius 5 km (cap 20 km)

Local run
1) Create a .env file based on .env.example. Do NOT commit real secrets.
2) Install deps:
   pip install -r requirements.txt
3) Start server:
   uvicorn app.main:app --host 0.0.0.0 --port 8080

WhatsApp webhook
- Verify (GET): /webhook/whatsapp?hub.mode=subscribe&hub.verify_token=<token>&hub.challenge=<challenge>
- Receive (POST): /webhook/whatsapp

Env vars (see .env.example)
- GCP_PROJECT, GCP_LOCATION, VERTEX_MODEL
- FIRESTORE_COLLECTION, CATEGORY_DEFAULT, DEFAULT_RADIUS_M, MAX_RADIUS_M
- WHATSAPP_PHONE_ID, WHATSAPP_VERIFY_TOKEN, WHATSAPP_TOKEN
- GOOGLE_MAPS_API_KEY (only if you later enable ETA)

Deployment (existing Cloud Run service)
- Build container image
- Deploy to service: frikkie-2-service (do not create a new service)
- Configure environment variables and Secret Manager bindings

Notes
- Do not store plaintext secrets in repo. Use Secret Manager + Cloud Run env.
- If default 5 km returns 0 results, code expands up to 20 km, then drops category filter as a last resort.
