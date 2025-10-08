# WhatsApp ➜ Vertex AI ➜ Cloud Run Bot

A minimal FastAPI service that:
- Verifies and receives WhatsApp Cloud API webhooks
- Generates a response using Vertex AI (Gemini)
- Sends the reply back via WhatsApp Cloud API
- Runs on Google Cloud Run

## Environment Variables

### For Local Development

Create a `.env` file in the project root and add your values:

```
WHATSAPP_TOKEN=EAA...ZDZD
WHATSAPP_PHONE_ID=718800947994191
WHATSAPP_VERIFY_TOKEN=frikkie-verify-token-247

GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_REGION=us-central1
VERTEX_MODEL=gemini-1.5-flash
```

Then load them in your shell before running the app:

```bash
export $(cat .env | xargs)
```

### For Cloud Run Deployment

You'll set these when deploying with the `gcloud run deploy` command (see deploy section below).

## Local Run

```
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Then expose `http://localhost:8080/webhook` to the internet using a tunnel (e.g. `ngrok`) and configure the WhatsApp webhook in the Meta developer dashboard. Ensure the verify token matches `WHATSAPP_VERIFY_TOKEN`.

## Deploy to Cloud Run

1. Authenticate and set project/region:

```
gcloud auth login
gcloud config set project $GOOGLE_CLOUD_PROJECT
gcloud config set run/region $GOOGLE_CLOUD_REGION
```

2. Build and deploy:

```
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/whatsapp-vertex-bot

gcloud run deploy whatsapp-vertex-bot \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/whatsapp-vertex-bot \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars WHATSAPP_TOKEN=$WHATSAPP_TOKEN \
  --set-env-vars WHATSAPP_PHONE_ID=$WHATSAPP_PHONE_ID \
  --set-env-vars WHATSAPP_VERIFY_TOKEN=$WHATSAPP_VERIFY_TOKEN \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT \
  --set-env-vars GOOGLE_CLOUD_REGION=$GOOGLE_CLOUD_REGION \
  --set-env-vars VERTEX_MODEL=${VERTEX_MODEL:-gemini-1.5-flash}
```

3. Grant the service account Vertex AI access:

- Ensure the Cloud Run service's identity has `Vertex AI User` role (`roles/aiplatform.user`).
- Enable the `Vertex AI API` (and it pulls other needed services):

```
gcloud services enable aiplatform.googleapis.com
```

4. Configure the WhatsApp webhook in Meta:

- Set callback URL to the Cloud Run URL: `https://<service-url>/webhook`
- Set verify token to `WHATSAPP_VERIFY_TOKEN`
- Subscribe to message events

### v1.8 Prompts

Two prompt files guide behavior (loaded at runtime if present):

- `prompts/system_prompt.txt` — overall assistant behavior
- `prompts/business_search_prompt.txt` — business search formatting and examples

If these files are absent, sensible defaults are used.

## Notes

- WhatsApp text messages are parsed; non-text messages are acknowledged with 200 but ignored.
- Responses are trimmed to 4096 chars to meet WhatsApp API limits.
- Customize prompt/behavior in `app/vertex.py`.

## Version History

 - v1.1
  - Firestore integration using ADC (same GCP project).
  - Support WhatsApp location messages; stores latest user location in `whatsapp_users`.
  - New command: `closest <type>` — finds nearest businesses from `businesses` collection and replies with a list.
- v1.0
  - Initial Cloud Run bot with webhook verify, reverse-text echo baseline, and Vertex AI scaffolding.