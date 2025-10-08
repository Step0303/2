#!/bin/bash

# Configuration
SERVICE_NAME="frikkie-2-service"
REGION="us-central1"
PROJECT_ID="albert-assist247"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting deployment to Cloud Run...${NC}"

# Load environment variables from .env if present (without exporting comments)
if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -I {} echo {})
  echo -e "${BLUE}Loaded environment variables from .env${NC}"
fi

# Set the project (optional, if you work with multiple projects)
gcloud config set project $PROJECT_ID

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "WHATSAPP_TOKEN=${WHATSAPP_TOKEN},WHATSAPP_PHONE_ID=${WHATSAPP_PHONE_ID},WHATSAPP_VERIFY_TOKEN=${WHATSAPP_VERIFY_TOKEN},GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-$PROJECT_ID},GOOGLE_CLOUD_REGION=${GOOGLE_CLOUD_REGION:-$REGION},VERTEX_MODEL=${VERTEX_MODEL:-gemini-1.5-flash}"

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deployment successful!${NC}"
    
    # Get and display the service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)")
    echo -e "${GREEN}Service URL: $SERVICE_URL${NC}"
else
    echo "Deployment failed!"
    exit 1
fi