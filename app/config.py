import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    # Web server
    port: int = int(os.getenv("PORT", "8080"))

    # WhatsApp Cloud API
    whatsapp_token: str = os.getenv("WHATSAPP_TOKEN", "")
    whatsapp_phone_id: str = os.getenv("WHATSAPP_PHONE_ID", "")
    whatsapp_verify_token: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")

    # Google Cloud / Vertex AI
    gcp_project: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or os.getenv("PROJECT_ID")
    gcp_location: str = os.getenv("GOOGLE_CLOUD_REGION", os.getenv("GCP_REGION", "us-central1"))
    vertex_model: str = os.getenv("VERTEX_MODEL", "gemini-1.5-flash")


settings = Settings()


