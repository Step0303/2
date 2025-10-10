from __future__ import annotations
import os
from dataclasses import dataclass


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # GCP
    project_id: str = os.getenv("GCP_PROJECT", "albert-assist247")
    location: str = os.getenv("GCP_LOCATION", "us-central1")

    # Vertex AI model (user requested Gemini "2.5 flash"; defaulting to a stable Flash model name)
    vertex_model: str = os.getenv("VERTEX_MODEL", "gemini-1.5-flash")

    # Firestore
    firestore_collection: str = os.getenv("FIRESTORE_COLLECTION", "businesses")

    # Search defaults
    category_default: str = os.getenv("CATEGORY_DEFAULT", "used cars").lower()
    default_radius_m: int = _get_env_int("DEFAULT_RADIUS_M", 5000)
    max_radius_m: int = _get_env_int("MAX_RADIUS_M", 20000)
    top_k_results: int = _get_env_int("TOP_K_RESULTS", 10)
    distance_matrix_top_k: int = _get_env_int("DISTANCE_MATRIX_TOP_K", 0)  # 0 = disabled
    session_ttl_hours: int = _get_env_int("SESSION_TTL_HOURS", 24)

    # WhatsApp
    whatsapp_phone_id: str | None = os.getenv("WHATSAPP_PHONE_ID")
    whatsapp_verify_token: str | None = os.getenv("WHATSAPP_VERIFY_TOKEN")
    whatsapp_token: str | None = os.getenv("WHATSAPP_TOKEN")

    # Google Maps
    maps_api_key: str | None = os.getenv("GOOGLE_MAPS_API_KEY")


settings = Settings()
