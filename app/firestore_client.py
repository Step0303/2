from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from google.cloud import firestore

from .config import settings


def _get_client() -> firestore.Client:
    # Uses ADC in Cloud Run; locally uses gcloud auth or env credentials
    return firestore.Client(project=settings.gcp_project)


@dataclass
class Business:
    id: str
    name: str
    type: str
    latitude: float
    longitude: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def upsert_whatsapp_user_location(phone: str, latitude: float, longitude: float) -> None:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone)
    doc.set({
        "phone": phone,
        # Store under location.coordinate to match project schema
        "location": {"coordinate": {"lat": latitude, "lng": longitude}},
    }, merge=True)


def get_user_location(phone: str) -> Optional[Tuple[float, float]]:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    loc = data.get("location") or {}
    # Prefer location.coordinate.{lat,lng}; fall back to previous shapes
    coord = loc.get("coordinate") if isinstance(loc, dict) else None
    lat = None
    lng = None
    if isinstance(coord, dict):
        lat = coord.get("lat")
        lng = coord.get("lng")
    if lat is None or lng is None:
        lat = loc.get("lat") if isinstance(loc, dict) else None
        lng = loc.get("lng") if isinstance(loc, dict) else None
    if lat is None or lng is None:
        return None
    return float(lat), float(lng)


def find_closest_businesses(
    user_lat: float,
    user_lng: float,
    business_type: str,
    *,
    limit: int = 1,
    max_radius_km: float = 50.0,
) -> List[Tuple[Business, float]]:
    client = _get_client()
    # Match against normalized_tags array (case-insensitive):
    tag = (business_type or "").strip().lower()
    q = client.collection("businesses").where("normalized_tags", "array_contains", tag)
    docs = q.stream()

    results: List[Tuple[Business, float]] = []
    for d in docs:
        data = d.to_dict() or {}
        # location.coordinate.{lat,lng} preferred
        lat = None
        lng = None
        loc = data.get("location") or {}
        if isinstance(loc, dict):
            coord = loc.get("coordinate")
            if isinstance(coord, dict):
                lat = coord.get("lat")
                lng = coord.get("lng")
            if lat is None or lng is None:
                lat = loc.get("lat")
                lng = loc.get("lng")
        # fallbacks
        lat = lat or data.get("latitude")
        lng = lng or data.get("longitude")
        name = data.get("name") or d.id
        if lat is None or lng is None:
            continue
        dist = haversine_km(user_lat, user_lng, float(lat), float(lng))
        # Apply radius cap
        if max_radius_km is not None and dist > float(max_radius_km):
            continue
        results.append((Business(id=d.id, name=name, type=business_type, latitude=float(lat), longitude=float(lng)), dist))

    results.sort(key=lambda x: x[1])
    return results[:limit]



# ===== Conversation logging (v1.3) =====
def log_message(phone: str, sender: str, text: str) -> None:
    """Append a message to the conversation log in Firestore.

    Collection: processed_messages
      - phone: str (WhatsApp sender number)
      - sender: "user" | "bot"
      - text: message content
      - ts: server timestamp
    """
    client = _get_client()
    client.collection("processed_messages").add({
        "phone": phone,
        "sender": sender,
        "text": text,
        "ts": firestore.SERVER_TIMESTAMP,
    })


def fetch_conversation(phone: str, limit: int = 100) -> List[Tuple[str, str]]:
    """Return the last N (sender, text) pairs in chronological order."""
    client = _get_client()
    q = (
        client.collection("processed_messages")
        .where("phone", "==", phone)
        .order_by("ts")
        .limit(limit)
    )
    rows: List[Tuple[str, str]] = []
    for d in q.stream():
        data = d.to_dict() or {}
        rows.append((str(data.get("sender", "")), str(data.get("text", ""))))
    return rows


def clear_conversation(phone: str) -> None:
    """Delete up to 1000 recent messages for the user."""
    client = _get_client()
    q = (
        client.collection("processed_messages")
        .where("phone", "==", phone)
        .order_by("ts")
        .limit(1000)
    )
    batch = client.batch()
    count = 0
    for d in q.stream():
        batch.delete(d.reference)
        count += 1
        if count % 400 == 0:
            batch.commit()
            batch = client.batch()
    batch.commit()

