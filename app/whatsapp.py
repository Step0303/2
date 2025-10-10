import json
from typing import Any, Dict, Optional, Tuple


def parse_inbound(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts minimal fields from WhatsApp webhook payload:
    - user_id (wa_id)
    - text (str | None)
    - location (lat, lng | None)
    - message_id (for idempotency)
    """
    try:
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None
        msg = messages[0]
        contacts = value.get("contacts", [])
        wa_id = contacts[0].get("wa_id") if contacts else msg.get("from")
        message_id = msg.get("id")

        text: Optional[str] = None
        location: Optional[Tuple[float, float]] = None

        if msg.get("type") == "text":
            text = msg.get("text", {}).get("body", "").strip()
        elif msg.get("type") == "location":
            loc = msg.get("location", {})
            lat = loc.get("latitude")
            lng = loc.get("longitude")
            if lat is not None and lng is not None:
                location = (float(lat), float(lng))
        else:
            # Other types (interactive, image, etc.) can be handled later
            pass

        return {
            "user_id": wa_id,
            "text": text,
            "location": location,
            "message_id": message_id,
        }
    except Exception:
        return None


def extract_query_text(payload: Dict[str, Any]) -> str:
    try:
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return ""
        msg = messages[0]
        if msg.get("type") == "text":
            return msg.get("text", {}).get("body", "")
        return ""
    except Exception:
        return ""
