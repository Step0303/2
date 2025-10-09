from typing import Any, Dict, Optional
import logging
import httpx

from .config import settings
from .firestore_client import upsert_whatsapp_user_location


logger = logging.getLogger(__name__)


def extract_text_message(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract the sender phone number and text body from WhatsApp webhook payload.

    Returns a dict with keys: "from" (phone number as string) and "text" (message body),
    or None if no text message is found.
    """
    try:
        entries = payload.get("entry", [])
        if not entries:
            logger.debug("No 'entry' in payload when parsing text message")
            return None
        changes = entries[0].get("changes", [])
        if not changes:
            logger.debug("No 'changes' in entry[0] when parsing text message")
            return None
        value = changes[0].get("value", {})
        messages = value.get("messages", [])
        if not messages:
            # Sometimes the webhook contains different keys (e.g., statuses/contacts). Log for visibility.
            logger.debug("No 'messages' in change value when parsing text message; keys=%s", list(value.keys()))
            return None
        msg = messages[0]
        msg_type = msg.get("type")
        if msg_type != "text":
            logger.debug("Message type is not 'text' (type=%s); skipping", msg_type)
            return None
        text_body = msg.get("text", {}).get("body")
        from_number = msg.get("from")
        if not text_body or not from_number:
            logger.debug("Missing text body or from number (body=%s, from=%s)", bool(text_body), from_number)
            return None
        logger.info("Parsed text message from %s: %s", from_number, (text_body[:100] + '...') if len(text_body) > 100 else text_body)
        return {"from": from_number, "text": text_body}
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to parse WhatsApp payload: %s", exc)
        return None


def extract_location_message(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract a WhatsApp location message with latitude/longitude.

    Returns dict with keys: "from", "lat", "lng" or None.
    """
    try:
        entries = payload.get("entry", [])
        if not entries:
            return None
        changes = entries[0].get("changes", [])
        if not changes:
            return None
        value = changes[0].get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None
        # Iterate over messages and find the first with a location payload
        for msg in messages:
            loc = msg.get("location")
            if not isinstance(loc, dict):
                continue
            from_number = msg.get("from")
            lat = loc.get("latitude") or loc.get("lat")
            lng = loc.get("longitude") or loc.get("lng")
            if from_number and lat is not None and lng is not None:
                logger.info("Parsed WhatsApp location for %s: lat=%s lng=%s", from_number, lat, lng)
                return {"from": from_number, "lat": float(lat), "lng": float(lng)}
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to parse WhatsApp location payload: %s", exc)
        return None


def parse_text_location_command(text: str) -> Optional[str]:
    """Parse a text command that sets location and return the address if present.

    Supported patterns:
      - "location <address>"
      - "set my location to <address>"
      - "set my location as <address>"
      - "update my location to <address>"
    """
    t = (text or "").strip()
    if not t:
        return None
    lower = t.lower()
    if lower.startswith("location ") and len(t.split(" ", 1)) == 2:
        return t.split(" ", 1)[1].strip()
    try:
        import re
        m = re.match(r"^(?:set|update)\s+my\s+location\s+(?:to|as)\s+(.+)$", lower)
        if m:
            # Return original-cased tail using slice length from match
            start = len(t) - len(lower)  # usually 0
            # Use matched group length to slice from original text end
            addr_lower = m.group(1)
            # Find the address substring in original text by ending alignment
            addr = t[-len(addr_lower):] if len(addr_lower) <= len(t) else addr_lower
            return addr.strip()
    except Exception:
        pass
    return None


async def send_whatsapp_reply(to_number: str, body_text: str) -> None:
    """Send a text message reply via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v19.0/{settings.whatsapp_phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": body_text[:4096]},
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=headers, json=data)
        try:
            resp.raise_for_status()
            logger.info("WhatsApp reply sent to %s | response=%s", to_number, resp.text)
        except httpx.HTTPStatusError as exc:
            logger.error("Failed to send WhatsApp message: %s | %s", exc, resp.text)


