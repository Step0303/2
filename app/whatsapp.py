from typing import Any, Dict, Optional
import logging
import httpx

from .config import settings


logger = logging.getLogger(__name__)


def extract_text_message(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract the sender phone number and text body from WhatsApp webhook payload.

    Returns a dict with keys: "from" (phone number as string) and "text" (message body),
    or None if no text message is found.
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
        msg = messages[0]
        if msg.get("type") != "text":
            return None
        text_body = msg.get("text", {}).get("body")
        from_number = msg.get("from")
        if not text_body or not from_number:
            return None
        return {"from": from_number, "text": text_body}
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to parse WhatsApp payload: %s", exc)
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


