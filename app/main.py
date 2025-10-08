import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .config import settings
from .whatsapp import extract_text_message, send_whatsapp_reply, extract_location_message, parse_text_location_command
from .firestore_client import (
    log_message, fetch_conversation, clear_conversation,
    set_debug_enabled, is_debug_enabled,
    upsert_whatsapp_user_location, find_business_by_name,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Vertex AI Bot")


@app.get("/webhook")
async def verify_webhook(
    request: Request,
    # Accept non-hub params as a fallback (some tools use these)
    mode: str | None = None,
    challenge: str | None = None,
    verify_token: str | None = None,
):
    """Endpoint for WhatsApp webhook verification (GET).

    Meta commonly sends hub.* query params (e.g., hub.mode). These are not valid
    Python identifiers, so we read them from the raw query params. We also accept
    plain params (mode/challenge/verify_token) as a fallback.
    """
    qp = request.query_params
    hub_mode = qp.get("hub.mode")
    hub_challenge = qp.get("hub.challenge")
    hub_verify_token = qp.get("hub.verify_token")

    mode_val = hub_mode or mode
    challenge_val = hub_challenge or challenge
    token_val = hub_verify_token or verify_token

    if mode_val == "subscribe" and token_val == settings.whatsapp_verify_token and challenge_val:
        return PlainTextResponse(challenge_val)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_message(request: Request) -> Response:
    payload: Dict[str, Any] = await request.json()

    # v1.6: handle shared location payloads and save lat/lng
    loc_payload = extract_location_message(payload)
    if loc_payload:
        upsert_whatsapp_user_location(loc_payload["from"], loc_payload["lat"], loc_payload["lng"])
        await send_whatsapp_reply(loc_payload["from"], "Location saved")
        return Response(status_code=200)

    # Handle text messages
    msg = extract_text_message(payload)
    if not msg:
        # Return 200 immediately for unsupported events to avoid retries
        return Response(status_code=200)

    user_number = msg["from"]
    user_text = msg["text"].strip()

    lowered = user_text.lower()

    # Toggle debug mode
    if lowered == "debug":
        current = is_debug_enabled(user_number)
        set_debug_enabled(user_number, not current)
        state = "enabled" if not current else "disabled"
        await send_whatsapp_reply(user_number, f"Debug {state}")
        return Response(status_code=200)

    # System commands (always run first)
    debug_on = is_debug_enabled(user_number)

    if lowered == "ping":
        if debug_on:
            await send_whatsapp_reply(user_number, "frikkie is online and ready")
            log_message(user_number, "bot", "pong")
            return Response(status_code=200)
        # debug off → fall through to fallback
    if lowered == "version":
        if debug_on:
            await send_whatsapp_reply(user_number, settings.version)
            log_message(user_number, "bot", settings.version)
            return Response(status_code=200)
        # debug off → fall through to fallback
    if lowered == "reset":
        if debug_on:
            clear_conversation(user_number)
            await send_whatsapp_reply(user_number, "Conversation reset")
            log_message(user_number, "bot", "Conversation reset")
            return Response(status_code=200)
        # debug off → fall through to fallback
    if lowered == "history":
        if debug_on:
            rows = fetch_conversation(user_number, limit=100)
            if not rows:
                await send_whatsapp_reply(user_number, "No history yet.")
                return Response(status_code=200)
            lines = ["Conversation history (latest 100):"]
            for sender, text in rows:
                lines.append(f"{sender}: {text}")
            transcript = "\n".join(lines)
            await send_whatsapp_reply(user_number, transcript[:4096])
            log_message(user_number, "bot", "[sent history]")
            return Response(status_code=200)
        # debug off → fall through to fallback

    # v1.6: text 'location <address>' → geocode + save (simple placeholder without external API)
    # NOTE: Real geocoding requires an API (e.g., Google Geocoding). Here we store a stub and acknowledge.
    addr = parse_text_location_command(user_text)
    if addr:
        # In a real integration, call geocoder to resolve (lat,lng) from address.
        # For now, we store a placeholder (0.0, 0.0) with the address for traceability.
        upsert_whatsapp_user_location(user_number, 0.0, 0.0)
        await send_whatsapp_reply(user_number, f"Location saved for: {addr}")
        return Response(status_code=200)

    # v1.6: 'business <name>' → lookup by name and return entry
    if lowered.startswith("business ") and len(user_text.split(" ", 1)) == 2:
        biz_name = user_text.split(" ", 1)[1].strip()
        found = find_business_by_name(biz_name)
        if not found:
            await send_whatsapp_reply(user_number, "Business not found")
            return Response(status_code=200)
        # Build a compact reply of fields
        parts = [f"Found: {found.get('name', found.get('__id'))}"]
        for k, v in found.items():
            if k == "__id" or k == "name":
                continue
            parts.append(f"{k}: {v}")
        reply_biz = "\n".join(parts)
        await send_whatsapp_reply(user_number, reply_biz[:4096])
        return Response(status_code=200)

    # Fallback behavior: simple presence response
    reply = "I am here"
    # Log user + bot messages
    log_message(user_number, "user", user_text)
    log_message(user_number, "bot", reply)
    await send_whatsapp_reply(user_number, reply)
    return Response(status_code=200)


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


