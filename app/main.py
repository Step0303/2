import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .config import settings
from .whatsapp import extract_text_message, extract_location_message, send_whatsapp_reply
from .firestore_client import (
    log_message, fetch_conversation, clear_conversation,
    set_debug_enabled, is_debug_enabled,
    upsert_whatsapp_user_location, get_user_location, find_closest_businesses,
)
from .vertex import VertexAIClient


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

    # v1.9: Handle WhatsApp native location messages first and persist last-known location
    loc_msg = extract_location_message(payload)
    if loc_msg:
        user_number = loc_msg["from"]
        upsert_whatsapp_user_location(user_number, loc_msg["lat"], loc_msg["lng"])
        log_message(user_number, "user", f"[shared location] lat={loc_msg['lat']} lng={loc_msg['lng']}")
        await send_whatsapp_reply(
            user_number,
            "Got it – I saved your location. Ask me something like 'Find the closest pharmacy'.",
        )
        log_message(user_number, "bot", "[acknowledged and saved location]")
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

    # v1.7: removed 'location <address>' command

    # v1.9: If the user asks for something "closest" or "near me", use last-known location
    if ("closest" in lowered) or ("near me" in lowered) or ("nearest" in lowered):
        # Heuristic to extract a business type keyword
        import re
        biz_type = ""
        if "closest" in lowered:
            # take words after 'closest'
            after = lowered.split("closest", 1)[1]
            words = re.findall(r"[a-zA-Z]+", after)
            biz_type = words[0] if words else ""
        if not biz_type and "nearest" in lowered:
            after = lowered.split("nearest", 1)[1]
            words = re.findall(r"[a-zA-Z]+", after)
            biz_type = words[0] if words else ""
        if not biz_type and "near me" in lowered:
            before = lowered.split("near me", 1)[0]
            words = re.findall(r"[a-zA-Z]+", before)
            biz_type = words[-1] if words else ""

        loc = get_user_location(user_number)
        if not loc:
            await send_whatsapp_reply(
                user_number,
                "Could you please share your current location using the WhatsApp location pin? Then ask again.",
            )
            log_message(user_number, "bot", "Requested location pin for closest search")
            return Response(status_code=200)

        lat, lng = loc
        matches = find_closest_businesses(lat, lng, biz_type or "", limit=3, max_radius_km=50.0)
        if not matches:
            await send_whatsapp_reply(user_number, "I couldn't find matching places nearby in my list.")
            log_message(user_number, "bot", "No nearby results")
            return Response(status_code=200)

        lines = ["Here are the closest options:"]
        for b, dist_km in matches:
            lines.append(f"- {b.name} — {dist_km:.1f} km away")
        reply = "\n".join(lines)
        log_message(user_number, "user", user_text)
        log_message(user_number, "bot", reply)
        await send_whatsapp_reply(user_number, reply)
        return Response(status_code=200)

    # v1.8: Otherwise, if the message appears to be a general business search, use the business prompt
    if any(kw in lowered for kw in ["business ", "find ", "restaurant", "garage", "pharmacy", "hospital", "doctor", "shop", "cafe", "coffee"]):
        client = VertexAIClient()
        try:
            ai_reply = client.generate_business_reply(user_text, user_number=user_number)
        except Exception:
            ai_reply = "No results yet. Please specify the area or type."
        log_message(user_number, "user", user_text)
        log_message(user_number, "bot", ai_reply)
        await send_whatsapp_reply(user_number, ai_reply)
        return Response(status_code=200)

    # v1.7: Fallback behavior → call Vertex AI (Gemini 2.5 Flash)
    client = VertexAIClient()
    try:
        ai_reply = client.generate_reply(user_text, user_number=user_number)
    except Exception:
        ai_reply = "I’m here! How can I help today?"

    # Log user + bot messages
    log_message(user_number, "user", user_text)
    log_message(user_number, "bot", ai_reply)
    await send_whatsapp_reply(user_number, ai_reply)
    return Response(status_code=200)


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


