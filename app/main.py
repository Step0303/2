import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .config import settings
from .whatsapp import (
    extract_text_message,
    extract_location_message,
    parse_text_location_command,
    send_whatsapp_reply,
)
from .firestore_client import (
    log_message, fetch_conversation, clear_conversation,
    set_debug_enabled, is_debug_enabled,
    upsert_whatsapp_user_location, get_user_location, find_closest_businesses,
)
from .vertex import VertexAIClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Vertex AI Bot")

@app.get("/debug/list_nearby")
def debug_list_nearby(lat: float, lng: float, radius_km: float = 5.0, api_key: str = None):
    """Debug endpoint: list businesses within radius_km of lat/lng and show extracted coords.

    Protected by DEBUG_API_KEY env var. Returns a small JSON list of candidate businesses
    with id, name, extracted lat/lng, and computed distance.
    """
    from .firestore_client import _extract_lat_lng_from_doc, haversine_km
    import os
    key = os.environ.get("DEBUG_API_KEY")
    if key and api_key != key:
        return {"error": "invalid api_key"}
    from .firestore_client import _get_client
    client = _get_client()
    coll = client.collection("businesses")
    out = []
    for d in coll.limit(1000).stream():
        data = d.to_dict() or {}
        ex_lat, ex_lng = _extract_lat_lng_from_doc(data)
        if ex_lat is None or ex_lng is None:
            continue
        dist = haversine_km(float(lat), float(lng), float(ex_lat), float(ex_lng))
        if dist <= float(radius_km):
            out.append({"id": d.id, "name": data.get("name"), "extracted_lat": ex_lat, "extracted_lng": ex_lng, "distance_km": dist, "normalized_tags": data.get("normalized_tags"), "tags": data.get("tags")})
    out.sort(key=lambda x: x["distance_km"])
    return {"count": len(out), "results": out}


@app.get("/debug/sample_businesses")
def debug_sample_businesses(limit: int = 50, api_key: str = None):
    """Return the first `limit` businesses with their raw document fields for inspection.

    Protected by DEBUG_API_KEY env var.
    """
    import os
    from .firestore_client import _get_client
    key = os.environ.get("DEBUG_API_KEY")
    if key and api_key != key:
        return {"error": "invalid api_key"}
    client = _get_client()
    col = client.collection("businesses")
    out = []
    count = 0
    for d in col.limit(int(limit)).stream():
        data = d.to_dict() or {}
        # include the raw document for inspection
        out.append({"id": d.id, "doc": data})
        count += 1
    return {"count": count, "results": out}


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
    logger.debug("Received webhook payload keys: %s", list(payload.keys()))

    # v1.9: Handle WhatsApp native location messages first and persist last-known location
    loc_msg = extract_location_message(payload)
    if loc_msg:
        logger.info("Received native location message from %s", loc_msg.get("from"))
        user_number = loc_msg["from"]
        upsert_whatsapp_user_location(user_number, loc_msg["lat"], loc_msg["lng"])
        logger.info("Saved location for %s lat=%s lng=%s", user_number, loc_msg["lat"], loc_msg["lng"])  # debug trace
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
        logger.info("No text message parsed from payload; acknowledging 200")
        return Response(status_code=200)

    user_number = msg["from"]
    user_text = msg["text"].strip()

    lowered = user_text.lower()
    logger.info("Processing text message from %s: %s", user_number, user_text[:120])

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

    # v1.9: If user sets location via text command, geocode and save
    addr = parse_text_location_command(user_text)
    if addr:
        import httpx
        import urllib.parse
        api_key = settings.google_maps_api_key
        if not api_key:
            await send_whatsapp_reply(user_number, "I couldn't set your location because the Maps API key isn't configured.")
            return Response(status_code=200)
        try:
            q = urllib.parse.urlencode({"address": addr, "key": api_key})
            url = f"https://maps.googleapis.com/maps/api/geocode/json?{q}"
            async with httpx.AsyncClient(timeout=15) as hc:
                resp = await hc.get(url)
                resp.raise_for_status()
                data = resp.json()
            results = data.get("results") or []
            if not results:
                await send_whatsapp_reply(user_number, "I couldn't find that address. Please try sharing a location pin.")
                return Response(status_code=200)
            loc = results[0].get("geometry", {}).get("location", {})
            lat = float(loc.get("lat"))
            lng = float(loc.get("lng"))
            upsert_whatsapp_user_location(user_number, lat, lng)
            await send_whatsapp_reply(user_number, f"Got it – I've updated your location to {results[0].get('formatted_address','that area')}.")
            log_message(user_number, "bot", "[saved geocoded location]")
            return Response(status_code=200)
        except Exception:
            await send_whatsapp_reply(user_number, "I couldn't set your location right now. Please share a location pin instead.")
            return Response(status_code=200)

    # v1.9: If the user asks for something "closest" or "near me", use last-known location
    if ("closest" in lowered) or ("near me" in lowered) or ("nearest" in lowered):
        # Heuristic to extract a business type keyword
        import re
        biz_type = ""
        def words_after(token: str) -> list[str]:
            seg = lowered.split(token, 1)[1] if token in lowered else ""
            return re.findall(r"[a-zA-Z]+", seg)
        def words_before(token: str) -> list[str]:
            seg = lowered.split(token, 1)[0] if token in lowered else ""
            return re.findall(r"[a-zA-Z]+", seg)
        # Prefer a noun phrase of up to 3 words (e.g., 'ford dealership', 'used car')
        candidates: list[str] = []
        wa = words_after("closest")
        if wa:
            candidates.append(" ".join(wa[:3]).strip())
        wa = words_after("nearest")
        if wa:
            candidates.append(" ".join(wa[:3]).strip())
        wb = words_before("near me")
        if wb:
            tail = wb[-3:]
            candidates.append(" ".join(tail).strip())
        # Reduce candidates: longest non-empty first
        candidates = sorted([c for c in candidates if c], key=lambda s: -len(s))
        biz_type = candidates[0] if candidates else ""

        loc = get_user_location(user_number)
        if not loc:
            await send_whatsapp_reply(
                user_number,
                "Could you please share your current location using the WhatsApp location pin? Then ask again.",
            )
            log_message(user_number, "bot", "Requested location pin for closest search")
            return Response(status_code=200)

        lat, lng = loc
        logger.info("Closest search user=%s phrase='%s' lat=%s lng=%s", user_number, biz_type, lat, lng)  # debug trace
        # Try progressively larger radii, then fall back to DB list if still empty
        matches = find_closest_businesses(lat, lng, biz_type or "", limit=3, max_radius_km=50.0)
        if not matches:
            matches = find_closest_businesses(lat, lng, biz_type or "", limit=3, max_radius_km=200.0)
        if not matches:
            matches = find_closest_businesses(lat, lng, biz_type or "", limit=3, max_radius_km=None)
        logger.info("Closest search results user=%s phrase='%s' count=%d", user_number, biz_type, len(matches))  # debug trace
        if not matches:
            await send_whatsapp_reply(user_number, "I couldn't find matching places nearby in my list.")
            log_message(user_number, "bot", "No nearby results")
            return Response(status_code=200)
        # Log details of returned matches for easier debugging/verification
        for b, dkm in matches:
            logger.info("Closest match: %s (id=%s) dist_km=%.3f", b.name, b.id, dkm)

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
        # First check if we have user location to enhance business search
        loc = get_user_location(user_number)
        client = VertexAIClient()
        try:
            # Extract potential business type from query
            import re
            words = [w.lower() for w in re.findall(r"[a-zA-Z]+", user_text)]
            business_type = ""
            
            # Try to extract business type from common patterns
            for kw in ["find", "looking for", "need", "want", "search for"]:
                if kw in lowered:
                    idx = lowered.find(kw)
                    if idx >= 0:
                        business_type = lowered[idx + len(kw):].strip()
                        break
            
            # If we have location and business type, try to find closest businesses first
            if loc and business_type:
                lat, lng = loc
                matches = find_closest_businesses(lat, lng, business_type, limit=3, max_radius_km=50.0)
                if matches:
                    # Format results from Firestore directly
                    lines = ["Here are some businesses that match your search:"]
                    for b, dist_km in matches:
                        lines.append(f"- {b.name} — {dist_km:.1f} km away")
                    ai_reply = "\n".join(lines)
                else:
                    # Fall back to AI if no direct matches
                    ai_reply = client.generate_business_reply(user_text, user_number=user_number)
            else:
                # No location or business type, use AI
                ai_reply = client.generate_business_reply(user_text, user_number=user_number)
        except Exception as e:
            logger.error(f"Business search error: {e}")
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


