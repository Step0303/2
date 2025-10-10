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
    find_category_ids, resolve_tag_from_categories, infer_category_ids_from_business_tags,
    suggest_tag_candidates, set_pending_suggestions, get_pending_suggestions, clear_pending_suggestions, suggest_token_nearest,
    upsert_whatsapp_user_location_with_address, businesses_by_category_within_radius,
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


@app.get("/debug/find_closest")
def debug_find_closest(lat: float, lng: float, phrase: str = "", limit: int = 5, max_radius_km: float | None = 50.0, api_key: str = None):
    """Call find_closest_businesses and return the result for debugging.

    Protected by DEBUG_API_KEY if set.
    """
    import os
    key = os.environ.get("DEBUG_API_KEY")
    if key and api_key != key:
        return {"error": "invalid api_key"}
    from .firestore_client import find_closest_businesses
    matches = find_closest_businesses(float(lat), float(lng), phrase or "", limit=int(limit), max_radius_km=(None if str(max_radius_km).lower() == 'none' else float(max_radius_km)))
    out = []
    for b, d in matches:
        out.append({"id": b.id, "name": b.name, "latitude": b.latitude, "longitude": b.longitude, "distance_km": d})
    return {"count": len(out), "results": out}


@app.get("/debug/resolve_category")
def debug_resolve_category(phrase: str = "", api_key: str = None):
    """Return resolved category ids and slug for a free-text phrase (debug only)."""
    import os
    key = os.environ.get("DEBUG_API_KEY")
    if key and api_key != key:
        return {"error": "invalid api_key"}
    slug = resolve_tag_from_categories(phrase or "")
    ids = find_category_ids(phrase or "")
    # If categories collection didn't match, try inferring from businesses
    inferred = []
    if not ids:
        try:
            inferred = infer_category_ids_from_business_tags(phrase or "")
        except Exception:
            inferred = []
    return {"phrase": phrase, "slug": slug, "category_ids": ids, "inferred_category_ids": inferred}


@app.get("/debug/business")
def debug_get_business(id: str | None = None, ids: str | None = None, api_key: str = None):
    """Return raw business document(s) for the given id or comma-separated ids.

    Use `id` for a single id or `ids=comma,separated,ids` for multiple.
    """
    import os
    from .firestore_client import _get_client
    key = os.environ.get("DEBUG_API_KEY")
    if key and api_key != key:
        return {"error": "invalid api_key"}
    c = _get_client()
    wanted = []
    if id:
        wanted = [id]
    elif ids:
        wanted = [s.strip() for s in ids.split(",") if s.strip()]
    else:
        return {"error": "no id(s) provided"}
    out = []
    for bid in wanted:
        doc = c.collection("businesses").document(bid).get()
        if not doc.exists:
            out.append({"id": bid, "found": False})
            continue
        data = doc.to_dict() or {}
        # Only return a subset useful for debugging
        out.append({
            "id": bid,
            "found": True,
            "name": data.get("name"),
            "category": data.get("category"),
            "categories": data.get("categories"),
            "normalized_tags": data.get("normalized_tags"),
            "tags": data.get("tags"),
            "location": data.get("location"),
        })
    return {"count": len(out), "results": out}

@app.get("/debug/whatsapp_user")
def debug_whatsapp_user(phone: str, api_key: str | None = None):
    """Return stored whatsapp_users/{phone} fields useful for debugging.

    Protected by DEBUG_API_KEY if set.
    """
    import os
    from .firestore_client import _get_client
    key = os.environ.get("DEBUG_API_KEY")
    if key and api_key != key:
        return {"error": "invalid api_key"}
    c = _get_client()
    doc = c.collection("whatsapp_users").document(phone).get()
    if not doc.exists:
        return {"found": False}
    data = doc.to_dict() or {}
    # return only useful fields
    return {
        "found": True,
        "phone": data.get("phone"),
        "location": data.get("location"),
        "pending_suggestions": data.get("pending_suggestions"),
        "last_structured_query": data.get("last_structured_query"),
        "last_suggestions": data.get("last_suggestions"),
    }


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

    # Early: if the user is replying to a previous suggestion prompt with a number,
    # interpret it here so the rest of the flow uses the selected tag/business.
    pending_early = get_pending_suggestions(user_number)
    if pending_early:
        import re
        m = re.match(r"^\s*(\d+)\s*$", user_text)
        if m:
            idx = int(m.group(1)) - 1
            opts = pending_early.get("options") or []
            if 0 <= idx < len(opts):
                choice_obj = opts[idx]
                clear_pending_suggestions(user_number)
                log_message(user_number, "user", f"selected_suggestion:{choice_obj}")
                # If the stored option is a dict with a 'value' token, use that as the user's text
                if isinstance(choice_obj, dict) and choice_obj.get("value"):
                    user_text = choice_obj.get("value")
                    lowered = user_text.lower()
                elif isinstance(choice_obj, str):
                    user_text = choice_obj
                    lowered = user_text.lower()
                # otherwise, leave user_text unchanged and let downstream handle
            else:
                await send_whatsapp_reply(user_number, "Sorry, I didn't understand that selection. Reply with the number of your choice.")
                return Response(status_code=200)
        else:
            # Allow users to reply with the suggestion text itself (case-insensitive)
            opts = pending_early.get("options") or []
            matched = False
            for choice_obj in opts:
                # normalize possible stored forms
                val = None
                if isinstance(choice_obj, dict):
                    val = choice_obj.get("value") or choice_obj.get("business_name") or choice_obj.get("name")
                elif isinstance(choice_obj, str):
                    val = choice_obj
                if not val:
                    continue
                try:
                    if user_text.strip().lower() == str(val).strip().lower():
                        # treat as selection
                        clear_pending_suggestions(user_number)
                        log_message(user_number, "user", f"selected_suggestion:{choice_obj}")
                        if isinstance(choice_obj, dict) and choice_obj.get("value"):
                            user_text = choice_obj.get("value")
                        elif isinstance(choice_obj, str):
                            user_text = choice_obj
                        lowered = user_text.lower()
                        matched = True
                        break
                except Exception:
                    continue
            if not matched:
                await send_whatsapp_reply(user_number, "Sorry, I didn't understand that selection. Reply with the number of your choice.")
                return Response(status_code=200)

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
            # Save address + coords for v2.1
            formatted = results[0].get('formatted_address')
            upsert_whatsapp_user_location_with_address(user_number, lat, lng, formatted)
            await send_whatsapp_reply(user_number, f"Got it – I've updated your location to {formatted}.")
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
            # Prompt for location; user must provide to continue
            await send_whatsapp_reply(
                user_number,
                "Please share your current location (use the WhatsApp location pin) so I can find nearby businesses."
            )
            log_message(user_number, "bot", "Requested location pin for closest search")
            return Response(status_code=200)

        lat, lng = loc
        logger.info("Closest search user=%s phrase='%s' lat=%s lng=%s", user_number, biz_type, lat, lng)  # debug trace
        # New v2.2: LLM-first structured query
        client = VertexAIClient()
        spec, raw_text = client.generate_structured_query(biz_type or user_text, user_lat=lat, user_lng=lng)
        # persist raw LLM output for debugging
        from .firestore_client import set_last_structured_query
        try:
            set_last_structured_query(user_number, {"spec": spec, "raw": raw_text})
        except Exception:
            pass
        if not spec:
            # fallback to previous behavior if LLM didn't produce a query
            matches = find_closest_businesses(lat, lng, biz_type or "", limit=3, max_radius_km=50.0, allow_geo_fallback=False)
            if not matches:
                token_suggestions = suggest_token_nearest(lat, lng, biz_type or "", token_limit=5)
                if token_suggestions:
                    opts = [s for s in token_suggestions]
                    set_pending_suggestions(user_number, opts, biz_type or "")
                    lines = ["I didn't find an exact match. Did you mean one of these? Reply with the number:"]
                    for i, s in enumerate(opts, start=1):
                        lines.append(f"{i}. {s['business_name']} (matches '{s['value']}') — {s['distance_km']:.1f} km")
                    await send_whatsapp_reply(user_number, "\n".join(lines))
                    log_message(user_number, "bot", "prompted suggestions")
                    return Response(status_code=200)
                matches = find_closest_businesses(lat, lng, biz_type or "", limit=3, max_radius_km=None)
            if not matches:
                await send_whatsapp_reply(user_number, "I couldn't find matching places nearby in my list.")
                log_message(user_number, "bot", "No nearby results")
                return Response(status_code=200)
            # Format matches into a reply
            lines = ["Here are the closest options:"]
            for b, dist_km in matches:
                lines.append(f"- {b.name} — {dist_km:.1f} km away")
            reply = "\n".join(lines)
            log_message(user_number, "user", user_text)
            log_message(user_number, "bot", reply)
            await send_whatsapp_reply(user_number, reply)
            return Response(status_code=200)

        # Execute the structured query returned by the LLM
        try:
            from .firestore_client import execute_structured_query, set_last_suggestions
            results = execute_structured_query(spec, user_lat=lat, user_lng=lng)
        except Exception as exc:
            logger.exception("Structured query execution failed: %s", exc)
            await send_whatsapp_reply(user_number, "Sorry, I couldn't run that search right now.")
            return Response(status_code=200)

        if not results:
            # Ask LLM for alternative suggestions (did-you-mean) using nearby tags/names
            # Build candidate list by scanning a small set of businesses for tags/names
            from .firestore_client import _get_client
            c = _get_client()
            cand = set()
            for d in c.collection("businesses").limit(200).stream():
                data = d.to_dict() or {}
                for t in (data.get("normalized_tags") or []):
                    if isinstance(t, str) and t.strip():
                        cand.add(t.strip())
                name = str(data.get("name") or "").strip()
                if name:
                    cand.add(name)
            cand_list = list(cand)[:200]
            suggestions = client.suggest_alternatives(user_text, cand_list)
            if suggestions:
                # persist suggestions and pending options
                try:
                    set_last_suggestions(user_number, suggestions)
                except Exception:
                    logger.exception("Failed to persist suggestions")
                set_pending_suggestions(user_number, [ {"value": s, "business_name": s} for s in suggestions ], user_text)
                lines = ["I didn't find exact matches. Did you mean one of these? Reply with the number:"]
                for i, s in enumerate(suggestions, start=1):
                    lines.append(f"{i}. {s}")
                # if debug is enabled, echo the structured query and suggestions in logs
                if is_debug_enabled(user_number):
                    logger.info("Structured query for %s: %s", user_number, spec)
                    logger.info("Suggestions for %s: %s", user_number, suggestions)
                await send_whatsapp_reply(user_number, "\n".join(lines))
                log_message(user_number, "bot", "prompted suggestions")
                return Response(status_code=200)

        # Ask LLM to format the results into a nice WhatsApp reply
        formatted = client.format_results_with_llm(user_text, results, user_number=user_number)
        log_message(user_number, "user", user_text)
        log_message(user_number, "bot", formatted)
        await send_whatsapp_reply(user_number, formatted)
        return Response(status_code=200)

    # v1.8: Otherwise, if the message appears to be a general business search, use the business prompt
    if any(kw in lowered for kw in ["business ", "find ", "restaurant", "garage", "pharmacy", "hospital", "doctor", "shop", "cafe", "coffee"]):
        # First check if we have user location to enhance business search
        loc = get_user_location(user_number)
        client = VertexAIClient()
        # Check if the user is replying to a suggestions prompt with a number
        pending = get_pending_suggestions(user_number)
        if pending:
            # See if the user's message is a selection index
            import re
            m = re.match(r"^\s*(\d+)\s*$", user_text)
            if m:
                idx = int(m.group(1)) - 1
                opts = pending.get("options") or pending.get("options") or []
                if 0 <= idx < len(opts):
                    choice_obj = opts[idx]
                    # clear pending
                    clear_pending_suggestions(user_number)
                    log_message(user_number, "user", f"selected_suggestion:{choice_obj}")
                    # If the stored option is a dict with business_id, rerun a precise search
                    if isinstance(choice_obj, dict) and choice_obj.get("business_id"):
                        # perform search by business id (single result)
                        bid = choice_obj["business_id"]
                        # Fetch the business doc directly and return it
                        from .firestore_client import _get_client
                        c = _get_client()
                        doc = c.collection("businesses").document(bid).get()
                        if doc.exists:
                            data = doc.to_dict() or {}
                            latlng = data.get("location", {}).get("coordinates") or data.get("location", {})
                            # respond with the single match
                            await send_whatsapp_reply(user_number, f"Here is the business you selected: {data.get('name')} — located at {data.get('location', {}).get('address','unknown')}")
                            return Response(status_code=200)
                        else:
                            await send_whatsapp_reply(user_number, "Sorry, I couldn't find the selected business anymore.")
                            return Response(status_code=200)
                    else:
                        # If option is a simple tag string, set business_type to that tag
                        business_type = choice_obj if isinstance(choice_obj, str) else str(choice_obj)
                else:
                    await send_whatsapp_reply(user_number, "Sorry, I didn't understand that selection. Please reply with the number of your choice.")
                    return Response(status_code=200)
        try:
            client = VertexAIClient()
            # Ask LLM to produce a structured query
            user_lat = loc[0] if loc else None
            user_lng = loc[1] if loc else None
            spec = client.generate_structured_query(user_text, user_lat=user_lat, user_lng=user_lng)
            if spec:
                try:
                    from .firestore_client import execute_structured_query
                    results = execute_structured_query(spec, user_lat=user_lat, user_lng=user_lng)
                    ai_reply = client.format_results_with_llm(user_text, results, user_number=user_number)
                except Exception:
                    # if execution failed, fallback to the older behavior
                    ai_reply = client.generate_business_reply(user_text, user_number=user_number)
            else:
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


