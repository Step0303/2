from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

from .config import settings
from .whatsapp import parse_inbound
from . import firestore_client
from . import nlu

app = FastAPI(title="Frikkie Nearest Business Bot", version="0.1.0")


@app.get("/webhook/whatsapp")
async def whatsapp_verify(mode: Optional[str] = None, challenge: Optional[str] = None, verify_token: Optional[str] = None, hub_mode: Optional[str] = None, hub_verify_token: Optional[str] = None, hub_challenge: Optional[str] = None):
    token = settings.whatsapp_verify_token
    # Meta sends either as hub.* or plain query params depending on client
    provided_token = hub_verify_token or verify_token
    provided_mode = hub_mode or mode
    provided_challenge = hub_challenge or challenge
    if provided_mode == "subscribe" and provided_token == token:
        return PlainTextResponse(provided_challenge or "")
    raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(req: Request):
    payload: Dict[str, Any] = await req.json()
    inbound = parse_inbound(payload)
    if not inbound:
        return {"status": "ignored"}

    user_id = inbound["user_id"]
    text = inbound.get("text") or ""
    location = inbound.get("location")

    if not settings.whatsapp_token or not settings.whatsapp_phone_id:
        # Cannot respond without credentials
        raise HTTPException(status_code=500, detail="WhatsApp credentials not configured")

    # Determine category (default to 'used cars')
    category = settings.category_default
    extracted = nlu.extract_category(text)
    if extracted:
        category = extracted

    if not location:
        # Ask for location pin
        msg = "Please share your location so I can find the nearest used-car dealers (default 5 km, max 20 km)."
        _send_whatsapp_text(user_id, msg)
        return {"status": "asked_location"}

    lat, lng = location

    # Search Firestore: exact category first, then fallback and/or radius expansion
    results = firestore_client.search_nearby(
        lat=lat,
        lng=lng,
        radius_m=settings.default_radius_m,
        category_exact=category,
        expand_to_max_if_empty=True,
        max_radius_m=settings.max_radius_m,
        limit=max(settings.top_k_results, 10),
    )

    if not results:
        _send_whatsapp_text(user_id, "Sorry, I couldn't find nearby places. Try increasing the radius or moving the pin.")
        return {"status": "no_results"}

    # Format top N
    top = results[: min(settings.top_k_results, 3)]
    lines: List[str] = []
    for i, b in enumerate(top, start=1):
        km = b["distance_m"] / 1000.0
        name = b.get("name") or "Business"
        address = b.get("address") or ""
        lines.append(f"{i}. {name} — {km:.1f} km\n{address}")
    if len(results) > len(top):
        lines.append("\nReply 'more' to see additional results.")

    _send_whatsapp_text(user_id, "\n\n".join(lines))
    return {"status": "ok", "count": len(top)}


def _send_whatsapp_text(to_wa_id: str, message: str) -> None:
    url = f"https://graph.facebook.com/v20.0/{settings.whatsapp_phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_wa_id,
        "type": "text",
        "text": {"body": message},
    }
    try:
        requests.post(url, headers=headers, json=data, timeout=10)
    except Exception:
        # Avoid raising; webhook should return 200 to prevent retries storms
        pass
