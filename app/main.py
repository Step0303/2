import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .config import settings
from .vertex import VertexAIClient
from .whatsapp import extract_text_message, send_whatsapp_reply


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

    msg = extract_text_message(payload)
    if not msg:
        # Return 200 immediately for unsupported events to avoid retries
        return Response(status_code=200)

    user_number = msg["from"]
    user_text = msg["text"]

    try:
        vertex = VertexAIClient()
        reply = vertex.generate_reply(user_text)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Vertex AI generation failed: %s", exc)
        reply = "Sorry, I couldn't generate a response right now. Please try again later."

    await send_whatsapp_reply(user_number, reply)

    return Response(status_code=200)


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


