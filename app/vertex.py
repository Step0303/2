from typing import Optional
import logging

import vertexai
from vertexai.generative_models import GenerativeModel

from .config import settings


class VertexAIClient:
    """Thin wrapper around Vertex AI Gemini text generation."""

    def __init__(self, project_id: Optional[str] = None, location: Optional[str] = None, model_name: Optional[str] = None):
        self.project_id = project_id or settings.gcp_project
        self.location = location or settings.gcp_location
        self.model_name = model_name or settings.vertex_model

        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT (or equivalent) must be set for Vertex AI")

        vertexai.init(project=self.project_id, location=self.location)
        self._model = GenerativeModel(self.model_name)

    def generate_reply(self, user_text: str) -> str:
        """Generate a concise helpful reply to the user's WhatsApp message."""
        prompt = (
            "You are a helpful assistant responding over WhatsApp. "
            "Keep responses concise and friendly. If the message is a greeting, greet back. "
            "If the message asks a question, answer clearly."
            "\n\nUser message: " + user_text.strip()
        )

        try:
            response = self._model.generate_content(prompt)
        except Exception as exc:  # pragma: no cover
            logging.exception("Vertex AI generate_content failed: %s", exc)
            return "I’m here! How can I help today?"

        # Prefer unified .text, otherwise stitch candidate part texts
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        try:
            pieces: list[str] = []
            for cand in getattr(response, "candidates", []) or []:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content is not None else []
                for p in parts or []:
                    t = getattr(p, "text", None)
                    if isinstance(t, str) and t.strip():
                        pieces.append(t.strip())
            combined = "\n".join(pieces).strip()
            return combined or "I’m here! How can I help today?"
        except Exception as exc:  # pragma: no cover
            logging.exception("Vertex AI response parse error: %s", exc)
            return "I’m here! How can I help today?"


