from typing import Optional

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

        response = self._model.generate_content(prompt)
        # Prefer text output; fall back to empty string if unavailable
        text = getattr(response, "text", None) or ""
        return text.strip() or "I’m here! How can I help today?"


