from typing import Optional
import logging

import vertexai
from vertexai.generative_models import GenerativeModel

from .config import settings


class VertexAIClient:
    """Thin wrapper around Vertex AI Gemini text generation."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt_path: Optional[str] = "prompts/system_prompt.txt",
        business_prompt_path: Optional[str] = "prompts/business_search_prompt.txt",
    ):
        self.project_id = project_id or settings.gcp_project
        self.location = location or settings.gcp_location
        self.model_name = model_name or settings.vertex_model

        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT (or equivalent) must be set for Vertex AI")

        vertexai.init(project=self.project_id, location=self.location)
        self._model = GenerativeModel(self.model_name)

        # Load prompt files if present
        self.system_prompt = self._load_prompt(system_prompt_path)
        self.business_prompt = self._load_prompt(business_prompt_path)

    def generate_reply(self, user_text: str) -> str:
        """Generate a concise helpful reply to the user's WhatsApp message."""
        base = self.system_prompt or (
            "You are a helpful assistant responding over WhatsApp. "
            "Keep responses concise and friendly."
        )
        prompt = f"{base}\n\nUser message: {user_text.strip()}"

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

    def generate_business_reply(self, user_text: str) -> str:
        """Use the business-specific prompt to guide the model."""
        base = self.business_prompt or (
            "You help users search for local businesses. Ask for missing details, "
            "and respond with a clean, scannable list."
        )
        prompt = f"{base}\n\nUser query: {user_text.strip()}"
        try:
            response = self._model.generate_content(prompt)
        except Exception as exc:  # pragma: no cover
            logging.exception("Vertex AI business generate_content failed: %s", exc)
            return "Sorry, I couldn’t search right now. Please try again."

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        try:
            parts: list[str] = []
            for cand in getattr(response, "candidates", []) or []:
                content = getattr(cand, "content", None)
                pparts = getattr(content, "parts", []) if content is not None else []
                for p in pparts or []:
                    t = getattr(p, "text", None)
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            combined = "\n".join(parts).strip()
            return combined or "No results yet. Could you specify the area or type?"
        except Exception as exc:  # pragma: no cover
            logging.exception("Vertex AI business parse error: %s", exc)
            return "No results yet. Could you specify the area or type?"

    @staticmethod
    def _load_prompt(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            import os
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            return None
        except Exception:  # pragma: no cover
            return None


