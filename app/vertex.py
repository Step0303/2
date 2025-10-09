from typing import Optional
import logging

import vertexai
import google.auth
from vertexai.generative_models import GenerativeModel

from .config import settings
from .firestore_client import fetch_conversation, list_categories, search_businesses_by_tag


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
        # Resolve project id: explicit arg → env settings → ADC default project
        resolved_project = project_id or settings.gcp_project
        if not resolved_project:
            try:
                _, adc_project = google.auth.default()
                resolved_project = adc_project
            except Exception:  # pragma: no cover
                resolved_project = None
        self.project_id = resolved_project
        self.location = location or settings.gcp_location
        self.model_name = model_name or settings.vertex_model

        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT not found and default project unavailable for Vertex AI")

        vertexai.init(project=self.project_id, location=self.location)
        self._model = GenerativeModel(self.model_name)

        # Load prompt files if present
        self.system_prompt = self._load_prompt(system_prompt_path)
        self.business_prompt = self._load_prompt(business_prompt_path)

    def generate_reply(self, user_text: str, *, user_number: Optional[str] = None) -> str:
        """Generate a concise helpful reply to the user's WhatsApp message."""
        base = self.system_prompt or (
            "You are a helpful assistant responding over WhatsApp. "
            "Keep responses concise and friendly."
        )
        # Build context: conversation history and domain knowledge
        history_lines: list[str] = []
        if user_number:
            rows = fetch_conversation(user_number, limit=40)  # remember last 40 messages
            for sender, text in rows:
                history_lines.append(f"{sender}: {text}")
        history_block = "\n".join(history_lines)

        categories = list_categories(limit=50)
        categories_block = ", ".join(categories)

        prompt = (
            f"{base}\n\n"
            f"Conversation so far (oldest→newest):\n{history_block}\n\n"
            f"Available business categories: {categories_block}\n\n"
            f"User message: {user_text.strip()}"
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

    def generate_business_reply(self, user_text: str, *, user_number: Optional[str] = None) -> str:
        """Use the business-specific prompt to guide the model."""
        base = self.business_prompt or (
            "You help users search for local businesses. Ask for missing details, "
            "and respond with a clean, scannable list."
        )
        # Try to derive a tag from the user query (simple heuristic: last noun-ish word)
        # Then fetch matching businesses to ground the model before generation
        import re
        words = [w.lower() for w in re.findall(r"[a-zA-Z]+", user_text)]
        candidate = words[-1] if words else ""
        docs = search_businesses_by_tag(candidate, limit=5) if candidate else []
        corpus_lines: list[str] = []
        for d in docs:
            name = d.get("name", d.get("__id"))
            desc = d.get("description") or d.get("about") or ""
            tags = ", ".join([t for t in (d.get("normalized_tags") or []) if isinstance(t, str)])
            corpus_lines.append(f"- {name} | {desc} | tags: {tags}")
        corpus_block = "\n".join(corpus_lines) or "(no matching businesses found in DB)"

        prompt = (
            f"{base}\n\n"
            f"Database results for context:\n{corpus_block}\n\n"
            f"User query: {user_text.strip()}"
        )
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


