from __future__ import annotations
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel

from .config import settings


_model: Optional[GenerativeModel] = None


def get_model() -> GenerativeModel:
    global _model
    if _model is None:
        vertexai.init(project=settings.project_id, location=settings.location)
        _model = GenerativeModel(settings.vertex_model)
    return _model


def extract_category(query: str) -> Optional[str]:
    """Very light-weight extractor. Since the default is 'used cars',
    we only override if the model is confident about a different category.
    """
    if not query:
        return None
    try:
        model = get_model()
        prompt = (
            "Extract the business category from this user request. "
            "Return only a single lowercase noun phrase like 'used cars', 'pharmacy', or 'coffee shop'.\n\n"
            f"User: {query}\n"
        )
        resp = model.generate_content(prompt)
        text = resp.text.strip().lower() if hasattr(resp, "text") else None
        if not text:
            return None
        # Heuristic: if 'car' present, prefer 'used cars'
        if "car" in text and "used" in text:
            return "used cars"
        return text
    except Exception:
        return None
