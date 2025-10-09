from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

from google.cloud import firestore

from .config import settings


def _get_client() -> firestore.Client:
    # Uses ADC in Cloud Run; locally uses gcloud auth or env credentials
    return firestore.Client(project=settings.gcp_project)


@dataclass
class Business:
    id: str
    name: str
    type: str
    latitude: float
    longitude: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _extract_lat_lng_from_doc(data: dict) -> Tuple[Optional[float], Optional[float]]:
    """Robustly extract latitude and longitude from a business document dict.

    Supports several shapes found in Firestore records:
    - location: { coordinate: {lat, lng} } or { coordinate: GeoPoint }
    - location: { lat: .., lng: .. }
    - location: GeoPoint
    - coordinates: {lat, lng} or {latitude, longitude} or list [lat, lng] or [lng, lat]
    - top-level latitude / longitude fields
    Returns (lat, lng) as floats when found, else (None, None).
    """
    lat = None
    lng = None
    try:
        loc = data.get("location") if isinstance(data, dict) else None
        # location may be a dict with 'coordinate' or lat/lng
        if isinstance(loc, dict):
            coord = loc.get("coordinate") or loc.get("coordinates")
            if coord is not None:
                # GeoPoint-like: attributes 'latitude' and 'longitude'
                if hasattr(coord, "latitude") and hasattr(coord, "longitude"):
                    lat = getattr(coord, "latitude")
                    lng = getattr(coord, "longitude")
                elif isinstance(coord, dict):
                    lat = coord.get("lat") or coord.get("latitude")
                    lng = coord.get("lng") or coord.get("longitude")
            # flat lat/lng in location
            if (lat is None or lng is None) and isinstance(loc, dict):
                lat = lat or loc.get("lat") or loc.get("latitude")
                lng = lng or loc.get("lng") or loc.get("longitude")
        else:
            # location might itself be a GeoPoint-like object
            if loc is not None and hasattr(loc, "latitude") and hasattr(loc, "longitude"):
                lat = getattr(loc, "latitude")
                lng = getattr(loc, "longitude")

        # coordinates field: dict, list, or GeoPoint-like
        if (lat is None or lng is None) and data.get("coordinates") is not None:
            coord2 = data.get("coordinates")
            if hasattr(coord2, "latitude") and hasattr(coord2, "longitude"):
                lat = getattr(coord2, "latitude")
                lng = getattr(coord2, "longitude")
            elif isinstance(coord2, dict):
                lat = lat or coord2.get("lat") or coord2.get("latitude")
                lng = lng or coord2.get("lng") or coord2.get("longitude")
            elif isinstance(coord2, (list, tuple)) and len(coord2) >= 2:
                # try interpret as [lat, lng] first, else [lng, lat]
                try:
                    a = float(coord2[0])
                    b = float(coord2[1])
                    # Heuristic: if a is in [-90,90] treat as lat
                    if -90 <= a <= 90:
                        lat = lat or a
                        lng = lng or b
                    else:
                        # assume [lng, lat]
                        lat = lat or b
                        lng = lng or a
                except Exception:
                    pass

        # top-level fallbacks
        if lat is None or lng is None:
            lat = lat or data.get("lat") or data.get("latitude")
            lng = lng or data.get("lng") or data.get("longitude")

        if lat is None or lng is None:
            return None, None
        return float(lat), float(lng)
    except Exception:
        return None, None


def upsert_whatsapp_user_location(phone: str, latitude: float, longitude: float) -> None:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone)
    doc.set({
        "phone": phone,
        # Store both nested coordinate and flat lat/lng for compatibility
        "location": {
            "coordinate": {"lat": latitude, "lng": longitude},
            "lat": latitude,
            "lng": longitude,
        },
    }, merge=True)


def get_user_location(phone: str) -> Optional[Tuple[float, float]]:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    loc = data.get("location") or {}
    # Prefer location.coordinate.{lat,lng}; fall back to previous shapes
    coord = loc.get("coordinate") if isinstance(loc, dict) else None
    lat = None
    lng = None
    if isinstance(coord, dict):
        lat = coord.get("lat")
        lng = coord.get("lng")
    if lat is None or lng is None:
        lat = loc.get("lat") if isinstance(loc, dict) else None
        lng = loc.get("lng") if isinstance(loc, dict) else None
    if lat is None or lng is None:
        return None
    return float(lat), float(lng)


def list_categories(limit: int = 200) -> List[str]:
    """Return up to N category names from `categories` collection if present.

    Falls back to deriving categories from businesses' normalized_tags when
    a dedicated categories collection is not available.
    """
    client = _get_client()
    col = client.collection("categories")
    docs = list(col.limit(limit).stream())
    if docs:
        cats: List[str] = []
        for d in docs:
            data = d.to_dict() or {}
            name = data.get("name") or d.id
            if name:
                cats.append(str(name))
        return cats[:limit]

    # Fallback: mine tags from businesses
    tag_set = set()
    for d in client.collection("businesses").limit(500).stream():
        data = d.to_dict() or {}
        tags = data.get("normalized_tags") or []
        for t in tags or []:
            if isinstance(t, str) and t.strip():
                tag_set.add(t.strip())
    return sorted(tag_set)[:limit]


def _slugify(text: str) -> str:
    """Normalize free text to a slug used in categories and tags.

    Lowercase, replace non-alphanumeric with '-', collapse duplicates.
    """
    import re
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-+", "-", t).strip("-")
    return t


def resolve_tag_from_categories(free_text: str) -> Optional[str]:
    """Try map free text like 'Telecommunications/VOIP' to a canonical tag/slug.

    Looks up the `categories` collection where docs may have fields like
    'slug' and 'name'. Returns slug if found, else a slugified free-text.
    """
    client = _get_client()
    slug = _slugify(free_text)
    col = client.collection("categories")
    # Try slug exact match
    Docs = list(col.where("slug", "==", slug).limit(1).stream())
    if Docs:
        return slug
    # Try name exact (case-insensitive best effort via bounded scan)
    target = (free_text or "").strip().lower()
    for d in col.limit(200).stream():
        data = d.to_dict() or {}
        nm = str(data.get("name", "")).strip().lower()
        if nm == target:
            s = str(data.get("slug") or _slugify(nm))
            return s
    # Fallback: return slugified text
    return slug or None


def _tag_variants(text: str) -> List[str]:
    """Return possible tag spellings to match DB values.

    Includes the raw lowercase phrase (spaces preserved) and a slug form.
    """
    raw = (text or "").strip().lower()
    slug = _slugify(raw)
    variants: List[str] = []
    if raw:
        variants.append(raw)
    if slug and slug != raw:
        variants.append(slug)
    return variants


def search_businesses_by_tag(tag: str, *, limit: int = 5) -> List[dict]:
    """Return up to N businesses that contain the given normalized tag."""
    client = _get_client()
    t = (tag or "").strip().lower()
    if not t:
        return []
    # Try multiple variants (raw phrase and slug)
    variants = _tag_variants(t)
    # Query normalized_tags first
    docs_norm = []
    for v in variants:
        docs_norm.extend(list(client.collection("businesses").where("normalized_tags", "array_contains", v).limit(limit).stream()))
    results: List[dict] = []
    seen: Set[str] = set()
    for d in docs_norm:
        data = d.to_dict() or {}
        data["__id"] = d.id
        results.append(data)
        seen.add(d.id)
        if len(results) >= limit:
            return results
    # Then query legacy 'tags'
    remaining = max(0, limit - len(results))
    if remaining > 0:
        for v in variants:
            if len(results) >= limit:
                break
            docs_tags = client.collection("businesses").where("tags", "array_contains", v).limit(remaining).stream()
            for d in docs_tags:
                if d.id in seen:
                    continue
                data = d.to_dict() or {}
                data["__id"] = d.id
                results.append(data)
                seen.add(d.id)
                if len(results) >= limit:
                    break
    return results

def find_closest_businesses(
    user_lat: float,
    user_lng: float,
    business_type: str,
    *,
    limit: int = 1,
    max_radius_km: float = 50.0,
) -> List[Tuple[Business, float]]:
    client = _get_client()
    # Resolve text to a canonical tag/slug and search across tag fields
    base_text = (business_type or "").strip().lower()
    tag = resolve_tag_from_categories(base_text) or base_text
    variants = _tag_variants(tag)
    # Also consider tokenized parts of the business_type to match against
    # normalized_tags like 'car sales' where user may input 'car' or 'sales'.
    import re
    needle = (business_type or "").strip().lower()
    tokens_raw = [t for t in re.split(r"[^a-z0-9]+", needle) if len(t) > 0]
    # Keep tokens of length > 1 to avoid tiny words; keep also the full slug
    tokens = [t for t in tokens_raw if len(t) > 1]
    slug_variant = _slugify(needle)
    # Combine two queries (normalized_tags and tags) for all variants
    docs: List[firestore.DocumentSnapshot] = []
    ids: Set[str] = set()
    for v in variants:
        for d in client.collection("businesses").where("normalized_tags", "array_contains", v).stream():
            if d.id not in ids:
                docs.append(d)
                ids.add(d.id)
    # Also query by individual tokens to match tag elements like 'car' or 'sales'
    for t in tokens:
        for d in client.collection("businesses").where("normalized_tags", "array_contains", t).stream():
            if d.id not in ids:
                docs.append(d)
                ids.add(d.id)
    # Also include slug variant as a tag query
    if slug_variant:
        for d in client.collection("businesses").where("normalized_tags", "array_contains", slug_variant).stream():
            if d.id not in ids:
                docs.append(d)
                ids.add(d.id)
    # Expand with 'tags' results (avoid duplicates)
    for v in variants:
        for d in client.collection("businesses").where("tags", "array_contains", v).stream():
            if d.id not in ids:
                docs.append(d)
                ids.add(d.id)
    # Token-based queries for legacy 'tags' field as well
    for t in tokens:
        for d in client.collection("businesses").where("tags", "array_contains", t).stream():
            if d.id not in ids:
                docs.append(d)
                ids.add(d.id)
    if slug_variant:
        for d in client.collection("businesses").where("tags", "array_contains", slug_variant).stream():
            if d.id not in ids:
                docs.append(d)
                ids.add(d.id)
    # Fallback: bounded scan by name/tags substring contains
    if not docs:
        # Fallback: do a more forgiving keyword/slug match across name and tags.
        import re
        needle = (business_type or "").strip().lower()
        if needle:
            # Split into tokens (words) and keep only meaningful tokens (len>2)
            tokens_raw = [t for t in re.split(r"[^a-z0-9]+", needle) if len(t) > 2]
            # Remove generic stopwords that are not useful for business matching
            STOPWORDS = {"business", "place", "shop", "near", "me", "closest", "nearest", "the", "a", "an", "find", "for"}
            tokens = [t for t in tokens_raw if t not in STOPWORDS]
            # Also include the slugified full phrase as a variant
            slug_variant = _slugify(needle)
            variants = list(dict.fromkeys([*tokens, slug_variant] if slug_variant else tokens))
            pattern = re.compile("|".join(re.escape(v) for v in variants)) if variants else None
        else:
            pattern = None

        # Scan a bounded number of businesses but allow more documents to improve recall
        for d in client.collection("businesses").limit(500).stream():
            data = d.to_dict() or {}
            name = str(data.get("name", "")).lower()
            # include slugified name for matching against hyphenated tags
            name_slug = _slugify(name)
            tags = [str(x).lower() for x in (data.get("normalized_tags") or [])]
            tags += [str(x).lower() for x in (data.get("tags") or [])]
            if pattern and (pattern.search(name) or (name_slug and pattern.search(name_slug)) or any(pattern.search(t) for t in tags)):
                docs.append(d)

    # Geo-only fallback: if we still have no candidate docs, include any businesses within
    # the requested max_radius_km (or all businesses when max_radius_km is None). This helps
    # return exact/co-located businesses even when they lack matching tags or descriptive names.
    if not docs:
        for d in client.collection("businesses").limit(500).stream():
            data = d.to_dict() or {}
            # attempt to extract coordinates similar to the main loop
            lat = None
            lng = None
            loc = data.get("location") or {}
            if isinstance(loc, dict):
                coord = loc.get("coordinate")
                if isinstance(coord, dict):
                    lat = coord.get("lat")
                    lng = coord.get("lng")
                if lat is None or lng is None:
                    lat = loc.get("lat")
                    lng = loc.get("lng")
            if (lat is None or lng is None) and isinstance(data.get("coordinates"), dict):
                coord2 = data.get("coordinates") or {}
                lat = coord2.get("lat", lat)
                lng = coord2.get("lng", lng)
            lat = lat or data.get("latitude")
            lng = lng or data.get("longitude")
            if lat is None or lng is None:
                continue
            try:
                dist = haversine_km(user_lat, user_lng, float(lat), float(lng))
            except Exception:
                continue
            if max_radius_km is not None and dist > float(max_radius_km):
                continue
            if d.id not in ids:
                docs.append(d)
                ids.add(d.id)

    results: List[Tuple[Business, float]] = []
    for d in docs:
        data = d.to_dict() or {}
        # location.coordinate.{lat,lng} preferred
        lat = None
        lng = None
        loc = data.get("location") or {}
        if isinstance(loc, dict):
            coord = loc.get("coordinate")
            if isinstance(coord, dict):
                lat = coord.get("lat")
                lng = coord.get("lng")
            if lat is None or lng is None:
                lat = loc.get("lat")
                lng = loc.get("lng")
        # Support 'coordinates' object: coordinates.{lat,lng}
        if (lat is None or lng is None) and isinstance(data.get("coordinates"), dict):
            coord2 = data.get("coordinates") or {}
            lat = coord2.get("lat", lat)
            lng = coord2.get("lng", lng)
        # fallbacks
        lat = lat or data.get("latitude")
        lng = lng or data.get("longitude")
        name = data.get("name") or d.id
        if lat is None or lng is None:
            continue
        dist = haversine_km(user_lat, user_lng, float(lat), float(lng))
        # Apply radius cap
        if max_radius_km is not None and dist > float(max_radius_km):
            continue
        results.append((Business(id=d.id, name=name, type=business_type, latitude=float(lat), longitude=float(lng)), dist))

    results.sort(key=lambda x: x[1])
    # If we found matches via tags/names return them. Otherwise, as a final
    # fallback, return the nearest businesses purely by geo proximity. This
    # guarantees the user will see nearby options (e.g., Trellidor) even when
    # text/tag matching fails.
    if results:
        return results[:limit]

    # Final geo fallback: scan a larger set and return nearest docs within
    # max_radius_km (or all if max_radius_km is None). Limit the scan to 1000
    # documents to bound runtime.
    geo_candidates: List[Tuple[Business, float]] = []
    for d in client.collection("businesses").limit(1000).stream():
        data = d.to_dict() or {}
        lat, lng = _extract_lat_lng_from_doc(data)
        if lat is None or lng is None:
            continue
        try:
            dist = haversine_km(user_lat, user_lng, float(lat), float(lng))
        except Exception:
            continue
        if max_radius_km is not None and dist > float(max_radius_km):
            continue
        geo_candidates.append((Business(id=d.id, name=data.get("name") or d.id, type=business_type, latitude=float(lat), longitude=float(lng)), dist))

    geo_candidates.sort(key=lambda x: x[1])
    return geo_candidates[:limit]



# ===== Conversation logging (v1.3) =====
def log_message(phone: str, sender: str, text: str) -> None:
    """Append a message to the conversation log in Firestore.

    Collection: processed_messages
      - phone: str (WhatsApp sender number)
      - sender: "user" | "bot"
      - text: message content
      - ts: server timestamp
    """
    client = _get_client()
    client.collection("processed_messages").add({
        "phone": phone,
        "sender": sender,
        "text": text,
        "ts": firestore.SERVER_TIMESTAMP,
    })


def fetch_conversation(phone: str, limit: int = 100) -> List[Tuple[str, str]]:
    """Return up to N (sender, text) pairs in chronological order.

    Note: Avoids composite indexes by not ordering in the query; sorts client-side by ts.
    """
    client = _get_client()
    q = client.collection("processed_messages").where("phone", "==", phone).limit(1000)
    rows_full: List[Tuple[float, str, str]] = []  # (ts_seconds, sender, text)
    for d in q.stream():
        data = d.to_dict() or {}
        ts = data.get("ts")
        ts_seconds = float(ts.timestamp()) if hasattr(ts, "timestamp") else 0.0
        rows_full.append((ts_seconds, str(data.get("sender", "")), str(data.get("text", ""))))
    rows_full.sort(key=lambda t: t[0])
    trimmed = rows_full[-limit:]
    return [(sender, text) for _, sender, text in trimmed]


def clear_conversation(phone: str) -> None:
    """Delete up to 1000 messages for the user (no ordering to avoid index requirements)."""
    client = _get_client()
    q = client.collection("processed_messages").where("phone", "==", phone).limit(1000)
    batch = client.batch()
    count = 0
    for d in q.stream():
        batch.delete(d.reference)
        count += 1
        if count % 400 == 0:
            batch.commit()
            batch = client.batch()
    batch.commit()


# ===== Debug flag (v1.5) =====
def set_debug_enabled(phone: str, enabled: bool) -> None:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone)
    doc.set({"phone": phone, "debug": bool(enabled)}, merge=True)


def is_debug_enabled(phone: str) -> bool:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone).get()
    if not doc.exists:
        return False
    data = doc.to_dict() or {}
    return bool(data.get("debug", False))


# ===== Business lookup (v1.6) =====
def find_business_by_name(name: str) -> Optional[dict]:
    """Find a business document by name (case-insensitive best effort).

    Tries exact match on 'name'. If not found, scans up to 100 docs and matches
    case-insensitively client-side (to avoid extra indexes).
    Returns the first matching document dict with an added '__id' field.
    """
    client = _get_client()
    col = client.collection("businesses")
    # Exact match first
    exact = list(col.where("name", "==", name).limit(1).stream())
    if exact:
        d = exact[0]
        obj = d.to_dict() or {}
        obj["__id"] = d.id
        return obj
    # Best-effort case-insensitive scan (bounded)
    lower = name.strip().lower()
    count = 0
    for d in col.limit(100).stream():
        data = d.to_dict() or {}
        nm = str(data.get("name", ""))
        if nm.strip().lower() == lower:
            data["__id"] = d.id
            return data
        count += 1
    return None

