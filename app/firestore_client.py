from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict

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
        # location will include coordinate, lat/lng and optional address
        "location": {
            "coordinate": {"lat": latitude, "lng": longitude},
            "lat": latitude,
            "lng": longitude,
        },
    }, merge=True)


def upsert_whatsapp_user_location_with_address(phone: str, latitude: float, longitude: float, address: Optional[str] = None) -> None:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone)
    loc = {
        "coordinate": {"lat": latitude, "lng": longitude},
        "lat": latitude,
        "lng": longitude,
    }
    if address:
        loc["address"] = address
    doc.set({"phone": phone, "location": loc}, merge=True)


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


def set_pending_suggestions(phone: str, options: List[str], original: str) -> None:
    """Store pending suggestion options for a user in whatsapp_users doc."""
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone)
    doc.set({"phone": phone, "pending_suggestions": {"original": original, "options": options}}, merge=True)


def get_pending_suggestions(phone: str) -> Optional[Dict]:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    return data.get("pending_suggestions")


def clear_pending_suggestions(phone: str) -> None:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone)
    doc.set({"pending_suggestions": firestore.DELETE_FIELD}, merge=True)


def set_last_structured_query(phone: str, spec: dict) -> None:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone)
    doc.set({"phone": phone, "last_structured_query": spec}, merge=True)


def get_last_structured_query(phone: str) -> Optional[dict]:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    return data.get("last_structured_query")


def set_last_suggestions(phone: str, suggestions: List[str]) -> None:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone)
    doc.set({"phone": phone, "last_suggestions": suggestions}, merge=True)


def get_last_suggestions(phone: str) -> Optional[List[str]]:
    client = _get_client()
    doc = client.collection("whatsapp_users").document(phone).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    return data.get("last_suggestions")


def suggest_tag_candidates(free_text: str, limit: int = 5) -> List[str]:
    """Return up to `limit` tag suggestions for the free_text.

    Heuristic: tokenize free_text, scan a bounded number of businesses and
    count matching normalized_tags that contain any token (substring). Return
    the most frequent tags as suggestions.
    """
    import re
    if not free_text or not str(free_text).strip():
        return []
    tokens = [t for t in re.split(r"[^a-z0-9]+", free_text.lower()) if len(t) > 1]
    if not tokens:
        return []
    counts: Dict[str, int] = {}
    client = _get_client()
    # Scan a bounded set of businesses to keep cost/time reasonable
    for d in client.collection("businesses").limit(500).stream():
        data = d.to_dict() or {}
        tags = [str(x).lower() for x in (data.get("normalized_tags") or [])]
        for tag in tags:
            for t in tokens:
                if t in tag:
                    counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return []
    # sort by frequency then alphabetically
    sorted_tags = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in sorted_tags[:limit]]


def suggest_token_nearest(user_lat: float, user_lng: float, free_text: str, token_limit: int = 5):
    """For each token in free_text, find the nearest business that contains the token
    (in normalized_tags or name). Return up to token_limit suggestions as a list of
    dicts: {value: token, business_id, business_name, distance_km} ordered by distance.
    """
    import re
    tokens = [t for t in re.split(r"[^a-z0-9]+", (free_text or "").lower()) if len(t) > 1]
    if not tokens:
        return []
    client = _get_client()
    # For each token keep the nearest business found
    nearest_for_token = {}
    for d in client.collection("businesses").limit(500).stream():
        data = d.to_dict() or {}
        name = str(data.get("name") or "").lower()
        tags = [str(x).lower() for x in (data.get("normalized_tags") or [])]
        lat, lng = _extract_lat_lng_from_doc(data)
        if lat is None or lng is None:
            continue
        try:
            dist = haversine_km(user_lat, user_lng, float(lat), float(lng))
        except Exception:
            continue
        for t in tokens:
            matched = False
            if any(t in tag for tag in tags):
                matched = True
            elif t in name:
                matched = True
            if not matched:
                continue
            prev = nearest_for_token.get(t)
            if prev is None or dist < prev[0]:
                nearest_for_token[t] = (dist, d.id, data.get("name") or d.id)

    # Build suggestion list ordered by distance
    suggestions = []
    for t, (dist, bid, bname) in sorted(nearest_for_token.items(), key=lambda kv: kv[1][0])[:token_limit]:
        suggestions.append({"value": t, "business_id": bid, "business_name": bname, "distance_km": dist})
    return suggestions


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


def find_category_ids(free_text: str) -> List[str]:
    """Return a list of category document ids that match the free_text.

    Matches by slug (field 'slug') or case-insensitive name. Returns [] when
    no matching category is found.
    """
    if not free_text or not str(free_text).strip():
        return []
    client = _get_client()
    slug = _slugify(free_text)
    col = client.collection("categories")
    ids: List[str] = []
    # Try slug exact match (fast)
    for d in col.where("slug", "==", slug).stream():
        ids.append(d.id)
    if ids:
        return ids
    # Bounded scan for name match (case-insensitive)
    target = str(free_text).strip().lower()
    for d in col.limit(500).stream():
        data = d.to_dict() or {}
        name = str(data.get("name") or "").strip().lower()
        if not name:
            continue
        if name == target or target in name:
            ids.append(d.id)
    return ids


def infer_category_ids_from_business_tags(free_text: str, scan_limit: int = 500) -> List[str]:
    """When `categories` collection does not contain the phrase, try to infer
    category ids by scanning businesses whose `normalized_tags` or `tags`
    contain the phrase (or slug). Return a list of candidate category ids
    ordered by frequency (most frequent first).
    """
    if not free_text or not str(free_text).strip():
        return []
    client = _get_client()
    import re
    slug = _slugify(free_text)
    tokens = [t for t in re.split(r"[^a-z0-9]+", free_text.lower()) if len(t) > 1]
    counts: Dict[str, int] = {}
    scanned = 0
    for d in client.collection("businesses").limit(int(scan_limit)).stream():
        scanned += 1
        data = d.to_dict() or {}
        tags = [str(x).lower() for x in (data.get("normalized_tags") or [])]
        tags += [str(x).lower() for x in (data.get("tags") or [])]
        name = str(data.get("name") or "").lower()
        matched = False
        # match slug or any token in tags or name
        if slug and any(slug == t or slug in t for t in tags):
            matched = True
        if not matched:
            for t in tokens:
                if any(t in tag for tag in tags) or t in name:
                    matched = True
                    break
        if not matched:
            continue
        cat = data.get("category")
        if not cat:
            continue
        counts[str(cat)] = counts.get(str(cat), 0) + 1
    if not counts:
        return []
    # return category ids ordered by frequency
    sorted_ids = [cid for cid, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    return sorted_ids


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


def execute_structured_query(spec: dict, *, user_lat: Optional[float] = None, user_lng: Optional[float] = None, max_scan: int = 500, max_return: int = 50) -> List[dict]:
    """Execute a validated structured query spec (from LLM) against Firestore.

    Spec schema (partial):
      {"collection":"businesses", "filters": [{"field":"normalized_tags","op":"array_contains","value":"pharmacy"}, ...],
       "limit": 20, "geo": {"lat": -26.7, "lng": 27.5, "radius_km": 10.0} }

    This function validates collection and ops, translates simple filters to
    Firestore queries, performs a bounded scan, applies geo filtering server-side,
    and returns a list of document dicts (with selected fields).
    """
    allowed_collections = {"businesses", "categories"}
    allowed_ops = {"==", "array_contains", "in"}
    # Basic validation
    coll = spec.get("collection")
    if coll not in allowed_collections:
        raise ValueError("collection not allowed")
    filters = spec.get("filters") or []
    for f in filters:
        if not isinstance(f, dict) or "field" not in f or "op" not in f:
            raise ValueError("invalid filter")
        if f["op"] not in allowed_ops:
            raise ValueError("op not allowed")

    client = _get_client()
    colref = client.collection(coll)

    # Try to translate simple filters into Firestore where clauses when possible
    query = None
    use_manual_scan = False
    try:
        # Start with the first filter as a base query
        if filters:
            # Only translate filters that are simple equality/array_contains/in
            q = colref
            for f in filters:
                field = f.get("field")
                op = f.get("op")
                val = f.get("value")
                if op == "array_contains":
                    q = q.where(field, "array_contains", val)
                elif op == "==":
                    q = q.where(field, "==", val)
                elif op == "in":
                    q = q.where(field, "in", list(val) if isinstance(val, (list, tuple)) else [val])
                else:
                    use_manual_scan = True
                    break
            query = q
        else:
            # no filters => will require scan (dangerous)
            use_manual_scan = True
    except Exception:
        use_manual_scan = True

    docs = []
    scanned = 0
    limit = int(spec.get("limit") or max_return)
    # Perform query or manual scan
    if query is not None and not use_manual_scan:
        # Execute the Firestore query, but cap total docs scanned
        for d in query.limit(max_scan).stream():
            scanned += 1
            data = d.to_dict() or {}
            data["__id"] = d.id
            docs.append(data)
            if len(docs) >= max_scan:
                break
    else:
        # Full scan (bounded) across collection
        for d in colref.limit(max_scan).stream():
            scanned += 1
            data = d.to_dict() or {}
            data["__id"] = d.id
            docs.append(data)

    # Apply geo filtering if requested
    geo = spec.get("geo") or {}
    radius = geo.get("radius_km")
    lat = geo.get("lat") or user_lat
    lng = geo.get("lng") or user_lng
    out: List[dict] = []
    for data in docs:
        # compute distance if possible
        dlat, dlng = _extract_lat_lng_from_doc(data)
        dist = None
        if dlat is not None and dlng is not None and lat is not None and lng is not None:
            try:
                dist = haversine_km(float(lat), float(lng), float(dlat), float(dlng))
            except Exception:
                dist = None
        # if radius specified and no distance or distance > radius then skip
        if radius is not None and dist is not None and float(dist) > float(radius):
            continue
        # Attach computed distance for later formatting
        if dist is not None:
            data["__distance_km"] = dist
        out.append(data)

    # Sort by distance if present
    out.sort(key=lambda x: x.get("__distance_km", 999999))

    # Trim results to requested limit
    # If no results and filters contained string tokens, try a name/description/tag substring fallback
    if not out:
        # Build token set from filter values
        import re
        tokens: Set[str] = set()
        for f in filters:
            val = f.get("value")
            if isinstance(val, str):
                for t in re.split(r"[^a-z0-9]+", val.lower()):
                    if len(t) > 1:
                        tokens.add(t)
        # If we have tokens, scan and match against name/description/tags
        if tokens:
            out2: List[dict] = []
            for d in colref.limit(max_scan).stream():
                data = d.to_dict() or {}
                name = str(data.get("name") or "").lower()
                desc = str(data.get("description") or data.get("about") or "").lower()
                tags = [str(x).lower() for x in (data.get("normalized_tags") or [])]
                matched = False
                for t in tokens:
                    if t in name or t in desc or any(t in tag for tag in tags):
                        matched = True
                        break
                if not matched:
                    continue
                lat2, lng2 = _extract_lat_lng_from_doc(data)
                dist2 = None
                if lat is not None and lng is not None and lat2 is not None and lng2 is not None:
                    try:
                        dist2 = haversine_km(float(lat), float(lng), float(lat2), float(lng2))
                    except Exception:
                        dist2 = None
                if radius is not None and dist2 is not None and float(dist2) > float(radius):
                    continue
                if dist2 is not None:
                    data["__distance_km"] = dist2
                data["__id"] = d.id
                out2.append(data)
            out2.sort(key=lambda x: x.get("__distance_km", 999999))
            return out2[:limit]
    return out[:limit]


def businesses_by_category_within_radius(cat_id: str, user_lat: float, user_lng: float, radius_km: float = 10.0, limit: int = 100) -> List[dict]:
    """Return full business documents whose 'category' equals cat_id and are within radius_km of user coords.

    Returns list of dicts with added keys '__id' and '__distance_km'.
    """
    client = _get_client()
    out: List[dict] = []
    try:
        q = client.collection("businesses").where("category", "==", cat_id).limit(limit).stream()
    except Exception:
        # Fallback to scanning if the query fails
        q = client.collection("businesses").stream()
    for d in q:
        data = d.to_dict() or {}
        lat, lng = _extract_lat_lng_from_doc(data)
        if lat is None or lng is None:
            continue
        try:
            dist = haversine_km(user_lat, user_lng, float(lat), float(lng))
        except Exception:
            continue
        if radius_km is not None and dist > float(radius_km):
            continue
        data["__id"] = d.id
        data["__distance_km"] = dist
        out.append(data)
    out.sort(key=lambda x: x.get("__distance_km", 9999))
    return out

def find_closest_businesses(
    user_lat: float,
    user_lng: float,
    business_type: str,
    *,
    limit: int = 1,
    max_radius_km: float = 50.0,
    allow_geo_fallback: bool = True,
) -> List[Tuple[Business, float]]:
    client = _get_client()
    # Resolve text to a canonical tag/slug and also try to map to category ids
    base_text = (business_type or "").strip().lower()
    # Phrase synonyms / expansions: common user phrases -> preferred normalized tags
    # This helps map conversational phrases like 'used car' -> 'used cars'/'car sales'
    SYNONYMS = {
        "used car": ["used cars", "car sales", "car dealership", "vehicle sales"],
        "used cars": ["used cars", "car sales", "car dealership", "vehicle sales"],
        "second hand car": ["used cars", "car sales", "car dealership"],
        "pre owned car": ["used cars", "car sales"],
        "pre-owned car": ["used cars", "car sales"],
        "used vehicle": ["used cars", "vehicle sales", "car sales"],
        "used truck": ["truck sales", "vehicle sales"],
        "used van": ["van sales", "vehicle sales"],
    }
    expansion_variants: List[str] = []
    for key, ex_list in SYNONYMS.items():
        if key in base_text:
            # add slug/raw variants for each expansion term, preserving order
            for ex in ex_list:
                for v in _tag_variants(ex):
                    if v not in expansion_variants:
                        expansion_variants.append(v)
            break
    # Try to map the free text to category ids (term_id in your DB)
    cat_ids = find_category_ids(base_text)
    # If we couldn't resolve a category id from the categories collection,
    # attempt to infer one from businesses that have the exact normalized tag.
    # This helps when the categories collection is missing entries but businesses
    # are tagged consistently (e.g., many businesses with normalized_tag
    # 'security barriers' share the same `category` id).
    if not cat_ids:
        # sample businesses for each exact tag variant and collect their categories
        sample_counts: dict = {}
        total_samples = 0
        # use variants of the tag (raw phrase and slug)
        tag_variants = _tag_variants(base_text)
        for v in tag_variants:
            try:
                # limit to 50 samples per variant to bound cost
                for d in client.collection("businesses").where("normalized_tags", "array_contains", v).limit(50).stream():
                    data = d.to_dict() or {}
                    cat = data.get("category")
                    if cat is None:
                        continue
                    total_samples += 1
                    try:
                        key = str(cat)
                    except Exception:
                        continue
                    sample_counts[key] = sample_counts.get(key, 0) + 1
            except Exception:
                continue
        # Pick a dominant category if it exists (at least 2 samples and >=40% of samples)
        if total_samples >= 2 and sample_counts:
            best_cat, best_count = max(sample_counts.items(), key=lambda t: t[1])
            if best_count >= 2 and (best_count / float(total_samples)) >= 0.4:
                cat_ids = [best_cat]
    if cat_ids:
        # Collect businesses whose 'category' field explicitly matches one of the
        # resolved category ids. We enforce strict matching (string equality or
        # membership when the business stores categories as a list).
        docs: List[firestore.DocumentSnapshot] = []
        ids: Set[str] = set()
        # Use 'in' query when number of ids is reasonable, otherwise loop.
        try:
            if 1 <= len(cat_ids) <= 10:
                for d in client.collection("businesses").where("category", "in", cat_ids).stream():
                    if d.id not in ids:
                        docs.append(d)
                        ids.add(d.id)
            else:
                for cid in cat_ids:
                    for d in client.collection("businesses").where("category", "==", cid).stream():
                        if d.id not in ids:
                            docs.append(d)
                            ids.add(d.id)
        except Exception:
            # Some Firestore deployments may not support 'in' or may error; fall
            # back to per-id queries.
            ids.clear()
            docs.clear()
            for cid in cat_ids:
                for d in client.collection("businesses").where("category", "==", cid).stream():
                    if d.id not in ids:
                        docs.append(d)
                        ids.add(d.id)

        # Defensive filter: ensure the document's category field indeed matches
        # one of the resolved cat_ids (string equality or membership in list).
        filtered_docs: List[firestore.DocumentSnapshot] = []
        for d in docs:
            data = d.to_dict() or {}
            cat_field = data.get("category")
            matches = False
            if isinstance(cat_field, (list, tuple, set)):
                for c in cat_field:
                    try:
                        if str(c) in cat_ids:
                            matches = True
                            break
                    except Exception:
                        continue
            else:
                try:
                    if str(cat_field) in cat_ids:
                        matches = True
                except Exception:
                    matches = False
            if matches:
                filtered_docs.append(d)

        # If we found category-matched businesses, compute distances and return
        # only these (no tag/name/geo fallbacks).
        if filtered_docs:
            results: List[Tuple[Business, float]] = []
            for d in filtered_docs:
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
                name = data.get("name") or d.id
                results.append((Business(id=d.id, name=name, type=business_type, latitude=float(lat), longitude=float(lng)), dist))
            results.sort(key=lambda x: x[1])
            return results[:limit]
    # If no category mapping, fall back to tag-based search
    tag = resolve_tag_from_categories(base_text) or base_text
    # Build prioritized variants: first use expansion_variants (if any), then
    # the resolved tag variants (full phrase/slug)
    variants: List[str] = []
    if expansion_variants:
        variants.extend(expansion_variants)
    # then the canonical tag variants for the original phrase
    for v in _tag_variants(tag):
        if v not in variants:
            variants.append(v)

    # Prefer exact normalized_tags matches for the full phrase or slug.
    # This avoids earlier tokenized queries matching unrelated businesses
    # (e.g., 'car' matching many automotive businesses).
    exact_tag_docs: List[firestore.DocumentSnapshot] = []
    exact_ids: Set[str] = set()
    for v in variants:
        for d in client.collection("businesses").where("normalized_tags", "array_contains", v).stream():
            if d.id not in exact_ids:
                exact_tag_docs.append(d)
                exact_ids.add(d.id)
    if exact_tag_docs:
        results: List[Tuple[Business, float]] = []
        for d in exact_tag_docs:
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
            name = data.get("name") or d.id
            results.append((Business(id=d.id, name=name, type=business_type, latitude=float(lat), longitude=float(lng)), dist))
        results.sort(key=lambda x: x[1])
        return results[:limit]
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
    if not docs and allow_geo_fallback:
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
    if not allow_geo_fallback:
        return []
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

