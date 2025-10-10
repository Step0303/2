from __future__ import annotations
from typing import Any, Dict, List, Optional

from google.cloud import firestore

from .config import settings
from .geo import geohash_prefixes, haversine_meters


db: Optional[firestore.Client] = None


def _client() -> firestore.Client:
    global db
    if db is None:
        db = firestore.Client(project=settings.project_id)
    return db


def _range_query(col_ref, prefix: str):
    # '~' is lexicographically after all base32 chars used in geohash
    return col_ref.where("geohash", ">=", prefix).where("geohash", "<", prefix + "~")


def search_nearby(
    lat: float,
    lng: float,
    radius_m: int,
    category_exact: Optional[str] = None,
    expand_to_max_if_empty: bool = True,
    max_radius_m: Optional[int] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Search businesses near a point within radius.
    - Exact category match first (case-insensitive in 'categories' array).
    - If none, optionally expand radius up to max or drop category filter.
    Returns sorted by distance ascending.
    """
    col = _client().collection(settings.firestore_collection)

    def run(radius: int, require_category: bool) -> List[Dict[str, Any]]:
        prefixes = geohash_prefixes(lat, lng, radius)
        snapshots = []
        for p in prefixes:
            q = _range_query(col, p)
            snapshots.extend(list(q.stream()))
        # De-duplicate by doc id
        seen = set()
        docs = []
        for s in snapshots:
            if s.id in seen:
                continue
            seen.add(s.id)
            d = s.to_dict() or {}
            loc = d.get("location") or d.get("coordinates") or {}
            # Support either GeoPoint in 'location' or dict in 'coordinates'
            if hasattr(loc, "latitude") and hasattr(loc, "longitude"):
                lat2, lng2 = float(loc.latitude), float(loc.longitude)
            else:
                lat2, lng2 = float(loc.get("lat", 0)), float(loc.get("lng", 0))
            if not lat2 and not lng2:
                continue
            dist = haversine_meters(lat, lng, lat2, lng2)
            if dist > radius:
                continue
            d_out = {
                "id": s.id,
                "name": d.get("name") or d.get("Name") or "",
                "categories": [c.lower() for c in (d.get("categories") or d.get("category") or [])],
                "address": d.get("address") or d.get("physical_address") or d.get("location_str") or "",
                "contact": d.get("contact") or d.get("details") or {},
                "rating": d.get("rating"),
                "price_level": d.get("price_level"),
                "distance_m": dist,
                "lat": lat2,
                "lng": lng2,
                "raw": d,
            }
            docs.append(d_out)
        if require_category and category_exact:
            cat = category_exact.lower()
            docs = [d for d in docs if cat in (d.get("categories") or [])]
        docs.sort(key=lambda x: x["distance_m"])
        return docs[:limit]

    # 1) Exact category within requested radius
    results = run(radius_m, require_category=bool(category_exact))

    # 2) If empty, optionally expand radius then drop category
    if not results and expand_to_max_if_empty:
        max_r = max_radius_m or settings.max_radius_m
        if radius_m < max_r:
            results = run(max_r, require_category=bool(category_exact))
        if not results:
            results = run(max_r, require_category=False)

    return results
