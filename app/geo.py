import math
from typing import List, Tuple

import geohash2 as geohash


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def precision_for_radius(radius_m: int) -> int:
    """Choose geohash precision based on search radius.
    Smaller radius -> higher precision. These are approximate.
    """
    if radius_m <= 500:  # ~150m cells
        return 7
    if radius_m <= 5000:  # ~1.2km cells
        return 6
    return 5  # ~4.9km cells


def geohash_prefixes(lat: float, lng: float, radius_m: int) -> List[str]:
    prec = precision_for_radius(radius_m)
    center = geohash.encode(lat, lng, precision=prec)
    neigh = geohash.neighbors(center)
    return [center] + neigh
