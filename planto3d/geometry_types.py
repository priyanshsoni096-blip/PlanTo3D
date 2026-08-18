"""Structured floor plan geometry passed between pipeline stages.

Coordinates are in feet once calibration has run, and in pixels before it.
Whichever unit is in play, one plan is internally consistent; conversion to
metres happens at the mesh stage.

``to_dict``/``from_dict`` produce plain Python values so a plan can be written
to JSON between stages -- coordinates frequently arrive as numpy scalars from
OpenCV, which are not JSON-serializable on their own.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, get_args

Point = tuple[float, float]
OpeningType = Literal["door", "window"]
OPENING_TYPES = get_args(OpeningType)

# Fewest vertices that can enclose an area; anything less cannot be a room.
MIN_POLYGON_VERTICES = 3


@dataclass
class Wall:
    """A straight wall segment, measured along its centreline."""

    start: Point
    end: Point
    thickness: float

    def length(self) -> float:
        return math.dist(self.start, self.end)

    def to_dict(self) -> dict:
        return {
            "start": [float(self.start[0]), float(self.start[1])],
            "end": [float(self.end[0]), float(self.end[1])],
            "thickness": float(self.thickness),
        }

    @staticmethod
    def from_dict(d: dict) -> "Wall":
        return Wall(
            start=(float(d["start"][0]), float(d["start"][1])),
            end=(float(d["end"][0]), float(d["end"][1])),
            thickness=float(d["thickness"]),
        )


@dataclass
class Room:
    """A closed region. ``label`` is empty until OCR supplies a name."""

    polygon: list[Point]
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.polygon) < MIN_POLYGON_VERTICES:
            raise ValueError(
                f"polygon needs at least {MIN_POLYGON_VERTICES} vertices to enclose "
                f"a room, got {len(self.polygon)}"
            )

    def contains(self, point: Point) -> bool:
        """Whether a point lies inside the polygon (ray casting)."""
        x, y = point
        inside = False
        count = len(self.polygon)
        for i in range(count):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % count]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
        return inside

    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds as (left, top, right, bottom)."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def to_dict(self) -> dict:
        return {
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "label": self.label,
        }

    @staticmethod
    def from_dict(d: dict) -> "Room":
        return Room(
            polygon=[(float(x), float(y)) for x, y in d["polygon"]],
            label=d["label"],
        )


@dataclass
class Opening:
    """A door or window interrupting a wall.

    ``position`` is the distance from that wall's ``start`` to the opening's
    centre; ``width`` is measured along the wall.
    """

    wall_id: int
    position: float
    width: float
    type: OpeningType

    def __post_init__(self) -> None:
        if self.type not in OPENING_TYPES:
            raise ValueError(f"type must be one of {OPENING_TYPES}, got {self.type!r}")

    def to_dict(self) -> dict:
        return {
            "wall_id": int(self.wall_id),
            "position": float(self.position),
            "width": float(self.width),
            "type": self.type,
        }

    @staticmethod
    def from_dict(d: dict) -> "Opening":
        return Opening(
            wall_id=int(d["wall_id"]),
            position=float(d["position"]),
            width=float(d["width"]),
            type=d["type"],
        )


@dataclass
class FloorPlan:
    """Everything extracted from one floor.

    ``footprint`` outlines the storey's built extent and is what floor slabs
    and the roof are generated from; walls alone leave a building open above
    and below.
    """

    walls: list[Wall] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    footprint: list[Point] = field(default_factory=list)
    planting: list[list[Point]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "walls": [w.to_dict() for w in self.walls],
            "rooms": [r.to_dict() for r in self.rooms],
            "openings": [o.to_dict() for o in self.openings],
            "footprint": [[float(x), float(y)] for x, y in self.footprint],
            "planting": [
                [[float(x), float(y)] for x, y in region] for region in self.planting
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "FloorPlan":
        return FloorPlan(
            walls=[Wall.from_dict(w) for w in d["walls"]],
            rooms=[Room.from_dict(r) for r in d["rooms"]],
            openings=[Opening.from_dict(o) for o in d["openings"]],
            footprint=[(float(x), float(y)) for x, y in d.get("footprint", [])],
            planting=[
                [(float(x), float(y)) for x, y in region]
                for region in d.get("planting", [])
            ],
        )
