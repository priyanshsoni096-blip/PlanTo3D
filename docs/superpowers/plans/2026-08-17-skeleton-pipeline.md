# PlanTo3D Skeleton Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin, end-to-end version of the PlanTo3D pipeline (rasterize → segment → extract geometry → calibrate scale → crude 3D extrude) and validate it against the real Soni Residence ground floor page, before investing in floor alignment, the viewer, model fine-tuning, or the Blender/photoreal passes.

**Architecture:** Single-floor, single-pass pipeline. Segmentation uses inference against a pretrained CubiCasa5k-style checkpoint (no fine-tuning yet — that's a later plan, once the skeleton proves the rest of the pipeline works). Geometry extraction and scale calibration are pure, independently-testable functions operating on the mask and source image. Output is a crude `.glb` (walls only, no cut openings, no roof yet) plus a 2D overlay PNG for visual validation.

**Tech Stack:** Python 3.11+, OpenCV (`opencv-python`), PyTorch + `segmentation-models-pytorch` (U-Net+ResNet), pytesseract (OCR — requires system Tesseract install), trimesh, `pdf2image` (requires system poppler install), pytest.

**Spec:** [docs/superpowers/specs/2026-08-17-planto3d-design.md](../specs/2026-08-17-planto3d-design.md)

## Global Constraints

- Input floor plans are clean, digital-quality CAD-drafted drawings, not hand-drawn scans (spec Scope).
- Internal geometry representation is in feet; convert to meters only at the mesh-export stage; mesh is Y-up (spec Data flow & formats).
- Room labels come from OCR text, never from the segmentation model's class output (spec Component 3 — CubiCasa5K's taxonomy has no "Temple"/"Verandah"/"Dress-Toilet" categories).
- On any per-room extraction failure (OCR read fails, polygon doesn't close), log a warning and skip that room — never raise/crash the pipeline (spec Error handling).
- JSON floor representation shape is exactly:
  ```json
  {"walls": [{"start": [x, y], "end": [x, y], "thickness": t}],
   "rooms": [{"polygon": [[x, y], ...], "label": "BEDROOM"}],
   "openings": [{"wall_id": id, "position": p, "width": w, "type": "door|window"}]}
  ```

---

### Task 1: Project scaffolding and PDF ingestion

**Files:**
- Create: `pyproject.toml`
- Create: `planto3d/__init__.py`
- Create: `planto3d/ingest.py`
- Create: `tests/test_ingest.py`
- Create: `data/soni_residence/` (directory, empty — real PDF is user-supplied, not committed)
- Create: `.gitignore`

> **STATUS: COMPLETE** — implemented with a revised interface. The planned
> fixed `crop_box` was replaced by consensus border detection across the page
> set, because a fixed box is brittle and, more importantly, per-page
> detection misread the ground floor: a plot boundary running 7px inside the
> drawing's bottom border was picked up as the border, clipping the landscape
> and parking. Since all sheets share one template, intersecting each page's
> candidate border lines isolates the frame and drops plan features, and
> yields one box for every page — which is what keeps floors on a shared
> origin for the alignment stage. Steps 1–6 below are superseded by the
> interface block that follows; kept for provenance.

**Interfaces (as built):**
- Produces: `ingest.rasterize_pdf(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[Path]` — renders each PDF page to `page-{n}.png`, returns paths in order.
- Produces: `ingest.detect_drawing_region(images: list[np.ndarray]) -> tuple[int, int, int, int]` — one (left, top, right, bottom) box shared by all pages, excluding sheet border, option/area band, and title block. Raises `ValueError` on an empty set or a blank sheet.
- Produces: `ingest.crop_pages(image_paths: list[Path]) -> list[Path]` — crops every page to the shared region, writing `<stem>_cropped.png`.
- Verified on the real sheet set: all three floors crop to an identical 855×587 from box `x 159-1013, y 41-627`.

- [ ] **Step 1: Set up project files**

`pyproject.toml`:
```toml
[project]
name = "planto3d"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opencv-python>=4.9",
    "pdf2image>=1.17",
    "pytesseract>=0.3.10",
    "trimesh>=4.0",
    "numpy>=1.26",
    "torch>=2.2",
    "segmentation-models-pytorch>=0.3",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
data/soni_residence/*.pdf
data/soni_residence/*.png
data/cubicasa5k/
*.glb
```

`planto3d/__init__.py`: empty file.

- [ ] **Step 2: Write the failing test for rasterization and cropping**

```python
# tests/test_ingest.py
from pathlib import Path
import numpy as np
from PIL import Image
from planto3d.ingest import crop_title_block


def test_crop_title_block_removes_specified_region(tmp_path):
    img = Image.fromarray(np.full((100, 200, 3), 255, dtype=np.uint8))
    img_path = tmp_path / "page.png"
    img.save(img_path)

    cropped_path = crop_title_block(img_path, crop_box=(0, 80, 200, 100))

    cropped = Image.open(cropped_path)
    assert cropped.size == (200, 80)
    assert cropped_path.name == "page_cropped.png"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'planto3d.ingest'`

- [ ] **Step 4: Implement ingest.py**

```python
# planto3d/ingest.py
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image


def rasterize_pdf(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    output_paths = []
    for i, page in enumerate(pages, start=1):
        out_path = output_dir / f"page_{i}.png"
        page.save(out_path)
        output_paths.append(out_path)
    return output_paths


def crop_title_block(image_path: Path, crop_box: tuple[int, int, int, int]) -> Path:
    image = Image.open(image_path)
    cropped = image.crop(crop_box)
    out_path = image_path.with_name(f"{image_path.stem}_cropped.png")
    cropped.save(out_path)
    return out_path
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore planto3d/__init__.py planto3d/ingest.py tests/test_ingest.py
git commit -m "feat: add project scaffolding and PDF ingestion"
```

---

### Task 2: Shared geometry data types and JSON serialization

**Files:**
- Create: `planto3d/geometry_types.py`
- Create: `tests/test_geometry_types.py`

**Interfaces:**
- Consumes: nothing (foundational types).
- Produces: `Wall`, `Room`, `Opening`, `FloorPlan` dataclasses, each with `.to_dict()` and `FloorPlan.from_dict(d) -> FloorPlan`, matching the JSON shape in Global Constraints exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_types.py
from planto3d.geometry_types import Wall, Room, Opening, FloorPlan


def test_floorplan_roundtrips_through_dict():
    plan = FloorPlan(
        walls=[Wall(start=(0.0, 0.0), end=(10.0, 0.0), thickness=0.5)],
        rooms=[Room(polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], label="BEDROOM")],
        openings=[Opening(wall_id=0, position=5.0, width=3.0, type="door")],
    )

    d = plan.to_dict()
    restored = FloorPlan.from_dict(d)

    assert d["walls"][0]["start"] == [0.0, 0.0]
    assert d["rooms"][0]["label"] == "BEDROOM"
    assert d["openings"][0]["type"] == "door"
    assert restored == plan
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_geometry_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'planto3d.geometry_types'`

- [ ] **Step 3: Implement geometry_types.py**

```python
# planto3d/geometry_types.py
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Wall:
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float

    def to_dict(self) -> dict:
        return {"start": list(self.start), "end": list(self.end), "thickness": self.thickness}

    @staticmethod
    def from_dict(d: dict) -> "Wall":
        return Wall(start=tuple(d["start"]), end=tuple(d["end"]), thickness=d["thickness"])


@dataclass
class Room:
    polygon: list[tuple[float, float]]
    label: str

    def to_dict(self) -> dict:
        return {"polygon": [list(p) for p in self.polygon], "label": self.label}

    @staticmethod
    def from_dict(d: dict) -> "Room":
        return Room(polygon=[tuple(p) for p in d["polygon"]], label=d["label"])


@dataclass
class Opening:
    wall_id: int
    position: float
    width: float
    type: Literal["door", "window"]

    def to_dict(self) -> dict:
        return {"wall_id": self.wall_id, "position": self.position, "width": self.width, "type": self.type}

    @staticmethod
    def from_dict(d: dict) -> "Opening":
        return Opening(wall_id=d["wall_id"], position=d["position"], width=d["width"], type=d["type"])


@dataclass
class FloorPlan:
    walls: list[Wall] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "walls": [w.to_dict() for w in self.walls],
            "rooms": [r.to_dict() for r in self.rooms],
            "openings": [o.to_dict() for o in self.openings],
        }

    @staticmethod
    def from_dict(d: dict) -> "FloorPlan":
        return FloorPlan(
            walls=[Wall.from_dict(w) for w in d["walls"]],
            rooms=[Room.from_dict(r) for r in d["rooms"]],
            openings=[Opening.from_dict(o) for o in d["openings"]],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_geometry_types.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add planto3d/geometry_types.py tests/test_geometry_types.py
git commit -m "feat: add shared geometry data types with JSON roundtrip"
```

---

### Task 3: Geometry extraction from a segmentation mask

**Files:**
- Create: `planto3d/extract.py`
- Create: `tests/test_extract.py`

**Interfaces:**
- Consumes: `Wall`, `Room`, `Opening`, `FloorPlan` from Task 2.
- Produces: `extract.extract_walls(mask: np.ndarray, wall_class: int) -> list[Wall]` and `extract.extract_rooms(mask: np.ndarray, room_class: int) -> list[Room]`, both operating in pixel coordinates (calibration to feet happens in Task 4).

- [ ] **Step 1: Write the failing test against a synthetic mask**

```python
# tests/test_extract.py
import numpy as np
from planto3d.extract import extract_walls, extract_rooms

WALL, ROOM, BG = 1, 2, 0


def test_extract_walls_finds_single_rectangular_wall_loop():
    mask = np.full((50, 50), BG, dtype=np.uint8)
    mask[10:40, 10:12] = WALL   # left wall
    mask[10:40, 38:40] = WALL   # right wall
    mask[10:12, 10:40] = WALL   # top wall
    mask[38:40, 10:40] = WALL   # bottom wall

    walls = extract_walls(mask, wall_class=WALL)

    assert len(walls) >= 4
    for wall in walls:
        assert wall.thickness > 0


def test_extract_rooms_finds_single_closed_room():
    mask = np.full((50, 50), BG, dtype=np.uint8)
    mask[12:38, 12:38] = ROOM

    rooms = extract_rooms(mask, room_class=ROOM)

    assert len(rooms) == 1
    assert len(rooms[0].polygon) >= 4
    assert rooms[0].label == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'planto3d.extract'`

- [ ] **Step 3: Implement extract.py**

```python
# planto3d/extract.py
import cv2
import numpy as np
from planto3d.geometry_types import Wall, Room


def extract_walls(mask: np.ndarray, wall_class: int) -> list[Wall]:
    binary = (mask == wall_class).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    walls: list[Wall] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        length, thickness = max(w, h), min(w, h)
        if thickness <= 0:
            continue
        angle_rad = np.radians(angle if w >= h else angle + 90)
        dx = (length / 2) * np.cos(angle_rad)
        dy = (length / 2) * np.sin(angle_rad)
        start = (cx - dx, cy - dy)
        end = (cx + dx, cy + dy)
        walls.append(Wall(start=start, end=end, thickness=thickness))
    return walls


def extract_rooms(mask: np.ndarray, room_class: int) -> list[Room]:
    binary = (mask == room_class).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rooms: list[Room] = []
    for contour in contours:
        if cv2.contourArea(contour) < 16:
            continue
        polygon = [(float(p[0][0]), float(p[0][1])) for p in contour]
        rooms.append(Room(polygon=polygon, label=""))
    return rooms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add planto3d/extract.py tests/test_extract.py
git commit -m "feat: extract wall and room geometry from segmentation mask"
```

---

### Task 4: Scale calibration and room labeling via OCR

**Files:**
- Create: `planto3d/calibrate.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: nothing new (operates on raw OCR text strings — kept decoupled from pytesseract's image API so the parsing logic is unit-testable without a real image or Tesseract installed).
- Produces: `calibrate.parse_dimension_text(text: str) -> tuple[float, float] | None` (feet, feet); `calibrate.compute_scale_factor(pixel_length: float, feet_length: float) -> float` (pixels per foot); `calibrate.read_room_labels_and_dimensions(image: np.ndarray) -> list[dict]` — each dict has `text`, `bbox` (x, y, w, h in pixels), calls pytesseract internally.

- [ ] **Step 1: Write the failing test for dimension parsing and scale computation**

```python
# tests/test_calibrate.py
import numpy as np
from planto3d.calibrate import parse_dimension_text, compute_scale_factor, read_room_labels_and_dimensions


def test_parse_dimension_text_reads_feet_and_inches():
    assert parse_dimension_text('15\'0"X18\'0"') == (15.0, 18.0)
    assert parse_dimension_text("BEDROOM 15'0\"X18'0\"") == (15.0, 18.0)
    assert parse_dimension_text("7'6\"X27'0\"") == (7.5, 27.0)


def test_parse_dimension_text_returns_none_for_non_dimension_text():
    assert parse_dimension_text("KITCHEN") is None


def test_compute_scale_factor_returns_pixels_per_foot():
    assert compute_scale_factor(pixel_length=300.0, feet_length=15.0) == 20.0


def test_read_room_labels_and_dimensions_returns_list_of_dicts(monkeypatch):
    import planto3d.calibrate as calibrate_module

    def fake_image_to_data(image, output_type):
        return {
            "text": ["", "BEDROOM", "15'0\"X18'0\""],
            "left": [0, 100, 100],
            "top": [0, 200, 220],
            "width": [0, 80, 90],
            "height": [0, 15, 15],
            "conf": ["-1", "95", "93"],
        }

    monkeypatch.setattr(calibrate_module.pytesseract, "image_to_data", fake_image_to_data)
    monkeypatch.setattr(calibrate_module.pytesseract, "Output", type("Output", (), {"DICT": "dict"}))

    results = read_room_labels_and_dimensions(np.zeros((300, 300, 3), dtype=np.uint8))

    assert {"text": "BEDROOM", "bbox": (100, 200, 80, 15)} in results
    assert {"text": "15'0\"X18'0\"", "bbox": (100, 220, 90, 15)} in results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'planto3d.calibrate'`

- [ ] **Step 3: Implement calibrate.py**

```python
# planto3d/calibrate.py
import re
import numpy as np
import pytesseract

DIMENSION_PATTERN = re.compile(r"(\d+)'(\d+)\"?\s*[Xx]\s*(\d+)'(\d+)\"?")


def parse_dimension_text(text: str) -> tuple[float, float] | None:
    match = DIMENSION_PATTERN.search(text)
    if not match:
        return None
    ft1, in1, ft2, in2 = (int(g) for g in match.groups())
    return (ft1 + in1 / 12.0, ft2 + in2 / 12.0)


def compute_scale_factor(pixel_length: float, feet_length: float) -> float:
    return pixel_length / feet_length


def read_room_labels_and_dimensions(image: np.ndarray) -> list[dict]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    results = []
    for i, text in enumerate(data["text"]):
        stripped = text.strip()
        if not stripped or int(data["conf"][i]) < 0:
            continue
        bbox = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        results.append({"text": stripped, "bbox": bbox})
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibrate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add planto3d/calibrate.py tests/test_calibrate.py
git commit -m "feat: add OCR-based scale calibration and room label reading"
```

---

### Task 5: Wire room labels onto extracted room polygons

**Files:**
- Create: `planto3d/label_rooms.py`
- Create: `tests/test_label_rooms.py`

**Interfaces:**
- Consumes: `Room` from Task 2; OCR result dicts (`{"text", "bbox"}`) from Task 4.
- Produces: `label_rooms.assign_labels(rooms: list[Room], ocr_results: list[dict]) -> list[Room]` — for each room, finds the OCR text whose bbox center falls inside the room polygon and is *not* a dimension string (per Global Constraints: labels come from OCR, not the model); returns new `Room` objects with `.label` set, or logs a warning and leaves `label=""` if none found (per spec Error handling: skip/flag rather than crash).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_label_rooms.py
import logging
from planto3d.geometry_types import Room
from planto3d.label_rooms import assign_labels


def test_assign_labels_matches_text_inside_room_polygon():
    room = Room(polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], label="")
    ocr_results = [
        {"text": "BEDROOM", "bbox": (40, 40, 50, 15)},
        {"text": "15'0\"X18'0\"", "bbox": (40, 60, 50, 15)},
    ]

    labeled = assign_labels([room], ocr_results)

    assert labeled[0].label == "BEDROOM"


def test_assign_labels_warns_and_leaves_blank_when_no_match(caplog):
    room = Room(polygon=[(0, 0), (10, 0), (10, 10), (0, 10)], label="")

    with caplog.at_level(logging.WARNING):
        labeled = assign_labels([room], ocr_results=[])

    assert labeled[0].label == ""
    assert "no label found" in caplog.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_label_rooms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'planto3d.label_rooms'`

- [ ] **Step 3: Implement label_rooms.py**

```python
# planto3d/label_rooms.py
import logging
from dataclasses import replace
from planto3d.geometry_types import Room
from planto3d.calibrate import parse_dimension_text

logger = logging.getLogger(__name__)


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def assign_labels(rooms: list[Room], ocr_results: list[dict]) -> list[Room]:
    labeled_rooms = []
    for room in rooms:
        match_text = ""
        for result in ocr_results:
            if parse_dimension_text(result["text"]) is not None:
                continue
            x, y, w, h = result["bbox"]
            center = (x + w / 2, y + h / 2)
            if _point_in_polygon(center, room.polygon):
                match_text = result["text"]
                break
        if not match_text:
            logger.warning("no label found for room with polygon %s", room.polygon)
        labeled_rooms.append(replace(room, label=match_text))
    return labeled_rooms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_label_rooms.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add planto3d/label_rooms.py tests/test_label_rooms.py
git commit -m "feat: assign OCR-derived labels to extracted room polygons"
```

---

### Task 6: Segmentation inference wrapper (pretrained checkpoint)

**Files:**
- Create: `planto3d/segment.py`
- Create: `tests/test_segment.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `segment.Segmenter` class with `Segmenter(checkpoint_path: Path)` constructor and `.predict(image: np.ndarray) -> np.ndarray` returning a per-pixel class-index mask (`int64`, same H×W as input). Class indices: `0=background, 1=wall, 2=room, 3=door, 4=window` (module-level constants `BACKGROUND, WALL, ROOM, DOOR, WINDOW`).

- [ ] **Step 1: Write the failing test using a stub model (no real checkpoint needed)**

```python
# tests/test_segment.py
import numpy as np
import torch
from planto3d.segment import Segmenter, WALL, BACKGROUND


class _StubModel(torch.nn.Module):
    def forward(self, x):
        batch, _, h, w = x.shape
        logits = torch.zeros((batch, 5, h, w))
        logits[:, WALL, :, : w // 2] = 10.0
        logits[:, BACKGROUND, :, w // 2:] = 10.0
        return logits


def test_segmenter_predict_returns_class_index_mask_matching_input_size(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "fake.pt"
    checkpoint_path.write_bytes(b"unused")

    monkeypatch.setattr("planto3d.segment._load_model", lambda path: _StubModel())

    segmenter = Segmenter(checkpoint_path)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = segmenter.predict(image)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.int64
    assert (mask[:, :32] == WALL).all()
    assert (mask[:, 32:] == BACKGROUND).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_segment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'planto3d.segment'`

- [ ] **Step 3: Implement segment.py**

```python
# planto3d/segment.py
from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp

BACKGROUND, WALL, ROOM, DOOR, WINDOW = 0, 1, 2, 3, 4
NUM_CLASSES = 5


def _load_model(checkpoint_path: Path) -> torch.nn.Module:
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


class Segmenter:
    def __init__(self, checkpoint_path: Path):
        self.model = _load_model(checkpoint_path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = self.model(tensor)
        mask = torch.argmax(logits, dim=1).squeeze(0).numpy().astype(np.int64)
        return mask
```

**Note for the engineer:** this task tests the wrapper's *interface* with a stub model, not real segmentation accuracy — no pretrained checkpoint is downloaded or evaluated here. Sourcing/fine-tuning an actual CubiCasa5k-style checkpoint is out of scope for this plan (see spec Component 2 and "Known risks"); track it as a follow-on plan once this skeleton validates end-to-end.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_segment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add planto3d/segment.py tests/test_segment.py
git commit -m "feat: add segmentation inference wrapper with stub-testable interface"
```

---

### Task 7: Crude 3D extrusion to `.glb`

**Files:**
- Create: `planto3d/extrude.py`
- Create: `tests/test_extrude.py`

**Interfaces:**
- Consumes: `Wall` from Task 2.
- Produces: `extrude.walls_to_mesh(walls: list[Wall], wall_height_ft: float = 9.0, scale_factor: float = 1.0) -> trimesh.Trimesh` — converts each wall (in pixel coordinates) to a box mesh, scaled from pixels to feet via `scale_factor` (pixels per foot from Task 4) then feet to meters (`* 0.3048`), combined into one mesh, Y-up. `extrude.export_glb(mesh: trimesh.Trimesh, output_path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extrude.py
from pathlib import Path
import numpy as np
from planto3d.geometry_types import Wall
from planto3d.extrude import walls_to_mesh, export_glb


def test_walls_to_mesh_produces_nonempty_mesh_with_expected_height():
    wall = Wall(start=(0.0, 0.0), end=(200.0, 0.0), thickness=10.0)  # pixels

    mesh = walls_to_mesh([wall], wall_height_ft=9.0, scale_factor=20.0)  # 20 px/ft

    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    expected_height_m = 9.0 * 0.3048
    assert np.isclose(mesh.bounds[1][1] - mesh.bounds[0][1], expected_height_m, atol=0.01)


def test_export_glb_writes_file(tmp_path):
    wall = Wall(start=(0.0, 0.0), end=(200.0, 0.0), thickness=10.0)
    mesh = walls_to_mesh([wall], wall_height_ft=9.0, scale_factor=20.0)
    out_path = tmp_path / "test.glb"

    export_glb(mesh, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extrude.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'planto3d.extrude'`

- [ ] **Step 3: Implement extrude.py**

```python
# planto3d/extrude.py
from pathlib import Path
import numpy as np
import trimesh
from planto3d.geometry_types import Wall

FEET_TO_METERS = 0.3048


def walls_to_mesh(walls: list[Wall], wall_height_ft: float, scale_factor: float) -> trimesh.Trimesh:
    height_m = wall_height_ft * FEET_TO_METERS
    box_meshes = []

    for wall in walls:
        start = np.array(wall.start) / scale_factor * FEET_TO_METERS
        end = np.array(wall.end) / scale_factor * FEET_TO_METERS
        thickness_m = wall.thickness / scale_factor * FEET_TO_METERS

        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            continue

        box = trimesh.creation.box(extents=[length, thickness_m, height_m])

        angle = np.arctan2(direction[1], direction[0])
        rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        midpoint_xz = (start + end) / 2
        translation = trimesh.transformations.translation_matrix(
            [midpoint_xz[0], midpoint_xz[1], height_m / 2]
        )
        box.apply_transform(rotation)
        box.apply_transform(translation)
        box_meshes.append(box)

    combined = trimesh.util.concatenate(box_meshes)
    # Convert from XZ-ground/Z-up construction to Y-up for the mesh output.
    combined.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return combined


def export_glb(mesh: trimesh.Trimesh, output_path: Path) -> None:
    mesh.export(str(output_path), file_type="glb")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extrude.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add planto3d/extrude.py tests/test_extrude.py
git commit -m "feat: add crude wall extrusion to glb mesh export"
```

---

### Task 8: 2D overlay visualization and end-to-end skeleton script

**Files:**
- Create: `planto3d/overlay.py`
- Create: `tests/test_overlay.py`
- Create: `scripts/run_skeleton.py`

**Interfaces:**
- Consumes: `FloorPlan` from Task 2.
- Produces: `overlay.draw_overlay(original_image: np.ndarray, floor_plan: FloorPlan, scale_factor: float) -> np.ndarray` — draws walls (red lines) and room polygons (green outline) from `floor_plan` (in feet) onto a copy of `original_image` (pixel coordinates), converting feet back to pixels via `scale_factor`. `scripts/run_skeleton.py` is a manually-run orchestration script (not unit tested — it's an integration entrypoint), documented below.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_overlay.py
import numpy as np
from planto3d.geometry_types import FloorPlan, Wall, Room
from planto3d.overlay import draw_overlay


def test_draw_overlay_returns_image_same_size_as_input_and_modifies_pixels():
    original = np.zeros((100, 100, 3), dtype=np.uint8)
    plan = FloorPlan(
        walls=[Wall(start=(1.0, 1.0), end=(4.0, 1.0), thickness=0.5)],
        rooms=[Room(polygon=[(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)], label="BEDROOM")],
        openings=[],
    )

    result = draw_overlay(original, plan, scale_factor=10.0)

    assert result.shape == original.shape
    assert not np.array_equal(result, original)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'planto3d.overlay'`

- [ ] **Step 3: Implement overlay.py**

```python
# planto3d/overlay.py
import cv2
import numpy as np
from planto3d.geometry_types import FloorPlan

RED = (0, 0, 255)
GREEN = (0, 255, 0)


def draw_overlay(original_image: np.ndarray, floor_plan: FloorPlan, scale_factor: float) -> np.ndarray:
    result = original_image.copy()

    for wall in floor_plan.walls:
        start_px = tuple(int(c * scale_factor) for c in wall.start)
        end_px = tuple(int(c * scale_factor) for c in wall.end)
        cv2.line(result, start_px, end_px, RED, thickness=2)

    for room in floor_plan.rooms:
        points_px = np.array([[int(x * scale_factor), int(y * scale_factor)] for x, y in room.polygon], dtype=np.int32)
        cv2.polylines(result, [points_px], isClosed=True, color=GREEN, thickness=2)
        if room.label and len(points_px) > 0:
            cv2.putText(result, room.label, tuple(points_px[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_overlay.py -v`
Expected: PASS

- [ ] **Step 5: Write the end-to-end orchestration script**

```python
# scripts/run_skeleton.py
"""
Manual integration entrypoint for the thin skeleton pipeline.
Not covered by pytest — requires a real segmentation checkpoint,
a real floor plan PDF, and a system Tesseract install.

Usage:
    python scripts/run_skeleton.py <pdf_path> <checkpoint_path> <output_dir>
"""
import sys
from pathlib import Path
import cv2
import numpy as np

from planto3d.ingest import rasterize_pdf
from planto3d.segment import Segmenter, WALL, ROOM
from planto3d.extract import extract_walls, extract_rooms
from planto3d.calibrate import read_room_labels_and_dimensions, parse_dimension_text, compute_scale_factor
from planto3d.label_rooms import assign_labels
from planto3d.geometry_types import FloorPlan
from planto3d.extrude import walls_to_mesh, export_glb
from planto3d.overlay import draw_overlay


def main(pdf_path: str, checkpoint_path: str, output_dir: str) -> None:
    out_dir = Path(output_dir)
    pages = rasterize_pdf(Path(pdf_path), out_dir)
    page_path = pages[0]  # ground floor page for the skeleton run

    image = cv2.imread(str(page_path))
    segmenter = Segmenter(Path(checkpoint_path))
    mask = segmenter.predict(image)

    walls = extract_walls(mask, wall_class=WALL)
    rooms = extract_rooms(mask, room_class=ROOM)

    ocr_results = read_room_labels_and_dimensions(image)
    rooms = assign_labels(rooms, ocr_results)

    dimension_matches = [r for r in ocr_results if parse_dimension_text(r["text"]) is not None]
    if not dimension_matches or not walls:
        raise RuntimeError("no dimension text or walls found; cannot calibrate scale")
    feet_w, _ = parse_dimension_text(dimension_matches[0]["text"])
    pixel_length = np.linalg.norm(np.array(walls[0].end) - np.array(walls[0].start))
    scale_factor = compute_scale_factor(pixel_length, feet_w)

    plan = FloorPlan(walls=walls, rooms=rooms, openings=[])

    overlay_image = draw_overlay(image, plan, scale_factor)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay_image)

    mesh = walls_to_mesh(walls, wall_height_ft=9.0, scale_factor=scale_factor)
    export_glb(mesh, out_dir / "skeleton.glb")

    print(f"Wrote {out_dir / 'overlay.png'} and {out_dir / 'skeleton.glb'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
```

- [ ] **Step 6: Commit**

```bash
git add planto3d/overlay.py tests/test_overlay.py scripts/run_skeleton.py
git commit -m "feat: add 2D overlay visualization and end-to-end skeleton script"
```

---

### Task 9: Run the full test suite and validate against the real floor plan

**Files:**
- No new files — verification task.

- [ ] **Step 1: Run the full automated test suite**

Run: `pytest -v`
Expected: All tests from Tasks 1–8 PASS.

- [ ] **Step 2: Manually run the skeleton script against the real Soni Residence ground floor page**

This step requires a real segmentation checkpoint (not yet sourced — see Task 6 note) and system installs of Tesseract and poppler. Document the exact command to run once those are available:

```bash
python scripts/run_skeleton.py "data/soni_residence/DOC-20260817-WA0027.PDF" <checkpoint_path> data/soni_residence/output
```

- [ ] **Step 3: Visually inspect `overlay.png` against the original ground floor page**

Confirm wall lines and room outlines roughly track the real walls/rooms (per spec's visual-overlay success criterion). Note any systematic misalignment for the follow-on plan.

- [ ] **Step 4: Open `skeleton.glb` in a glTF viewer (e.g. https://gltf-viewer.donmccurdy.com/) to confirm a crude 3D wall layout appears**

- [ ] **Step 5: Commit any notes on observed accuracy**

```bash
git add docs/superpowers/plans/2026-08-17-skeleton-pipeline.md
git commit -m "docs: record skeleton pipeline validation notes" --allow-empty
```
