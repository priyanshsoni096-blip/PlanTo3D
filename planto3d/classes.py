"""Segmentation class indices shared by the model and the geometry stages.

Kept in their own module so geometry code can name classes without importing
the segmentation wrapper, which pulls in torch.

CubiCasa5K annotates 80+ categories; those are collapsed to these five when
building training masks.
"""

BACKGROUND = 0
WALL = 1
ROOM = 2
DOOR = 3
WINDOW = 4

NUM_CLASSES = 5

CLASS_NAMES = {
    BACKGROUND: "background",
    WALL: "wall",
    ROOM: "room",
    DOOR: "door",
    WINDOW: "window",
}
