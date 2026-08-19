"""Segmentation class indices shared by the model and the geometry stages.

Kept in their own module so geometry code can name classes without importing
the segmentation wrapper, which pulls in torch.

Most floor plans carry no room names. Of 21 CubiCasa plans reconstructed in
a batch run, OCR read a room name on three; the rest print nothing but a
disclaimer and a watermark, and their function is legible only from fixtures
-- a toilet, a hob, a sauna bench. Since floor finishes, planting, railings
and wet areas all hang off knowing what a room is for, a text-only route to
room function leaves the great majority of plans bare.

So the model is asked for the room type directly. CubiCasa annotates spaces
with a type it collapses into these classes, which are chosen for what they
change downstream rather than for architectural tidiness: a bath and a
kitchen both want a tiled floor and are worth separating from a bedroom,
while a study and a library want the same floor as each other and are not.

Indices 0-4 are unchanged from the five-class scheme, so a checkpoint trained
before the room types existed still loads and still predicts sensibly -- it
simply never emits anything above WINDOW, and every interior it finds arrives
as the generic ROOM.
"""

BACKGROUND = 0
WALL = 1
ROOM = 2
DOOR = 3
WINDOW = 4

# Room types. A plan is readable without these; it is only plain.
BEDROOM = 5
KITCHEN = 6
BATH = 7
STORAGE = 8
CIRCULATION = 9
OUTDOOR = 10

NUM_CLASSES = 11

CLASS_NAMES = {
    BACKGROUND: "background",
    WALL: "wall",
    ROOM: "room",
    DOOR: "door",
    WINDOW: "window",
    BEDROOM: "bedroom",
    KITCHEN: "kitchen",
    BATH: "bath",
    STORAGE: "storage",
    CIRCULATION: "circulation",
    OUTDOOR: "outdoor",
}

# Every class that encloses floor area. Geometry asks this rather than
# comparing against ROOM, so a plan segmented by an older five-class
# checkpoint and one segmented by a newer model both work.
ROOM_CLASSES = frozenset(
    {ROOM, BEDROOM, KITCHEN, BATH, STORAGE, CIRCULATION, OUTDOOR}
)

# Room types whose floors get wet, and are therefore tiled.
WET_CLASSES = frozenset({KITCHEN, BATH})

# What a predicted room type means to the feature system, which until now was
# fed only by OCR. The names match the categories in ``features`` so a
# predicted type and a read label arrive downstream in the same form.
CLASS_FEATURES = {
    KITCHEN: "wet",
    BATH: "wet",
    OUTDOOR: "open",
}
