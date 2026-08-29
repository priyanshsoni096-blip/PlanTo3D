"""PlanTo3D web interface: upload a floor plan PDF, get a 3D model.

Run with:  python app.py
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from planto3d.corrections import CATEGORY_LABELS, NO_CHANGE, apply_room_corrections
from planto3d.extrude import DEFAULT_WALL_HEIGHT_FT
from planto3d.pipeline import PipelineResult, build, draw_overlay, extract
from planto3d.preview import render_views
from planto3d.segment import load_segmenter
from planto3d.design import (
    CREATIVITY,
    LANDSCAPING,
    STYLES,
    TIMES,
    TONES,
    Design,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TITLE = "PlanTo3D"
DESCRIPTION = (
    "Upload a 2D architectural floor plan — PDF, PNG or JPEG — and get back a "
    "3D model of the building. Walls, rooms, doors and windows are read "
    "straight off the drawing. Where the plan prints room dimensions the model "
    "is measured in real feet; where it does not, the scale is inferred from "
    "standard door widths and reported as such."
)

# A trained checkpoint is used when one is present; otherwise the classical
# baseline, which only reads cleanly drafted CAD sheets.
MODEL_DIR = Path(__file__).parent / "models"




def _checkpoint() -> Path | None:
    override = os.environ.get("PLANTO3D_CHECKPOINT")
    if override:
        return Path(override)
    return next(iter(sorted(MODEL_DIR.glob("*.pt"))), None)


CHECKPOINT = _checkpoint()
SEGMENTER = load_segmenter(CHECKPOINT)
SEGMENTER_NAME = (
    f"trained model (`{CHECKPOINT.name}`)" if CHECKPOINT else "classical baseline"
)


def detect(uploads, workdir: Path) -> tuple[list, list, PipelineResult]:
    """Extract geometry and labels, and show what was found for review.

    This is everything through labeling -- what ``planto3d.pipeline.
    extract()`` calls the pause point -- with nothing built yet, so a
    human can fix up ``result.floors[i].plan.rooms[j].label`` before
    ``convert()`` turns it into geometry.
    """
    if not uploads:
        raise gr.Error("Upload a floor plan first — a PDF, PNG or JPEG.")

    if isinstance(uploads, (str, Path)):
        uploads = [uploads]

    # Several images are one storey each; a single file speaks for itself.
    # Images are collected into a folder so they arrive in upload order.
    if len(uploads) > 1:
        source = workdir / "floors"
        source.mkdir()
        for index, upload in enumerate(uploads):
            shutil.copy(upload, source / f"{index:02d}{Path(upload).suffix.lower()}")
    else:
        source = Path(uploads[0])

    try:
        result = extract(source, workdir, segmenter=SEGMENTER)
    except Exception as error:
        raise gr.Error(f"Could not process this plan: {error}") from error

    overlays = []
    for floor in result.floors:
        path = workdir / f"overlay-{floor.index}.png"
        cv2.imwrite(str(path), draw_overlay(floor))
        overlays.append((str(path), f"Floor {floor.index + 1}"))

    table_rows = []
    for floor in result.floors:
        for room_index, room in enumerate(floor.plan.rooms):
            table_rows.append(
                [
                    floor.index + 1,
                    room_index,
                    room.label or "(unlabelled)",
                    room.category or "(none predicted)",
                    NO_CHANGE,
                ]
            )

    return overlays, table_rows, result


def _start_detect(uploads):
    """Gradio click handler: Detect always gets its own fresh workdir.

    A new ``tempfile.mkdtemp()`` per click -- not one made at import time
    or reused across clicks -- is what makes a second Detect safe even
    while the first plan's rows are still sitting in the table: this
    handler's outputs (overlay, table, state, workdir) are all replaced
    wholesale, so nothing from the previous detection survives for a
    stale row to point into.
    """
    workdir = Path(tempfile.mkdtemp(prefix="planto3d_"))
    overlays, table_rows, result = detect(uploads, workdir)
    return overlays, table_rows, result, workdir


def convert(wall_height_ft, correction_table, extracted_result, workdir, *choices):
    """Apply any corrections and build the model — the second click."""
    return convert_with_details(
        wall_height_ft, correction_table, extracted_result, workdir, *choices
    )[:4]


def convert_with_details(
    wall_height_ft: float,
    correction_table,
    extracted_result,
    workdir,
    style: str = "modern",
    colour: str = "warm",
    time: str = "day",
    landscaping: str = "basic",
    creativity: str = "balanced",
):
    """As ``convert``, plus what the drawing turned out to contain.

    The extra detail is what a photoreal pass needs to describe this house
    rather than a generic one -- how many storeys, and which rooms were
    named. Kept separate so the desktop app's outputs stay as they are.
    """
    if extracted_result is None or workdir is None:
        raise gr.Error("Detect a plan first — click Detect before Build.")

    # Each row identifies its room by (floor, room #), not by position, so
    # a user deleting or reordering rows in the UI can't misapply an
    # override to the wrong room -- it just drops that room's correction.
    #
    # Override is free text (this Gradio version has no per-column
    # dropdown), so it is matched case- and whitespace-insensitively: a
    # user typing "Open", " open ", or "(No Change)" plainly means the
    # same thing as "open" or "(no change)", and a blank cell -- clearing
    # it rather than retyping "(no change)" -- reads just as naturally as
    # "leave this room alone". Only text that still doesn't match any
    # known category after normalizing is a genuine mistake worth an error.
    corrections = {}
    for row in correction_table or []:
        floor_number, room_index, _label, _predicted, override = row
        normalized = str(override).strip().lower()
        if normalized in ("", NO_CHANGE):
            continue
        if normalized not in CATEGORY_LABELS:
            raise gr.Error(
                f"'{override}' isn't a category PlanTo3D knows. Choose one "
                f"of: {', '.join(CATEGORY_LABELS)}; or leave it as "
                f"{NO_CHANGE!r}."
            )
        corrections[(int(floor_number) - 1, int(room_index))] = normalized
    result = apply_room_corrections(extracted_result, corrections)

    design = Design(
        style=style,
        colour=colour,
        time=time,
        landscaping=landscaping,
        creativity=creativity,
    )
    try:
        result = build(
            result,
            workdir,
            wall_height_ft=wall_height_ft,
            palette=design.palette(),
            site=design.site(),
        )
    except Exception as error:
        raise gr.Error(f"Could not build this plan: {error}") from error

    if result.model_path is None:
        raise gr.Error(
            "Could not build a model from this drawing — no walls were found. "
            "PlanTo3D expects a clean, digital floor plan rather than a photo "
            "or a heavily compressed scan."
        )

    views = render_views(
        result.model_path, workdir, resolution=(1000, 750), lighting=design.lighting()
    )
    view_gallery = [
        (str(views[name]), name.title())
        for name in ("top", "front", "back", "left", "right", "aerial")
        if name in views
    ]

    SCALE_SOURCES = {
        "dimensions": "measured from the printed room dimensions",
        "doors": "inferred from door widths — this drawing carries no readable "
        "dimensions, so a standard 2'6\" door was used as the reference "
        "(typically accurate to within a few percent)",
        "walls": "inferred from wall thickness — no dimensions or doors could be "
        "read, so a standard 9\" wall was used as the reference",
        "ratio": "assumed from a typical 1:150 drafting ratio — nothing "
        "measurable was found on this drawing",
    }
    scale_line = (
        f"**Scale:** {result.scale:.1f} pixels per foot, "
        f"{SCALE_SOURCES.get(result.scale_source, result.scale_source)}"
    )
    if result.scale_assumed:
        scale_line += (
            "\n\n> The model's **proportions are correct**, but its absolute "
            "size is inferred rather than measured."
        )

    # Room names drive the floor finishes, planting, paving, railings and
    # stairs. When none are read the model still builds, but comes out far
    # plainer than the plan deserves -- and the usual cause is an image too
    # small for OCR rather than anything wrong with the drawing.
    named = sum(len(floor.named_rooms) for floor in result.floors)
    warning = ""
    if named == 0:
        warning = (
            "\n\n> **No room names could be read.** Floor finishes, planting, "
            "paving, railings and stairs all come from the labels, so the "
            "model is much plainer than it should be. This is nearly always "
            "an input resolution problem: upload the original PDF, or images "
            "at least 2000px across. Screenshots are usually too small."
        )
    elif named < result.room_count / 3:
        warning = (
            f"\n\n> Only {named} of {result.room_count} rooms could be named, "
            "so some finishes and features are missing. A higher-resolution "
            "scan would read more of them."
        )

    lines = [
        f"**Segmenter:** {SEGMENTER_NAME}",
        scale_line + warning,
        f"**Total:** {result.wall_count} wall segments, {result.room_count} rooms "
        f"across {len(result.floors)} floors",
        "",
        "| Floor | Walls | Rooms | Names read |",
        "| --- | --- | --- | --- |",
    ]
    for floor in result.floors:
        names = ", ".join(floor.named_rooms) or "_none_"
        lines.append(
            f"| {floor.index + 1} | {len(floor.plan.walls)} | "
            f"{len(floor.plan.rooms)} | {names} |"
        )

    hero = str(views.get("aerial", result.model_path))
    details = {
        "storeys": len(result.floors),
        "conditioning": design.conditioning(),
        "labels": [label for floor in result.floors for label in floor.named_rooms],
    }
    return (
        hero,
        str(result.model_path),
        view_gallery,
        "\n".join(lines),
        details,
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="Floor plan",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"],
                    file_count="multiple",
                    type="filepath",
                    height=180,
                )
                gr.Markdown(
                    "PDF, PNG or JPEG. Upload one file per storey — ground "
                    "floor first — or a single multi-page PDF.\n\n"
                    "**Prefer the original PDF.** Room names drive the floor "
                    "finishes, planting and railings, and reading them needs "
                    "resolution — a screenshot is usually too small."
                )
                height_input = gr.Slider(
                    minimum=7.0,
                    maximum=14.0,
                    value=DEFAULT_WALL_HEIGHT_FT,
                    step=0.5,
                    label="Storey height (feet)",
                    info="Floor plans do not state ceiling height, so set it here.",
                )

                # A drawing fixes the geometry and says nothing about the
                # building: not what it is clad in, not what hour it is
                # seen at, not whether there is a garden. Five choices,
                # because an earlier version offered a colour picker per
                # surface and that is a spreadsheet rather than a choice.
                with gr.Accordion("How it should look", open=True):
                    style_input = gr.Radio(
                        list(STYLES), value="modern", label="Style"
                    )
                    colour_input = gr.Radio(
                        list(TONES), value="warm", label="Colour"
                    )
                    time_input = gr.Radio(
                        list(TIMES),
                        value="day",
                        label="Time of day",
                    )
                    landscaping_input = gr.Radio(
                        list(LANDSCAPING),
                        value="basic",
                        label="Landscaping",
                        info="None leaves the building alone against the sky, "
                        "which is what a massing study wants.",
                    )
                    creativity_input = gr.Radio(
                        list(CREATIVITY),
                        value="balanced",
                        label="Creativity",
                        info="How far the photoreal pass may stray from the "
                        "plan. Strict holds the geometry and looks like a "
                        "shaded model; creative invents.",
                    )

                convert_button = gr.Button("Detect", variant="primary")

            with gr.Column(scale=2):
                # Two views of the same model, because they do different jobs.
                # The interactive viewer lights a scene brightly enough that
                # desaturated materials -- masonry, concrete -- saturate to
                # white, and it offers no way to turn that down. It stays for
                # orbiting the geometry; the rendered image beside it is what
                # the building actually looks like.
                render_output = gr.Image(
                    label="Rendered view", type="filepath", height=340
                )
                model_output = gr.Model3D(
                    label="Interactive model — drag to orbit",
                    clear_color=[0.1, 0.1, 0.12, 1.0],
                    height=300,
                )

        summary_output = gr.Markdown()
        view_output = gr.Gallery(
            label="Views — top, front, back, left, right, aerial",
            columns=3,
            height=460,
        )
        overlay_output = gr.Gallery(
            label="What was detected (walls in red, rooms in green)",
            columns=3,
            height=420,
        )

        # column_count/static_columns replace col_count in this Gradio
        # version (6.24.0); static_columns locks the four identifying
        # columns and leaves only Override editable. There is no per-
        # column dropdown datatype here, so Override is free text --
        # the caption below spells out the valid values.
        correction_output = gr.Dataframe(
            headers=["Floor", "Room #", "Detected label", "Predicted type", "Override"],
            datatype=["number", "number", "str", "str", "str"],
            label="Review — correct anything the plan reader got wrong",
            interactive=True,
            column_count=5,
            static_columns=[0, 1, 2, 3],
        )
        gr.Markdown(
            "Type into **Override** to relabel a room (case and spacing "
            "don't matter): "
            f"{', '.join(f'`{c}`' for c in CATEGORY_LABELS)}, "
            f"or leave it blank / `{NO_CHANGE}` to leave the room alone."
        )
        build_button = gr.Button("Build & render", variant="primary")

        # Detect fills these; Build reads them. Threading workdir through
        # explicitly (rather than re-deriving it from a floor's image path,
        # which for a single-image upload is the user's own file, not a
        # path inside the run's own temp directory) is what keeps Build
        # writing its .glb into the same place Detect already extracted to.
        extracted_state = gr.State(value=None)
        workdir_state = gr.State(value=None)

        convert_button.click(
            fn=_start_detect,
            inputs=[pdf_input],
            outputs=[overlay_output, correction_output, extracted_state, workdir_state],
        )

        build_button.click(
            fn=convert,
            inputs=[
                height_input,
                correction_output,
                extracted_state,
                workdir_state,
                style_input,
                colour_input,
                time_input,
                landscaping_input,
                creativity_input,
            ],
            outputs=[
                render_output,
                model_output,
                view_output,
                summary_output,
            ],
        )

    return demo


if __name__ == "__main__":
    # Gradio 6 moved theme from the Blocks constructor to launch().
    build_interface().launch(theme=gr.themes.Soft())
