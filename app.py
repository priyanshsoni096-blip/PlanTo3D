"""PlanTo3D web interface: upload a floor plan PDF, get a 3D model.

Run with:  python app.py
"""

import logging
import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from planto3d.extrude import DEFAULT_WALL_HEIGHT_FT
from planto3d.pipeline import draw_overlay, run
from planto3d.preview import render_views
from planto3d.segment import load_segmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TITLE = "PlanTo3D"
DESCRIPTION = (
    "Upload a 2D architectural floor plan PDF and get back a 3D model of the "
    "building. Walls, rooms and dimensions are read straight off the drawing: "
    "the scale comes from the printed room sizes, so the model is measured in "
    "real feet rather than guessed."
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


def convert(pdf_file, wall_height_ft: float):
    """Run the pipeline and return the model, overlays, and a summary."""
    if pdf_file is None:
        raise gr.Error("Upload a floor plan PDF first.")

    workdir = Path(tempfile.mkdtemp(prefix="planto3d_"))

    try:
        result = run(
            Path(pdf_file),
            workdir,
            segmenter=SEGMENTER,
            wall_height_ft=wall_height_ft,
        )
    except Exception as error:
        raise gr.Error(f"Could not process this plan: {error}") from error

    if result.model_path is None:
        raise gr.Error(
            "Could not build a model from this drawing — no walls were found. "
            "PlanTo3D expects a clean, digital floor plan rather than a photo "
            "or a heavily compressed scan."
        )

    views = render_views(result.model_path, workdir, resolution=(1000, 750))
    view_gallery = [
        (str(views[name]), name.title())
        for name in ("top", "front", "back", "left", "right", "aerial")
        if name in views
    ]

    overlays = []
    for floor in result.floors:
        path = workdir / f"overlay-{floor.index}.png"
        cv2.imwrite(str(path), draw_overlay(floor))
        overlays.append((str(path), f"Floor {floor.index + 1}"))

    if result.scale_assumed:
        scale_line = (
            f"**Scale: assumed** at {result.scale:.1f} pixels per foot — no room "
            "dimensions could be read from this drawing, so a typical 1:150 "
            "drafting ratio was used. **The model's proportions are correct but "
            "its absolute size is a guess.**"
        )
    else:
        scale_line = (
            f"**Scale read from the drawing:** {result.scale:.1f} pixels per foot"
        )

    lines = [
        f"**Segmenter:** {SEGMENTER_NAME}",
        scale_line,
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

    return str(result.model_path), view_gallery, overlays, "\n".join(lines)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="Floor plan PDF", file_types=[".pdf"], type="filepath"
                )
                height_input = gr.Slider(
                    minimum=7.0,
                    maximum=14.0,
                    value=DEFAULT_WALL_HEIGHT_FT,
                    step=0.5,
                    label="Storey height (feet)",
                    info="Floor plans do not state ceiling height, so set it here.",
                )
                convert_button = gr.Button("Convert to 3D", variant="primary")

            with gr.Column(scale=2):
                model_output = gr.Model3D(label="3D model", clear_color=[0.1, 0.1, 0.12, 1.0])

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

        convert_button.click(
            fn=convert,
            inputs=[pdf_input, height_input],
            outputs=[model_output, view_output, overlay_output, summary_output],
        )

    return demo


if __name__ == "__main__":
    # Gradio 6 moved theme from the Blocks constructor to launch().
    build_interface().launch(theme=gr.themes.Soft())
