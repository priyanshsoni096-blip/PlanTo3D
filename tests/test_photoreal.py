import numpy as np
import pytest
from PIL import Image

from planto3d.design import STYLES, TIMES, TONES, Design
from planto3d.geometry_types import FloorPlan, Opening, Wall
from planto3d.materials import build_scene, export_scene
from planto3d.photoreal import (
    MAX_PROMPT_TOKENS,
    NEGATIVE_PROMPT,
    _tokens,
    build_guides,
    build_prompt,
    edge_guide,
)

FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


@pytest.fixture
def model(tmp_path):
    walls = [
        Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0)
        for i in range(4)
    ]
    plan = FloorPlan(
        walls=walls,
        footprint=list(FOOTPRINT),
        openings=[Opening(wall_id=0, position=200.0, width=120.0, type="window")],
    )
    path = tmp_path / "house.glb"
    export_scene(build_scene([plan, plan], wall_height_ft=9.0, scale=20.0), path)
    return path


class TestPrompt:
    def test_names_the_storey_count(self):
        assert "3-storey" in build_prompt(3)

    def test_mentions_features_the_plan_actually_has(self):
        prompt = build_prompt(3, ["BEDROOM", "PARKING", "TERRACE GARDEN"])

        assert "cars" in prompt
        assert "lawn" in prompt
        assert "terrace" in prompt.lower()

    def test_does_not_invent_features_the_plan_lacks(self):
        # A stylization stops resembling its subject when the prompt asks for
        # things the building does not have.
        prompt = build_prompt(2, ["BEDROOM", "KITCHEN"])

        assert "cars" not in prompt
        assert "lawn" not in prompt

    def test_labels_are_optional(self):
        assert build_prompt(2)

    def test_the_prompt_needs_no_3d_stack(self):
        # The generation environment installs diffusion packages, not trimesh
        # or shapely, and only ever wants the prompt. Importing this module
        # must not drag in the mesh libraries.
        import subprocess
        import sys

        script = (
            "import sys;"
            "sys.modules['trimesh'] = None;"
            "sys.modules['shapely'] = None;"
            "from planto3d.photoreal import build_prompt, NEGATIVE_PROMPT;"
            "assert build_prompt(3);"
            "print('ok')"
        )
        finished = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )

        assert finished.returncode == 0, finished.stderr


class TestGuides:
    def test_all_three_guides_are_produced(self, model, tmp_path):
        guides = build_guides(model, tmp_path, resolution=(256, 192))

        assert set(guides) == {"render", "depth", "edges"}
        assert all(path.exists() for path in guides.values())

    def test_the_guides_share_one_camera(self, model, tmp_path):
        # A depth map and an edge map shot from different angles fight, and
        # the diffusion model resolves the conflict by warping the building.
        guides = build_guides(model, tmp_path, resolution=(256, 192))

        sizes = {Image.open(path).size for path in guides.values()}
        assert len(sizes) == 1

    def test_depth_is_single_channel_with_near_surfaces_bright(self, model, tmp_path):
        guides = build_guides(model, tmp_path, resolution=(256, 192))
        depth = np.asarray(Image.open(guides["depth"]))

        assert depth.ndim == 2  # greyscale, as depth ControlNets expect
        assert depth.max() > depth.min()

    def test_background_reads_as_far_away_not_near(self, model, tmp_path):
        # Undrawn pixels must be black. Left bright, the model reads the sky
        # as a surface pressed against the camera.
        guides = build_guides(model, tmp_path, resolution=(256, 192))
        depth = np.asarray(Image.open(guides["depth"]))

        assert depth[0, 0] == 0

    def test_edges_are_binary_and_trace_the_building(self, model, tmp_path):
        guides = build_guides(model, tmp_path, resolution=(256, 192))
        edges = np.asarray(Image.open(guides["edges"]))

        assert set(np.unique(edges)) <= {0, 255}
        assert (edges == 255).any()

    def test_a_missing_render_fails_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            edge_guide(tmp_path / "absent.png", tmp_path / "edges.png")


class TestThePromptFitsWhatTheEncoderReads:
    """CLIP reads 77 tokens and silently discards the rest.

    The prompt ran to about 144, so half went -- and being the tail, the
    half that went was everything derived from the drawing: the cars, the
    lawn, the terrace, the balconies. The generic opening survived and the
    specific ending did not, which is exactly backwards.
    """

    RICH = ["PARKING", "TERRACE GARDEN", "LANDSCAPE", "BALCONY", "SWIMMING POOL", "TEMPLE"]

    def test_it_fits(self):
        assert _tokens(build_prompt(3, self.RICH)) <= MAX_PROMPT_TOKENS

    def test_it_fits_with_nothing_to_say(self):
        assert _tokens(build_prompt(1, [])) <= MAX_PROMPT_TOKENS

    def test_the_negative_prompt_fits_too(self):
        assert _tokens(NEGATIVE_PROMPT) <= MAX_PROMPT_TOKENS

    @pytest.mark.parametrize(
        "label, phrase",
        [
            ("PARKING", "cars"),
            ("LANDSCAPE", "lawn"),
            ("TERRACE GARDEN", "terrace"),
            ("BALCONY", "balcon"),
            ("SWIMMING POOL", "pool"),
        ],
    )
    def test_what_the_drawing_shows_survives_the_budget(self, label, phrase):
        # The whole point. These are the only phrases the drawing justifies,
        # so they must outrank the generic ones rather than being cut for
        # them.
        assert phrase in build_prompt(3, self.RICH).lower(), phrase

    def test_the_subject_and_its_light_are_never_dropped(self):
        # A building of the wrong material is wrong in every frame, and a
        # dusk render without described lighting is just a dark one.
        prompt = build_prompt(3, self.RICH).lower()

        assert "residence" in prompt
        assert "limestone" in prompt
        assert "amber" in prompt

    def test_nothing_is_claimed_that_the_drawing_does_not_show(self):
        plain = build_prompt(2, ["BEDROOM", "KITCHEN"]).lower()

        for absent in ("pool", "cars", "balcon", "terrace"):
            assert absent not in plain, absent

    def test_the_storey_count_comes_through(self):
        assert "4-storey" in build_prompt(4, [])


def _design(**overrides):
    base = dict(
        style="modern", colour="warm", time="day",
        landscaping="basic", creativity="balanced",
    )
    base.update(overrides)
    return Design(**base)


def test_every_style_changes_the_prompt():
    # A traditional house described as "modern" is the wrong building.
    prompts = {
        style: build_prompt(2, ["BALCONY"], design=_design(style=style))
        for style in STYLES
    }
    assert len(set(prompts.values())) == len(STYLES), prompts


def test_every_time_of_day_changes_the_prompt():
    prompts = {
        time: build_prompt(2, ["BALCONY"], design=_design(time=time))
        for time in TIMES
    }
    assert len(set(prompts.values())) == len(TIMES), prompts


def test_every_tone_changes_the_prompt():
    prompts = {
        tone: build_prompt(2, ["BALCONY"], design=_design(colour=tone))
        for tone in TONES
    }
    assert len(set(prompts.values())) == len(TONES), prompts


def test_night_is_not_described_as_dusk():
    night = build_prompt(2, None, design=_design(time="night"))
    assert "dusk" not in night
    assert "night" in night


def test_the_prompt_still_fits_the_token_budget():
    # CLIP reads 77 tokens and silently drops the rest, tail first -- which
    # is where the drawing-derived detail lives. Every combination must fit.
    labels = ["BALCONY", "TERRACE GARDEN", "PARKING", "SWIMMING POOL"]
    for style in STYLES:
        for tone in TONES:
            for time in TIMES:
                design = _design(style=style, colour=tone, time=time)
                prompt = build_prompt(3, labels, design=design)
                assert _tokens(prompt) <= MAX_PROMPT_TOKENS, (style, tone, time)


def test_no_design_keeps_the_previous_wording():
    # Every existing caller passes no design and must be unaffected.
    assert "modern luxury residence at dusk" in build_prompt(2, ["BALCONY"])
