"""The demo must be able to say how the building should look.

The renderer takes a style, a colour tone and how much landscaping to put
around the building, and `notebooks/run_on_colab.ipynb` passes all three.
`scripts/demo.py` did not, so the one command that shows the project off
always rendered the default scheme on bare ground -- grey masonry, no
planting -- while "the render should be beautiful" is one of the project's
three goals.
"""

from planto3d.design import LANDSCAPING, STYLES, TONES


def test_the_demo_offers_every_style_tone_and_landscaping():
    from scripts.demo import build_parser

    parser = build_parser()
    choices = {action.dest: action.choices for action in parser._actions if action.choices}

    assert set(choices["style"]) == set(STYLES)
    assert set(choices["colour"]) == set(TONES)
    assert set(choices["landscaping"]) == set(LANDSCAPING)


def test_the_demo_shows_the_project_at_its_best_by_default():
    # A demo opening on bare ground teaches the wrong thing about a project
    # whose point is that the render is worth looking at.
    from scripts.demo import build_parser

    arguments = build_parser().parse_args([])

    assert arguments.landscaping == "premium"
    assert arguments.style in STYLES
    assert arguments.colour in TONES


def test_the_choices_reach_the_renderer():
    from planto3d.design import Design

    design = Design(style="luxury", colour="dark", landscaping="premium")
    plain = Design(style="minimalist", colour="light", landscaping="none")

    assert design.palette() != plain.palette()
    assert design.site() != plain.site()
