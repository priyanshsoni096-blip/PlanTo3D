"""A plan is the same building whatever size it is rendered at.

Every size in the extraction stages used to be an absolute pixel count,
measured on drawings around 28-30 pixels per foot. That is a resolution,
not a property of any building, and it made the pipeline quietly dependent
on one: the same CubiCasa plan reconstructed with 15 walls at half size and
40 at double, and its inferred scale went from 8% out to 48% out.

They are multiples of the drawing's own wall thickness now -- the one
length a floor plan always contains and always draws to scale.
"""

import numpy as np
import pytest

from planto3d.classes import BACKGROUND, ROOM, WALL
from planto3d.extract import (
    REFERENCE_GAUGE,
    extract_footprint,
    extract_rooms,
    extract_walls,
    wall_gauge,
)


def _plan(gauge: int, size: int = 20):
    """A four-room building drawn at a given wall thickness.

    Every length is a multiple of the gauge, so the same building is
    described identically at any of them -- which is the whole point.
    """
    span = gauge * size
    mask = np.full((span, span), BACKGROUND, dtype=np.int64)

    inner = gauge * 2
    mask[inner : span - inner, inner : span - inner] = ROOM

    # Perimeter.
    mask[inner - gauge : inner, inner - gauge : span - inner + gauge] = WALL
    mask[span - inner : span - inner + gauge, inner - gauge : span - inner + gauge] = WALL
    mask[inner - gauge : span - inner + gauge, inner - gauge : inner] = WALL
    mask[inner - gauge : span - inner + gauge, span - inner : span - inner + gauge] = WALL

    # A cross of partitions, each interrupted by a doorway three gauges wide.
    middle = span // 2
    mask[middle : middle + gauge, inner : middle - gauge] = WALL
    mask[middle : middle + gauge, middle + gauge * 2 : span - inner] = WALL
    mask[inner : middle - gauge, middle : middle + gauge] = WALL
    mask[middle + gauge * 2 : span - inner, middle : middle + gauge] = WALL
    return mask


GAUGES = [12, 18, 24, 36, 48]


class TestTheGauge:
    @pytest.mark.parametrize("gauge", GAUGES)
    def test_it_measures_the_drawn_wall(self, gauge):
        # Loosely, and it reads a little high on purpose. Where two walls
        # meet the mask is thicker than either of them, and the distance
        # transform counts those junctions along with the walls. What
        # matters is that it lands near the drawn thickness and moves with
        # it, not that it recovers it exactly -- everything downstream is a
        # tolerance, not a measurement.
        measured = wall_gauge(_plan(gauge))

        assert measured == pytest.approx(gauge, rel=0.45)

    def test_it_tracks_the_drawing_rather_than_the_page(self):
        measured = [wall_gauge(_plan(g)) for g in GAUGES]

        assert measured == sorted(measured)
        assert measured[-1] > measured[0] * 2

    def test_a_mask_with_no_wall_falls_back(self):
        empty = np.full((200, 200), BACKGROUND, dtype=np.int64)

        assert wall_gauge(empty) == REFERENCE_GAUGE

    def test_a_trace_of_wall_is_not_measured(self):
        # A handful of stray pixels says nothing about the building, and a
        # wrong gauge is worse than the reference: it rescales every
        # threshold at once.
        speck = np.full((400, 400), BACKGROUND, dtype=np.int64)
        speck[10:13, 10:13] = WALL

        assert wall_gauge(speck) == REFERENCE_GAUGE


class TestTheSameBuildingAtEverySize:
    def test_the_same_walls_are_found(self):
        # The building is identical in every proportion; only the number of
        # pixels differs. The wall count has to follow the building.
        counts = [len(extract_walls(_plan(g))) for g in GAUGES]

        assert max(counts) - min(counts) <= 2, counts

    def test_the_same_rooms_are_found(self):
        counts = [len(extract_rooms(_plan(g))) for g in GAUGES]

        assert len(set(counts)) == 1, counts

    def test_doorways_are_bridged_at_every_size(self):
        # A doorway is about three wall thicknesses wide whatever the
        # drawing's resolution. Measured in fixed pixels, the same doorway
        # was a gap to bridge on one sheet and two separate walls on
        # another.
        for gauge in GAUGES:
            walls = extract_walls(_plan(gauge))
            horizontal = [
                w for w in walls if abs(w.end[0] - w.start[0]) > abs(w.end[1] - w.start[1])
            ]
            assert horizontal, gauge

    def test_the_footprint_is_the_same_shape(self):
        # Compared as a fraction of the page, since the page grows with the
        # drawing.
        shares = []
        for gauge in GAUGES:
            mask = _plan(gauge)
            outline = extract_footprint(mask)
            assert outline, gauge
            xs = [p[0] for p in outline]
            ys = [p[1] for p in outline]
            page = mask.shape[0]
            shares.append(((max(xs) - min(xs)) / page, (max(ys) - min(ys)) / page))

        widths = [s[0] for s in shares]
        assert max(widths) - min(widths) < 0.1, shares
