"""The transforms that decide whether the model reads a sixth thousand plans.

There was none of this. A segmentation model shown 5,000 plans learns
those 5,000 plans; what makes it read one it has not seen is having been
shown each of them drawn differently.

Every transform here exists because it matches something real that arrives
at the pipeline and is currently read badly -- sheets printed sideways,
messaging-app JPEGs, the same building drawn at another resolution.
"""

import cv2
import numpy as np
import pytest

from planto3d.classes import WALL
from training.augment import PROBABILITIES, augment, blur, compress, exposure, flip, rescale, rotate, unfill_walls


def _drawing(size=96):
    """An asymmetric plan, so a rotation or a flip is detectable."""
    rng = np.random.default_rng(0)
    image = rng.integers(90, 200, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), np.uint8)
    mask[10:16, 8:80] = 1  # a wall along the top
    mask[20:70, 8:14] = 1  # and down the left
    mask[22:68, 20:78] = 2  # a room
    mask[12:14, 30:40] = 4  # a window in the top wall
    return image, mask


class TestGeometryMovesBothTogether:
    @pytest.mark.parametrize("turns", [1, 2, 3])
    def test_a_quarter_turn_turns_the_mask_with_it(self, turns):
        image, mask = _drawing()

        turned_image, turned_mask = rotate(image, mask, turns)

        assert turned_image.shape == image.shape
        # The same pixels, in the same places, relative to each other.
        assert np.array_equal(turned_mask, np.rot90(mask, turns))

    def test_four_quarter_turns_come_back(self):
        image, mask = _drawing()

        result = (image, mask)
        for _ in range(4):
            result = rotate(*result, 1)

        assert np.array_equal(result[1], mask)

    def test_a_flip_mirrors_both(self):
        image, mask = _drawing()

        _, flipped = flip(image, mask, horizontal=True)

        assert np.array_equal(flipped, np.flip(mask, 1))

    def test_rescaling_keeps_the_shape_and_the_classes(self):
        image, mask = _drawing()

        scaled_image, scaled_mask = rescale(image, mask, 0.7, (0.5, 0.5))

        assert scaled_image.shape == image.shape
        assert scaled_mask.shape == mask.shape

    def test_rescaling_thickens_what_it_magnifies(self):
        # The point of it: the same wall arrives sometimes thick and
        # sometimes thin, so the model stops relying on either.
        image, mask = _drawing()

        _, scaled = rescale(image, mask, 0.5, (0.0, 0.0))

        assert (scaled == 1).sum() > (mask == 1).sum()


class TestNoTransformInventsAClass:
    """Nearest-neighbour throughout.

    Anything smoother blends a wall and a room into a door -- a class index
    that was never labelled anywhere on the drawing.
    """

    @pytest.mark.parametrize("seed", range(12))
    def test_only_labelled_classes_survive(self, seed):
        image, mask = _drawing()
        original = set(np.unique(mask))

        _, result = augment(image, mask, np.random.default_rng(seed))

        assert set(np.unique(result)) <= original


class TestThePhotometricOnesLeaveTheMaskAlone:
    def test_exposure_changes_only_the_image(self):
        image, _ = _drawing()

        assert not np.array_equal(exposure(image, 1.3, 0.8), image)

    def test_compression_leaves_artefacts(self):
        # Plans arrive as messaging-app JPEGs where every wall carries
        # ringing along its edge, and the thin classes are the first thing
        # the encoder throws away.
        image, _ = _drawing()

        assert not np.array_equal(compress(image, 30), image)

    def test_blurring_softens(self):
        image, _ = _drawing()

        softened = blur(image, 2.0)

        assert softened.std() < image.std()

    def test_no_blur_is_no_change(self):
        image, _ = _drawing()

        assert np.array_equal(blur(image, 0.0), image)


class TestTheWholeSet:
    def test_shapes_survive_every_draw(self):
        image, mask = _drawing()

        for seed in range(30):
            result_image, result_mask = augment(
                image.copy(), mask.copy(), np.random.default_rng(seed)
            )
            assert result_image.shape == image.shape
            assert result_mask.shape == mask.shape

    def test_it_actually_varies(self):
        # An augmentation that rarely fires is not augmentation.
        image, mask = _drawing()

        changed = sum(
            not np.array_equal(
                augment(image.copy(), mask.copy(), np.random.default_rng(seed))[1], mask
            )
            for seed in range(20)
        )

        assert changed >= 12, f"only {changed}/20 draws changed the drawing"

    def test_turning_everything_off_changes_nothing(self):
        image, mask = _drawing()
        # Taken from the set itself rather than listed here, so adding a
        # transform cannot leave this test quietly checking five of six.
        off = dict.fromkeys(PROBABILITIES, 0.0)

        result_image, result_mask = augment(
            image.copy(), mask.copy(), np.random.default_rng(0), off
        )

        assert np.array_equal(result_image, image)
        assert np.array_equal(result_mask, mask)

    def test_the_same_seed_gives_the_same_drawing(self):
        image, mask = _drawing()

        first = augment(image.copy(), mask.copy(), np.random.default_rng(7))
        second = augment(image.copy(), mask.copy(), np.random.default_rng(7))

        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])


class TestUnfillWalls:
    """Some offices draw a wall as two lines with nothing between them.

    Measured with scripts/convention_stress.py, that convention costs 0.214
    of wall IoU and takes recall from 0.899 to 0.689 -- much the worst of
    the eight tested, and the only one left needing the model rather than
    the reader.
    """

    def _drawing(self):
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        image[40:160, 40:60] = 20
        mask[40:160, 40:60] = WALL
        return image, mask

    def test_a_filled_wall_comes_back_hollow(self):
        image, mask = self._drawing()
        wall = mask == WALL

        before = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[wall] < 100).mean()
        after = (cv2.cvtColor(unfill_walls(image, mask), cv2.COLOR_BGR2GRAY)[wall] < 100).mean()

        assert before > 0.9
        assert after < 0.3

    def test_the_wall_keeps_its_edges(self):
        # Hollow, not absent. An outline wall still has two lines.
        image, mask = self._drawing()
        out = cv2.cvtColor(unfill_walls(image, mask), cv2.COLOR_BGR2GRAY)

        assert (out[mask == WALL] < 100).any()

    def test_the_labels_are_untouched(self):
        # The wall is still a wall; only how it is drawn changes. Altering
        # the mask here would teach the model that a hollow wall is not one.
        image, mask = self._drawing()
        original = mask.copy()

        unfill_walls(image, mask)

        assert np.array_equal(mask, original)

    def test_a_drawing_with_no_walls_is_returned_unchanged(self):
        image = np.full((60, 60, 3), 255, dtype=np.uint8)
        mask = np.zeros((60, 60), dtype=np.uint8)

        assert np.array_equal(unfill_walls(image, mask), image)
