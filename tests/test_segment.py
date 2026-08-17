import numpy as np
import pytest

torch = pytest.importorskip("torch")

from planto3d.classes import NUM_CLASSES, WALL
from planto3d.segment import Segmenter, load_segmenter


@pytest.fixture
def checkpoint(tmp_path):
    """A real, tiny checkpoint in the format training/train.py writes."""
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=NUM_CLASSES
    )
    path = tmp_path / "unet.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "encoder": "resnet18",
            "num_classes": NUM_CLASSES,
            "size": 64,
            "val_dice": 0.5,
        },
        path,
    )
    return path


class TestSegmenter:
    def test_mask_comes_back_at_the_page_resolution(self, checkpoint):
        # The network takes a fixed square, but wall coordinates, room
        # polygons and OCR boxes all live in the page's own frame. A mask at
        # the network's resolution would silently misplace every one of them.
        segmenter = Segmenter(checkpoint, device="cpu")
        page = np.full((300, 500, 3), 255, dtype=np.uint8)

        mask = segmenter.predict(page)

        assert mask.shape == (300, 500)

    def test_mask_holds_only_valid_class_indices(self, checkpoint):
        # Resizing the mask back up must not interpolate between indices and
        # invent a class the model never predicted.
        segmenter = Segmenter(checkpoint, device="cpu")
        page = np.random.randint(0, 255, (137, 211, 3), dtype=np.uint8)

        mask = segmenter.predict(page)

        assert mask.dtype == np.int64
        assert set(np.unique(mask).tolist()) <= set(range(NUM_CLASSES))

    def test_output_feeds_the_geometry_extractors(self, checkpoint):
        from planto3d.extract import extract_rooms, extract_walls

        mask = Segmenter(checkpoint, device="cpu").predict(
            np.full((200, 200, 3), 255, dtype=np.uint8)
        )

        # Shape and dtype must satisfy the extractors, whatever the content.
        assert isinstance(extract_walls(mask), list)
        assert isinstance(extract_rooms(mask), list)

    def test_greyscale_pages_are_accepted(self, checkpoint):
        segmenter = Segmenter(checkpoint, device="cpu")

        mask = segmenter.predict(np.full((100, 120), 255, dtype=np.uint8))

        assert mask.shape == (100, 120)

    def test_the_callable_form_matches_predict(self, checkpoint):
        segmenter = Segmenter(checkpoint, device="cpu")
        page = np.full((80, 90, 3), 200, dtype=np.uint8)

        assert np.array_equal(segmenter(page), segmenter.predict(page))

    def test_input_size_is_read_from_the_checkpoint(self, checkpoint):
        assert Segmenter(checkpoint, device="cpu").input_size == 64

    def test_a_missing_checkpoint_fails_immediately(self, tmp_path):
        # Fail on construction, not on the first page, so a bad path is not
        # discovered part-way through processing a document.
        with pytest.raises(FileNotFoundError):
            Segmenter(tmp_path / "absent.pt")


class TestLoadSegmenter:
    def test_falls_back_to_the_classical_baseline(self):
        from planto3d.classical import classical_mask

        assert load_segmenter(None) is classical_mask

    def test_returns_the_trained_model_when_given_one(self, checkpoint):
        assert isinstance(load_segmenter(checkpoint), Segmenter)

    def test_both_forms_are_interchangeable_to_the_pipeline(self, checkpoint):
        page = np.full((120, 160, 3), 255, dtype=np.uint8)

        baseline = load_segmenter(None)(page)
        trained = load_segmenter(checkpoint)(page)

        assert baseline.shape == trained.shape == (120, 160)
        assert baseline.dtype == trained.dtype
