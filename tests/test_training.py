import numpy as np
import pytest

torch = pytest.importorskip("torch")

from planto3d.classes import BACKGROUND, DOOR, NUM_CLASSES, ROOM, WALL
from training.metrics import dice_score, iou_score, per_class_iou


class TestMetrics:
    def test_a_perfect_prediction_scores_one(self):
        target = torch.tensor([[WALL, ROOM], [ROOM, BACKGROUND]])

        assert dice_score(target, target) == pytest.approx(1.0, abs=1e-4)
        assert iou_score(target, target) == pytest.approx(1.0, abs=1e-4)

    def test_a_completely_wrong_prediction_scores_zero(self):
        target = torch.full((4, 4), WALL)
        prediction = torch.full((4, 4), ROOM)

        assert dice_score(prediction, target) == pytest.approx(0.0, abs=1e-4)
        assert iou_score(prediction, target) == pytest.approx(0.0, abs=1e-4)

    def test_averaging_is_per_class_not_pooled(self):
        # 99 background pixels and 1 door. Missing the door entirely still
        # gets 99% of pixels right, so a pooled score would look excellent.
        target = torch.full((10, 10), BACKGROUND)
        target[0, 0] = DOOR
        prediction = torch.full((10, 10), BACKGROUND)

        # Background is perfect, door is zero -> the average must show it.
        assert dice_score(prediction, target) < 0.6

    def test_classes_absent_everywhere_do_not_inflate_the_score(self):
        # Only two classes appear. Scoring the missing three as perfect would
        # push the average up regardless of how the model actually did.
        target = torch.tensor([[WALL, WALL], [ROOM, ROOM]])
        prediction = torch.tensor([[WALL, ROOM], [ROOM, ROOM]])

        score = iou_score(prediction, target)

        assert score < 0.8

    def test_iou_is_never_greater_than_dice(self):
        target = torch.tensor([[WALL, ROOM], [ROOM, BACKGROUND]])
        prediction = torch.tensor([[WALL, WALL], [ROOM, BACKGROUND]])

        assert iou_score(prediction, target) <= dice_score(prediction, target) + 1e-6

    def test_per_class_iou_reports_none_for_absent_classes(self):
        target = torch.full((4, 4), WALL)

        scores = per_class_iou(target, target)

        assert scores[WALL] == pytest.approx(1.0, abs=1e-4)
        assert scores[DOOR] is None
        assert len(scores) == NUM_CLASSES


class TestDataset:
    def _sample(self, root, folder, size=(40, 60)):
        directory = root / folder
        directory.mkdir(parents=True)
        import cv2

        cv2.imwrite(str(directory / "F1_scaled.png"), np.full((*size, 3), 255, np.uint8))
        (directory / "model.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<g class="Wall"><polygon points="0,0 20,0 20,20 0,20"/></g>'
            "</svg>"
        )

    def test_yields_tensors_the_network_expects(self, tmp_path):
        from training.dataset import CubiCasaDataset

        self._sample(tmp_path, "colorful/1")
        split = tmp_path / "train.txt"
        split.write_text("/colorful/1/\n")

        dataset = CubiCasaDataset(tmp_path, split, size=64)
        image, mask = dataset[0]

        assert image.shape == (3, 64, 64)
        assert image.dtype == torch.float32
        assert mask.shape == (64, 64)
        assert mask.dtype == torch.int64

    def test_masks_keep_valid_class_indices_after_resizing(self, tmp_path):
        # Smooth interpolation would blend a wall and a room into a door.
        from training.dataset import CubiCasaDataset

        self._sample(tmp_path, "colorful/1")
        split = tmp_path / "train.txt"
        split.write_text("/colorful/1/\n")

        _, mask = CubiCasaDataset(tmp_path, split, size=97)[0]

        assert set(mask.unique().tolist()) <= set(range(NUM_CLASSES))

    def test_an_empty_split_is_an_error_not_a_silent_no_op(self, tmp_path):
        from training.dataset import CubiCasaDataset

        split = tmp_path / "train.txt"
        split.write_text("")

        with pytest.raises(ValueError):
            CubiCasaDataset(tmp_path, split)
