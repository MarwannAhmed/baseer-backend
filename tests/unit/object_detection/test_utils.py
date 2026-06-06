import numpy as np

from app.features.object_detection.utils import compute_iou, non_maximum_suppression


def test_compute_iou_disjoint_boxes_returns_zero():
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]

    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_identical_boxes_returns_one():
    box1 = [5, 5, 15, 15]
    box2 = [5, 5, 15, 15]

    assert compute_iou(box1, box2) == 1.0


def test_compute_iou_partial_overlap_returns_expected_ratio():
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]

    iou = compute_iou(box1, box2)
    expected = 25 / (100 + 100 - 25)

    assert np.isclose(iou, expected)


def test_non_maximum_suppression_returns_empty_for_no_boxes():
    boxes, scores, labels = non_maximum_suppression([], [], [], iou_threshold=0.5)

    assert boxes == []
    assert scores == []
    assert labels == []
