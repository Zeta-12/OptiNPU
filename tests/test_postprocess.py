import numpy as np
from PIL import Image

from optinpu.postprocess import decode, preprocess


def test_preprocess_shape_and_range():
    img = Image.fromarray(np.full((100, 200, 3), 255, dtype=np.uint8))
    x = preprocess(img, (64, 32))
    assert x.shape == (1, 3, 64, 32)
    assert x.dtype == np.float32
    assert np.allclose(x, 1.0)


def test_decode_topk_threshold_and_scaling():
    scores = np.zeros((4, 3), dtype=np.float32)
    scores[2, 1] = 0.9
    scores[0, 2] = 0.5
    scores[3, 0] = 0.1  # below threshold
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0, 0, 1, 1], [0.5, 0.5, 1.0, 1.2], [0, 0, 1, 1]], dtype=np.float32)
    dets = decode(scores, boxes, orig_size=(200, 100), class_names=["a", "b", "c"], conf=0.3)
    assert dets.names == ["b", "c"]
    np.testing.assert_allclose(dets.scores, [0.9, 0.5])
    np.testing.assert_allclose(dets.boxes[0], [100, 50, 200, 100])  # clipped to the image
    np.testing.assert_allclose(dets.boxes[1], [0, 0, 100, 50])


def test_decode_empty():
    dets = decode(np.zeros((5, 2), np.float32), np.zeros((5, 4), np.float32), (10, 10), ["a", "b"])
    assert len(dets) == 0
