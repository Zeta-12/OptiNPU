"""COCO mAP evaluation of a Detector on any backend."""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import asdict, dataclass

from optinpu.data.coco import CocoSubset

_STAT_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl")


@dataclass
class EvalResult:
    model: str
    device: str
    precision: str
    num_images: int
    AP: float
    AP50: float
    AP75: float
    APs: float
    APm: float
    APl: float
    seconds_per_image: float  # end-to-end, including image decoding and pre/post-processing

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_coco(
    detector,
    subset: CocoSubset,
    device: str = "npu",
    precision: str = "fp16",
    verbose: bool = False,
) -> EvalResult:
    """Standard COCO bbox mAP. Keeps the top-300 (query, class) pairs per image, no threshold."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    results = []
    t0 = time.perf_counter()
    for image_id in subset.image_ids:
        dets = detector.predict(subset.image_path(image_id), device=device, precision=precision, conf=0.0)
        for (x0, y0, x1, y1), score, label in zip(dets.boxes, dets.scores, dets.labels):
            results.append({
                "image_id": image_id,
                "category_id": subset.category_ids[int(label)],
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "score": float(score),
            })
    elapsed = (time.perf_counter() - t0) / len(subset.image_ids)

    quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with quiet:
        gt = COCO(str(subset.annotations))
        dt = gt.loadRes(results) if results else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = subset.image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    stats = dict(zip(_STAT_NAMES, (float(s) for s in ev.stats[:6])))
    return EvalResult(detector.name, device, precision, len(subset.image_ids), **stats, seconds_per_image=elapsed)


def format_eval_table(results: list[EvalResult]) -> str:
    header = f"{'model':<14} {'device':<6} {'prec':<5} {'AP':>6} {'AP50':>6} {'AP75':>6} {'s/img':>6}"
    rows = [
        f"{r.model:<14} {r.device:<6} {r.precision:<5} {r.AP:>6.3f} {r.AP50:>6.3f} {r.AP75:>6.3f} {r.seconds_per_image:>6.3f}"
        for r in results
    ]
    return "\n".join([header, *rows])
