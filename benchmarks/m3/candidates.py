"""M3: OptiDet-N variants at higher resolution within the N tier (3.0-3.8 ms INT8, NPU only), keeping
the ImageNet RepVGG-A0 backbone: its last full stage (14 layers) is cut to fewer layers, whose
weights load_repvgg_a0 still provides (deployed form, random weights).

    python -m benchmarks.m1.run M3      # on mains power
"""

from __future__ import annotations

from benchmarks.m1.blocks import Block
from benchmarks.m2.candidates import build

A0_WIDTHS = (48, 48, 96, 192, 384)


def candidates() -> list[Block]:
    blocks = []
    for stage4 in (10, 8, 6):
        for res in (480, 512):
            depths = (1, 2, 4, stage4, 1)
            blocks.append(Block(
                "M3", f"optidet_a0d{stage4}_n64_{res}",
                lambda d=depths, r=res: build(A0_WIDTHS, d, 64, r),
                (1, 3, res, res), {"backbone": f"a0d{stage4}", "neck": 64, "res": res},
            ))
    blocks.append(Block("M3", "optidet_a0_n64_480", lambda: build(A0_WIDTHS, (1, 2, 4, 14, 1), 64, 480),
                        (1, 3, 480, 480), {"backbone": "a0", "neck": 64, "res": 480}))
    return blocks
