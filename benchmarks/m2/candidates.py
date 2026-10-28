"""OptiDet architecture candidates for the NPU latency sweep (deployed form, random weights).

Measured with the M1 machinery: python -m benchmarks.m1.run M2
"""

from __future__ import annotations

from benchmarks.m1.blocks import Block

BACKBONES = {
    "tiny": ((32, 32, 64, 128, 256), (1, 1, 2, 4, 1)),
    "a0s": ((32, 48, 96, 192, 384), (1, 2, 4, 8, 1)),
    "a0": ((48, 48, 96, 192, 384), (1, 2, 4, 14, 1)),
    "wide": ((32, 64, 128, 256, 512), (1, 1, 3, 4, 1)),
}
NECK_WIDTHS = (64, 96, 128)
RESOLUTIONS = (416, 512, 640)


def build(widths, depths, neck, res):
    from optinpu.models.optidet import OptiDet, OptiDetConfig

    model = OptiDet(OptiDetConfig(widths=widths, depths=depths, neck_width=neck, head_width=neck, imgsz=(res, res)))
    return model.deploy()


def candidates() -> list[Block]:
    blocks = []
    for name, (widths, depths) in BACKBONES.items():
        for neck in NECK_WIDTHS:
            for res in RESOLUTIONS:
                blocks.append(Block(
                    "M2", f"optidet_{name}_n{neck}_{res}",
                    lambda w=widths, d=depths, n=neck, r=res: build(w, d, n, r),
                    (1, 3, res, res), {"backbone": name, "neck": neck, "res": res},
                ))
    return blocks
