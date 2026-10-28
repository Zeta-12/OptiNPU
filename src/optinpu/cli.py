"""Command line interface: optinpu devices | models | predict | bench | export | eval | train | profile."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="optinpu")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list inference devices")
    sub.add_parser("models", help="list registered models")

    p = sub.add_parser("predict", help="detect objects in an image")
    p.add_argument("model")
    p.add_argument("image")
    p.add_argument("--device", default="npu", help="cpu, gpu, npu, or hybrid npu+gpu / npu+cpu")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--save", help="write the annotated image here")
    p.add_argument("--checkpoint", help="best.pt from `optinpu train`")

    b = sub.add_parser("bench", help="latency / throughput per device")
    b.add_argument("model")
    b.add_argument("--devices", default="cpu,gpu,npu")
    b.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    b.add_argument("--iterations", type=int, default=100)
    b.add_argument("--stages", action="store_true", help="also measure cumulative stage latency on NPU")

    e = sub.add_parser("export", help="export to OpenVINO IR (int8 calibrates on COCO by default)")
    e.add_argument("model")
    e.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    e.add_argument("--force", action="store_true", help="re-export even if the IR exists")

    v = sub.add_parser("eval", help="COCO val2017 mAP on a subset")
    v.add_argument("model")
    v.add_argument("--devices", default="npu")
    v.add_argument("--precisions", default="fp16")
    v.add_argument("--images", type=int, default=500)

    t = sub.add_parser("train", help="fine-tune on a COCO-format dataset (iGPU if available)")
    t.add_argument("model")
    t.add_argument("--train", nargs=2, metavar=("JSON", "IMAGES"), required=True)
    t.add_argument("--val", nargs=2, metavar=("JSON", "IMAGES"))
    t.add_argument("--run", default="train")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--device")
    t.add_argument("--amp", action="store_true", help="bf16 autocast")

    n = sub.add_parser("lint", help="check NPU compatibility: static rules, compilation, accuracy vs CPU")
    n.add_argument("model", help="registered model name, or path to an OpenVINO .xml")
    n.add_argument("--image", help="real input for the accuracy check (default: noise)")
    n.add_argument("--device", default="npu")
    n.add_argument("--raw", action="store_true", help="ignore the model's known NPU config")
    n.add_argument("--images", type=int, default=0,
                   help="judge accuracy on detections over the first N COCO val2017 images (recommended: 30)")

    f = sub.add_parser("profile", help="per-layer time on a device")
    f.add_argument("model")
    f.add_argument("--device", default="npu")
    f.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    f.add_argument("--top", type=int, default=15)

    args = parser.parse_args(argv)

    if args.command == "devices":
        from optinpu.devices import available_devices

        for name, full in available_devices().items():
            print(f"{name:<4} {full}")
    elif args.command == "models":
        from optinpu.models import get_model, list_models

        for name in list_models():
            spec = get_model(name)
            note = f"  NPU config {spec.npu_config}" if spec.npu_config else ""
            print(f"{name:<14} {spec.source}{note}")
    elif args.command == "predict":
        from optinpu import Detector
        from optinpu.postprocess import load_image

        detector = Detector.from_checkpoint(args.model, args.checkpoint) if args.checkpoint else Detector(args.model)
        dets = detector.predict(args.image, device=args.device, conf=args.conf)
        for box, name, score in zip(dets.boxes, dets.names, dets.scores):
            print(f"{name:<16} {score:.2f}  " + " ".join(f"{v:.0f}" for v in box))
        if args.save:
            dets.draw(load_image(args.image)).save(args.save)
    elif args.command == "bench":
        from optinpu import Detector
        from optinpu.bench import format_table

        detector = Detector(args.model)
        devices = args.devices.split(",")
        print(format_table(detector.benchmark(devices, precision=args.precision, iterations=args.iterations)))
        if args.stages:
            for label, ms in detector.benchmark_stages(precision=args.precision).items():
                print(f"{label:<10} {ms:8.2f} ms")
    elif args.command == "export":
        from optinpu import Detector

        print(Detector(args.model).export(precision=args.precision, force=args.force))
    elif args.command == "eval":
        from optinpu import Detector
        from optinpu.evaluate import format_eval_table

        detector = Detector(args.model)
        results = [
            detector.evaluate(None, device, precision, num_images=args.images)
            for precision in args.precisions.split(",")
            for device in args.devices.split(",")
        ]
        print(format_eval_table(results))
    elif args.command == "train":
        from optinpu import Detector

        detector = Detector(args.model)
        result = detector.train(
            tuple(args.train), tuple(args.val) if args.val else None, run=args.run,
            epochs=args.epochs, batch=args.batch, lr=args.lr, device=args.device, amp=args.amp,
        )
        print(f"best epoch {result.best_epoch}, checkpoint {result.checkpoint}")
        print(f"NPU IR: {detector.export(precision='fp16')}")
    elif args.command == "lint":
        if args.model.endswith(".xml"):
            from optinpu.lint import lint

            report = lint(args.model, args.device)
        else:
            from optinpu import Detector

            images = None
            if args.images:
                from optinpu.data import coco_val_subset

                images = coco_val_subset(args.images).image_paths()
            report = Detector(args.model).lint(args.image, args.device, raw=args.raw, images=images)
        print(report)
        raise SystemExit(0 if report.ok else 1)
    elif args.command == "profile":
        from optinpu import Detector
        from optinpu.profile import format_profile, profile

        ir = Detector(args.model).export(precision=args.precision)
        print(format_profile(profile(ir, args.device), top=args.top))


if __name__ == "__main__":
    main()
