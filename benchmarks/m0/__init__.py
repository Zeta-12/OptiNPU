"""M0: reference accuracy / latency of existing real-time detectors on this machine (CPU, iGPU, NPU).

Third-party models with licenses incompatible with optinpu (DEIMv2: non-commercial, YOLO26: AGPL)
are exported to ONNX in a separate virtualenv (.venv-export) by export_models.py and only measured
here as OpenVINO IRs: none of their code is imported by optinpu.
"""
