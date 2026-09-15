from __future__ import annotations

import importlib.util
import platform
import sys


PACKAGES = ("torch", "torchvision", "ultralytics", "cv2", "networkx", "neo4j")


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for package in PACKAGES:
        status = "OK" if importlib.util.find_spec(package) else "NOT INSTALLED"
        print(f"{package}: {status}")

    if importlib.util.find_spec("torch"):
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()

