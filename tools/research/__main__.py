"""No-argument research entry. Mode is source-locked in settings.py."""
import json

from . import settings
from .workflows.pipeline import inspect, step


def main():
    if settings.ACTION == "inspect":
        result = inspect()
    elif settings.ACTION in ("behavior", "worker"):
        result = step()
    else:
        raise ValueError("unsupported source-configured research mode")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
