"""No-argument, read-only local status; research execution is MCP-controlled."""
import json
from . import settings
from .workflows.pipeline import inspect


def main():
    if settings.ACTION != "inspect":
        raise RuntimeError("Research execution is MCP-only; use research_start/research_resume")
    print(json.dumps(inspect(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
