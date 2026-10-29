"""Export requirements-kaggle.txt from uv.lock, omitting Kaggle-preinstalled packages."""
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / "pyproject.toml", "rb") as f:
        preinstalled = tomllib.load(f)["tool"]["kaggle-export"]["preinstalled"]
    cmd = ["uv", "export", "--frozen", "--no-hashes", "--no-dev", "--no-emit-project",
           "--no-header", "--no-annotate", "--output-file", str(ROOT / "requirements-kaggle.txt")]
    for name in sorted(set(preinstalled)):
        cmd += ["--prune", name]
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
