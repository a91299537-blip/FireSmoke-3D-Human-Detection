#!/usr/bin/env python3
"""Install the paper-specific files into a clean OpenPCDet checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "233f849829b6ac19afb8af8837a0246890908755"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpcdet-root", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Confirm replacement of upstream files in the target checkout.",
    )
    return parser.parse_args()


def git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    target = args.openpcdet_root.resolve()
    overlay = Path(__file__).resolve().parents[1] / "openpcdet_overlay"

    if not args.force:
        raise SystemExit("Refusing to replace files without --force.")
    if not (target / "pcdet").is_dir() or not (target / "tools").is_dir():
        raise SystemExit(f"Not an OpenPCDet checkout: {target}")

    head = git_head(target)
    if head is not None and head != EXPECTED_COMMIT:
        raise SystemExit(
            f"Expected OpenPCDet {EXPECTED_COMMIT}, but found {head}. "
            "Use a clean checkout at the pinned commit."
        )

    copied = 0
    for source in sorted(overlay.rglob("*")):
        if not source.is_file():
            continue
        destination = target / source.relative_to(overlay)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    print(f"Installed {copied} files into {target}")


if __name__ == "__main__":
    main()

