"""Filesystem helpers for notebooks that run on Windows, macOS, and Linux."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _remove_existing(path: Path) -> None:
    """Remove a file, directory, or symlink without following symlinks."""
    if not os.path.lexists(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def link_or_copy(source: str | Path, destination: str | Path) -> str:
    """Replace destination with a symlink, falling back to a real copy.

    Windows commonly disables symlink creation for non-elevated users. The
    fallback keeps notebook behavior portable, at the cost of extra disk space.
    Returns ``"symlink"`` or ``"copy"`` so callers can report the choice.
    """
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().absolute()
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing(dst)

    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return "symlink"
    except (OSError, NotImplementedError):
        # A failed symlink call may leave a partial entry on some filesystems.
        _remove_existing(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copy"
