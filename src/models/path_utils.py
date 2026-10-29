"""Hàm hỗ trợ hệ thống tệp cho notebook chạy trên Windows, macOS và Linux."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _remove_existing(path: Path) -> None:
    """Xóa tệp, thư mục hoặc liên kết tượng trưng mà không lần theo liên kết."""
    if not os.path.lexists(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def link_or_copy(source: str | Path, destination: str | Path) -> str:
    """Thay đích đến bằng liên kết tượng trưng hoặc sao chép nếu không tạo được.

    Windows thường không cho người dùng thường tạo liên kết tượng trưng. Cách
    dự phòng giúp notebook hoạt động trên nhiều hệ điều hành nhưng tốn thêm dung
    lượng. Trả về ``"symlink"`` hoặc ``"copy"`` để bên gọi thông báo lựa chọn.
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
        # Trên một số hệ thống tệp, thao tác tạo liên kết lỗi vẫn có thể để lại mục dở dang.
        _remove_existing(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copy"
