"""`python -m yaro` 入口：等价于安装后的 `yaro` 命令。"""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
