"""PyInstaller 打包入口（冻结成 YARO.exe 后从这里启动）。

行为：
- 双击 ``YARO.exe``（没有任何参数）→ 直接进 ``yaro app``（开窗口）；
- 带参数运行 ``YARO.exe gui`` / ``YARO.exe doctor`` 等 → 等价于源码下的 ``python -m yaro ...``。

这个文件不属于 ``yaro`` 包，只在打包时用到；运行时通过 ``packaging/yaro.spec``
把它作为入口脚本。
"""

from __future__ import annotations

import sys

from yaro.cli import main

if __name__ == "__main__":
    # 双击时 sys.argv 只有 exe 自己：默认给 app，避免出现"闪一下就没了"。
    argv = ["app"] if len(sys.argv) <= 1 else sys.argv[1:]
    raise SystemExit(main(argv))
