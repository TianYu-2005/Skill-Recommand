#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动本地演示服务。

    python run_server.py --skills data/clawhub --port 8000

显式插入项目根目录是刻意的：有些 Python 环境以隔离模式启动（忽略 ``PYTHONPATH``、
不把当前目录加进 ``sys.path``），不这样处理会报 ``ModuleNotFoundError``。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from skill_router.server import main  # noqa: E402

# 推荐条数与拒答阈值由后端配置（TOP_K / MIN_SCORE），入口脚本不暴露它们

if __name__ == "__main__":
    raise SystemExit(main())
