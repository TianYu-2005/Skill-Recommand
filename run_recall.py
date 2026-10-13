#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 1 层召回的免安装入口。

    python run_recall.py --skills data/skills --query "帮我把 pdf 简历压到一页"

包安装后（``pip install -e .``）也可以直接 ``python -m skill_router.recall``
或使用 ``skill-recall`` 命令。

显式插入项目根目录是刻意的：有些 Python 环境以隔离模式启动（忽略 ``PYTHONPATH``、
不把当前目录加进 ``sys.path``），不这样处理会报 ``ModuleNotFoundError: skill_router``。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from skill_router.recall import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
