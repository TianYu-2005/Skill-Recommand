#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口：单条请求走完整三层链路。

    python run_route.py --skills data/clawhub --query "帮我把 pdf 简历压到一页"
    python run_route.py --skills data/clawhub --query "红烧肉怎么做" --min-score 0.4

显式插入项目根目录是刻意的：有些 Python 环境以隔离模式启动（忽略 ``PYTHONPATH``、
不把当前目录加进 ``sys.path``），不这样处理会报 ``ModuleNotFoundError``。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from skill_router.router import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
