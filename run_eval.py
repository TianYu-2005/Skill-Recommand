#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估入口。

    python run_eval.py --skills data/clawhub --eval data/eval.jsonl
    python run_eval.py --skills data/clawhub --eval data/eval.jsonl --sweep 0.5,0.7,0.8,0.9

显式插入项目根目录是刻意的：有些 Python 环境以隔离模式启动（忽略 ``PYTHONPATH``、
不把当前目录加进 ``sys.path``），不这样处理会报 ``ModuleNotFoundError``。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from skill_router.evaluate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
