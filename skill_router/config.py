"""配置加载。

优先级：**真实环境变量 > `.env` 文件 > 代码默认值**。

`.env` 放真实配置（已 gitignore），仓库里追踪的是 `.env.example` 模板 —— 模板只有
键名与说明、没有值，所以密钥不会入库，新人也能一眼看懂要配什么。

库本身（`skill_router.skills` / `skill_router.recall`）不做任何自动加载，
读配置只发生在命令行入口，避免 import 时产生副作用。
"""

from __future__ import annotations

import os
from typing import Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENV_FILE = os.path.join(PROJECT_ROOT, ".env")


def parse_env(text: str) -> Dict[str, str]:
    """解析 `.env` 内容：``KEY=VALUE``，支持 ``#`` 注释、空行与引号包裹的值。"""
    values: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            values[key] = value.strip().strip("\"'")
    return values


def load_env(path: Optional[str] = None, override: bool = False) -> Dict[str, str]:
    """把配置文件载入环境变量，返回实际读到的键值对。

    默认**不覆盖**已存在的变量 —— 真实环境变量优先级最高，这样临时覆盖很方便：

        OPENAI_API_KEY=xxx python run_recall.py --dense

    配置文件位置：显式传入的 ``path`` > ``SKILL_ROUTER_ENV`` > 项目根 `.env`。
    """
    env_file = path or os.environ.get("SKILL_ROUTER_ENV") or DEFAULT_ENV_FILE
    if not os.path.isfile(env_file):
        return {}

    with open(env_file, "r", encoding="utf-8") as handle:
        values = parse_env(handle.read())

    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values
