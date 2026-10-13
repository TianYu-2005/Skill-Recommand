"""第 0 层 · 技能加载。

技能的本质是一个文件夹 + 一个 `SKILL.md`。整条链路只吃 name 与 description 两项，
所以这里的工作只有一件：把这两项从文件里可靠地抠出来。

支持两种技能来源：

* **目录**（递归查找 `SKILL.md`）—— 技能库的实际形态；
* **`skills.json`**（`[{"id", "name", "description"}]`）—— 便于测试与外部接入。

描述质量决定系统上限：`description` 写成「做什么 + 何时用 + 触发词」时，
召回与精排都最稳。这里只负责原样读取，不做任何加工。
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$")
_BLOCK_MARKERS = {"|", "|-", "|+", ">", ">-", ">+"}


@dataclass(frozen=True)
class Skill:
    """一个技能。

    ``id`` 取自技能文件夹名（冲突时退化为相对路径），是后续评测标注要引用的稳定标识。
    """

    id: str
    name: str
    description: str
    source: str = ""

    @property
    def text(self) -> str:
        """进索引的文本 —— 只有 name 和 description，索引对象就是它俩的原文。"""
        return f"{self.name}\n{self.description}".strip()

    def __str__(self) -> str:  # pragma: no cover - 仅用于打印
        return f"{self.id}({self.name})"


def parse_frontmatter(text: str) -> Dict[str, Any]:
    """取 `SKILL.md` 头部的 YAML frontmatter。

    装了 pyyaml 就用它；没装则退回下面的极简解析器 —— 只认 `key: value` 与块标量，
    够 name / description 用，不追求 YAML 完备。整条链路因此没有强制依赖。
    """
    # 带 BOM 的文件（Windows 编辑器常见）会让 \A 锚点匹配不上，先剥掉
    if text.startswith("\ufeff"):
        text = text[1:]
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    raw = match.group(1)
    try:
        import yaml  # type: ignore
    except ImportError:
        return _minimal_yaml(raw)
    try:
        data = yaml.safe_load(raw)
    except Exception:
        return _minimal_yaml(raw)
    return data if isinstance(data, dict) else {}


def _minimal_yaml(raw: str) -> Dict[str, str]:
    """无 pyyaml 时的回退解析器：处理顶层 `key: value` 与块标量。

    块标量按 YAML 语义区分：``|`` 保留换行（literal），``>`` 折叠成空格（folded）。
    结果是给召回用的文本，两者最终都会被折成单行，但解析本身不该含糊。
    """
    out: Dict[str, str] = {}
    key: Optional[str] = None
    buf: List[str] = []
    folded = False

    def flush() -> None:
        nonlocal folded
        if key is not None:
            out[key] = (" ".join(buf) if folded else "\n".join(buf)).strip()
        folded = False

    for line in raw.splitlines():
        match = _KEY_RE.match(line)
        if match and not line[:1].isspace():
            flush()
            key = match.group(1)
            value = match.group(2).strip()
            if value in _BLOCK_MARKERS:
                folded = value.startswith(">")
                buf = []
            else:
                buf = [value.strip("\"'")]
        elif key is not None and line.strip():
            buf.append(line.strip())
    flush()
    return out


def _read_text(path: str) -> str:
    """按常见编码读文本。

    真实技能库里的 `SKILL.md` 编码并不统一 —— 用 GBK 写中文描述的相当常见。
    一个文件读不动就整个库加载失败，代价太大，所以这里逐级退让，
    最后用 ``replace`` 兜底，保证任何字节序列都能读出一个字符串。
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_skill(skill_md: str, root: Optional[str] = None) -> Skill:
    """读一个 `SKILL.md`，产出 :class:`Skill`。"""
    folder = os.path.basename(os.path.dirname(os.path.abspath(skill_md)))
    meta = parse_frontmatter(_read_text(skill_md))

    skill_id = folder
    if root:
        rel = os.path.relpath(os.path.dirname(os.path.abspath(skill_md)), root)
        rel = rel.replace(os.sep, "/")
        if rel not in (".", ""):
            skill_id = f"{rel}" if os.path.basename(rel) == folder else f"{rel}/{folder}"

    name = str(meta.get("name") or folder).strip()
    # name 不该是多行。出现换行说明上游把后续字段错误地缩进了 —— 在 YAML 里它们会
    # 变成 name 的续行，不截断就会把一整段 frontmatter 灌进索引。
    if "\n" in name:
        name = name.splitlines()[0].strip()

    description = " ".join(str(meta.get("description") or "").split())
    return Skill(id=skill_id, name=name or folder, description=description, source=skill_md)


def load_skills(path: str) -> List[Skill]:
    """从目录（递归找 `SKILL.md`）或 `skills.json` 加载技能库。"""
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return _load_json(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"技能库路径不存在: {path}")

    root = os.path.abspath(path)
    skills: List[Skill] = []
    seen: Dict[str, int] = {}
    for skill_md in sorted(glob.glob(os.path.join(root, "**", "SKILL.md"), recursive=True)):
        skill = load_skill(skill_md, root)
        if skill.id in seen:
            # 文件夹同名：用相对路径消歧，保证 id 在库内唯一
            skill = Skill(skill.id, skill.name, skill.description, skill.source)
            seen[f"{skill.id}#{len(skills)}"] = len(skills)
        seen[skill.id] = len(skills)
        skills.append(skill)
    return skills


def _load_json(path: str) -> List[Skill]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"skills.json 必须是数组: {path}")

    skills: List[Skill] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"skills.json 第 {index} 项不是对象: {path}")
        skill_id = str(row.get("id") or f"skill-{index}")
        skills.append(
            Skill(
                id=skill_id,
                name=str(row.get("name") or skill_id),
                description=" ".join(str(row.get("description") or "").split()),
                source=path,
            )
        )
    return skills
