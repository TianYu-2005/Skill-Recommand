"""技能推荐链路。

分层实现，逐层落地：

======  ==========  ================================================
层      模块        职责
======  ==========  ================================================
第 0 层  skills      技能加载：读 `SKILL.md`，产出 name / description
第 1 层  recall      召回：BM25 +（可选）向量 → RRF 融合
第 2 层  —           精排：Jev 并行判断相关度（待实现）
第 3 层  —           决策：排序 / 阈值 / 拒答 / 截断（待实现）
======  ==========  ================================================

当前可用的是第 0、1 层：给定一个技能库和一段请求，召回出 N′ 个候选。
"""

from .recall import (
    BM25,
    DenseRecall,
    Embedder,
    OpenAIEmbedder,
    RecallResult,
    Recaller,
    rrf,
    tokenize,
)
from .config import load_env
from .skills import Skill, load_skill, load_skills, parse_frontmatter

__all__ = [
    "BM25",
    "DenseRecall",
    "Embedder",
    "OpenAIEmbedder",
    "RecallResult",
    "Recaller",
    "Skill",
    "load_env",
    "load_skill",
    "load_skills",
    "parse_frontmatter",
    "rrf",
    "tokenize",
]
