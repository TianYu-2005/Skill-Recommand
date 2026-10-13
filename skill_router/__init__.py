"""技能推荐链路。

分层实现，逐层可单独使用：

======  ==========  ==================================================
层      模块        职责
======  ==========  ==================================================
第 0 层  skills      技能加载：读 `SKILL.md`，产出 name / description
第 1 层  recall      召回：BM25 +（可选）向量 → RRF 融合
第 2 层  rerank      精排：并行判断每条候选的相关度
第 3 层  decide      决策：排序 / 阈值 / 拒答 / 截断
—       router      编排：串起三层，并带出每层的中间数据
—       server      演示服务：把链路摊开给网页看
======  ==========  ==================================================

最短用法：

    from skill_router import build_router

    router = build_router("data/clawhub")
    result = router.route("帮我把 pdf 简历压到一页")

    result.library_size       # 技能库总量
    result.recalled           # 召回候选数（稀疏 + 稠密，融合去重后）
    result.kept               # 过阈值后剩几条
    result.recommendations    # 最终推荐
"""

from .config import load_env
from .decide import Decision, decide
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
from .rerank import Judge, MockJudge, SystemOneJudge, get_judge
from .router import Candidate, RouteResult, Router, build_router
from .skills import Skill, load_skill, load_skills, parse_frontmatter

__all__ = [
    "BM25",
    "Candidate",
    "Decision",
    "DenseRecall",
    "Embedder",
    "Judge",
    "MockJudge",
    "OpenAIEmbedder",
    "RecallResult",
    "Recaller",
    "RouteResult",
    "Router",
    "Skill",
    "SystemOneJudge",
    "build_router",
    "decide",
    "get_judge",
    "load_env",
    "load_skill",
    "load_skills",
    "parse_frontmatter",
    "rrf",
    "tokenize",
]
