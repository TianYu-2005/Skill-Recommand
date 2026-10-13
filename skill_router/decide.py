"""第 3 层 · 决策。

把模型给的概率变成最终结果。**这一步全是代码，不含任何模型判断** ——
模型负责回答「这条相关吗」，代码负责「要不要、给几条、什么时候拒答」。

    candidates + scores
      ├─ 1. 按概率降序
      ├─ 2. 过滤掉低于阈值的
      ├─ 3. 若结果为空 → 拒答（返回「无匹配」）
      └─ 4. 截断到 top_k

**拒答是必须的出口**：没有拒答能力的推荐系统一定会「硬凑」。

阈值必须自己标定 —— 不同模型的概率分布差别很大。实测 `decision-model-preview`
在 45 条候选上给出的最低分都在 0.38 以上，照抄 0.35 这种默认值等于永不拒答。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .skills import Skill


@dataclass
class Decision:
    """决策结果。

    ``filtered`` 是**过滤后、截断前**的条数。它与 ``len(skills)`` 的差值，就是被
    ``top_k`` 砍掉的部分 —— 只留后者会让这一步完全不可见（5 条过阈值、推 3 条，
    和 3 条过阈值、推 3 条，看起来会一模一样）。

    ``abstained`` 为真表示「无匹配」。它是**显式**的出口，而不是空结果的副作用 ——
    调用方应该据此告诉用户"没有合适的技能"，而不是什么都不说。
    """

    skills: List[Skill] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    abstained: bool = False
    filtered: int = 0

    def __len__(self) -> int:
        return len(self.skills)

    @property
    def ids(self) -> List[str]:
        return [skill.id for skill in self.skills]


def decide(
    candidates: Sequence[Skill],
    scores: Dict[str, float],
    *,
    top_k: int = 3,
    min_score: float = 0.0,
) -> Decision:
    """排序 → 阈值 → 拒答 → 截断。

    ``min_score`` 默认 0 表示不设阈值（有候选就给），真实阈值应当在标注集上标定。
    """
    ranked = sorted(candidates, key=lambda skill: -scores.get(skill.id, 0.0))
    kept = [
        (skill, scores.get(skill.id, 0.0))
        for skill in ranked
        if scores.get(skill.id, 0.0) >= min_score
    ]

    if not kept:
        return Decision(abstained=True)

    picked = kept[: max(1, top_k)]
    return Decision(
        skills=[skill for skill, _ in picked],
        scores=[value for _, value in picked],
        filtered=len(kept),
        abstained=False,
    )
