"""第 2 层 · 精排。

第 1 层把 N 收窄到几十条候选，这一层逐条判断「它是不是完成该请求的合适技能」。
模型**只给概率，不排序** —— 排序、阈值、拒答都是第 3 层代码的事。

协议是 **TypeSafe System One**：一次请求挂多个 noul 问题，服务端并行求值。
兼容该协议的服务都能接，换供应商只改 ``JUDGE_BASE_URL``：

===========================  ===================================================================
TypeSafe 官方                 ``https://api.typesafe.ai/v1``
阿里云百炼（decision-model）   ``https://{workspace}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1``
本地 Ollama 0.35+             ``http://localhost:11434/v1``
===========================  ===================================================================

**绝不逐候选发请求** —— 那是这条链路上最贵的错误。一次挂 N 个问题的边际成本几乎为零，
逐条发则延迟随候选数线性增长。候选过多时按 ``batch`` 分批**并发**，而不是串行。

模型侧的两条已知限制（来自官方 jaggedness 文档），设计时已经绕开：

1. **context rot** —— state 里的无关内容会直接损失准确率，所以每批只放该批候选；
2. **字面化理解** —— 问题按字面解读，所以 instructions 要写全，别指望它揣摩意图。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Protocol, Sequence

from .skills import Skill

SYSTEMONE_SUFFIX = "/systemone"


class Judge(Protocol):
    """判断器接口：给定请求与候选，返回 ``{技能 id: 相关概率 0~1}``。"""

    def score(self, query: str, candidates: Sequence[Skill]) -> Dict[str, float]:
        ...


def _batch(items: Sequence[Skill], size: int) -> List[Sequence[Skill]]:
    """按 size 切块。候选多于单批上限时靠它分批并发。"""
    return [items[index : index + size] for index in range(0, len(items), size)]


class SystemOneJudge:
    """TypeSafe System One 协议的客户端（裸 HTTP，零依赖）。

    参数：
        base_url：含 ``/v1`` 前缀的地址。完整端点 = ``base_url`` + ``/systemone``。
        api_key： Bearer token。
        model：  模型名。TypeSafe 用 ``jev-latest``，百炼用 ``decision-model-preview``。
        batch：  单次请求最多挂几个问题。官方建议 ≤16，延迟随问题数近线性增长。
    """

    DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch: int = 16,
        timeout: float = 60.0,
    ):
        self.base_url = (
            base_url or os.environ.get("JUDGE_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key = (
            api_key
            or os.environ.get("JUDGE_API_KEY")
            or os.environ.get("TYPESAFE_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or ""
        )
        self.model = model or os.environ.get("JUDGE_MODEL", "jev-latest")
        self.batch = max(1, batch)
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> Optional["SystemOneJudge"]:
        """没有 key 时返回 ``None`` —— 精排缺席，由调用方决定降级。"""
        if not (
            os.environ.get("JUDGE_API_KEY")
            or os.environ.get("TYPESAFE_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        ):
            return None
        return cls()

    def score(self, query: str, candidates: Sequence[Skill]) -> Dict[str, float]:
        if not candidates:
            return {}

        batches = _batch(list(candidates), self.batch)
        scores: Dict[str, float] = {}

        # 批数即并发数：总延迟约等于单批，而不是各批之和
        with ThreadPoolExecutor(max_workers=len(batches)) as pool:
            for chunk in pool.map(lambda part: self._one_batch(query, part), batches):
                scores.update(chunk)
        return scores

    def _one_batch(self, query: str, candidates: Sequence[Skill]) -> Dict[str, float]:
        # state 只放请求本身，候选进 instructions —— 这是官方 skill_suggestion
        # cookbook 的做法：让每个问题**自带上下文**（"它被描述为……"），
        # 而不是让模型去 state 的数组里按路径定位。
        #
        # 两个理由：
        #   1. 问题 ID 不会传给模型，指代越少越不容易错；
        #   2. jaggedness 文档点名的 context rot —— state 越大越容易分心，
        #      把候选堆在 state 里正是它警告的"充满无关细节的大 state"。
        state = {"query": query}
        questions = {
            f"cand_{index}": {
                "type": "noul",
                # 措辞是实测调出来的："是否是合适选择"比"能否直接完成"召回更好
                # （Recall@3 100% vs 92.9%），加 criteria 收紧反而只伤自己人。
                # 拒答能力不靠收紧措辞拿 —— 那是 gate 那一层该管的事。
                "instructions": (
                    f"技能「{skill.name}」是否是完成用户请求「{query}」的合适选择？"
                    f"它被描述为：{skill.description}"
                ),
            }
            for index, skill in enumerate(candidates)
        }

        body = json.dumps(
            {"model": self.model, "state": state, "questions": questions},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{SYSTEMONE_SUFFIX}",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"精排请求失败: HTTP {error.code} {detail}") from error

        answers = payload.get("answers") or {}
        return {
            skill.id: float((answers.get(f"cand_{index}") or {}).get("noul") or 0.0)
            for index, skill in enumerate(candidates)
        }


class MockJudge:
    """本地词重叠模拟器：不需要任何 key，只用来验证链路能不能跑通。

    概率**不具备参考价值** —— 它只会做词面重叠折算，跨语言场景下给不出有意义的分。
    真实判断质量完全取决于接入的决策模型。
    """

    def score(self, query: str, candidates: Sequence[Skill]) -> Dict[str, float]:
        from .recall import tokenize

        query_terms = set(tokenize(query))
        if not query_terms:
            return {skill.id: 0.0 for skill in candidates}

        scores: Dict[str, float] = {}
        for skill in candidates:
            overlap = len(query_terms & set(tokenize(skill.text)))
            scores[skill.id] = min(1.0, overlap / len(query_terms))
        return scores


def get_judge() -> Judge:
    """按配置返回可用的判断器：真实的优先，拿不到就退回模拟器。"""
    judge = SystemOneJudge.from_env()
    if judge is not None:
        return judge
    return MockJudge()
