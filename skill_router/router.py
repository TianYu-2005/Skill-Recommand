"""整条链路的编排层。

把第 1、2、3 层串起来，并且**把每一层的中间数据一并带出来**。
只给最终推荐的话，结果不对时只能看到「推荐错了」，看不到「哪一层错了」。

    Router(skills).route("帮我把 pdf 简历压到一页")
      ├─ 第 1 层 召回  → 候选 49 个（稀疏 10 + 稠密 40，融合去重后 49）
      ├─ 第 2 层 精排  → 49 条打分
      └─ 第 3 层 决策  → 过阈值 3 条 / 或拒答

分成两部分：
- :class:`Router` 跑链路，**无状态**，可并发调用，服务端持有一个实例即可；
- :class:`RouteResult` 是结果快照，``to_dict()`` 后直接喂给 HTTP 接口或前端。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import load_env
from .decide import decide
from .recall import DenseRecall, Recaller, build_dense_channel
from .rerank import Judge, get_judge
from .skills import Skill, load_skills

DEFAULT_PREVIEW = 30
DESC_LIMIT = 260


@dataclass
class Candidate:
    """一条候选在整个链路里的履历 —— 前端靠它把漏斗的每一格填满。"""

    id: str
    name: str
    description: str
    score: float = 0.0
    recalled_by: List[str] = field(default_factory=list)  # sparse / dense / full
    recall_rank: int = 0  # 融合后的名次，从 1 起
    picked: bool = False  # 是否进了最终推荐


@dataclass
class RouteResult:
    """一次完整路由的结果快照。"""

    query: str
    library_size: int
    top_k: int = 3
    min_score: float = 0.0
    mode: str = "empty"  # full-injection / hybrid / sparse / dense / empty
    sparse_count: int = 0
    dense_count: int = 0
    recalled: int = 0  # 融合去重后的候选数
    scored: int = 0  # 精排实际打分的条数
    kept: int = 0  # 过阈值后剩下的条数（**截断前**，与推荐数可能不等）
    abstained: bool = False
    candidates: List[Candidate] = field(default_factory=list)
    recommendations: List[Candidate] = field(default_factory=list)
    elapsed_ms: int = 0
    judge: str = "-"
    judge_model: str = "-"
    dense_model: Optional[str] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _sources(recalled) -> Dict[str, List[str]]:
    """每个候选被哪几路捞了回来。全量注入没有通路可言，统一记为 full。"""
    sources: Dict[str, List[str]] = {}
    for channel, ids in (recalled.channels or {}).items():
        for skill_id in ids:
            sources.setdefault(skill_id, []).append(channel)
    if not sources:
        for skill in recalled.skills:
            sources[skill.id] = ["full"]
    return sources


def _candidate(skill, scores, sources, rank_of, picked) -> Candidate:
    return Candidate(
        id=skill.id,
        name=skill.name,
        description=skill.description[:DESC_LIMIT],
        score=round(scores.get(skill.id, 0.0), 4),
        recalled_by=sources.get(skill.id, []),
        recall_rank=rank_of.get(skill.id, 0),
        picked=skill.id in picked,
    )


def _ranked(
    skills: Sequence[Skill], scores: Dict[str, float]
) -> Tuple[List[Skill], Dict[str, int]]:
    """按精排分降序排一次，并给出 id → 名次（从 1 起）。

    缺分数的候选按 0 处理、不丢弃 —— 丢弃会静默改变候选集，让「它为什么不见了」
    变成无从追查的事。
    """
    ordered = sorted(skills, key=lambda skill: -scores.get(skill.id, 0.0))
    return ordered, {skill.id: position for position, skill in enumerate(ordered, 1)}


class Router:
    """三层链路的编排器。无状态，可并发调用。"""

    def __init__(
        self,
        skills: Sequence[Skill],
        *,
        sparse_k: int = 10,
        dense_k: int = 40,
        small_n: int = 60,
        dense: Optional[DenseRecall] = None,
        judge: Optional[Judge] = None,
        top_k: int = 3,
        min_score: float = 0.0,
        preview: int = DEFAULT_PREVIEW,
    ):
        self.skills = list(skills)
        self.top_k = top_k
        self.min_score = min_score
        self.preview = preview
        self.recaller = Recaller(
            self.skills, sparse_k=sparse_k, dense_k=dense_k, small_n=small_n, dense=dense
        )
        self.judge = judge if judge is not None else get_judge()

    @property
    def dense_model(self) -> Optional[str]:
        embedder = getattr(self.recaller.dense, "_embedder", None)
        return getattr(embedder, "model", None)

    def warmup(self) -> None:
        """把稠密通路的全库向量先算出来。

        向量是惰性构建的，不预热的话**第一个真实请求**要等全库 embedding 走完。
        """
        dense = self.recaller.dense
        if dense is not None and dense.available:
            dense._ensure_vectors()  # noqa: SLF001 - 预热就是它的用途

    def route(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RouteResult:
        """跑完整三层。``top_k`` / ``min_score`` 传了就覆盖实例默认值。

        逐次覆盖是给前端做对比用的 —— 同一段请求换个阈值，立刻能看到不同结果。
        """
        top_k = self.top_k if top_k is None else top_k
        min_score = self.min_score if min_score is None else min_score

        started = time.perf_counter()
        result = RouteResult(
            query=(query or "").strip(),
            library_size=len(self.skills),
            top_k=top_k,
            min_score=min_score,
            judge=type(self.judge).__name__,
            judge_model=str(getattr(self.judge, "model", "-")),
            dense_model=self.dense_model,
        )
        if not result.query:
            result.error = "请求为空"
            return result

        recalled = self.recaller.recall(result.query)
        result.mode = recalled.mode
        result.sparse_count = len(recalled.channels.get("sparse", ()))
        result.dense_count = len(recalled.channels.get("dense", ()))
        result.recalled = len(recalled.skills)

        sources = _sources(recalled)
        scores: Dict[str, float] = {}
        if recalled.skills:
            try:
                scores = self.judge.score(result.query, recalled.skills)
            except Exception as error:  # noqa: BLE001 - 精排是外部调用，失败不该让整条链路炸掉
                # 召回结果照样返回：能看到「捞到了 45 条」，比只报一句失败有用。
                # 但**不能置 abstained** —— 那会把「精排坏了」误报成「没有合适技能」。
                result.error = f"精排失败：{error}"
                ranked, rank_of = _ranked(recalled.skills, {})
                result.candidates = [
                    _candidate(skill, {}, sources, rank_of, set())
                    for skill in ranked[: self.preview]
                ]
                result.elapsed_ms = _ms(started)
                return result

        result.scored = len(scores)

        decision = decide(recalled.skills, scores, top_k=top_k, min_score=min_score)
        result.abstained = decision.abstained
        result.kept = decision.filtered

        ranked, rank_of = _ranked(recalled.skills, scores)
        picked = set(decision.ids)

        # 候选池也带上 picked：前端据此区分「推荐」「过了阈值但被 top_k 截断」
        result.candidates = [
            _candidate(skill, scores, sources, rank_of, picked)
            for skill in ranked[: self.preview]
        ]
        result.recommendations = [
            _candidate(skill, scores, sources, rank_of, picked) for skill in decision.skills
        ]
        result.elapsed_ms = _ms(started)
        return result


def build_router(
    skills_dir: str,
    *,
    top_k: int = 3,
    min_score: float = 0.0,
    dense: Optional[bool] = None,
    sparse_k: int = 10,
    dense_k: int = 40,
    small_n: int = 60,
    preview: int = DEFAULT_PREVIEW,
) -> Router:
    """按当前环境组装 Router：加载技能库，条件满足时接上稠密通路。

    ``dense`` 显式传入时优先；为 ``None`` 时看 ``USE_EMBED``。
    没有 embedding key 就静默退化为纯 BM25，链路照跑。
    """
    skills = load_skills(skills_dir)
    if dense is None:
        dense = os.environ.get("USE_EMBED") == "1"

    channel = build_dense_channel([skill.text for skill in skills], dense)

    return Router(
        skills,
        sparse_k=sparse_k,
        dense_k=dense_k,
        small_n=small_n,
        dense=channel,
        top_k=top_k,
        min_score=min_score,
        preview=preview,
    )


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def _recall_line(result: RouteResult) -> str:
    if result.mode == "full-injection":
        return f"全量注入（库小，跳过检索）→ 候选 {result.recalled}"
    if result.mode == "hybrid":
        return (
            f"混合 · 稀疏 {result.sparse_count} + 稠密 {result.dense_count}"
            f" → 融合去重 {result.recalled}"
        )
    if result.mode in ("sparse", "dense"):
        label = "稀疏" if result.mode == "sparse" else "稠密"
        return f"{label} → 候选 {result.recalled}"
    return f"无候选（{result.mode}）"


def format_result(result: RouteResult) -> str:
    lines = [
        f"请求    : {result.query}",
        f"技能库  : {result.library_size} 个技能",
        f"召回    : {_recall_line(result)}",
        f"精排    : {result.judge} / {result.judge_model} → {result.scored} 条打分",
    ]
    if result.abstained:
        lines.append(f"决策    : 阈值 {result.min_score:.2f} → 拒答（无匹配）")
    else:
        lines.append(
            f"决策    : 阈值 {result.min_score:.2f} → 过阈值 {result.kept} 条，"
            f"推荐 {len(result.recommendations)} 条"
        )
    lines.append(f"耗时    : {result.elapsed_ms} ms")
    if result.error:
        lines.append(f"错误    : {result.error}")
    lines.append("-" * 74)

    if result.recommendations:
        for position, item in enumerate(result.recommendations, 1):
            sources = "+".join(item.recalled_by) or "-"
            lines.append(
                f"  推荐 {position}  {item.score:.3f}  {item.id}   [召回#{item.recall_rank} {sources}]"
            )
            if item.description:
                lines.append(f"            {item.description[:70]}")
    elif not result.error:
        lines.append("  无推荐 —— 所有候选都低于阈值。")

    if result.candidates:
        lines.append("")
        lines.append(f"  候选池前 {len(result.candidates)} 条：")
        for item in result.candidates:
            mark = "✓" if item.score >= result.min_score else " "
            lines.append(f"   {mark} {item.score:.3f}  {item.id}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env()

    parser = argparse.ArgumentParser(
        prog="python run_route.py",
        description="单条请求走完整三层链路",
    )
    parser.add_argument("--skills", default=os.environ.get("SKILLS_DIR", "data/skills"))
    parser.add_argument("--query", required=True, help="用户请求")
    parser.add_argument(
        "--top-k", type=int, default=int(os.environ.get("TOP_K", 3)), help="最终推荐条数"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=float(os.environ.get("MIN_SCORE", "0.4")),
        help="拒答阈值，由评测标定（默认 0.4）",
    )
    parser.add_argument("--sparse-k", type=int, default=int(os.environ.get("SPARSE_K", 10)))
    parser.add_argument("--dense-k", type=int, default=int(os.environ.get("DENSE_K", 40)))
    parser.add_argument("--small-n", type=int, default=int(os.environ.get("SMALL_N", 60)))
    parser.add_argument(
        "--dense", action="store_true", default=os.environ.get("USE_EMBED") == "1"
    )
    parser.add_argument("--no-dense", dest="dense", action="store_false")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = parser.parse_args(argv)

    try:
        router = build_router(
            args.skills,
            top_k=args.top_k,
            min_score=args.min_score,
            dense=args.dense,
            sparse_k=args.sparse_k,
            dense_k=args.dense_k,
            small_n=args.small_n,
        )
    except FileNotFoundError as error:
        print(f"加载失败：{error}", file=sys.stderr)
        return 2

    if not router.skills:
        print(f"技能库为空：{args.skills}", file=sys.stderr)
        return 2

    result = router.route(args.query)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_result(result))
    return 1 if result.error else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
