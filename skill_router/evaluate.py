"""评估：把「感觉」换成「数字」。

没有评估，所有调优都是拍脑袋。这个模块用一份标注集量三件事：

1. **召回闸门** —— 正解有没有进候选池（``Recall@召回``）。漏了就彻底没戏。
2. **端到端** —— 最终结果里有没有正解、排得够不够前（``Recall@top_k`` / ``MRR``）。
3. **拒答** —— 该说「没有」的时候有没有说。

两级指标配合着看，能定位问题出在哪一层：

=================  ==========================================
闸门高、端到端低    精排或阈值的问题
闸门本身就低        召回的问题（上一层没捞到，后面怎么调都白搭）
=================  ==========================================

标注集格式（``eval.jsonl``，每行一条）：

    {"query": "帮我把简历压到一页", "expect": ["resume-skill-id"]}

``expect`` 为空数组表示「这条请求不该匹配任何技能」，用来衡量拒答能力。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import load_env
from .decide import Decision, decide
from .recall import RecallResult, Recaller, build_dense_channel
from .rerank import Judge, get_judge
from .skills import Skill, load_skills


@dataclass
class EvalRow:
    query: str
    expect: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """一次评估的原始计数，比率用属性算，避免出现除零。"""

    min_score: float = 0.0
    total: int = 0
    with_expect: int = 0
    no_expect: int = 0
    recall_hit: int = 0
    topk_hit: int = 0
    reciprocal_rank: float = 0.0
    abstain_ok: int = 0
    false_reject: int = 0
    misses: List[str] = field(default_factory=list)

    @property
    def recall_rate(self) -> float:
        return self.recall_hit / self.with_expect if self.with_expect else 0.0

    @property
    def topk_rate(self) -> float:
        return self.topk_hit / self.with_expect if self.with_expect else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank / self.with_expect if self.with_expect else 0.0

    @property
    def abstain_rate(self) -> float:
        return self.abstain_ok / self.no_expect if self.no_expect else 0.0


def load_eval(path: str) -> List[EvalRow]:
    """读标注集。每行一个 JSON 对象。"""
    rows: List[EvalRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON: {error}") from error
            expect = payload.get("expect") or []
            rows.append(
                EvalRow(
                    query=str(payload.get("query") or "").strip(),
                    expect=[str(item) for item in expect] if isinstance(expect, list) else [str(expect)],
                )
            )
    return [row for row in rows if row.query]


@dataclass
class _Cached:
    """一条标注的中间结果：召回池 + 精排概率。

    阈值扫描时复用它 —— 召回和精排是花钱的，不该为每个阈值各跑一遍。
    """

    row: EvalRow
    recalled: RecallResult
    scores: Dict[str, float]


def run_pipeline(recaller: Recaller, judge: Judge, rows: Sequence[EvalRow]) -> List[_Cached]:
    """把召回与精排跑一遍，结果缓存下来供不同阈值复用。"""
    cached: List[_Cached] = []
    for index, row in enumerate(rows, 1):
        print(f"\r  召回+精排 {index}/{len(rows)} …", end="", file=sys.stderr, flush=True)
        recalled = recaller.recall(row.query)
        scores = judge.score(row.query, recalled.skills) if recalled.skills else {}
        cached.append(_Cached(row, recalled, scores))
    print("\r" + " " * 40 + "\r", end="", file=sys.stderr, flush=True)
    return cached


def score_cached(cached: Sequence[_Cached], *, top_k: int, min_score: float) -> EvalReport:
    """在缓存结果上按给定阈值打分。"""
    report = EvalReport(min_score=min_score)

    for item in cached:
        row, recalled = item.row, item.recalled
        report.total += 1
        decision: Decision = decide(recalled.skills, item.scores, top_k=top_k, min_score=min_score)

        # 该拒答的：只看有没有拒答
        if not row.expect:
            report.no_expect += 1
            if decision.abstained:
                report.abstain_ok += 1
            else:
                report.misses.append(f"[未拒答] {row.query} → {decision.ids[:2]}")
            continue

        report.with_expect += 1

        # 闸门：正解进候选池了吗
        if set(row.expect) & set(recalled.ids):
            report.recall_hit += 1
        else:
            report.misses.append(f"[召回漏] {row.query}")

        # 端到端
        if decision.abstained:
            report.false_reject += 1
            report.misses.append(f"[误拒答] {row.query}")
            continue

        hits = [position for position, skill_id in enumerate(decision.ids) if skill_id in row.expect]
        if hits:
            report.topk_hit += 1
            report.reciprocal_rank += 1.0 / (hits[0] + 1)
        else:
            report.misses.append(f"[精排错] {row.query} → {decision.ids[:2]}")

    return report


def evaluate(
    recaller: Recaller,
    judge: Judge,
    rows: Sequence[EvalRow],
    *,
    top_k: int = 3,
    min_score: float = 0.0,
) -> EvalReport:
    """跑一遍完整链路并打分。"""
    return score_cached(run_pipeline(recaller, judge, rows), top_k=top_k, min_score=min_score)


def sweep(
    recaller: Recaller,
    judge: Judge,
    rows: Sequence[EvalRow],
    thresholds: Sequence[float],
    *,
    top_k: int = 3,
) -> List[EvalReport]:
    """阈值扫描：召回与精排只跑一次，对每个阈值重新决策。"""
    cached = run_pipeline(recaller, judge, rows)
    return [score_cached(cached, top_k=top_k, min_score=t) for t in thresholds]


def format_report(report: EvalReport, top_k: int) -> str:
    lines = [
        f"阈值 min_score={report.min_score}",
        f"  召回闸门 Recall@召回 : {report.recall_hit}/{report.with_expect} = {report.recall_rate:.1%}",
        f"  端到端   Recall@top{top_k}  : {report.topk_hit}/{report.with_expect} = {report.topk_rate:.1%}",
        f"  MRR                 : {report.mrr:.3f}",
        f"  拒答     {report.abstain_ok}/{report.no_expect} = {report.abstain_rate:.1%}",
        f"  误拒答              : {report.false_reject}",
    ]
    if report.misses:
        lines.append("  问题样本（前 8 条）:")
        lines.extend(f"    {item}" for item in report.misses[:8])
    return "\n".join(lines)


def _build_recaller(args: argparse.Namespace, skills: Sequence[Skill]) -> Recaller:
    dense = build_dense_channel([skill.text for skill in skills], args.dense)
    if args.dense and dense is None:
        print("提示：未检测到 embedding key，本次仅用 BM25。", file=sys.stderr)
    return Recaller(
        skills,
        sparse_k=args.sparse_k,
        dense_k=args.dense_k,
        small_n=args.small_n,
        dense=dense,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env()

    parser = argparse.ArgumentParser(
        prog="python run_eval.py",
        description="在标注集上评估整条链路",
    )
    parser.add_argument("--skills", default=os.environ.get("SKILLS_DIR", "data/skills"))
    parser.add_argument("--eval", dest="eval_path", required=True, help="标注集 jsonl")
    parser.add_argument(
        "--top-k", type=int, default=int(os.environ.get("TOP_K", 3)), help="最终推荐条数"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=float(os.environ.get("MIN_SCORE", "0.4")),
        help="拒答阈值；配合 --sweep 扫描时会被逐个覆盖",
    )
    parser.add_argument("--sweep", default="", help="逗号分隔的阈值列表，做扫描")
    parser.add_argument("--sparse-k", type=int, default=int(os.environ.get("SPARSE_K", 10)))
    parser.add_argument("--dense-k", type=int, default=int(os.environ.get("DENSE_K", 40)))
    parser.add_argument("--small-n", type=int, default=int(os.environ.get("SMALL_N", 60)))
    parser.add_argument(
        "--dense", action="store_true", default=os.environ.get("USE_EMBED") == "1"
    )
    args = parser.parse_args(argv)

    try:
        skills = load_skills(args.skills)
        rows = load_eval(args.eval_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"加载失败：{error}", file=sys.stderr)
        return 2

    if not skills or not rows:
        print("技能库或标注集为空。", file=sys.stderr)
        return 2

    recaller = _build_recaller(args, skills)
    judge = get_judge()

    print(f"技能库  : {args.skills}（{len(skills)} 个技能）")
    print(f"标注集  : {args.eval_path}（{len(rows)} 条，其中 {sum(1 for r in rows if not r.expect)} 条应拒答）")
    print(f"精排    : {type(judge).__name__} / {getattr(judge, 'model', '-')}")
    print("-" * 72)

    if args.sweep:
        thresholds = [float(item) for item in args.sweep.split(",") if item.strip()]
        reports = sweep(recaller, judge, rows, thresholds, top_k=args.top_k)
    else:
        reports = [evaluate(recaller, judge, rows, top_k=args.top_k, min_score=args.min_score)]

    print()
    for report in reports:
        print(format_report(report, args.top_k))
        print()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
