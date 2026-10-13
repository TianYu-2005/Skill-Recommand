"""编排层的测试。不需要网络，也不需要 API key。

编排层除了串联三层，还承担几条跨层约定 —— 「精排坏了不能误报成拒答」、
「过阈值数要在截断前统计」这类。它们都是只在真实调用时才暴露的规则，
所以值得单独锁住。
"""

from __future__ import annotations

import unittest

from skill_router.rerank import MockJudge
from skill_router.router import Router
from skill_router.skills import Skill

QUERY = "把简历压缩成 PDF"


class BoomJudge:
    """精排不可用时的替身。"""

    model = "boom"

    def score(self, query, candidates):
        raise RuntimeError("judge 挂了")


def library():
    return [
        Skill(id="pdf", name="PDF 简历压缩", description="把多页简历压缩成一页，导出 PDF"),
        Skill(id="xlsx", name="Excel 修复", description="修复损坏的 Excel 表格"),
        Skill(id="tts", name="文字转语音", description="把文字合成语音，支持多种音色"),
        Skill(id="git", name="Git 助手", description="撤销提交、查看历史、解决冲突"),
    ]


def build(**options) -> Router:
    options.setdefault("judge", MockJudge())
    return Router(library(), **options)


class TestStageReporting(unittest.TestCase):
    def test_reports_every_stage(self):
        result = build().route(QUERY)

        self.assertEqual(result.library_size, 4)
        self.assertEqual(result.recalled, 4)
        self.assertEqual(result.scored, 4)
        self.assertGreaterEqual(result.kept, 1)
        self.assertEqual(len(result.recommendations), 3)

    def test_small_library_skips_retrieval(self):
        """库不超过 small_n 时全量进下一层，检索这一步整个省掉。"""
        result = build().route(QUERY)
        self.assertEqual(result.mode, "full-injection")

    def test_forced_retrieval_narrows_candidates(self):
        result = build(small_n=1).route(QUERY)
        self.assertEqual(result.mode, "sparse")
        self.assertLess(result.recalled, 4)

    def test_blank_query_reports_error(self):
        result = build().route("   ")
        self.assertTrue(result.error)
        self.assertEqual(result.recommendations, [])


class TestDegradation(unittest.TestCase):
    def test_judge_failure_is_not_reported_as_abstain(self):
        """精排坏了要降级，**不能**误报成「没有合适技能」——两者语义完全不同。"""
        result = build(judge=BoomJudge()).route(QUERY)

        self.assertIn("精排失败", result.error)
        self.assertFalse(result.abstained)
        self.assertEqual(result.recommendations, [])
        self.assertEqual(result.scored, 0)

    def test_judge_failure_still_exposes_candidates(self):
        """召回结果照常返回：能看到「捞到了 4 条」，比只报一句失败有用。"""
        result = build(judge=BoomJudge()).route(QUERY)
        self.assertEqual(len(result.candidates), 4)


class TestDecisionPassThrough(unittest.TestCase):
    def test_top_k_is_honored(self):
        result = build().route(QUERY, top_k=1)
        self.assertEqual(len(result.recommendations), 1)

    def test_threshold_can_abstain(self):
        result = build().route("红烧肉怎么做", min_score=0.1)
        self.assertTrue(result.abstained)
        self.assertEqual(result.recommendations, [])

    def test_kept_counts_before_truncation(self):
        """kept 是过阈值数（截断前），与推荐数不是一回事。"""
        result = build().route(QUERY, top_k=1, min_score=0.0)
        self.assertEqual(result.kept, 4)
        self.assertEqual(len(result.recommendations), 1)

    def test_picked_marks_exactly_the_recommendations(self):
        result = build().route(QUERY, top_k=2)
        picked = [item.id for item in result.candidates if item.picked]
        self.assertEqual(picked, [item.id for item in result.recommendations])


if __name__ == "__main__":
    unittest.main()
