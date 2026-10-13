"""第 3 层决策的测试。不需要网络，也不需要 API key。"""

from __future__ import annotations

import unittest

from skill_router.decide import decide
from skill_router.skills import Skill


def make(skill_id: str, name: str = "", description: str = "") -> Skill:
    return Skill(id=skill_id, name=name or skill_id, description=description)


class TestOrdering(unittest.TestCase):
    def test_orders_by_score_desc(self):
        candidates = [make("a"), make("b"), make("c")]
        decision = decide(candidates, {"a": 0.2, "b": 0.9, "c": 0.5}, top_k=3)
        self.assertEqual(decision.ids, ["b", "c", "a"])

    def test_missing_score_counts_as_zero(self):
        """精排没给出分数的候选按 0 处理，而不是丢掉 —— 丢掉会静默改变候选集。"""
        decision = decide([make("a"), make("b")], {"a": 0.5}, top_k=2)
        self.assertEqual(decision.ids, ["a", "b"])
        self.assertEqual(decision.scores, [0.5, 0.0])

    def test_scores_align_with_skills(self):
        decision = decide([make("a"), make("b")], {"a": 0.3, "b": 0.8}, top_k=2)
        self.assertEqual(decision.ids, ["b", "a"])
        self.assertEqual(decision.scores, [0.8, 0.3])


class TestThreshold(unittest.TestCase):
    def test_filters_below_threshold(self):
        candidates = [make("a"), make("b"), make("c")]
        decision = decide(candidates, {"a": 0.9, "b": 0.3, "c": 0.1}, top_k=3, min_score=0.5)
        self.assertEqual(decision.ids, ["a"])

    def test_threshold_is_inclusive(self):
        """恰好等于阈值要保留，否则「阈值 0.4」实际语义会变成「大于 0.4」。"""
        decision = decide([make("a")], {"a": 0.4}, min_score=0.4)
        self.assertFalse(decision.abstained)
        self.assertEqual(decision.ids, ["a"])

    def test_zero_threshold_keeps_zero_score_candidates(self):
        """``min_score=0`` 是默认值，语义是「不设阈值」—— 0 分候选也会进来。"""
        decision = decide([make("a")], {"a": 0.0}, min_score=0.0)
        self.assertFalse(decision.abstained)
        self.assertEqual(decision.ids, ["a"])


class TestAbstain(unittest.TestCase):
    def test_all_below_threshold_abstains(self):
        decision = decide([make("a"), make("b")], {"a": 0.2, "b": 0.1}, min_score=0.5)
        self.assertTrue(decision.abstained)
        self.assertEqual(decision.ids, [])
        self.assertEqual(len(decision), 0)

    def test_no_candidates_abstains(self):
        """候选为空同样走拒答出口 —— 上游没捞到东西时，出口必须一致。"""
        self.assertTrue(decide([], {}).abstained)

    def test_kept_result_is_not_abstained(self):
        decision = decide([make("a")], {"a": 0.9}, min_score=0.5)
        self.assertFalse(decision.abstained)
        self.assertEqual(decision.ids, ["a"])


class TestTruncation(unittest.TestCase):
    def test_truncates_to_top_k(self):
        candidates = [make(str(index)) for index in range(10)]
        scores = {str(index): 1.0 - index / 100 for index in range(10)}
        decision = decide(candidates, scores, top_k=3)
        self.assertEqual(len(decision), 3)
        self.assertEqual(decision.ids, ["0", "1", "2"])

    def test_fewer_candidates_than_top_k(self):
        decision = decide([make("a")], {"a": 0.9}, top_k=5)
        self.assertEqual(len(decision), 1)

    def test_top_k_zero_still_returns_one(self):
        """top_k=0 是误传，但不该静默返回空 —— 取一条比撒谎好。"""
        decision = decide([make("a")], {"a": 0.5}, top_k=0)
        self.assertEqual(len(decision), 1)

    def test_filtered_counts_before_truncation(self):
        """filtered 是过滤后、截断前的条数 —— 少了它，top_k 砍掉多少就看不见。"""
        candidates = [make(str(index)) for index in range(6)]
        scores = {str(index): 0.9 - index * 0.1 for index in range(6)}
        decision = decide(candidates, scores, top_k=2, min_score=0.5)

        self.assertEqual(decision.filtered, 5)  # 0.9 ~ 0.5 都过线
        self.assertEqual(len(decision), 2)  # 但只推前 2 条

    def test_filtered_equals_length_when_not_truncated(self):
        decision = decide(
            [make("a"), make("b")], {"a": 0.9, "b": 0.8}, top_k=5, min_score=0.5
        )
        self.assertEqual(decision.filtered, 2)
        self.assertEqual(decision.filtered, len(decision))


if __name__ == "__main__":
    unittest.main()
