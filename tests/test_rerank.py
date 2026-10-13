"""第 2 层精排的测试。不需要网络，也不需要 API key。"""

from __future__ import annotations

import os
import unittest

from skill_router.rerank import MockJudge, SystemOneJudge, _batch
from skill_router.skills import Skill

ENV_KEYS = ("JUDGE_API_KEY", "TYPESAFE_API_KEY", "DASHSCOPE_API_KEY")


def make(skill_id: str, name: str, description: str) -> Skill:
    return Skill(id=skill_id, name=name, description=description)


class TestBatch(unittest.TestCase):
    def test_splits_evenly(self):
        self.assertEqual(_batch([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_keeps_remainder(self):
        self.assertEqual(_batch([1, 2, 3], 2), [[1, 2], [3]])

    def test_size_larger_than_input(self):
        self.assertEqual(_batch([1, 2], 10), [[1, 2]])

    def test_empty_input(self):
        self.assertEqual(_batch([], 3), [])


class TestSystemOneJudge(unittest.TestCase):
    def setUp(self):
        self._saved = {key: os.environ.pop(key, None) for key in ENV_KEYS}

    def tearDown(self):
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value

    def test_from_env_returns_none_without_key(self):
        """没有 key 时不该凭空造一个判断器出来。"""
        self.assertIsNone(SystemOneJudge.from_env())

    def test_from_env_picks_up_key(self):
        os.environ["JUDGE_API_KEY"] = "test-key"
        judge = SystemOneJudge.from_env()
        self.assertIsNotNone(judge)
        self.assertEqual(judge._api_key, "test-key")

    def test_defaults_target_typesafe(self):
        judge = SystemOneJudge(api_key="x")
        self.assertEqual(judge.base_url, "https://api.typesafe.ai/v1")
        self.assertEqual(judge.model, "jev-latest")

    def test_trailing_slash_is_trimmed(self):
        judge = SystemOneJudge(base_url="http://localhost:11434/v1/", api_key="x")
        self.assertEqual(judge.base_url, "http://localhost:11434/v1")

    def test_empty_candidates_short_circuits(self):
        """没有候选时不该发请求。"""
        judge = SystemOneJudge(api_key="x")
        self.assertEqual(judge.score("任意请求", []), {})


class TestMockJudge(unittest.TestCase):
    def test_overlapping_candidate_scores_higher(self):
        candidates = [
            make("hit", "PDF 简历压缩", "把多页简历压缩成一页"),
            make("miss", "容器编排配置", "写 docker-compose"),
        ]
        scores = MockJudge().score("把简历压到一页", candidates)

        self.assertGreater(scores["hit"], scores["miss"])
        self.assertEqual(scores["miss"], 0.0)

    def test_covers_every_candidate(self):
        candidates = [make(f"s{i}", f"技能{i}", f"描述{i}") for i in range(5)]
        scores = MockJudge().score("技能描述", candidates)
        self.assertEqual(set(scores), {skill.id for skill in candidates})

    def test_blank_query_yields_zeros(self):
        candidates = [make("a", "技能", "描述")]
        self.assertEqual(MockJudge().score("   ", candidates), {"a": 0.0})


if __name__ == "__main__":
    unittest.main()
