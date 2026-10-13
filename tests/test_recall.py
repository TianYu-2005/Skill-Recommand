"""第 1 层召回的测试。

全部离线可跑：不依赖网络、API key，也不依赖 pyyaml / openai 等可选包。
运行：``python -m unittest discover -s tests -t .``
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from skill_router.recall import BM25, DenseRecall, Recaller, rrf, tokenize
from skill_router.skills import (  # noqa: F401  (_minimal_yaml 用于直测回退解析器)
    Skill,
    _minimal_yaml,
    load_skills,
    parse_frontmatter,
)

DEMO_SKILLS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "skills"
)


def make(skill_id: str, name: str, description: str) -> Skill:
    return Skill(id=skill_id, name=name, description=description)


class PairedEmbedder:
    """把指定的一对（query, 目标文本）映射到同一方向，其余落到正交方向。

    用来在无 key 环境下验证稠密通路：目标与 query 词面完全不同，
    BM25 抓不到它，只能靠这一路捞回来。
    """

    def __init__(self, query: str, target_text: str):
        self.query = query
        self.target_text = target_text

    def embed(self, texts):
        return [[1.0, 0.0] if t in (self.query, self.target_text) else [0.0, 1.0] for t in texts]


# ---------------------------------------------------------------------------


class TestTokenize(unittest.TestCase):
    def test_latin_is_lowercased_and_split(self):
        self.assertEqual(tokenize("BM25 Okapi"), ["bm25", "okapi"])

    def test_cjk_yields_unigrams_and_bigrams(self):
        tokens = tokenize("简历压缩")
        for expected in ("简", "历", "简历", "压缩"):
            self.assertIn(expected, tokens)

    def test_bigrams_do_not_cross_latin_runs(self):
        """汉字被英文隔开时不能拼出跨段的假词，否则倒排会被污染。"""
        tokens = tokenize("简历pdf压缩")
        self.assertNotIn("历压", tokens)
        self.assertIn("简历", tokens)
        self.assertIn("压缩", tokens)
        self.assertIn("pdf", tokens)

    def test_empty_input(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "PDF 简历压缩为一页 简历排版",
            "PDF 表格抽取 导出 Excel",
            "代码审查 检查缺陷",
        ]
        self.bm25 = BM25(self.docs)

    def test_ranks_matching_document_first(self):
        ranked = self.bm25.rank("帮我把 pdf 简历压到一页")
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], 0)

    def test_documents_without_overlap_are_absent(self):
        ranked = self.bm25.rank("简历")
        self.assertNotIn(2, [index for index, _ in ranked])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.bm25.rank(""), [])

    def test_topk_limits_result(self):
        self.assertLessEqual(len(self.bm25.rank("pdf", topk=1)), 1)

    def test_ranking_is_deterministic(self):
        self.assertEqual(self.bm25.rank("pdf 表格"), self.bm25.rank("pdf 表格"))


class TestRRF(unittest.TestCase):
    def test_two_channel_agreement_beats_single_first_place(self):
        """两路都排第二，胜过只在单路排第一 —— 这正是 RRF 的价值。"""
        fused = rrf([[0, 1], [2, 1]])
        self.assertEqual(fused[0], 1)
        self.assertEqual(sorted(fused), [0, 1, 2])

    def test_rank_order_decides_not_score(self):
        self.assertEqual(rrf([[5, 1, 2], [5, 2, 1]])[0], 5)

    def test_empty_input(self):
        self.assertEqual(rrf([]), [])


class TestScaleAdaptation(unittest.TestCase):
    def test_small_library_returns_everything_without_retrieval(self):
        skills = [make(f"s{i}", f"技能{i}", f"负责第{i}类事务") for i in range(5)]
        result = Recaller(skills, small_n=60).recall("随便什么请求")

        self.assertEqual(result.mode, "full-injection")
        self.assertEqual(len(result), 5)
        self.assertEqual(result.channels, {})

    def test_library_above_threshold_goes_through_retrieval(self):
        skills = [make(f"s{i}", f"技能{i}", f"处理流水线编号 {i} 的任务") for i in range(40)]
        result = Recaller(skills, sparse_k=10, small_n=5).recall("处理流水线编号 7 的任务")

        self.assertEqual(result.mode, "sparse")
        self.assertLessEqual(len(result), 10)
        self.assertLess(len(result), len(skills))
        self.assertEqual(result.skills[0].id, "s7")
        self.assertIn("s7", result.channels["sparse"])

    def test_sparse_k_caps_that_channel_only(self):
        skills = [make(f"s{i}", f"技能{i}", f"处理编号 {i} 的任务") for i in range(40)]
        result = Recaller(skills, sparse_k=3, small_n=5).recall("处理编号 9 的任务")
        self.assertLessEqual(len(result.channels["sparse"]), 3)

    def test_empty_library(self):
        result = Recaller([], small_n=60).recall("任意请求")
        self.assertEqual(result.mode, "empty")
        self.assertEqual(len(result), 0)

    def test_blank_query_returns_nothing(self):
        skills = [make(f"s{i}", f"技能{i}", f"负责第{i}类事务") for i in range(40)]
        self.assertEqual(Recaller(skills, small_n=5).recall("   ").mode, "empty")


class TestDenseChannel(unittest.TestCase):
    def _library(self, target):
        skills = [make(f"s{i}", f"技能{i}", f"负责第{i}类事务") for i in range(20)]
        skills.append(target)
        return skills

    def test_dense_recovers_what_sparse_cannot(self):
        query = "帮我把那个玩意儿翻新一下"
        target = make("target", "同义表达技能", "语义相关但词面完全不同")
        dense = DenseRecall([s.text for s in self._library(target)], PairedEmbedder(query, target.text))

        result = Recaller(self._library(target), sparse_k=5, dense_k=5, small_n=5, dense=dense).recall(query)

        self.assertEqual(result.mode, "dense")
        self.assertEqual(result.skills[0].id, "target")

    def test_hybrid_mode_when_both_channels_hit(self):
        query = "技能"
        target = make("target", "同义表达技能", "语义相关但词面完全不同")
        dense = DenseRecall([s.text for s in self._library(target)], PairedEmbedder(query, target.text))

        result = Recaller(self._library(target), sparse_k=5, dense_k=5, small_n=5, dense=dense).recall(query)

        self.assertEqual(result.mode, "hybrid")
        self.assertEqual(set(result.channels), {"sparse", "dense"})

    def test_missing_dense_degrades_to_sparse(self):
        skills = [make(f"s{i}", f"技能{i}", f"处理编号 {i} 的任务") for i in range(40)]
        result = Recaller(skills, sparse_k=5, dense_k=5, small_n=5, dense=None).recall("处理编号 7 的任务")
        self.assertEqual(result.mode, "sparse")
        self.assertNotIn("dense", result.channels)

    def test_unavailable_dense_is_skipped(self):
        skills = [make(f"s{i}", f"技能{i}", f"处理编号 {i} 的任务") for i in range(40)]
        dense = DenseRecall([s.text for s in skills], None)  # 没有 embedder
        result = Recaller(skills, sparse_k=5, dense_k=5, small_n=5, dense=dense).recall("处理编号 7 的任务")
        self.assertFalse(dense.available)
        self.assertEqual(result.mode, "sparse")

    def test_fusion_keeps_results_from_both_channels(self):
        """融合只排序不淘汰：dense 独有捞回的候选不能被 sparse 的结果挤掉。"""
        query = "处理编号 7 的任务"
        skills = [make(f"s{i}", f"技能{i}", f"处理编号 {i} 的任务") for i in range(20)]
        target = skills[19]
        dense = DenseRecall([s.text for s in skills], PairedEmbedder(query, target.text))

        result = Recaller(skills, sparse_k=3, dense_k=3, small_n=5, dense=dense).recall(query)

        self.assertIn(target.id, result.ids)
        self.assertIn("dense", result.channels)


class TestSkillLoading(unittest.TestCase):
    def test_loads_demo_library(self):
        skills = load_skills(DEMO_SKILLS)

        self.assertGreaterEqual(len(skills), 10)
        ids = {skill.id for skill in skills}
        self.assertIn("resume-pdf-one-page", ids)
        for skill in skills:
            self.assertTrue(skill.name, skill.id)
            self.assertTrue(skill.description, f"{skill.id} 缺少 description")

    def test_index_text_is_name_plus_description(self):
        skill = make("x", "技能名", "描述文本")
        self.assertEqual(skill.text, "技能名\n描述文本")

    def test_loads_skills_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "skills.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"id": "a", "name": "技能 A", "description": "描述 A"}], handle)

            skills = load_skills(path)
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].id, "a")

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_skills(os.path.join("no", "such", "dir"))

    def test_parse_frontmatter_reads_name_and_description(self):
        meta = parse_frontmatter('---\nname: 甲\ndescription: 乙\n---\n\n正文\n')
        self.assertEqual(meta.get("name"), "甲")
        self.assertEqual(meta.get("description"), "乙")

    def test_minimal_yaml_handles_block_scalar(self):
        """没有 pyyaml 时的回退解析器：块标量要能读出来。"""
        meta = _minimal_yaml("name: 甲\ndescription: |\n  第一行\n  第二行\n")
        self.assertEqual(meta["name"], "甲")
        self.assertEqual(meta["description"], "第一行\n第二行")

    def test_loads_skill_md_encoded_in_gb18030(self):
        """真实技能库编码不统一：GBK 写的中文描述不能把整库拖垮。"""
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "gbk-skill")
            os.makedirs(folder)
            body = "---\nname: 中文技能\ndescription: 把多页简历压缩成一页\n---\n\n正文\n"
            with open(os.path.join(folder, "SKILL.md"), "wb") as handle:
                handle.write(body.encode("gb18030"))

            skills = load_skills(tmp)

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "中文技能")
            self.assertIn("简历", skills[0].description)

    def test_frontmatter_survives_utf8_bom(self):
        meta = parse_frontmatter("\ufeff---\nname: 甲\ndescription: 乙\n---\n\n正文\n")
        self.assertEqual(meta.get("name"), "甲")
        self.assertEqual(meta.get("description"), "乙")

    def test_quoted_values_are_unquoted(self):
        meta = parse_frontmatter("---\nname: \"中文公文写作\"\ndescription: '单引号描述'\n---\n")
        self.assertEqual(meta.get("name"), "中文公文写作")
        self.assertEqual(meta.get("description"), "单引号描述")

    def test_folded_scalar_folds_to_spaces(self):
        """`>` 是折叠标量，换行应变空格；`|` 才是保留换行。"""
        folded = parse_frontmatter("---\nname: 甲\ndescription: >\n  第一行\n  第二行\n---\n")
        literal = parse_frontmatter("---\nname: 甲\ndescription: |\n  第一行\n  第二行\n---\n")
        self.assertEqual(folded.get("description"), "第一行 第二行")
        self.assertEqual(literal.get("description"), "第一行\n第二行")

    def test_crlf_line_endings(self):
        meta = parse_frontmatter("---\r\nname: 甲\r\ndescription: 乙\r\n---\r\n\r\n正文\r\n")
        self.assertEqual(meta.get("name"), "甲")
        self.assertEqual(meta.get("description"), "乙")

    def test_space_before_colon(self):
        meta = parse_frontmatter("---\nname : 甲\ndescription : 乙\n---\n")
        self.assertEqual(meta.get("name"), "甲")
        self.assertEqual(meta.get("description"), "乙")

    def test_unterminated_frontmatter_is_ignored(self):
        """没有结束分隔符时不猜，交给上层回退到文件夹名。"""
        self.assertEqual(parse_frontmatter("---\nname: 甲\ndescription: 乙\n"), {})

    def test_misindented_frontmatter_does_not_pollute_name(self):
        """上游把后续字段缩进进 name 时，整段 frontmatter 不能灌进索引。"""
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "misindented")
            os.makedirs(folder)
            body = (
                "---\n"
                "name: 命理大师\n"
                "  version: 1.2.6\n"
                "  description: 全体系命理大师\n"
                "metadata:\n"
                '  displayName: "命理大师"\n'
                "---\n\n正文\n"
            )
            with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as handle:
                handle.write(body)

            skills = load_skills(tmp)

            self.assertEqual(skills[0].name, "命理大师")
            self.assertNotIn("version", skills[0].name)


class TestRecallEndToEnd(unittest.TestCase):
    """用真实技能库跑一遍召回，确认正解确实被捞进候选池。"""

    @classmethod
    def setUpClass(cls):
        cls.skills = load_skills(DEMO_SKILLS)

    def recall(self, query: str, k: int = 3):
        # small_n=5 强制走检索分支（演示库只有 10 个技能，否则会全量注入）
        return Recaller(self.skills, sparse_k=k, dense_k=k, small_n=5).recall(query)

    def test_resume_query(self):
        result = self.recall("帮我把 pdf 简历压到一页")
        self.assertEqual(result.skills[0].id, "resume-pdf-one-page")

    def test_pdf_table_query_does_not_return_resume(self):
        result = self.recall("从这个 pdf 附件里把表格提出来")
        self.assertEqual(result.skills[0].id, "pdf-table-extract")

    def test_ops_query(self):
        result = self.recall("容器起不来，连不上数据库")
        self.assertEqual(result.skills[0].id, "docker-compose")

    def test_ambiguous_query_still_returns_candidates(self):
        result = self.recall("帮我看看这段代码")
        self.assertTrue(len(result) > 0)
        self.assertEqual(result.skills[0].id, "code-review")


if __name__ == "__main__":
    unittest.main()
