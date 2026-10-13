"""第 1 层 · 召回。

把 N 个技能收窄到 N′ 个候选，**宁多勿漏**。这一层全是检索算法 ——
没有训练，也没有模型判断（稠密通路调用的是预训练模型的推理接口，训练发生在厂商侧）。

    query ──┬── 稀疏 BM25 ──┐
            │               ├── RRF ──→ top N′
            └── 稠密向量 ───┘

**规模自适应**：候选数很少（N <= small_n）时直接跳过检索，全库进下一层。
同一套代码覆盖小库和大库，不必为小库单独设计一套流程。

命令行：

    python -m skill_router.recall --skills data/skills --query "帮我把 pdf 简历压到一页"
    python -m skill_router.recall --skills data/skills --query "..." --small-n 5   # 强制走检索
    USE_EMBED=1 python -m skill_router.recall --skills data/skills --query "..."    # 加稠密通路
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from .config import load_env
from .skills import Skill, load_skills

# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------

_LATIN = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """中英混排的廉价分词：英文按词切，中文切单字 + 二元组。

    刻意不引分词器：BM25 只需要词面重叠，「简历 / 压缩 / 一页」这类信号用
    单字加二元组就能吃进来，而且零依赖、零训练。

    中文二元组只在**连续的 CJK 片段内**生成 —— 若把全文汉字抽出来两两拼接，
    「简历 pdf 压缩」会拼出「历压」这种并不存在的词，反而污染倒排。
    """
    text = (text or "").lower()
    tokens = _LATIN.findall(text)
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


# ---------------------------------------------------------------------------
# 稀疏通路：BM25
# ---------------------------------------------------------------------------


class BM25:
    """Okapi BM25，纯 Python 实现。零第三方依赖、零模型。"""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._n = len(docs)
        self._tf: List[Counter] = []
        self._len: List[int] = []
        doc_freq: Counter = Counter()

        for doc in docs:
            tokens = tokenize(doc)
            self._tf.append(Counter(tokens))
            self._len.append(len(tokens))
            doc_freq.update(set(tokens))

        self._avgdl = (sum(self._len) / self._n) if self._n else 1.0
        self._idf = {
            term: math.log(1.0 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def rank(self, query: str, topk: Optional[int] = None) -> List[Tuple[int, float]]:
        """返回 ``[(文档下标, 分数)]``，按分数降序。

        与 query 没有任何词面重叠的文档分数为 0，**不会出现在结果里** ——
        让 0 分候选挤进候选池只会给下一层添乱。
        """
        terms = set(tokenize(query))
        if not terms or not self._n:
            return []

        scored: List[Tuple[int, float]] = []
        for index in range(self._n):
            term_freq = self._tf[index]
            length = self._len[index] or 1
            norm = self.k1 * (1 - self.b + self.b * length / self._avgdl)
            total = 0.0
            for term in terms:
                freq = term_freq.get(term)
                if not freq:
                    continue
                total += self._idf.get(term, 0.0) * freq * (self.k1 + 1) / (freq + norm)
            if total > 0:
                scored.append((index, total))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:topk] if topk else scored


# ---------------------------------------------------------------------------
# 稠密通路：向量召回
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """预训练 embedding 模型的推理接口。换厂商只需换实现。"""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class OpenAIEmbedder:
    """OpenAI 兼容的 embedding 接口。

    只发一个 ``POST {base_url}/embeddings``，用标准库完成 —— 因此 OpenAI、Ollama、
    硅基流动、智谱等任何兼容端点都能直接接，且不引入任何依赖。

    训练发生在厂商侧，这里只做推理调用 —— 本项目不训练任何模型。
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = float(os.environ.get("EMBED_TIMEOUT", 60))

    @classmethod
    def from_env(cls) -> Optional["OpenAIEmbedder"]:
        """没有 key 时返回 ``None`` —— 稠密通路静默缺席，链路照跑。"""
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        return cls()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"embedding 请求失败: HTTP {error.code} {detail}") from error

        # 按 index 排序，保证返回顺序与输入一一对应
        rows = sorted(body.get("data") or [], key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in rows]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


class DenseRecall:
    """稠密召回：离线把全库向量算好，在线只对 query 求一次向量再算余弦。

    向量是**惰性**构建的，首次 rank 时才 embedding 全库；
    没有 embedder 时 :attr:`available` 为假，调用方直接跳过这一路。
    """

    def __init__(
        self,
        texts: Sequence[str],
        embedder: Optional[Embedder],
        batch: int = 64,
    ):
        self._texts = list(texts)
        self._embedder = embedder
        self._batch = max(1, batch)
        self._vectors: Optional[List[List[float]]] = None

    @property
    def available(self) -> bool:
        return self._embedder is not None and bool(self._texts)

    def rank(self, query: str, topk: Optional[int] = None) -> List[Tuple[int, float]]:
        if not self.available:
            return []
        vectors = self._ensure_vectors()
        query_vec = self._embedder.embed([query])[0]

        scored = [(_cosine(query_vec, vector), index) for index, vector in enumerate(vectors)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        ranked = [(index, score) for score, index in scored]
        return ranked[:topk] if topk else ranked

    def _ensure_vectors(self) -> List[List[float]]:
        """惰性构建全库向量。分批发送，避免把整库塞进单个请求。"""
        if self._vectors is None:
            vectors: List[List[float]] = []
            for start in range(0, len(self._texts), self._batch):
                vectors.extend(self._embedder.embed(self._texts[start : start + self._batch]))
            self._vectors = vectors
        return self._vectors


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------


def rrf(rank_lists: Sequence[Sequence[int]], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion。

    只吃名次、不吃分数，所以天然免疫两路召回的量纲差异 ——
    BM25 的分数是无界实数，余弦相似度落在 [-1, 1]，两者没法直接相加。
    """
    fused: Dict[int, float] = {}
    for ranks in rank_lists:
        for position, index in enumerate(ranks):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + position + 1)
    return sorted(fused, key=lambda index: (-fused[index], index))


# ---------------------------------------------------------------------------
# 召回入口
# ---------------------------------------------------------------------------


@dataclass
class RecallResult:
    """召回结果。

    ``channels`` 保留每一路的名次，用来回答「正解是靠哪一路捞回来的」——
    调召回时没有它就只能瞎猜。
    """

    skills: List[Skill]
    mode: str = "full-injection"
    channels: Dict[str, List[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.skills)

    @property
    def ids(self) -> List[str]:
        return [skill.id for skill in self.skills]


class Recaller:
    """第 1 层：把 N 收窄到 N′。

    参数：
        sparse_k：BM25 一路召回多少条，默认 10。
        dense_k： 稠密一路召回多少条，默认 40。
                  两路名额不对称是有意的：BM25 只认词面，跨语言时几乎全是噪声，
                  给它更少的名额是在限制它的影响面（并不能让它变准）。
        small_n： 规模阈值。N <= small_n 时全量返回，跳过检索，默认 60。
        dense：   可选的稠密通路；缺失或不可用时自动退化为纯 BM25。

    两路的结果**全部**进候选池（去重后按 RRF 排序），这里不做淘汰 ——
    候选池大小的上界就是 ``sparse_k + dense_k``，该裁掉哪条是第 2 层精排的事。
    """

    def __init__(
        self,
        skills: Sequence[Skill],
        *,
        sparse_k: int = 10,
        dense_k: int = 40,
        small_n: int = 60,
        dense: Optional[DenseRecall] = None,
    ):
        self.skills = list(skills)
        self.sparse_k = sparse_k
        self.dense_k = dense_k
        self.small_n = small_n
        self.dense = dense
        self.sparse = BM25([skill.text for skill in self.skills]) if self.skills else None

    def recall(self, query: str) -> RecallResult:
        if not self.skills:
            return RecallResult([], "empty", {})
        if not (query or "").strip():
            return RecallResult([], "empty", {})

        # 规模自适应：小库直接全量进下一层，检索这层的工作量省掉
        if len(self.skills) <= self.small_n:
            return RecallResult(list(self.skills), "full-injection", {})

        ranks: Dict[str, List[int]] = {}
        sparse_rank = [index for index, _ in self.sparse.rank(query, self.sparse_k)]
        if sparse_rank:
            ranks["sparse"] = sparse_rank
        if self.dense is not None and self.dense.available:
            dense_rank = [index for index, _ in self.dense.rank(query, self.dense_k)]
            if dense_rank:
                ranks["dense"] = dense_rank

        # 只有真正给出结果的通路才算数：一路空转不该污染 mode 的判定
        if not ranks:
            return RecallResult([], "empty", {})

        # 融合只负责排序，不负责淘汰 —— 两路捞到的候选全部留下，
        # 谁相关由第 2 层精排判断。在这里截断会把单路的结果整体挤出去。
        fused = rrf(list(ranks.values()))
        mode = "hybrid" if len(ranks) > 1 else next(iter(ranks))
        channels = {
            name: [self.skills[index].id for index in indexes] for name, indexes in ranks.items()
        }
        return RecallResult([self.skills[index] for index in fused], mode, channels)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def _build_recaller(args: argparse.Namespace, skills: Sequence[Skill]) -> Recaller:
    dense: Optional[DenseRecall] = None
    if args.dense:
        embedder = OpenAIEmbedder.from_env()
        if embedder is None:
            print(
                "提示：未检测到 OPENAI_API_KEY 或未安装 openai，稠密通路已跳过，本次仅用 BM25。",
                file=sys.stderr,
            )
        else:
            dense = DenseRecall([skill.text for skill in skills], embedder)
    return Recaller(
        skills,
        sparse_k=args.sparse_k,
        dense_k=args.dense_k,
        small_n=args.small_n,
        dense=dense,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 优先级：真实环境变量 > .env > 代码默认值
    load_env()

    parser = argparse.ArgumentParser(
        prog="python -m skill_router.recall",
        description="第 1 层 · 召回：BM25 +（可选）向量 → RRF 融合",
    )
    parser.add_argument(
        "--skills",
        default=os.environ.get("SKILLS_DIR", "data/skills"),
        help="技能库目录或 skills.json（默认 data/skills）",
    )
    parser.add_argument("--query", "-q", required=True, help="自然语言请求")
    parser.add_argument(
        "--sparse-k", type=int, default=int(os.environ.get("SPARSE_K", 10)), help="BM25 一路召回条数"
    )
    parser.add_argument(
        "--dense-k", type=int, default=int(os.environ.get("DENSE_K", 40)), help="稠密一路召回条数"
    )
    parser.add_argument(
        "--small-n", type=int, default=int(os.environ.get("SMALL_N", 60)), help="不超过此规模则全量返回"
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        default=os.environ.get("USE_EMBED") == "1",
        help="开启稠密通路（需 OPENAI_API_KEY）",
    )
    args = parser.parse_args(argv)

    try:
        skills = load_skills(args.skills)
    except (FileNotFoundError, ValueError) as error:
        print(f"加载技能库失败：{error}", file=sys.stderr)
        return 2
    if not skills:
        print(f"技能库为空：{args.skills}", file=sys.stderr)
        return 2

    result = _build_recaller(args, skills).recall(args.query)

    print(f"技能库 : {args.skills}（{len(skills)} 个技能）")
    print(f"请求   : {args.query}")
    if result.mode == "full-injection":
        print(f"模式   : 全量注入（{len(skills)} <= small_n={args.small_n}，跳过检索）")
    else:
        print(f"模式   : {result.mode}（召回 {len(result)} 条，来自 {', '.join(result.channels)}）")
    print("-" * 72)

    for position, skill in enumerate(result.skills, start=1):
        marks = [
            f"{name}#{indexes.index(skill.id) + 1}"
            for name, indexes in result.channels.items()
            if skill.id in indexes
        ]
        suffix = f"   [{', '.join(marks)}]" if marks else ""
        print(f"{position:>3}. {skill.id:<34}{skill.name}{suffix}")

    if not result.skills:
        print("（无候选）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
