# Skill-Recommand

技能推荐链路：给定一个技能库和一段自然语言请求，推荐最合适的技能。
目标、约束与整体设计见 [`docs/技术路线说明.md`](docs/技术路线说明.md)。

## 进度

| 层 | 内容 | 状态 |
|---|---|---|
| 第 0 层 | 技能加载（读 `SKILL.md`） | 已完成 |
| 第 1 层 | 召回（BM25 + 向量 → RRF 融合） | 已完成 |
| 第 2 层 | 精排（Jev 并行判断相关度） | 待实现 |
| 第 3 层 | 决策（排序 / 阈值 / 拒答 / 截断） | 待实现 |

## 跑一下

零依赖、零 key 即可运行（Python 3.8+）：

```bash
# 小库：跳过检索，全量进下一层
python run_recall.py --skills data/skills --query "帮我把 pdf 简历压到一页"

# 强制走检索分支（把规模阈值调小）
python run_recall.py --skills data/skills --query "帮我把 pdf 简历压到一页" --small-n 5 --sparse-k 5

# 接自己的技能库
python run_recall.py --skills ~/.workbuddy/skills --query "线上服务突然报错变多，帮我查日志"
```

跑测试：

```bash
python -m unittest discover -s tests -t .
```

> 若 `python -m skill_router.recall` 报 `ModuleNotFoundError`，说明当前环境以隔离模式启动
> Python（忽略 `PYTHONPATH`、不把当前目录加进 `sys.path`），改用 `python run_recall.py`，
> 或先 `pip install -e .`。

## 配置

真实配置写在项目根的 `.env`（已 gitignore），仓库里只追踪**无值的模板**：

```bash
cp .env.example .env     # 然后填入真实值
```

优先级：**真实环境变量 > `.env` > 代码默认值**。临时覆盖不必改文件：

```bash
OPENAI_API_KEY=xxx python run_recall.py --skills data/clawhub --query "..." --dense
```

稠密召回（可选）只需要一个 OpenAI 兼容的 embedding 端点，**不需要任何 SDK**：

| 渠道 | 配置 |
|---|---|
| 硅基流动 | `OPENAI_BASE_URL=https://api.siliconflow.cn/v1`、`EMBED_MODEL=BAAI/bge-m3` |
| 本地 Ollama | `OPENAI_BASE_URL=http://localhost:11434/v1`、`OPENAI_API_KEY=ollama`、`EMBED_MODEL=bge-m3` |
| OpenAI | `OPENAI_BASE_URL` 留空、`EMBED_MODEL=text-embedding-3-small` |

不配置时稠密通路静默缺席，链路自动退化为纯 BM25。

## 第 1 层在做什么

```
query ──┬── 稀疏 BM25 (top 10) ──┐
        │                          ├── RRF 排序 ──→ 候选池（不淘汰）
        └── 稠密向量 (top 40) ────┘
```

- **分词**：英文按词切，中文切单字 + 二元组。中文二元组只在**连续的汉字片段内**生成，
  不会跨过英文拼出「历压」这类并不存在的词。
- **BM25**：纯 Python 实现，零依赖。与 query 没有任何词面重叠的文档直接出局，不占候选位。
- **稠密通路（可选）**：调预训练 embedding 模型的推理接口 —— 离线算全库向量、在线算 query 向量。
  缺 key 时这一路静默缺席。它是**跨语言场景的唯一指望**：中文请求配英文技能描述时，
  BM25 的词面重叠为零，只能靠向量把语义接上。
- **两路名额不对称**：BM25 只认词面，跨语言时命中的多是同语言里的**弱相关项**，
  所以给它更少名额（默认 10 vs 40），限制它的影响面。
- **RRF 融合**：只吃名次、不吃分数，因此免疫两路召回的量纲差异
  （BM25 分数是无界实数，余弦相似度落在 [-1, 1]，两者没法直接相加）。
  **融合只负责排序，不负责淘汰** —— 两路捞到的候选全部进候选池，
  上界是 `sparse_k + dense_k`。裁掉哪条是第 2 层精排该判断的事，
  在这里截断会把整路的结果挤出去。
- **规模自适应**：`N <= small_n`（默认 60）时跳过检索，全库直接进下一层。
  小库上检索没有收益，何必多算一遍。

全量注入模式下输出是全库顺序（没有排序）—— 因为这一层在这里不做任何判断，
排序是第 2、3 层的事。

## 参数

命令行参数优先，未指定时读同名环境变量（可写在 `.env`）：

| 参数 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `--skills` | `SKILLS_DIR` | `data/skills` | 技能库目录，或 `skills.json` |
| `--query` / `-q` | — | 必填 | 自然语言请求 |
| `--sparse-k` | `SPARSE_K` | 10 | BM25 一路召回条数 |
| `--dense-k` | `DENSE_K` | 40 | 稠密一路召回条数 |
| `--small-n` | `SMALL_N` | 60 | 不超过此规模则全量返回，跳过检索 |
| `--dense` | `USE_EMBED=1` | 关 | 开启稠密通路 |

## 目录

```
skill_router/
├── config.py      配置加载（.env）
├── skills.py      第 0 层：技能加载（解析 SKILL.md）
└── recall.py      第 1 层：召回
data/skills/       演示技能库（10 个，中文描述）
data/clawhub/      从 ClawHub 下载的技能（已 gitignore，用 scripts/fetch_clawhub.py 拉取）
scripts/           工具脚本
tests/             离线测试
run_recall.py      免安装入口
.env.example       配置模板（真实配置 .env 不入库）
```

## 依赖

**核心零依赖**：第 0、1 层只用标准库，稠密召回也不需要 SDK。

唯一可选的是 `pyyaml` —— 用于 `SKILL.md` frontmatter 的完整 YAML 解析；
缺失时自动退回内置的极简解析器，够 `name` / `description` 用。

```bash
pip install pyyaml        # 或
pip install -e ".[yaml]"
```
