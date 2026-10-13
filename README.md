# Skill-Recommand

技能推荐链路：给定一个技能库和一段自然语言请求，推荐最合适的技能。
目标、约束与整体设计见 [`docs/技术路线说明.md`](docs/技术路线说明.md)。

## 进度

| 层 | 内容 | 状态 |
|---|---|---|
| 第 0 层 | 技能加载（读 `SKILL.md`） | 已完成 |
| 第 1 层 | 召回（BM25 + 向量 → RRF 融合） | 已完成 |
| 第 2 层 | 精排（Jev 并行判断相关度） | 已完成 |
| 第 3 层 | 决策（排序 / 阈值 / 拒答 / 截断） | 已完成 |
| 编排 | 三层串联 + 中间数据 | 已完成 |
| 演示 | 本地网页（三层漏斗可视化） | 已完成 |

## 跑一下

```bash
# 端到端：单条请求走完三层
python run_route.py --skills data/clawhub --query "帮我把 pdf 简历压到一页"

# 网页演示
python run_server.py --skills data/clawhub --port 8000
# 然后打开 http://127.0.0.1:8000

# 只看第 1 层
python run_recall.py --skills data/clawhub --query "帮我把 pdf 简历压到一页"

# 在标注集上评估整条链路
python run_eval.py --skills data/clawhub --eval data/eval.jsonl --min-score 0.4
```

零依赖即可运行（Python 3.8+）。不配任何 key 时精排退回词面重叠的 `MockJudge`，
稠密通路静默缺席，链路照跑。

跑测试：

```bash
python -m unittest discover -s tests -t .
```

> 若 `python -m skill_router.router` 报 `ModuleNotFoundError`，说明当前环境以隔离模式启动
> Python（忽略 `PYTHONPATH`、不把当前目录加进 `sys.path`），改用 `python run_route.py`，
> 或先 `pip install -e .`。

## 网页演示

`python run_server.py` 起一个只用标准库的服务，页面把三层的中间过程摊开：

```
技能库 223  →  召回 45  →  精排 45  →  推荐 3
             稀疏10·稠密40   模型打分    阈值 0.40
```

- 页面打开是落地态（标题与输入框居中），输入请求后切换为结果态；
- 漏斗四个阶段实时显示数量与该层细节（哪几路召回、用的什么模型、当前阈值）；
- 推荐卡片标注每条技能的召回来路（稀疏 / 稠密）与召回名次；
- 下方候选池列出精排得分、相关度条、召回来路，并用颜色区分三种去向：
  推荐 / 过了阈值但被 `top_k` 截掉 / 未过阈值。

页面打开是落地态（标题与输入框居中），跑过一次后切换到结果态。

接口只有两个：`GET /api/meta`（技能库与链路配置）和 `POST /api/route`
（请求体 `{"query": "...", "top_k": 3}`，`top_k` 可选）。

**拒答阈值不接受客户端覆盖** —— 它是标定出来的链路参数（`MIN_SCORE`），放给客户端调
等于让调用方绕过标定结果。推荐条数只决定「最多给几条」，属于展示偏好，可以逐次指定。

## 链路在做什么

```
                  ┌─ 第 1 层 召回 ─┐   ┌─ 第 2 层 精排 ─┐   ┌─ 第 3 层 决策 ─┐
请求 ──→ 223 技能 ─┤ BM25 top10    ├─→ 模型判断相关度 ├─→ 排序 / 阈值 / 拒答 ─→ 推荐 3 条
                  └ 向量 top40    ┘   └──────────────┘   └────────────────┘
                        RRF 融合
```

### 第 1 层 · 召回

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

### 第 2 层 · 精排

把每条候选作为一个 **noul 问题**并行送进模型：

> 「技能『X』是否是完成用户请求『Y』的合适选择？它被描述为：……」

模型对每个问题返回一个概率。候选进 `instructions` 而不是 `state`，
让每个问题自带上下文 —— 指代越少越不容易错，也避开大 state 带来的分心。

批量上限默认 16，批数即并发数，总延迟约等于单批而不是各批之和。

### 第 3 层 · 决策

**这一步全是代码，不含任何模型判断**：

```
candidates + scores
  ├─ 1. 按概率降序
  ├─ 2. 过滤掉低于阈值的
  ├─ 3. 若结果为空 → 拒答（返回「无匹配」）
  └─ 4. 截断到 top_k
```

拒答是**显式的出口**，不是空结果的副作用 —— 调用方应当据此告诉用户「没有合适的技能」，
而不是硬凑一个。阈值必须自己标定：不同模型的概率分布差别很大。

## 配置

真实配置写在项目根的 `.env`（已 gitignore），仓库里只追踪**无值的模板**：

```bash
cp .env.example .env     # 然后填入真实值
```

优先级：**真实环境变量 > `.env` > 代码默认值**。临时覆盖不必改文件：

```bash
USE_EMBED=1 python run_route.py --skills data/clawhub --query "..."
```

稠密召回（可选）只需要一个 OpenAI 兼容的 embedding 端点，**不需要任何 SDK**：

| 渠道 | 配置 |
|---|---|
| 硅基流动 | `OPENAI_BASE_URL=https://api.siliconflow.cn/v1`、`EMBED_MODEL=BAAI/bge-m3` |
| 本地 Ollama | `OPENAI_BASE_URL=http://localhost:11434/v1`、`OPENAI_API_KEY=ollama`、`EMBED_MODEL=bge-m3` |
| OpenAI | `OPENAI_BASE_URL` 留空、`EMBED_MODEL=text-embedding-3-small` |

不配置时稠密通路静默缺席，链路自动退化为纯 BM25。

## 参数

命令行参数优先，未指定时读同名环境变量（可写在 `.env`）：

| 参数 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `--skills` | `SKILLS_DIR` | `data/skills` | 技能库目录 |
| `--query` | — | 必填 | 自然语言请求 |
| `--sparse-k` | `SPARSE_K` | 10 | BM25 一路召回条数 |
| `--dense-k` | `DENSE_K` | 40 | 稠密一路召回条数 |
| `--small-n` | `SMALL_N` | 60 | 不超过此规模则全量返回，跳过检索 |
| `--dense` | `USE_EMBED=1` | 关 | 开启稠密通路 |
| `--top-k` | `TOP_K` | 3 | 最终推荐条数 |
| `--min-score` | `MIN_SCORE` | 0.4 | 拒答阈值，由评测标定，服务端独占 |

## 目录

```
skill_router/
├── config.py      配置加载（.env）
├── skills.py      第 0 层：技能加载（解析 SKILL.md）
├── recall.py      第 1 层：召回
├── rerank.py      第 2 层：精排
├── decide.py      第 3 层：决策
├── router.py      编排：串起三层，带出中间数据
├── evaluate.py    在标注集上评估
└── server.py      本地演示服务
web/index.html     单页前端
data/skills/       演示技能库（10 个，中文描述）
data/clawhub/      从 ClawHub 下载的技能（已 gitignore，用 scripts/fetch_clawhub.py 拉取）
data/eval.jsonl    标注集
scripts/           工具脚本（ClawHub 批量下载）
tests/             离线测试（不联网、不需要 key）
run_recall.py      第 1 层入口
run_route.py       端到端入口
run_server.py      演示服务入口
run_eval.py        评估入口
.env.example       配置模板（真实配置 .env 不入库）
```

## 依赖

**核心零依赖**：四层链路、编排与演示服务只用标准库，稠密召回也不需要 SDK。

唯一可选的是 `pyyaml` —— 用于 `SKILL.md` frontmatter 的完整 YAML 解析；
缺失时自动退回内置的极简解析器，够 `name` / `description` 用。

```bash
pip install pyyaml        # 或
pip install -e ".[yaml]"
```
