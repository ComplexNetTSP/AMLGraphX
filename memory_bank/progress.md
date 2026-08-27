# AMLGraphX 当前进度

更新时间：2026-08-28

## Git 状态

- 当前开发分支：`feature/tabular`
- 远端同步目标：`origin/feature/tabular`
- 精确提交与工作区状态以 `git log -1`、`git status` 为准，避免本文件记录过期 hash。
- 本阶段不直接向 `main` 推送。

## 已完成的功能

### 数据集基础层

位置：`src/amlgraphx/datasets/`

- `Dataset`：下载型数据集的最小抽象接口。
- `TaskType.TRANSACTION_CLASSIFICATION`：交易级分类任务。
- `LabelLevel.TRANSACTION`：交易级标签。
- `DatasetSource.HUGGING_FACE`：Hugging Face 数据源。
- `DatasetMetadata`：使用 Pydantic 校验数据集名称、任务、标签级别、来源、许可证、仓库 ID 和预期文件。
- `clean_lazy_frame()`：使用 Polars LazyFrame 做基础清洗、缺失值处理和时间字段解析。

### Hugging Face 下载层

位置：`src/amlgraphx/datasets/download.py`

- 使用 `huggingface_hub.snapshot_download()` 或 `hf_hub_download()`。
- 支持匿名访问公开仓库，也保留本地 Hugging Face 登录配置的自动使用能力。
- 不在代码中硬编码 token。
- 支持 revision、allow_patterns、cache_dir、local_dir 和 ZIP 解压。
- 默认缓存目录：`~/.cache/amlgraphx/`。
- 支持安全的 ZIP 路径检查、预期文件校验和 CSV/Parquet 文件发现。
- 会忽略 ZIP 中的 `__MACOSX` 和 `._*` 资源文件，避免误读 macOS 元数据文件。

### 已接入的数据集

#### IBM AML

类：`IBMAML`

仓库：`OsamaMIT/IBM-AML-HI`

这是原始数据集的第三方 Hugging Face 镜像，许可证为 `CDLA-Sharing-1.0`。当前显式支持：

- `hi-small`
- `hi-medium`
- `hi-large`
- `li-small`
- `li-medium`
- `li-large`

每个变体都显式映射交易、账户和模式文件，不动态猜测文件名。可用接口：

```python
dataset = IBMAML("hi-small")
dataset.download()
transactions = dataset.transactions()
accounts = dataset.accounts()
patterns = dataset.patterns()
```

`transactions()`、`accounts()` 和 `patterns()` 返回 Polars LazyFrame；同时提供 `transaction_path()`、`accounts_path()` 和 `patterns_path()`。

#### PaySim

类：`PaySim`

仓库：`LordNR/AMLGraphX-Paysim`

这是第三方 Hugging Face 镜像，README 标示原始数据集许可证为 `CC-BY-SA-4.0`。适配器会：

- 从 ZIP 中发现真实交易 CSV。
- 跳过 macOS 资源文件。
- 保留原始 `step` 字段。
- 将 PaySim 的模拟小时 `step` 映射为锚定 Unix epoch 的逻辑 `timestamp`，以支持事务图构建。

#### SAML-D

类：`SAML`

仓库：`LordNR/AMLGraphX-SAML-D`

这是第三方 Hugging Face 镜像，README 标示原始数据集许可证为 `CC-BY-NC-SA-4.0`。适配器支持动态发现 CSV/Parquet，并解析交易中的来源、目标和时间字段。

以下数据集目前没有在本阶段接入：DGraphFin、Elliptic2、AMLSim 及其他未确认有可靠 Hugging Face 来源的数据集。

### Graph views

位置：`src/amlgraphx/graphs.py`

- `build_account_graph()`：将账户作为节点、交易作为有向边。
- `build_transaction_graph()`：将交易作为节点，按时间窗口和资金流向建立事务关系边。
- 两个 builder 都支持 Polars DataFrame 和 LazyFrame。
- 自动解析常见的来源、目标、交易 ID 和时间字段别名。
- 交易 ID 缺失、为空或重复时生成稳定的唯一 ID。
- 账户元数据 join 前会清理 ID 两侧空白。
- 同时存在 `Date` 和 `Time` 时，优先保留完整时间戳，再组合日期与 time-only 字段。

### Canonical transaction schema

位置：`src/amlgraphx/datasets/schema.py`

- `normalize_transactions()` 为数据集增加统一的 `transaction_id`、`source`、`target`、`timestamp`、`amount` 和 `label` 字段。
- 原始字段保持不变，数据集特有字段继续保留。
- 支持 IBM AML 的 `Account_duplicated_0`、PaySim 的 `nameOrig`/`nameDest`、SAML-D 的 `Sender_account`/`Receiver_account` 等实际字段别名。
- SAML-D 的 `Date` 与 time-only `Time` 会组合为完整时间戳。
- 已添加合成 schema 测试，并完成 IBM AML、PaySim、SAML-D 的真实下载 smoke test；真实数据均使用临时目录并在测试后删除。

## 真实数据验证

所有真实下载都使用 `/tmp` 下独立的 `TemporaryDirectory`，每个数据集验证完成后确认临时目录已删除；没有把真实数据提交到仓库。

验证方式：每个数据集先 lazy loading，再完整构建账户图和事务图。

| 数据集 | 账户图（节点 / 边） | 事务图（节点 / 边） |
|---|---:|---:|
| IBM HI-Small | 515,080 / 5,078,345 | 5,078,345 / 2,176,494 |
| IBM HI-Medium | 2,076,999 / 31,898,238 | 31,898,238 / 45,072,272 |
| IBM HI-Large | 2,099,541 / 179,702,229 | 179,702,229 / 210,485,361 |
| IBM LI-Small | 705,903 / 6,924,049 | 6,924,049 / 3,882,235 |
| IBM LI-Medium | 2,032,061 / 31,251,483 | 31,251,483 / 44,238,663 |
| IBM LI-Large | 2,054,390 / 176,066,557 | 176,066,557 / 207,001,490 |
| PaySim | 9,073,900 / 6,362,620 | 6,362,620 / 56 |
| SAML-D | 855,460 / 9,504,852 | 9,504,852 / 170,886 |

## Review comments 与实际修复

在完成所有真实数据验证后，确认两条 PR review comment 描述的是通用 graph API 的有效边界问题，因此补充了修复和回归测试：

1. 完整 datetime 字符串不能再与 `Date` 字段重复拼接。
2. 账户元数据 ID 必须在 join 前去除两侧空白。

验证时还发现 PaySim 的实际 ZIP 包含 macOS 资源文件，并且 PaySim 原始 schema 没有 timestamp；这两个实际兼容性问题也已修复。

## 自动化验证

- `uv run pytest`：38 passed
- `uv run ruff check .`：通过
- `uv run ruff format --check src tests`：通过
- `git diff --check`：通过

## 后续建议

- 新增数据集时，优先确认原始来源、许可证和可靠 Hugging Face 仓库，再新增一个轻量适配器。
- 复用 `Dataset`、`DatasetMetadata`、`HuggingFaceDownloader` 和 `clean_lazy_frame`，不要为单个数据集重复实现下载与清洗框架。
- 新适配器至少应有 metadata、下载路径、LazyFrame 交易读取和合成小文件测试。
- 如果需要新的图语义，应继续扩展独立 graph builder，而不是把图构建逻辑塞进 dataset adapter。

## Temporal transaction graph data module

位置：`src/amlgraphx/data/`

- canonical transaction schema 的实现已从 `datasets/schema.py` 移至
  `data/schema.py`；`datasets` 继续导出 `normalize_transactions()`，旧模块保留轻量兼容导入。
- `TransactionGraphDataModule` 按“完整事务图 → 时间诱导切分 → split 内滑动窗口”组织数据。
- `split_transaction_graph()` 使用半开时间区间建立 train、validation 和 test
  诱导子图，并删除跨 split 的边。
- `sliding_snapshots()` 支持独立的 `window_size`、`stride` 和
  `drop_last`，窗口使用 `[start_time, end_time)` 语义。
- `GraphSnapshot.edge_index` 使用 PyG 风格的 `[2, E]` 局部节点索引，不构造
  dense adjacency matrix；节点和边属性继续保留为 Polars 表。
- 新增 temporal split、跨区间边隔离、滑动窗口、稀疏边索引和配置校验测试。
- 自动化验证：`42 passed`，Ruff lint 与 `src tests` format check 通过。
- 真实数据验证使用独立临时目录并在完成后删除：PaySim 为
  `6,362,620 / 56`（节点/边），IBM HI-Small 为
  `5,078,345 / 2,176,494`；两个数据集均完成 split 和 snapshot smoke test。

## 双语代码注释与 Polars 说明

位置：`src/amlgraphx/graph/graphs.py`、`src/amlgraphx/data/`

- 为账户图、交易图、schema 统一和 temporal snapshot 流程补充中英文模块说明、
  类/函数 docstring 及关键算法注释。
- 在实际使用位置解释了 `pl.col`、`with_columns`、`filter`、`select`、
  `join`（包括 `semi`/`inner`/`left`）、`unique`、`coalesce`、`concat_str`、
  `collect_schema` 和 `LazyFrame.collect()`，并加入小例子。
- 未改变运行逻辑；验证结果为 `42 passed`，相关 Ruff 检查和
  `git diff --check` 通过。

## Native tabular graph features

位置：`src/amlgraphx/tabular/`、`rust/tabular/`

- 新增 `amlgraphx.tabular.GraphFeaturePreprocessor`，提供 SnapML 风格的
  `fit`、`partial_fit`、`transform`、`fit_transform`、`get_params` 和
  `set_params` 接口；输入为
  `[edge_id, source_id, target_id, timestamp, ...numeric_features]`。
- 图状态是带时间窗和最大边数淘汰策略的有向多重图；重复的 active edge ID
  与 SnapML 一样会被忽略，时间窗下界使用排他语义。
- 支持 fan in/out、degree in/out、scatter-gather、temporal cycle、长度受限
  simple cycle，以及四个端点/方向的 account statistics。Python 层额外提供
  `transform_causal()`：逐行处理已排序事件，避免普通批处理内的未来可见性。
- Rust 后端使用每个实例私有的有界 Rayon 线程池：图插入/淘汰串行，特征行在
  不可变图状态上并行计算；没有全局 worker pool、mutex 或共享可变输出缓冲。
  这保持结果顺序和确定性，并避免锁顺序、死锁和跨实例状态污染。
- 小型确定性 fixture 与随机简单多重图无关的 batch 对齐 SnapML 1.17.2；覆盖
  空 batch schema、时间淘汰、重复 edge ID、严格因果模式、串行/并行一致性和
  两个 Python 线程并发的独立实例。
- Rust `cargo test` 为 5 passed，Python 全套测试为 50 passed；Ruff、Rust fmt
  和 Clippy（warnings-as-errors）均通过。
- 用论文的配置（128 行 batch、scatter-gather 为 6 小时、其余模式为 1 天、
  vertex stats 使用 timestamp 与 amount）对真实 PaySim、IBM HI-Small、
  IBM LI-Small 各执行 512 条、4 个 batch 的 smoke test，三者均返回
  `(512, 182)` 且所有数值有限。真实 ZIP、解压数据和缓存均在 `/tmp` 临时目录
  中使用后删除。

## GFP paper / SnapML compatibility review

位置：`rust/tabular/`、`tests/test_tabular_graph_features.py`、
`examples/graph_feature_preprocessor_ibm_aml.py`

- 通过 MarkItDown 审阅 GFP 论文、IBM Z AML 参考流水线，以及 venv 中 SnapML
  1.17.2 的 `GraphFeaturePreprocessor` 包装层。确认公共 API、128 行 batch、
  预热 `fit`、动态多重图和“写入串行 / 不可变图并行读取”的总体设计一致。
- 修正了与 SnapML 的三个可复现边界差异：fan/degree 对等时间戳的并发前缀、
  pattern window 按候选事件而非 batch 最大时间戳滑动，以及跨 batch 同时刻事件
  只作为历史图状态而不重复发射模式；账户 ratio 采用 SnapML 的 `degree / fan`。
  新增三个基于 SnapML 的确定性回归测试，tabular 测试现为 11 passed。
- 发布一个端到端的中英双语注释示例：下载 IBM HI-Small、惰性加载 canonical
  transactions、统一账户 ID、GFP history warm-up 和 128-row transform batch；
  严格因果 `transform_causal()` 的速度/泄漏权衡也在示例中说明。
- release 实测（4,000 条高连接度交易、fan+degree）：1 worker 4.0224 s，4 workers
  1.0100 s（3.98x）；每实例 Rayon pool 的 OS 线程数同时增加，验证 native 多线程
  实际生效。最终验证：Rust `cargo test` 5 passed、Clippy `-D warnings` 通过、
  Python 全套 53 passed、Ruff check 通过、`git diff --check` 通过。

## Full graph chronological node masks

位置：`src/amlgraphx/split/temporal.py`、
`examples/paysim_transaction_graph.py`

- 新增 `TemporalNodeMasks` 和 `build_temporal_node_masks()`，在一张完整的
  `TransactionGraph` 上按半开时间区间生成 PyTorch boolean masks，不删除节点或边。
- 保留原有 `TemporalSplit`、`apply_temporal_split()` 和
  `split_transaction_graph()` 的独立诱导子图行为，避免改变已有协议和兼容性。
- example 已从 PaySim 改为 IBM HI-Small，并展示完整图、chronological masks 和
  snapshot；支持传入临时 `cache_dir` 以便可重复验证和清理数据。
- 真实数据验证：IBM HI-Small 为 `5,078,345 / 7,853,196`、IBM LI-Small 为
  `6,924,049 / 14,149,999`（节点/边，`edge_delta=4 hours`）；两个数据集的 mask
  均覆盖全部节点且互不重叠，临时下载目录均已删除。
- 更新后的 HI-Small example 使用临时缓存真实运行成功，数据已删除。
- 最终 Python 验证：`54 passed`；Ruff check、format check 和 `compileall` 均通过。

## Native parallel transaction graph construction

位置：`rust/graph/temporal_edges.rs`、`rust/graph/mod.rs`、
`src/amlgraphx/graph/graphs.py`

- `TransactionGraph.from_transactions()` 保持公共 API 和 Polars schema 不变，
  但 temporal successor matching 与 edge position emission 已改为始终调用私有
  Rust kernel；没有 Python fallback、backend selector 或逐行 PyO3 调用。
- Python/Polars 继续负责字段解析、清洗、时间排序和共享 categorical account code；
  Rust 一次接收 contiguous `u32/u32/i64` 数组，并返回三条 contiguous `i64`
  edge arrays。时间语义保持 `(current_time, current_time + delta]`，同时间交易不连边，
  上界包含，输出顺序确定。
- kernel 使用进程级有界 Rayon pool；输入在释放 GIL 前复制，worker 只读共享索引并写入
  自己的 chunk-local vectors，最后按 chunk 顺序合并。小于 4,096 行时走串行 kernel，
  避免并行调度成本；没有 Mutex、RwLock、channel、atomic、unsafe 或手动线程生命周期。
- 修复测试发现的极大 `delta` 中间加法溢出：使用 `i128::saturating_add` 并截断到
  `i64::MAX`，避免 worker panic；PyO3 边界仍使用 `catch_unwind` 将意外 panic 转成
  `RuntimeError`。
- release benchmark：100 万节点/999,999 边 native kernel，4 workers 中位数
  `0.069249 s`，1 worker `0.163123 s`（约 2.36x）；20 万节点/399,898 边完整
  graph API 为 `0.223968 s`，等价 Python reference 为 `0.975437 s`（4.36x）。
- 全量真实验证（`delta=4h`）：IBM HI-Small 为
  `5,078,345 / 7,853,196`（节点/边，11.109 s），IBM LI-Small 为
  `6,924,049 / 14,149,999`（16.004 s）。两次下载均使用独立 `/tmp`
  `TemporaryDirectory`，退出后断言目录已删除。
- 新增 Rust temporal boundary、overflow、validation、determinism 测试，以及 Python
  native/reference 等价、空 edge schema、错误映射和 4 个 concurrent callers 测试。
  最终验证：Rust 9 tests passed，Clippy `-D warnings` 通过，Python 58 tests passed，
  Ruff lint、相关 format check 和 compileall 通过。

## IBM graph thread-scaling experiment

实验条件：64 logical CPUs、约 753 GiB available memory；四个变体各下载一次，
随后每个线程数在独立 Python 进程中重新扫描同一份数据。`delta=4h`、
`POLARS_MAX_THREADS=64` 固定，`OMP_NUM_THREADS=MKL_NUM_THREADS=1`，只改变
Rust Rayon 的 `RAYON_NUM_THREADS`（1、2、4、8、16、32、64）。每个条件执行一次
完整 `build_transaction_graph()`，所以结果适合扩展性判断，不应当解释为严格统计中位数。

| variant | 1 thread | 2 | 4 | 8 | 16 | 32 | 64 | best speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HI-Small (7,853,196 edges) | 12.450s | 12.102s | 11.661s | 11.284s | 11.253s | 11.074s | 11.156s | 1.12x @32 |
| LI-Small (14,149,999 edges) | 17.717s | 16.756s | 16.509s | 16.726s | 16.435s | 15.648s | 15.843s | 1.13x @32 |
| HI-Medium (168,731,526 edges) | 104.386s | — | 96.199s | 92.170s | 92.073s | 88.973s | 88.980s | 1.17x @32 |
| LI-Medium (164,698,987 edges) | 100.411s | — | 87.502s | 88.023s | 89.717s | 89.646s | 86.074s | 1.17x @64 |

- Medium 构图峰值 RSS 约为 26.4–29.5 GiB；64 线程相对 1 线程约增加
  9–10%，没有出现无界线程或内存失控。所有线程数的节点/边计数完全一致。
- 端到端最佳线程数没有稳定地随 CPU 数量增加而增加：Small 在 32 附近平台，
  HI-Medium 在 32 附近平台，LI-Medium 的单次测试 64 最佳。Rust edge kernel
  的并行收益被 CSV 读取、Polars 清洗、排序、categorical 编码和结果物化部分稀释。
- 实用默认建议为 `RAYON_NUM_THREADS=32`；如果 LI-Medium 类负载更常见且内存余量
  足够，可单独测试 64。设置必须发生在 Python 进程启动前，例如
  `RAYON_NUM_THREADS=32 uv run python examples/paysim_transaction_graph.py`。
- 四个实验均使用临时目录 `/tmp/amlgraphx-thread-small-*` 或
  `/tmp/amlgraphx-thread-medium-*`，实验结束后目录已删除。

## Account temporal representations and PyG interoperability

位置：`src/amlgraphx/graph/temporal/`、`src/amlgraphx/graph/pyg.py`、
`examples/account_graph_representations.py`

- `prepare_graph()` 现在完整支持 account-as-node 的 time-aware static、snapshot
  和 event stream。static/snapshot 每笔交易仍是一条独立有向边，金额、标签、
  时间和数据集特有字段全部保留；snapshot 使用边时间的半开窗口，不聚合平行边。
- 新增 `AccountEventStream`，按时间稳定排序交易事件并保留完整 Polars 事件表；
  transaction-as-node event stream 因需要额外 node-arrival 语义而继续显式拒绝。
- 检查当前 venv 的 PyG 2.8.0 源码后确认不需要自定义子类：`Data` 原生接收
  `edge_attr`、`time` 和任意 `**kwargs`，`TemporalData` 原生提供
  `src/dst/t/msg`。`to_pyg_data()` 统一转换 `AccountGraph` 与
  `TransactionGraph`，使用标准 `Data.edge_time` 自定义属性；
  `to_pyg_temporal_data()` 返回标准 `TemporalData`。
- PyG 特征列必须由研究员显式选择并为数值类型，避免隐藏的类别编码或归一化；
  Polars `Duration` edge feature 会转换为秒。字符串 ID 与完整原始特征继续留在
  AMLGraphX 的 Polars graph/stream 对象中。
- 合成验证：Python 全套 `64 passed`，包含 account 三种表示、两种 node type 的
  `Data`、event stream 的 `TemporalData`、edge feature/time、PyG Batch 和类别字段
  显式拒绝测试。
- IBM 全量真实 smoke test（`edge_delta=4h`、snapshot `bin_size=stride=1 day`）：
  HI-Small account graph `515,080 / 5,078,345`、首个 account snapshot
  `437,065 / 1,114,921`、transaction graph `5,078,345 / 7,853,196`、首个
  transaction snapshot `1,114,921 / 488,435`；LI-Small 对应为
  `705,903 / 6,924,049`、`599,154 / 1,524,807`、
  `6,924,049 / 14,149,999`、`1,524,807 / 741,009`。两个数据集的 static、
  snapshot、event stream 与 PyG tensor shape 全部通过。
- 真实数据使用 `/tmp/amlgraphx-ibm-small-representations-*` 临时目录；测试结束后
  已确认目录不存在，没有保留下载数据。

## PR #4 review follow-up

- `snapml` 只用于测试中的 parity oracle，已从运行时依赖移动到 `dev` dependency
  group，普通安装不再受其平台 wheel 可用性限制。
- GFP vertex statistics 现在按 `vertex_stats_tw` 截断；重复
  `vertex_stats_feats` 会被显式拒绝；严格因果转换要求输入时间严格递增，且必须晚于
  当前保留历史，避免从未来 state 读取特征。
- 交易图 native kernel 会拒绝无法由 `int64` 纳秒表达的 `time_delta`，而不发生
  overflow/wrap。
- `TransactionGraphDataModule` 的 validation/test snapshot 会从完整图保留
  `edge_delta` lookback，并通过局部 `target_mask` 区分历史 context 与评估目标；
  原有 `split_transaction_graph()` 仍保留严格诱导子图协议。
