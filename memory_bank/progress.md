# AMLGraphX 当前进度

更新时间：2026-08-25

## Git 状态

- 当前分支：`feature/datasets`
- 当前提交：`5d2266e Fix dataset graph compatibility`
- 远端：`origin/feature/datasets` 已同步
- 已跟踪文件：无未提交修改
- 本地未跟踪支持文件：`.codegraph/`、`AGENTS.md`、`memory_bank/`
- 本阶段没有向 `main` 推送

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
