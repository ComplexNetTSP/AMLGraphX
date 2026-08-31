# IBM AML CSV 字段记录

## 当前数据集层功能

AMLGraphX 当前提供一个可复用的数据集抽象层，主要能力包括：

- 使用 `Dataset` 抽象类统一数据集接口：`metadata`、`download()` 和
  `transactions()`。
- 使用 Pydantic 校验数据集元数据；交易记录不转换为 Pydantic 对象。
- 使用 `huggingface_hub` 下载 Hugging Face 数据集，支持 revision、缓存目录、
  本地目录和指定文件。
- 默认缓存目录为 `~/.cache/amlgraphx/`，已存在且完整的数据集会复用本地缓存。
- 使用安全 ZIP 解压、文件查找和预期文件完整性校验。
- 使用 Polars 的 `scan_csv()` / `scan_parquet()` 返回 `pl.LazyFrame`，避免在
  调用数据集方法时立即加载完整大文件。
- 使用 `clean_lazy_frame()` 做保守清洗：删除空行和缺失 source/target 的行，
  数值缺失值使用中位数填充，字符串缺失值使用 `UNKNOWN` 填充，并尝试惰性
  解析时间字段，同时保留原始时间列。
- 通过数据集注册表和 `load_dataset()` 提供统一入口，支持 `ibm-aml`、
  `paysim` 和 `saml-d`，以及 IBM 的名称别名。

当前尚未实现图结构构建、特征工程、数据集划分、采样器、模型训练或
PyTorch Geometric/DGL 转换。

## 公共加载示例

```python
from amlgraphx.datasets import IBMAML, load_dataset

dataset = IBMAML("hi-small")
transactions = dataset.transactions()  # polars.LazyFrame
accounts = dataset.accounts()          # polars.LazyFrame
patterns = dataset.patterns()          # polars.LazyFrame

dataset = load_dataset("ibm-aml", variant="hi-small")
transactions = dataset.transactions()
```

`download()` 返回本地数据集根目录；`transaction_path()`、`accounts_path()`
和 `patterns_path()` 返回对应文件路径。PaySim 和 SAML-D 适配器也提供
`download()`、`transaction_path()` 和 `transactions()`。

## 元数据和任务定义

目前支持的枚举值为：

| 类型 | 值 |
| --- | --- |
| `TaskType` | `transaction_classification` |
| `LabelLevel` | `transaction` |
| `DatasetSource` | `huggingface` |

IBM AML、PaySim 和 SAML-D 当前都按交易级分类任务描述。Hugging Face 仓库
均为项目使用的第三方镜像，不表示由原始数据集发布方维护。

本记录来自 Hugging Face 仓库 `LordNR/AMLGraphX-IBM-AML` 的六个 ZIP
变体。每个变体都按顺序完成了下载、解压和 Polars `scan_csv()` schema
读取，随后删除了临时缓存和解压目录。ZIP 中的 macOS `._*` 元数据文件
未计入字段记录。

## 交易 CSV

六个变体的交易表字段一致：

| 字段 | Polars 类型 |
| --- | --- |
| `Timestamp` | `String` |
| `From Bank` | `Int64` |
| `Account` | `String` |
| `To Bank` | `Int64` |
| `Account_duplicated_0` | `String` |
| `Amount Received` | `Float64` |
| `Receiving Currency` | `String` |
| `Amount Paid` | `Float64` |
| `Payment Currency` | `String` |
| `Payment Format` | `String` |
| `Is Laundering` | `Int64` |

`Account_duplicated_0` 是 Polars 对原始重复账户列的标准化名称；加载器将
它作为目标账户列识别。

## 账户 CSV

六个变体的账户表字段一致：

| 字段 | Polars 类型 |
| --- | --- |
| `Bank Name` | `String` |
| `Bank ID` | `Int64` |
| `Account Number` | `String` |
| `Entity ID` | `String` |
| `Entity Name` | `String` |

## 六个变体文件与大小

| 变体 | 交易文件 | 交易大小 | 账户文件 | 账户大小 |
| --- | --- | ---: | --- | ---: |
| `hi-small` | `HI-Small_Trans.csv` | 475,664,283 B | `HI-Small_accounts.csv` | 34,053,187 B |
| `hi-medium` | `HI-Medium_Trans.csv` | 3,031,783,420 B | `HI-Medium_accounts.csv` | 145,008,642 B |
| `hi-large` | `HI-Large_Trans.csv` | 17,052,760,651 B | `HI-Large_accounts.csv` | 147,691,860 B |
| `li-small` | `LI-Small_Trans.csv` | 650,422,357 B | `LI-Small_accounts.csv` | 47,248,016 B |
| `li-medium` | `LI-Medium_Trans.csv` | 2,976,436,978 B | `LI-Medium_accounts.csv` | 141,781,157 B |
| `li-large` | `LI-Large_Trans.csv` | 16,742,513,790 B | `LI-Large_accounts.csv` | 144,349,585 B |

每个变体还包含对应的 `*_Patterns.txt` 文件；它不是 CSV，因此没有列字段
schema。

## PaySim

来源仓库：`LordNR/AMLGraphX-Paysim`。实际文件为 `paysim.csv`，大小为
493,534,783 B。

| 字段 | Polars 类型 |
| --- | --- |
| `step` | `Int64` |
| `type` | `String` |
| `amount` | `Float64` |
| `nameOrig` | `String` |
| `oldbalanceOrg` | `Float64` |
| `newbalanceOrig` | `Float64` |
| `nameDest` | `String` |
| `oldbalanceDest` | `Float64` |
| `newbalanceDest` | `Float64` |
| `isFraud` | `Int64` |
| `isFlaggedFraud` | `Int64` |

## SAML-D

来源仓库：`LordNR/AMLGraphX-SAML-D`。实际文件为 `SAML-D.csv`，大小为
996,168,850 B。

| 字段 | Polars 类型 |
| --- | --- |
| `Time` | `String` |
| `Date` | `String` |
| `Sender_account` | `Int64` |
| `Receiver_account` | `Int64` |
| `Amount` | `Float64` |
| `Payment_currency` | `String` |
| `Received_currency` | `String` |
| `Sender_bank_location` | `String` |
| `Receiver_bank_location` | `String` |
| `Payment_type` | `String` |
| `Is_laundering` | `Int64` |
| `Laundering_type` | `String` |

PaySim 和 SAML-D 也均按单个临时目录顺序完成下载、解压和 schema 读取，
之后已删除临时缓存与解压文件。
