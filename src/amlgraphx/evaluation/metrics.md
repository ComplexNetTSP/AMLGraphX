# Fraud 与 AML 评估指标

本文件定义 AMLGraphX 中评估 fraud（欺诈）与 AML（反洗钱）风险评分时的常用指标、
计算方式和适用场景。对应的 NumPy 无状态 API 位于 ``evaluation.metrics``；Torch
API 位于 ``evaluation.torch_metrics``，其中每个指标都是独立、可实例化的
``torchmetrics.Metric`` class。两者都不负责数据切分、阈值调优或训练。

```python
from amlgraphx.evaluation import evaluate_binary_risk_scores

result = evaluate_binary_risk_scores(
    y_test,
    risk_score,
    threshold=0.5,  # 必须在 validation set 上选择，或预先规定
    top_fractions=(0.001, 0.01),
)
print(result.as_dict())
```

神经网络训练可按 TSL 的方式选择并命名需要记录的 metric。每个 class 累积多个
batch，再在 epoch / validation / test 结束时统一 ``compute()``：

```python
from amlgraphx.evaluation import (
    AveragePrecision,
    F1,
    Precision,
    PrecisionAtK,
    Recall,
    RecallAtK,
    RocAuc,
)

metrics = {
    "precision": Precision(threshold=0.5),
    "f1": F1(threshold=0.5),
    "average_precision": AveragePrecision(),
    "roc_auc": RocAuc(),
    "precision_at_100": PrecisionAtK(100),
    "recall_at_100": RecallAtK(100),
}

for risk_score, target in validation_loader:
    for metric in metrics.values():
        metric.update(risk_score, target)  # score first, then binary target

results = {name: metric.compute() for name, metric in metrics.items()}
```

`AveragePrecision`、`RocAuc` 和所有 ``@K`` metric 会保留完整 split 的 detached
score / target state，确保全局排序正确；不能分别计算每个 batch 后再取平均。

## 先确定评估对象

每条待评估记录都需要有一个明确的**评分单位**。它可以是一笔交易、一个账户、
一个地址、一个图节点，或一条图边；一个实验只能在同一种单位上计算一组指标。

记第 `i` 条记录的真实标签为 `y_i ∈ {0, 1}`，其中 `1` 表示 fraud / 可疑 AML，
模型风险评分为 `s_i`。评分越大，表示越应被优先调查。所有排序都按 `s_i` 降序；
若评分相同，必须采用可复现的次级排序（例如稳定的 `transaction_id`）。

在开始计算前，报告必须写清：

- 评分单位及其总数 `N`；
- 正类的业务定义和正类数 `P = Σ y_i`；
- 数据切分协议（随机、时间、严格因果或 transductive）；
- `K` 是固定调查数量，还是总体中的 `p%`；
- 阈值是否只在 validation set 上选择。

图表示不会改变指标的数学定义，但会改变分母。例如 transaction-level
`Precision@1%` 的分母是排名前 1% 的交易，而 account-level 指标的分母是账户。
二者不能直接混合比较。

## 基础计数与阈值指标

给定阈值 `τ`，将 `s_i ≥ τ` 判为正类。由此得到：

| 计数 | 含义 |
|---|---|
| `TP` | 正确命中的 fraud / AML 正类 |
| `FP` | 被告警但实际为负类的记录 |
| `FN` | 未被告警但实际为正类的记录 |
| `TN` | 正确排除的负类记录 |

| 指标 | 计算 | 回答的问题 | AML / fraud 中的用途 |
|---|---|---|---|
| Precision | `TP / (TP + FP)` | 被调查的告警中有多少是真的？ | 衡量误报负担与调查效率。 |
| Recall / TPR | `TP / (TP + FN)` | 已知风险中抓到了多少？ | 衡量覆盖率。 |
| F1 | `2 × Precision × Recall / (Precision + Recall)` | Precision 与 Recall 的平衡如何？ | 常报告**少数类 F1**，必须同时给出阈值。 |
| Fβ | `(1 + β²)PR / (β²P + R)` | 更重视 Precision 或 Recall 时表现如何？ | `β > 1` 更重视漏报，`β < 1` 更重视误报。 |
| FPR | `FP / (FP + TN)` | 正常记录中有多少被错误告警？ | 可作为运营负担的补充，不宜单独作主指标。 |

`Precision`、`Recall` 和 `F1` 都依赖阈值。阈值不得根据 test set 的标签挑选；
应在 validation set 上按业务政策确定，例如“每天最多调查 500 个账户”或
“Precision 至少为 20%”。

### 为什么 Accuracy 通常不是主指标

`Accuracy = (TP + TN) / N`。在正类极少时，一个永远预测“正常”的模型也可能有很高
Accuracy，却有 `Recall = 0`。因此 Accuracy 可以作为完整性信息报告，但不应用于
AML / fraud 模型选择或核心结论。

## 排名与调查预算指标

在 AML 与 fraud 工作流中，风险评分通常用于决定有限调查资源的优先级，而不是直接
执行一个固定分类阈值。令 `TopK` 为评分最高的 `K` 条记录：

| 指标 | 计算 | 解释 |
|---|---|---|
| Precision@K | `Σ(i ∈ TopK) y_i / K` | 调查前 `K` 个告警时，命中率是多少？ |
| Recall@K | `Σ(i ∈ TopK) y_i / P` | 全部已知风险中，前 `K` 个已覆盖多少？ |
| F1@K | 用 `Precision@K` 与 `Recall@K` 计算 F1 | 固定调查容量下的综合表现。 |
| Lift@K | `Precision@K / (P / N)` | 相对随机抽查，前 `K` 的命中率提升了多少倍？ |

当预算以比例表达时，先定义 `K = ceil(p × N)`，再计算相应的 `@K` 指标。报告中应
写作 `Precision@0.1%` 或 `Recall@1%`，不能只写模糊的 “top performance”。

推荐至少报告两个贴近实际调查能力的预算点，例如：

```text
Precision@0.1%, Recall@0.1%
Precision@1%,   Recall@1%
```

`Recall@K` 的分母始终是**评估集内全部已知正类 `P`**，不是 `K`，也不是被告警的
记录数。若评估集没有正类，Precision、Recall、F1 和 PR-AUC 的含义不成立；实现应
返回明确的缺失/错误状态，而不是悄悄用零掩盖问题。

## 阈值无关的排序指标

| 指标 | 计算思想 | 何时使用 | 注意事项 |
|---|---|---|---|
| PR-AUC / Average Precision | 沿全部阈值汇总 Precision–Recall 曲线 | AML、极端不平衡 fraud 的首选全局排序指标 | 必须说明是曲线面积还是 `average precision` 的离散实现；两者数值可能不同。 |
| ROC-AUC | 随机抽取一对正、负样本时，正样本评分更高的概率（平分相同评分） | 便于与历史文献横向比较 | 在负类极多时可能看起来很好，却对应很低的实际 Precision。 |

实践中建议将 **PR-AUC 作为 AML / 高不平衡 fraud 的主排序指标**，ROC-AUC 作为
补充。PR 曲线也应随结果保存，因为它显示了 Precision 与 Recall 随阈值变化的业务
权衡；单个 AUC 无法替代这一信息。

## 按场景选择指标

### 1. 监督式 AML / fraud

训练集含标签，模型学习分类概率或风险评分；validation 与 test 也有已冻结的真值标签。

**推荐主报告：**

```text
PR-AUC
Precision@0.1%、Recall@0.1%
Precision@1%、Recall@1%
少数类 Precision、Recall、F1（阈值来自 validation）
```

**补充报告：** ROC-AUC、PR curve、Lift@K、混淆矩阵和阈值/调查容量。

对于相对不那么稀有的 fraud 数据，ROC-AUC、F1 与 Accuracy 在文献中仍很常见；
但只要正类比例低或审核资源有限，仍应优先 PR-AUC 和预算指标。

### 2. 无监督 / 异常检测：离线评估时有标签

训练过程不使用 fraud / AML 标签，但基准数据的 validation/test 标签可以用于**评估**。
此时不应因为模型是无监督就换成另一套分类指标：将 anomaly score 当作风险评分，
使用与监督模型相同的排序指标。

**推荐主报告：**

```text
PR-AUC
Precision@K、Recall@K（K 由调查预算定义）
```

**补充报告：** ROC-AUC、F1@K，或在 validation set 上确定异常阈值后的少数类
Precision / Recall / F1。

无监督方法的阈值尤其不能从 test labels 调整；若需要阈值，应使用训练分数、
validation 分数或事先确定的告警容量规则。

### 3. 无监督线上运行：当前没有真值标签

没有已确认标签时，ROC-AUC、PR-AUC、Recall 和 F1 都**不能计算**。此时报告的是运营
观测，而不是已完成的分类效果评估：

| 运营指标 | 计算 | 边界 |
|---|---|---|
| Review yield | `已确认正例数 / 已完成审核数` | 只代表已审核队列；若审核对象不是随机抽样，不能当作全体 Precision。 |
| Alert volume | 每个时间窗口进入队列的告警数 | 必须和实际调查容量一起报告。 |
| Queue coverage | `已审核告警数 / 产生告警数` | 衡量积压，不代表模型准确率。 |
| Score / alert stability | 相邻窗口的分数分布、告警量或 Top-K 重叠度 | 用于发现漂移，不代表抓获率。 |

待标签成熟后，应回填到按时间冻结的离线评估集，重新计算 PR-AUC、Top-K
Precision/Recall 等效果指标。若只审核 Top-K，高风险选择偏差会使直接估计全体
Precision 变得不可靠；需要随机审核一部分未告警记录，或明确报告该限制。

## 时间与图数据的额外规则

AMLGraphX 的评价必须与图和时间语义一致：

- **时间切分优先。** 用未来交易、未来图边、未来聚合特征或未来标签评估过去目标会产生泄漏。
- **历史 context 与 target 分开。** 可以用过去记录构造图或预热模型，但这些记录不一定是当前预测目标。
- **图的全局可见性要声明。** 若所有节点和边都可见、仅掩盖 test 标签，这是 transductive 协议，不能称为严格因果评估。
- **只在目标集合上计算。** 有 lookback context 的 snapshot / temporal 模型，应只用其 `target_mask` 对应的记录作为指标分母。
- **不聚合不同任务层级。** transaction、account 和 subgraph 的指标应分别报告；只有定义了明确映射和去重规则后才能比较。

## AMLGraphX 的最小结果表

除非研究问题要求其他内容，一个可比较的 AML / fraud 实验至少应留下：

| 字段 | 示例 |
|---|---|
| 任务单位 | transaction |
| 正类定义 | `label = 1` 表示已标注 laundering transaction |
| 切分 | strict chronological; train / validation / test = 70 / 10 / 20 |
| 评分 | test risk score, descending; stable tie-breaker = `transaction_id` |
| 主指标 | PR-AUC, Precision@0.1%, Recall@0.1%, Precision@1%, Recall@1% |
| 阈值指标 | minority Precision / Recall / F1 at validation-selected threshold |
| 补充指标 | ROC-AUC, Lift@K, PR curve |
| 置信信息 | 随机种子、重复运行或 bootstrap 区间（如果适用） |

## 文献依据与术语

这套选择基于本仓库中审阅的两篇论文：

- *Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation*：该综述发现 AML 研究常报告 Precision、Recall、F1 与 ROC-AUC，但指出 Accuracy 在不平衡任务中不合适，并建议更重视 PR-AUC 和调查预算下的结果。
- *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*：该实验将少数类 F1、Precision、Recall 与 PR 曲线作为不平衡 AML 分类的主要分析工具。

“PR-AUC”在不同工具中可能指梯形积分的 PR 曲线面积，也可能指 Average Precision。
AMLGraphX 将来实现该指标时，必须在 API 名称、文档和实验结果中明确具体定义与所用后端。
