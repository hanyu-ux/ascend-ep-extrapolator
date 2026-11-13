# 源算子 bound 的证据口径

bound判据版本：`20261110.1.evidence_roofline`；热点统计接口：核心／生成器 `20261113.1.exclusive_hcpre`。运行时保持Python标准库依赖；Attention准备识别单列为同目录模块，不安装外部 Skill，不调用设备，保留整层时间、外推公式和共同源rank；新热点按完整层区间统计，旧JSON保留所存阶段内热点口径。

公开检索、实际代码审查与不采用的规则见 [Skill 调研](operator-bound-skill-survey.md)。这里把统计事实、候选判断和定量模型分开，不把某个百分比的最大值当作已定位根因。

## 报告如何读

方法1、方法2和方法3的CSA、HCA、SWA各自源实测Top8后依次展示：流水表、完整高级指标与候选依据、Roofline 定量表、模型依据与缺失项。包含这些Top8中的Attention/MoE；不是声称每个算子均有硬件证据。新报告从 `projection.layer_mean_methods[].operator_hotspots[].kernels` 读取热点，层内阶段独立保留（新11项将HCPre单列，旧10项保持原证据），不计算逐算子互斥覆盖或目标算子贡献。源PMU不乘目标层数和外推倍率。

| 层级 | 说明 | 能得出的结论 |
| --- | --- | --- |
| 原始事实 | 实际导出的流水比、核内时间、调用数、逐字段覆盖与范围 | 被统计的流水有多忙、哪些指标缺失 |
| 流水筛查 | 同核竞争流水、短调用、波动、联合覆盖、核内时差 | 计算/搬运/标量活动候选，置信度至多中 |
| 显式 Roofline | 同一源逻辑 NPU、同计算 dtype 峰值、有效工作量、同存储层级流量 | 给定假设下偏计算/存储侧及理想下限，不是因果认证 |
| 深入定位 | msOpProf 每核负载、指令流水、cache、真实流量、依赖与代码 | 后续才能核对等待、重叠、负载不均或具体优化收益 |

`bound_verified` 始终为 `false`。没有增加“未识别就默认 compute-bound”的分支。

## 流水与单位

| 指标 | 含义与限制 |
| --- | --- |
| AIC MAC | 矩阵指令流水活动周期占 AIC 核侧统计周期的比例；不是机器峰值 FLOPs 利用率 |
| AIV VEC | 向量指令流水活动周期占 AIV 核侧统计周期的比例；不能拿 `1−VEC` 算 C/V 空泡 |
| Scalar AIC/AIV | 各自核侧标量/控制指令流水活动；不是服务器 CPU 占用，也不直接证明控制瓶颈 |
| MTE1 | 片内供数相关通路，例如 L1→L0A/L0B；具体还需核对架构和指令 |
| MTE2 | GM 向核内存储供数相关通路；经过 cache 时，GM 逻辑读取量不等于 HBM 物理读取量 |
| MTE3 | 逻辑写出或片内搬运；例如 UB→GM，部分架构还有 UB→L1，不能统一命名为写显存 |
| Fixpipe | 矩阵结果搬运及随路处理；目标可为 GM/L1，部分架构还有 UB，不能统一称为片内搬运 |
| 核内时间/总周期 | 实际导出的核侧统计范围；不等于全任务墙钟，也不能把差值直接归为等待 |
| 流量 `*_datas_kb` | 原导出 KB 数值；文档未确认二进制/十进制时不自动换 bytes；按路径保留 |
| 带宽 `*_bw_gbps` | 原导出的 GB/s；不同通路/核侧不相加，不自动当成整芯片 HBM 带宽 |
| `*_fops` | 原导出的运算计数；可能是每核/子核及指令工作量，不自动当成有效算法 FLOPs，不再次乘2 |

新解析器识别有明确单位和文档依据的流量、通路带宽、cache、指令工作量字段，并附 `field_info` 来源/范围/estimate 标签。只有实际列存在才显示；msOpProf 每核导出和 msprof 任务级导出不可仅凭同名拼接。各通路的使用范围见 [Ascend C 数据通路](https://asc.gitcode.com/api/SIMD-API/c_api/data_move_guide/overview/data_path.html) 和 [msOpProf 性能字段](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md)。

比例按唯一原生整次调用的 Duration 加权，保留 min/max、逐指标样本数、时长覆盖。新的流量/工作量还额外保留 `sample_sum` 和 `sample_mean`；这些只属于字段已知的导出行，不能把加权 bytes 均值除以调用均时求实际带宽。层边界裁剪只用于热点耗时，不能裁出局部PMU。同一调用跨层边界或旧阶段被引用时，不重复计为独立采集。

## 候选判据

AIC 的基础竞争项为 MAC、Scalar、MTE1、MTE2、Fixpipe；AIV 为 VEC、Scalar、MTE2、MTE3。AIC MTE3 有实测则继续作为竞争项，但缺少它不自动降低 AIC 完整性。该清单是核角色相关筛查，不是已验证某个芯片型号的完整数据通路。具体架构未知时保留拓扑限制，绝不搬用其他 NPU 的峰值。

必需项齐备时才比较主导候选：计算≥90%且领先搬运/标量≥10个百分点；搬运≥75%且领先计算≥15、标量≥10个百分点；标量≥60%且领先两者≥10个百分点。计算与搬运均≥85%且相差不足10个百分点时提示同时高活动。**这些是本工具版本化启发式，不是硬件阈值或官方认证标准。** 缺少竞争项时只能写某流水偏忙，不能把缺值当0。

置信度还受以下事实限制：必需项缺失或逐项/联合时长覆盖不足80%；原调用时长≤20μs的部分达到一半；指标 min/max 波动≥35个百分点；无法核对原生逐调用诊断。不同字段分别有80%覆盖，不等于它们在同一批调用同时存在，因而单列联合覆盖。核内时差只在对应活动侧时间可比时检查，MIX 需两侧，异常仍称“时间差异”，不直接归因同步或空泡。

通信/路由复合、同步控制、AICPU 分流处理；不从 MAC 低推断 HBM/CPU bound。AIC/AIV 分母不同，比例不能直接相加或相减。流水可并行工作，所有比值合计可以超过100%。

## 已有 JSON 的离线更新

```bash
python -u build_ep128_leadership_excel.py --report ep128_report_7/model_architecture_report.json --refresh-bound --out "ep128_report_7/专业bound.xlsx"
```

生成Excel、同名HTML、`专业bound.bound.json`，原JSON不变；不读CSV／trace，不改已有周期、热点耗时和rank。旧JSON已有的阶段内热点按兼容口径展示，保留其层内十阶段展开，不再绘制逐算子目标贡献。重新处理原CSV才能得到新的完整层Top8；PMU刷新不会重排或重建采样热点。旧 JSON 未存 `sample_diagnostics` 时，联合覆盖、短调用和逐调用时差无法从均值恢复，对应检查未知、置信度降低。原本未采集的 MTE1 或访存量不会凭空补出。

可同时加 `--align-source-rank`，先按已存逐 rank 估算对齐，再更新证据。缺少新 rank 的算子证据仍未知。直接重绘不加刷新参数会保留原 JSON 所存判据；主表的新 MTE1/Fixpipe 列仍可显示已有值。

## Roofline 需要显式依据

先从真实报告导出带 case/共同源 rank/算子完整指纹的空模板：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report_7/model_architecture_report.json --bound-template bound_profile.local.json --out "ep128_report_7/bound模板预览.xlsx"
```

模板中所有峰值、工作量和流量都是 null；不会把示例硬件常数套到真实数据。仅为已核实的算子填写，其余保持 null。以下是**合成示例**，数字不代表任何生产 NPU：

```json
{
  "schema_version": 1,
  "cases": {
    "case-a": {
      "source_rank": "dp1",
      "hardware": {
        "name": "SYNTHETIC ONLY",
        "architecture": "synthetic",
        "scope": "logical_npu",
        "peak_tflops_by_dtype": {"FP16": 100},
        "memory_bandwidth_gbps": 1000,
        "memory_level": "HBM",
        "provenance": "合成教学数据，不用于生产估算"
      },
      "operators": {
        "op_替换为模板中的完整指纹": {
          "compute_dtype": "FP16",
          "flops_per_call": 1000000000,
          "memory_bytes_per_call": 1000000,
          "memory_bytes_basis": "measured",
          "memory_level": "HBM",
          "workload_basis": "constant_per_call",
          "provenance": "同rank同指纹全部原调用有效工作固定的证据；同层级流量测量方法"
        }
      }
    }
  }
}
```

`compute_dtype` 是实际执行计算的精度，不能仅取输入存储 dtype。FP8、INT8、INT4 不自动按 FP16 的固定倍数折算。峰值按每个 rank 对应的**一个逻辑 NPU**填写，不能填双芯物理卡或全系统峰值。稀疏/稠密、频率、功耗模式、是否将一次乘加计2 FLOPs必须与工作量口径一致，写明 provenance。字段名字、大小写/量纲错误会报错，不静默猜测。

`memory_bytes_basis` 为 `measured` 或 `model`；后者整体置信度低，实得 GB/s 也是流量模型除以时间，不是实测带宽。operator 的 `memory_level` 必须与 hardware 同名字段匹配，否则标记模型不一致；字段声明仍需真实测量与 provenance 支持，不能把 L1/UB 通路流量声明成 HBM。当前不会自动把 `*_main_memory_datas(KB)` 当成全 NPU HBM bytes，因为导出粒度与 KB 换算仍需核对。

`constant_per_call` 必须有依据。GMM 的预分配/填充 M 不能当真实有效行，group_list 不在 CSV 时不推算；packed/NZ 权重物理维度不自动还原 N/K。Attention 的 KV 长度、mask、sparse/gather 和融合语义不齐时不按名字猜 FLOPs。每次有效工作量变化的指纹暂不做均值 Roofline，应补逐调用元数据或在 msOpProf 单算子视图核对，不能把整批容量伪装为有效工作量。

核实后重绘：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report_7/model_architecture_report.json --bound-profile bound_profile.local.json --out "ep128_report_7/Roofline核对.xlsx"
```

也可以在新统计命令追加 `--bound-profile`。两种入口使用同一核心纯函数。profile 的 case、源 rank、完整指纹不匹配会拒绝，避免误用其他 rank 或目标硬件。

设整次原调用均时为 `t` 秒、有效工作量为 `F` FLOPs、同层级流量为 `B` bytes、同精度峰值为 `P` FLOP/s、该层级带宽为 `W` byte/s：

```text
实得算力 = F/t          算力峰值比例 = F/(tP)
实得带宽 = B/t          带宽峰值比例 = B/(tW)
算术强度 = F/B          Roofline拐点 = P/W
计算下限 = F/P          搬运下限 = B/W
理想下限 = max(F/P, B/W)
模型效率 = 理想下限/t
```

计算/搬运下限相差超过25%才给模型偏向，否则写接近。模型效率超过105%时标不一致，先核对工作量、计数单位、峰值和范围；保留异常值，不截成100%。这一模型也不覆盖所有片内通路、Scalar、同步或负载不均，所以其方向不等同于已证实算子瓶颈。全表不把模型结果当目标 EP128 实测。
