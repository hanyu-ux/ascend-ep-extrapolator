# Ascend EP Extrapolator

基于 Ascend 原生设备采样，按 HC 层区间外推 MoE 推理的完整周期、TPOT 和单物理卡吞吐，并导出可复核的 CSV、JSON、HTML 和 Excel。

当前实现针对已核对的 **DSpark：源模型 8 个主层 + 3 个草稿层、每轮一次草稿 forward**。示例场景为 EP16 → EP128、主模型 8 → 61 层、草稿仍为 3 层。这是一套带明确假设的工程估算工具；输出不是目标 EP128 的实测性能，也不适用于任意模型和采样结构。

**日常使用需要根目录这4个程序/配置文件；使用方法2时，还须保留随仓提供的 `data/communication-reference.csv`**。另行准备自己的采样数据。`tests/` 用于维护回归测试，`tools/` 是可选检查工具，均不参与默认估算流程。

```text
ascend-ep-extrapolator/
├── profile_ep128.py                  # 核心统计与外推
├── build_ep128_leadership_excel.py   # Excel / HTML 展示
├── ep128_xlsx_writer.py              # XLSX 写入器
├── profile_ep128_cases.example.json  # 配置模板
├── README.md
├── data/communication-reference.csv  # 方法2原始历史通信参考表
├── tests/                           # 回归测试
├── tools/                           # 可选检查工具
├── docs/                            # 详细说明、版本记录与 PPT
└── examples/                        # 合成数据示例
```

## 依赖与文件

离线运行需要 **Python 3.10+，仅使用标准库**；不需要安装 PyTorch、CANN、pandas、openpyxl 或 Office，也不需要在统计机器上连接 NPU。原始采样需要预先导出为 `kernel_details.csv`；同次 `trace_view.json` 可补充 Host 来源诊断。

| 文件 | 用途 |
| --- | --- |
| `profile_ep128.py` | 读取采样、识别层与 HC 周期、执行两种外推、输出证据和结果 |
| `build_ep128_leadership_excel.py` | 从核心报告生成 Excel 与配套 HTML，不重新估算 |
| `ep128_xlsx_writer.py` | 标准库 XLSX 写入器，支持可编辑的原生条形图和折线图 |
| `profile_ep128_cases.example.json` | 源/目标模型、拓扑、BS、草稿长度和有效输出长度的配置模板 |
| `data/communication-reference.csv` | 方法2使用的原始历史通信参考表，模板已默认指向此文件 |
| `examples/communication-reference.synthetic.csv` | **合成通信参考表**，仅演示格式和计算流程，不是历史硬件实测 |
| `tests/test_ep128*.py` | 维护代码时检查统计、公式、展示和文件格式，不是运行时依赖 |
| `tools/validate_ep128_capture.py` | 可选：独立核对原始采样和导出账本 |
| `tools/profile_ep128_comm_reference.py` | 可选：独立检查通信参考表并导出比值 |
| `docs/usage.md` | 输入配置与操作说明 |
| `docs/presentations/` | HC 周期拆解的一页 PPT、PDF 和预览图 |

三个运行脚本应放在同一目录。当前版本：统计 `20261110.3.rank_parallel`、Excel 生成器 `20261110.1.u_focus_step_pmu`、写入器 `20261104.1.line_charts`。

## 快速运行

从仓库根目录执行。下面使用 `python`；Windows 可替换为 `py -3`，Linux 环境也可使用 `python3`。

1. 复制配置模板：

   ```bash
   python -c "from shutil import copyfile; copyfile('profile_ep128_cases.example.json', 'cases.local.json')"
   ```

2. 编辑 `cases.local.json`：把示例输入路径替换为自己的采样目录，并核对模型、拓扑和输出 token 口径。模板将通信表指向随仓提供的 `data/communication-reference.csv`，配置副本仍放在仓库根目录时，无需另行复制通信表；使用前应核对历史表与本次场景的可比性。仓库的合成表仅用于测试和格式演示。示例数据路径可以采用以下结构；本仓库不提供原始采样：

   ```text
   data/
     communication-reference.csv                  # 随仓提供，方法2使用
     bs8_s7/
       dp0/ASCEND_PROFILER_OUTPUT/kernel_details.csv
       dp0/ASCEND_PROFILER_OUTPUT/trace_view.json    # 可选，须为同次同 rank
       dp1/ASCEND_PROFILER_OUTPUT/kernel_details.csv
       ...
   ```

   case 的 `path` 支持递归查找 `kernel_details.csv`。多个目录可用 `paths`；若目录名无法唯一表示全局 rank，使用显式 `inputs`，每个 rank 只选择一次采样：

   ```json
   {
     "name": "bs8-s7",
     "bs": 8,
     "num_speculative_tokens": 7,
     "synthetic_acceptance_length": 5.2,
     "inputs": [
       {"rank": "dp0", "path": "data/bs8_s7/dp0/ASCEND_PROFILER_OUTPUT/kernel_details.csv"},
       {"rank": "dp1", "path": "data/bs8_s7/dp1/ASCEND_PROFILER_OUTPUT/kernel_details.csv"}
     ]
   }
   ```

   上面只演示两个输入；实际应列出本次采样的全局 rank。JSON 中的相对路径（包括 `communication_reference`）相对于 **cases JSON 所在目录**；命令行 `--comm-reference` 的相对路径相对于 **运行命令时的工作目录**。将配置移到其他目录时，应同步调整路径。如需显式指定 trace，可在对应 `inputs` 项增加 `trace_path`。

3. 先核对输入清单，再运行统计：

   ```bash
   python -u profile_ep128.py cases.local.json --list-inputs
   python -u profile_ep128.py cases.local.json --out ep128_report --excel
   ```

   `--list-inputs` 只查找文件，不读取 CSV 内容。退出码为 0 不代表目标硬件性能已经验证；有 case 失败或主方法结果不可用时，统计命令会返回非零并保留具体原因。

多个 case 默认采用进程并行，自动最多同时运行2个 case；只处理1个 case 时直接运行。可显式调整：

```bash
python -u profile_ep128.py cases.local.json --out ep128_report --excel --workers 4
```

`--workers 1` 使用串行。自动预算会考虑 CPU 与可用内存；探测到可用内存不足4 GiB时自动使用1个进程。显式设置不会按内存自动改小。大采样每个进程可能使用数 GiB 内存，同时输出较大的证据 CSV；增加进程数须结合机器内存和磁盘吞吐。不同运行应使用不同 `--out` 目录。

`--parallel` 选择并行粒度，两种模式共用一个总进程上限，不叠加：

| 参数 | 行为 |
| --- | --- |
| `--parallel case --workers 9` | 最多同时分析9个 case；各 case 内的 rank 顺序处理。`case` 是默认模式。 |
| `--parallel rank --workers 4` | 一个全局进程池最多同时分析4个 rank；同一 case 内的 rank 可以并行，单个 case 也能使用多个进程。 |

rank 模式按 case 原顺序连续安排各自的 rank，分析结果暂存于私有临时目录；一个 case 的全部 rank 完成后，再按原 rank 顺序合并、校验重复 CSV、计算方法1/方法2并释放中间数据。临时盘需要容纳尚未汇总的 rank 证据。两种模式都保留全部 rank、统计口径、配置顺序和原来的最大均值 rank 选择规则；日志会标明 case、rank 和阶段。32GB Windows 机器可先用 `--parallel rank --workers 4` 观察实际内存与耗时，再调整进程数。

### 已有 JSON，只重绘 Excel

```bash
python -u build_ep128_leadership_excel.py --report ep128_report/model_architecture_report.json --out "ep128_report/10T模型推理估算.xlsx"
```

此命令仅读取 `--report` 指定的 JSON，不扫描新增 BS 目录，也不读取 cases 来补结果。新增 BS 后，先用更新的 cases 重新统计，再指向新 JSON。运行日志会打印报告路径、生成时间和各 case 的配置 BS，可据此检查是否读到了旧文件。

## 配置口径

| 字段 | 含义 |
| --- | --- |
| `bs` | **每个 DP rank/副本上的并发请求数**，不是整个系统请求数，也不是 MoE Dispatch 后的专家 token 行数 |
| `num_speculative_tokens` | 每组采样的草稿候选长度 S，旧名 `draft_tokens` 仍兼容；同层同时填写时必须相等 |
| `synthetic_acceptance_length` | 每组采样的配置平均输出长度，已含 bonus，范围 `[1, S+1]`；也接受兼容拼写 `synthetic_acception_length` |
| `effective_tokens_per_round` | 兼容旧配置的有效输出长度 Neff；与同组 SAL 同时提供时必须一致 |
| `acceptance` / `acceptance_semantics` | 兼容旧接受率配置；每组直接填写 S、SAL 时不需要提供 |
| `token_basis` | 旧显式 Neff 的来源说明；直接填写 SAL 时自动生成，不需要另外填写 |
| `base_ep` / `target_ep` | 源/目标专家并行组大小；EP 不自动等于 DP 或物理卡数 |
| `source_dp` / `target_dp` | 源/目标 DP 副本数量，用于系统吞吐折算 |
| `source_devices` / `target_devices` | **物理卡数**，用于单卡吞吐分母 |
| `source_chips_per_card` / `target_chips_per_card` | 每张物理卡实际参与的逻辑 NPU/芯片数，缺省为 1；双芯片参与应显式配置为 2 |
| `expected_ranks` | 源采样预期逻辑 rank 数，通常应等于源物理卡数 × 每卡参与芯片数 |
| `ranks_per_node` | 节点内参与的逻辑 rank 数，用于逻辑远端份额模型 |
| `trim_rounds` | 每个 rank 首尾裁剪的轮次数，用于去除不稳定边界轮次 |

模板不在 defaults 中填写接受率或固定 Neff；每个 case 单独填写 S 和 SAL。SAL 是**已包含 bonus 的配置假设**，不会再加一个 token，也不是设备 trace 测得的接受率。`with_stack=false` 是当前计时采样合同；不要将不同采样条件混为同一基线。

### 一次统计多组：每组填写两个参数

模型、拓扑等通用设置保留在 defaults；把 `num_speculative_tokens`、`synthetic_acceptance_length` 放到各自 case，不用填接受率、Neff 或 token_basis。例如同 BS8 比较四组独立采样：

```json
[
  {"name": "bs8-s3-r60", "bs": 8, "num_speculative_tokens": 3, "synthetic_acceptance_length": 2.8, "path": "data/bs8_s3_r60"},
  {"name": "bs8-s5-r60", "bs": 8, "num_speculative_tokens": 5, "synthetic_acceptance_length": 4.0, "path": "data/bs8_s5_r60"},
  {"name": "bs8-s7-r60", "bs": 8, "num_speculative_tokens": 7, "synthetic_acceptance_length": 5.2, "path": "data/bs8_s7_r60"},
  {"name": "bs8-s9-r60", "bs": 8, "num_speculative_tokens": 9, "synthetic_acceptance_length": 6.4, "path": "data/bs8_s9_r60"}
]
```

上面长度恰好对应之前60%的示例，但脚本直接使用每组填入的 SAL，不会强制60%。`synthetic_acception_length` 与标准拼写同义，只需选一种；同层同时填写却不一致会报错。

case 显式填写 SAL 且没有自己声明接受率时，会替换继承的旧固定 Neff/SAL，并清除继承的旧接受率策略，避免 defaults 的旧值干扰。同一个 case 若另外明确填写 Neff 或接受率，则仍须相互一致。原接受率模式和显式 Neff 旧配置继续兼容。

```bash
python -u profile_ep128.py cases.local.json --out ep128_report_multi --excel
```

每个 case 必须对应那组 S/接受长度实际采集的数据，不能反复使用同一 CSV 改 S 冒充不同实测。一个 case 的输入只能包含该次采样的各 rank；不要将所有不同 S 的共同父目录作为单个 case 扫描。总览、方法页、HTML 和最终 CSV 会标出 S、SAL/Neff；未填写接受率时，页面不再提示需要声明接受率，原 BS 趋势仍按不同 S/Neff 分组。

配置中的模型结构须与采样一致，包括隐藏维度、专家数、TopK、MoE 中间维度和完整 `compress_ratios` 序列。`schema_version=2` 是 JSON 结构版本，新增 BS 不改变它。

## 完整周期如何计算

当前层区间定义为 **本层首 HCPre 开始 → 本层末 HCPost 结束**。它包住 Attention 和 FFN 主体，但不等同于完整 Python `DecoderLayer.forward`；例如首个残差 clone 可能位于该统计起点之前。

```text
本轮主0首 HCPre
  → 主层区间 + 主层间衔接
  → U1：主模型末 HCPost → 草稿首 HCPre
  → 草稿层区间 + 草稿层间衔接
  → U2：草稿末 HCPost → 下一轮主0首 HCPre
```

源周期等于相邻两轮主0首 HCPre 起点之差，拆为五项：

```text
R源 = 主层区间合计 + 主层间衔接合计
    + 草稿层区间合计 + 草稿层间衔接合计 + U
U = U1 + U2
```

**各时间区间互斥、完整覆盖；区间内部允许多流 kernel 重叠执行。** 区间跨度包含任务间隔，不能直接解释为独占 kernel 忙时或硬件算力利用率。层间衔接也不能一概称为空闲。

当前示例中，主0仅从参考均值中排除，不从源周期删除，也不减少目标 61 层。HCA 参考主1/3/5/7，CSA 参考主2/4/6；草稿独立使用自身 SWA 均值。

```text
R目标 = 31 × 目标HCA层区间均值 + 30 × 目标CSA层区间均值
      + 3 × 目标草稿SWA层区间均值
      + 源主7段衔接之和 + 53 × 常规主衔接参考均值
      + 源草稿2段衔接之和 + U源
```

新增 53 段主衔接的参考均值取不触及主0的 6 段边界；原来的 0→1 衔接仍保留在源7段合计中。

脚本先**逐 rank、逐轮外推**，再对各 rank 的保留轮次求周期均值，最后选均值最大的 rank。各分项始终来自同一个选中 rank，不将不同 rank 的分项最大值拼接。两种方法及不同 BS 可以选中不同 rank。采样 rank 不全时会明确警告，不能声称已覆盖全系统最慢 rank。

## 两种外推方法

源层区间已经包含通信与路由计算，脚本仅添加目标相对源的变化量：

```text
W目标 = W源 + ΔDispatch + ΔCombine + Δ路由GMM
```

| 方法 | Dispatch / Combine 的处理 |
| --- | --- |
| 方法1：逻辑通信倍率 | 源耗时 × 逻辑远端份额倍率 × TopK 倍率，再对每次目标 Dispatch、Combine **分别加 14 μs** |
| 方法2：历史通信倍率 | 源耗时 × 对应历史通信比值 × 负载形状倍率；**不再叠加 14 μs** |

方法1在 `traffic_scope=interhost` 时使用 `(1−ranks_per_node/EP目标) / (1−ranks_per_node/EP源)` 作为远端份额倍率；这是逻辑份额假设，不是实际链路带宽测量。

方法2当前仅支持 EP16 → EP128。按匹配 BS 的 EP32/64/128 历史值，在 `log2(EP/32)` 轴做三点 OLS，再取“EP128 原参考值 / 拟合外推 EP16 值”作为通信比值。负载形状倍率为 `(K目标×H目标)/(K源×H源)`；示例 H 相同时简化为 TopK 倍率。

方法2所用的原始历史通信参考表已随仓提供：[`data/communication-reference.csv`](data/communication-reference.csv)，模板通过 `communication_reference` 默认引用。表中包含 BS `1、2、4、8、16、32、48、64、96、128、192、256`；其他 BS 不会自动插值匹配。也可通过 `communication_reference` 或 `--comm-reference` 改用经过核对的其他参考表。随仓表头为：

```csv
bs,ep32_send,ep32_return,ep32_total,ep64_send,ep64_return,ep64_total,ep128_send,ep128_return,ep128_total
```

表中 `send` 对应 Dispatch，`return` 对应 Combine；同一 BS 必须唯一，值必须为正，且应保持可比较的共同单位。不要填写虚构参考值来解锁结果。没有表、BS 不匹配或拟合无效时，方法2显示 `UNAVAILABLE`，方法1仍可运行。

仓库中的 [`communication-reference.synthetic.csv`](examples/communication-reference.synthetic.csv) 是另行生成的合成示例，不能替代上述真实参考表或作为性能依据。将它与真实采样混用，得到的方法2数值也没有真实性能含义。从仓库根目录显式指定随仓参考表，例如：

```bash
python -u profile_ep128.py cases.local.json --comm-reference data/communication-reference.csv --out ep128_report --excel
```

路由 GMM 当前按 TopK 比例修正；其余层内成本沿用源均值。固定 14 μs 和历史比值迁移均属于模型假设，不是对目标系统的校准结论。

## TPOT 与吞吐

以选中 rank 的目标平均周期 `R`（ms）和 `Neff` 为基础：

```text
TPOT（ms/token） = R / Neff
系统吞吐（token/s） = 1000 × BS × 目标DP × Neff / R
单物理卡吞吐（token/s/card） = 系统吞吐 / 目标物理卡数
```

例如 `R=52 ms`、`Neff=5.2`，则 `TPOT=10 ms/token`。**TPOT 不再除 BS 或 DP**；它是周期与有效输出长度折算值，不是 HTTP 客户端逐 token 时间戳直接测量值。只有目标 DP 等于目标物理卡数时，单卡公式才可简化为 `1000×BS×Neff/R`。

## U 的15阶段与外推边界

脚本默认尝试对源 U1、U2 识别连续的 8+7 个父阶段，并对 Markov 逐位置选 token 保留子阶段。15阶段仅用于解释 U，**不额外加到周期上，也不改变外推结果**。

本次增加下一主模型 KV/输入准备、主/草稿 metadata 的子阶段，以及每 rank、每个草稿 SWA 层的 10 段诊断。metadata 本体与前后的映射准备分开；原始 Task Type 保留 AI_CPU 等执行类型，busy 不等于 AICORE 有效算力。详见[结果读取说明](docs/usage.md#6-如何核对结果)。

方法1、方法2工作表在“轮次剩余区间耗时构成”下方，分别展示 CSA、HCA、SWA 层内10段，以及下一主 KV/输入、下一主 metadata 子阶段的五组独立表格和横向柱状图。前三组给出源单层均值、目标单层估算、目标同类全部层合计及占完整目标周期的比例；柱状图使用目标同类全部层的耗时，便于比较目标 10T 模型的主要开销。Dispatch、Combine 和专家段分别追加对应的通信、路由 GMM 增量；专家段中的激活及衔接不整体乘倍率。参考层、轮次和 rank 与原估算一致，默认主0不作参考。细分已包含在原目标周期内，不重复追加；源逐层诊断仍可在“层与组件实测”和“时间账本”查看。

每类层的10阶段图后还提供**算子下钻**表图：目标账本按真实算子覆盖、具名并行组合、任务间隔及有符号外推增减展开；源算子热点另列相交调用次数、阶段内执行累计和单次均值/P90，保留形状、dtype、stream及原始行号。图表预览以Top项加其余合计展示，完整明细保存在原报告JSON，不新增工作表。目标账本合计与父层估算闭合，源算子执行累计允许并发重叠，两者不能混加。

方法1、方法2及HTML中的算子标签保留完整父阶段、算子类型、输入形状和stream，长标签自动换行；原生名称、输出形状及dtype在来源列直接展示。已有算子证据的JSON可直接重新导出，无需重新采样或统计。

这些视图用于定位具体排查项，尚不证明真实关键路径或优化收益。通信、GMM增量沿用现有外推假设，单独展示；无任务记录的间隔不直接认定为CPU瓶颈。同名但不同形状或dtype的算子分别保留，不凭名称猜测Q/KV作用，也不拆分融合kernel内部耗时。旧JSON缺少算子证据时保留原阶段图并提示重新统计；必须用新版核心处理原CSV，单独重绘不能补出算子证据。详见[算子下钻的读取口径](docs/usage.md#算子下钻的读取口径)。

方法1、方法2页还提供“下一主模型 KV 与输入准备”和“下一主 metadata 及输入映射准备”的额外细分：表格、横向柱状图和源码函数入口对应。细项按真实设备区间拆成单算子覆盖、具名并发组合和无记录间隙，合计等于原父段；函数入口是带版本依据的源码映射，不是 Python 函数实测耗时。

两段的逐 step 表和折线使用本方法已选中的同一个 rank，展示各个保留 HC 周期的变化，并列均值、P50、P90和最大值。这里的 step 是保留周期及其原始 round_id，不冒充 trace 中的 `ProfilerStep`；不会逐点改选其他 rank。未识别 step 保留空值和原因，折线断开，不补0。

CSA、HCA、SWA 的源实测 Top8 热点后增加 PMU 指标和 bound 方向表，包含矩阵/向量计算、标量、读入/写出及片内搬运流水、核内时间、分核数和指标覆盖率等。指标按源 kernel **完整调用 Duration 加权**，不乘目标层数或外推倍率；跨阶段裁剪仅定位耗时，不能测得局部 PMU。bound 是有证据说明及置信度的流水倾向，MTE2 高不等于 HBM 已饱和，`cube_utilization` 也不是 MFU。需要原 CSV 已含相应 PMU 列；没有采集的数据继续显示未知。更新后应重新运行核心统计，旧 JSON 只重绘不能补出这些证据。

后两组 U 子阶段仍沿用源实测、保留一次，不乘目标层数。旧 JSON 缺少目标阶段估算时，需要用新版核心重新统计采样；仅重绘不能生成这些估算。

- 当前外推直接使用同 BS、同 rank、同轮的 `U目标=U源`，不乘 EP 倍率，也不乘主层数；没有证明 EP 扩大后 U 一定不变。
- U 包括输出处理、采样、上下文融合、KV/metadata/输入准备、主草稿切换以及其中的任务间隔，不是单一 Head、CPU 耗时或固定空闲。
- 有同次 trace 时补充 PyTorch/CANN/queue/设备来源关联；Host 同时段覆盖不等于该阶段独有的 CPU 成本。缺 trace 不阻断设备统计。
- 原生锚点不全、顺序不符或实现路径不支持时，保留整窗实测总耗时并标为 `UNRESOLVED`，不补零、不伪造15项分解。父阶段与子阶段不能重复相加。

若一侧 U 窗口全部轮次未识别，但原窗口实测总量和轮次证据完整，方法页会用“未识别整窗”与另一侧已匹配阶段共同绘图，标明部分识别并校验总量闭合；这不表示15阶段全部识别。最终 Norm 与词表投影之间存在抽行时，只有可核对的 Index/Gather 名称、输入输出形状及执行顺序才能建立边界；证据不足继续保留未知，并在原因中展示候选形状。

Context-KV 的 `KvCompressEpilog` 在融合结果有 padding 时，允许各草稿层按同一有效 context 行数裁剪后投影；payload 必须匹配投影输出，slots 长度必须匹配 payload 行数。已支持的一维 UINT8 缓存、BF16 payload、INT32 slots 签名保持不变，原位缓存输出按语义比较 dtype 别名。若仍未匹配，原因会列出失败项、投影与写回的 CSV 行号、形状和类型，便于直接核对已有采样；无需重新采样。

新增层衔接均值、U 平稳、路由效率与历史通信比值迁移都需要目标实测检验。路由器专家宽度增加、网络争用、慢 rank、调度、图模式、上下文长度和有效接受长度变化可能造成偏差。API server 数量等运行条件不同，也应作为采样背景记录，不能把跨 BS 差异全归于 BS。

## 输出与复核

主要结果位于 `--out` 目录：

| 输出 | 内容 |
| --- | --- |
| `最终结果.csv` | 两种方法的周期、TPOT、单卡吞吐与可用状态 |
| `model_architecture_report.json` | 核心结果、配置、源基线、逐 rank 汇总及证据索引，支持离线重绘 |
| `10T模型推理估算.xlsx` / 同名 `.html` | 总览、源基线、时间账本、两方法估算、公式假设与复核记录 |
| `layers.csv` / `components.csv` | 源层与组件实测 |
| `rank_estimates.csv` / `round_estimates.csv` | 逐 rank 与逐轮外推 |
| `boundary_accounting.csv` / `compute_evidence.csv` | 边界计账与路由 GMM 证据 |
| `step_residual_stages.csv` / `step_residual_substages.csv` | U15 父阶段与 Markov、下一主输入及 metadata 子阶段汇总 |
| `step_residual_focus_steps.csv` / `step_residual_focus_summary.csv` | 两个下一主准备阶段的逐 rank、逐 HC 周期变化及互斥细分汇总 |
| `swa_stage_summary.csv` | 每 case、rank、草稿层的 SWA 10 段统计 |
| `layer_stage_summary.csv` | 每 case、rank、主/草稿层的 CSA/HCA/SWA 10 段统计 |
| `step_residual_details/` | 各 rank 的原始设备明细、精确阶段边界、来源链、两段关注区间的逐步与分段证据，以及 `*-layer-stages.csv`、兼容的 `*-swa-stages.csv` |

Excel 的 BS 折线取 `config.bs` 数值排序；其他声明条件不同则分组。重复 BS 保留原值但不合并绘线；单点组只显示表格；未知值保留空白并断线，不补0。顶部索引列出全部采样组合，避免把单独分组误认为遗漏。

阅读结果时先看输入与 rank 覆盖，再看选中 rank、五项闭合、方法状态及假设。软件测试通过、源周期闭合，都不等于目标外推准确。

## PPT 归档

- [一页详解 PPT：HC周期拆解](docs/presentations/HC周期拆解_一页详解_20261104.pptx)
- [PDF版](docs/presentations/HC周期拆解_一页详解_20261104.pdf)
- [PNG预览](docs/presentations/HC周期拆解_一页详解_20261104.png)

PPT 使用可编辑形状和文字，展示源8+3周期、层区间边界、层间衔接、U15阶段以及8→61层外推关系。时间线宽度是结构示意，不代表实测耗时比例。

## 开发验证

后续脚本迭代统一使用 **新分支 → 提 PR → 验证 → 合并 PR**。PR 说明脚本功能、统计口径和软件回归结果；验证通过后按仓库所有者授权直接合并。运行时性能优化还须在本地完成正确性与设备 A/B 验证，详细采样证据和优化分析不上传。

远端维护脚本、测试、通用使用说明、既有 PPT 归档，以及经仓库所有者明确授权的 `data/communication-reference.csv`。详细 profiling 分析、其他原始实测数据与优化方案保存在本地，不提交到仓库或 PR 正文。

日常统计无需执行测试。维护代码时，在仓库根目录运行：

```bash
python -B -S -m unittest discover -s tests -p "test_ep128*.py" -v
```

测试涉及手算公式、rank 聚合、周期闭合、打包 FP4 原生证据路径、BS 分组/缺值处理与 XLSX 原生图表。相关 fixture 为合成数据，不能用于宣称目标硬件性能。若已安装 openpyxl，部分测试会额外校验第三方读取兼容性；它不是运行时依赖。

测试数量与结果见[版本记录](docs/versions.json)和对应 PR 的验证记录。跳过项包括 8 项未归档私有真实采样 fixture 和标准库隔离模式下的可选 openpyxl 兼容检查。PR 自动运行 Python 3.10、3.12 的标准库回归；真实采样的详细复核结果仅保存在本地。合成通信表为新建的演示数据，未复制原历史实测表。

需要独立复核时，可以从仓库根目录按需使用辅助工具：

```bash
python tools/validate_ep128_capture.py cases.local.json --stats-dir ep128_report
python tools/profile_ep128_comm_reference.py data/communication-reference.csv --out communication_reference_report
```

辅助工具各自读取指定输入并输出复核结果，不是主脚本运行前必须执行的步骤；详细选项见各工具的 `--help`。

原始 trace、采样目录、个人本地配置和生成报告由 `.gitignore` 排除。归档来源与 SHA-256 见 [版本记录](docs/versions.json)。
