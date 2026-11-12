# Ascend EP Extrapolator

基于 Ascend 原生设备采样，按 HC 层区间外推 MoE 推理的完整周期、TPOT 和单物理卡吞吐，并导出可复核的 CSV、JSON、HTML 和 Excel。

当前实现针对已核对的 **DSpark：源模型 8 个主层 + 3 个草稿层、每轮一次草稿 forward**。示例场景为 EP16 → EP128、主模型 8 → 61 层、草稿仍为 3 层。这是一套带明确假设的工程估算工具；输出不是目标 EP128 的实测性能，也不适用于任意模型和采样结构。

**日常使用需要根目录这4个程序/配置文件；使用方法2时，还须保留随仓提供的 `data/communication-reference.csv`**。另行准备自己的采样数据。`tests/` 用于维护回归测试，`tools/` 是可选检查工具，均不参与默认估算流程。

```text
ascend-ep-extrapolator/
├── profile_ep128.py                  # 核心统计与外推
├── build_ep128_leadership_excel.py   # Excel / HTML 展示
├── ep128_xlsx_writer.py              # XLSX 写入器
├── ep128_attention_prepare.py        # Attention准备业务细分识别（标准库）
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
| `profile_ep128.py` | 读取采样、识别层与 HC 周期、执行三种外推、输出证据和结果 |
| `build_ep128_leadership_excel.py` | 从核心报告生成 Excel 与配套 HTML，不重新估算 |
| `ep128_xlsx_writer.py` | 标准库 XLSX 写入器，支持可编辑的原生条形图和折线图 |
| `ep128_attention_prepare.py` | Attention准备业务分组、精确整数ns覆盖与保守归属 |
| `profile_ep128_cases.example.json` | 源/目标模型、拓扑、BS、草稿长度和有效输出长度的配置模板 |
| `data/communication-reference.csv` | 方法2使用的原始历史通信参考表，模板已默认指向此文件 |
| `examples/communication-reference.synthetic.csv` | **合成通信参考表**，仅演示格式和计算流程，不是历史硬件实测 |
| `tests/test_ep128*.py` | 维护代码时检查统计、公式、展示和文件格式，不是运行时依赖 |
| `tools/validate_ep128_capture.py` | 可选：独立核对原始采样和导出账本 |
| `tools/profile_ep128_comm_reference.py` | 可选：独立检查通信参考表并导出比值 |
| `docs/usage.md` | 输入配置与操作说明 |
| `docs/presentations/` | HC 周期拆解的一页 PPT、PDF 和预览图 |

主脚本、Excel生成器、XLSX写入器和Attention准备识别模块应放在同一目录。当前版本：统计 `20261112.1.layer_stages_restored`、Excel生成器 `20261112.2.case_table_labels`、写入器 `20261111.1.internal_links`。

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
     "acceptance": 0.65,
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

rank 模式按 case 原顺序连续安排各自的 rank，分析结果暂存于私有临时目录；一个 case 的全部 rank 完成后，再按原 rank 顺序合并、校验重复 CSV、计算方法1/方法2/方法3并释放中间数据。临时盘需要容纳尚未汇总的 rank 证据。两种并行模式保留全部 rank、配置顺序，并统一按外推前源整轮均值选定三种方法共用的 rank；日志会标明 case、rank 和阶段。32GB Windows 机器可先用 `--parallel rank --workers 4` 观察实际内存与耗时，再调整进程数。

### 已有 JSON，只重绘 Excel

```bash
python -u build_ep128_leadership_excel.py --report ep128_report/model_architecture_report.json --out "ep128_report/10T模型推理估算.xlsx"
```

此命令仅读取 `--report` 指定的 JSON，不扫描新增 BS 目录，也不读取 cases 来补结果。新增 BS 后，先用更新的 cases 重新统计，再指向新 JSON。运行日志会打印报告路径、生成时间和各 case 的配置 BS，可据此检查是否读到了旧文件。

报告同时展示两种横坐标：固定投机步长 S 看 BS，固定 BS 看 S。总览中每种外推方法分别画单卡吞吐、TPOT 和完整周期图：同方法、同指标的可比曲线合在一张图里，BS 横轴用不同 S/Neff 区分曲线，S 横轴用不同固定 BS 区分曲线；方法1/2/3分图，吞吐与时延各保留自己的单位。三个方法页同样展示这些指标，并各有整轮五项耗时随 BS、S 的合并图。完整周期（ITL代理）是独立总耗时折线，不能只看五项分量。源采样基线页保留自己的源指标和五项分解，HTML 同步展示，共为 10 个工作表。

图例中的曲线仅合并可比条件；模型、硬件/拓扑、rank覆盖或采样契约不同的组分别作图。各曲线缺少某个横坐标时留空断线，重复坐标的原始行保留，歧义曲线不取平均。不同 S 的有效输出长度继续取各 case 配置，不假设接受率固定。

方法页中，选中 rank 的逐轮阶段耗时明细集中到“源采样基线”，同 case/rank 只保留一份；方法页保留折线图、源数据跳转，以及原父阶段/细项/函数对照。折线使用源表的实际单元格引用，原 HC 轮次 ID、缺失点和长文本原因均保留。

总览、源采样基线和三个方法页的页首提供 case 跳转目录，第4行保留“本页完整图表目录”和跨页入口；完整目录按 case 列出每幅图和每张分析表，可直接点击并返回页首。HTML 有按章节/case 展开的可搜索图表目录和固定“返回图表目录”按钮，链接到折叠详情时会展开对应内容。

**ITL 代理值取完整周期，单位 ms/轮**，表示每轮输出间隔；采样没有逐 token 发送时间，不能把它当作实测逐 token ITL。TPOT 为周期除以该 case 配置的 SAL/Neff。固定 BS 的 S 对比保留每点自己的 SAL/Neff；未声明固定接受率时，不将变化只归因于 S。每个 case 的源基线与三种方法统一使用外推前源整轮均值最大的 rank，逐点列明；源五项使用该源 rank 的实际完整层边界，包含主0。旧 JSON 直接重绘保留旧口径，切换方式见下方。缺少边界时只留空受影响的分解，不借用目标数据。重复横坐标保留各行、不画合并曲线；缺值断线。

已有完整 JSON 可直接用新版脚本重绘上述视图，无需重跑 CSV 分析。各运行文件应保持同一版本集；本次需同步更新 `profile_ep128.py`、`build_ep128_leadership_excel.py` 和新增 `ep128_attention_prepare.py`；写入器沿用 `20261111.1.internal_links`，更旧版本需一并更新。

已有结果要补方法3，可直接使用保存的源证据，不重读九个 case 的CSV：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report_9/model_architecture_report.json --add-method3 --out "ep128_report_9/三方法估算.xlsx"
```

生成 `.method3.json`、Excel和HTML，原JSON不变；必要源证据缺失时方法3保持不可用。旧报告若各方法rank尚未统一，可同时加 `--align-source-rank` 显式对齐。方法3单独有估算sheet，并复用总览、BS/S趋势、整轮五项、阶段/算子/PMU、逐轮源图和跳转目录。新增统计默认包含方法3；单纯重绘旧JSON不会凭空补结果。

“写入HTML预览”之后的目录生成已改为按原始坐标一次组装，避免每个锚点都复制整份HTML。运行日志分别报告正文生成、各case、目录组装与文件写入及累计耗时，便于区分生成计算与磁盘写入。图表、明细和目录功能保留，运行时仍只依赖标准库。

已有 JSON 若仍是两种方法独立选 rank 的旧口径，可离线切换：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report/model_architecture_report.json --align-source-rank --out "ep128_report/共同源rank.xlsx"
```

此操作从源保留轮次选出共同 rank，再使用各方法已保存的该 rank 外推结果，另存 `共同源rank.source_rank.json`，并生成 Excel/HTML；原 JSON 不变，不读取 CSV。已存阶段与 U/逐 step 诊断随 rank 切换；新 rank 缺少算子/PMU证据时明确留空，不沿用旧 rank 明细。没有该 rank 的完整外推结果则报错，需要补统计，不能只改 rank 标签。直接重绘不加此参数时保留旧报告数据并注明旧口径。

## 配置口径

| 字段 | 含义 |
| --- | --- |
| `bs` | **每个 DP rank/副本上的并发请求数**，不是整个系统请求数，也不是 MoE Dispatch 后的专家 token 行数 |
| `num_speculative_tokens` | 每组采样的草稿候选长度 S，旧名 `draft_tokens` 仍兼容；同层同时填写时必须相等 |
| `synthetic_acceptance_length` | 每组采样的配置平均输出长度，已含 bonus，范围 `[1, S+1]`；也接受兼容拼写 `synthetic_acception_length` |
| `effective_tokens_per_round` | 兼容旧配置的有效输出长度 Neff；与同组 SAL 同时提供时必须一致 |
| `acceptance` / `acceptance_semantics` | 新接受率 r 为含 bonus 的有效输出比例，`SAL=(1+S)×r`；只写 acceptance 即采用新口径，规范化后语义为 `mean_output_fraction_including_bonus`。直接填 SAL 可省略这两项 |
| `token_basis` | 旧显式 Neff 的来源说明；直接填写 SAL 时自动生成，不需要另外填写 |
| `base_ep` / `target_ep` | 源/目标专家并行组大小；EP 不自动等于 DP 或物理卡数 |
| `source_dp` / `target_dp` | 源/目标 DP 副本数量，用于系统吞吐折算 |
| `source_devices` / `target_devices` | **物理卡数**，用于单卡吞吐分母 |
| `source_chips_per_card` / `target_chips_per_card` | 每张物理卡实际参与的逻辑 NPU/芯片数，缺省为 1；双芯片参与应显式配置为 2 |
| `expected_ranks` | 源采样预期逻辑 rank 数，通常应等于源物理卡数 × 每卡参与芯片数 |
| `ranks_per_node` | 节点内参与的逻辑 rank 数，用于逻辑远端份额模型 |
| `trim_rounds` | 每个 rank 首尾裁剪的轮次数，用于去除不稳定边界轮次 |

模板不在 defaults 中填写接受率或固定 Neff；每个 case 单独填写 S 和接受率，由脚本计算 SAL。模板 S=7、r=0.65 对应 SAL=5.2，保留原输出量。SAL 是**已包含 bonus 的配置假设**，不会再加一个 token，也不是设备 trace 测得的接受率。`with_stack=false` 是当前计时采样合同；不要将不同采样条件混为同一基线。

### 一次统计多组：每组填写两个参数

模型、拓扑等通用设置保留在 defaults；每个 case 填写 `num_speculative_tokens` 和 `acceptance`，即可按 **`SAL=(1+S)×r`** 计算有效输出量。r 可以逐 case 改变，写小数而非百分数。以下是固定新口径60%的四组独立采样：

```json
[
  {"name": "bs8-s3-r60", "bs": 8, "num_speculative_tokens": 3, "acceptance": 0.6, "path": "data/bs8_s3_r60"},
  {"name": "bs8-s5-r60", "bs": 8, "num_speculative_tokens": 5, "acceptance": 0.6, "path": "data/bs8_s5_r60"},
  {"name": "bs8-s7-r60", "bs": 8, "num_speculative_tokens": 7, "acceptance": 0.6, "path": "data/bs8_s7_r60"},
  {"name": "bs8-s9-r60", "bs": 8, "num_speculative_tokens": 9, "acceptance": 0.6, "path": "data/bs8_s9_r60"}
]
```

对应 SAL 依次为2.4、3.6、4.8、6.0。也可以继续直接填写 `synthetic_acceptance_length`，兼容拼写 `synthetic_acception_length`；只填 SAL 时不推断实测接受率，同时填写 SAL/Neff/r 则必须与新公式一致。保持 S=7、SAL=5.2 时，新口径 r 为65%；如果改成80%，SAL 为6.4。

现有有效输出合同仍为 `1≤SAL≤1+S`，所以新接受率须满足 `1/(1+S)≤r≤1`；S=0 时 r 只能为1。无效值会报错，不补一个 bonus、不静默钳位。

case 显式填写 SAL 且没有自己声明接受率时，会替换继承的固定 Neff/SAL，并清除继承的接受率策略。同一个 case 若另外明确填写 Neff 或接受率，仍须相互一致。默认接受率可复用，但不同 S 会重新按新公式计算 SAL；不把上一组派生的 SAL 继续沿用。

新分析拒绝仍生效的旧语义 `mean_accepted_draft_fraction`，避免将同一个60%悄悄改义。旧口径迁移时，如需保持原输出量，先按 `r新=SAL/(1+S)` 换算，再删除旧语义或改为 `mean_output_fraction_including_bonus`。旧 JSON 离线重绘保留原始指标，并明确标注历史 `SAL=1+S×r`，不会自动修改历史结果。

接受率只用于有效输出量、吞吐和TPOT换算；复用同一采样只改 r 属于固定每轮耗时的敏感性估算。它不会预测接受率变化导致的 KV、输入准备、上下文或调度耗时变化；验证真实影响仍需对应采样或校准模型。

```bash
python -u profile_ep128.py cases.local.json --out ep128_report_multi --excel
```

每个 case 必须对应那组 S/接受长度实际采集的数据，不能反复使用同一 CSV 改 S 冒充不同实测。一个 case 的输入只能包含该次采样的各 rank；不要将所有不同 S 的共同父目录作为单个 case 扫描。总览、方法页、HTML 和最终 CSV 会标出 S、SAL/Neff；未填写接受率时，页面不再提示需要声明接受率。BS 趋势中不同 S/Neff 各为一条曲线，可比曲线按方法、指标合图。

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

脚本先按每个 rank **外推前的完整源周期**，对首尾裁剪后保留轮次求平均，再选均值最大的 rank；并列时取源输入顺序中的第一个。完整周期包含主0、主/草稿层链、层间衔接和 U，不使用剔除主0后的参考层均值来选 rank。同一 case 的方法1/2/3固定共用此 rank；各方法仍保留全部 rank 的逐轮及均值外推结果，外推后不再改选。总览、五项分解、层阶段、U、逐 step 和算子/PMU均跟随该共同 rank。不同 BS/S 的 case 可各选自己的源 rank。

因此当前结果是“源采样最慢平均 rank 上的三种外推”，不保证等于各方法外推后所有 rank 的最大周期。采样 rank 不全时，最大值仅限已采 rank，不能声称已覆盖全系统最慢 rank。方法缺输入仍标不可用，不以其他 rank 结果替代。

## 三种外推方法

源层区间已经包含通信与路由计算，脚本仅添加目标相对源的变化量：

```text
W目标 = W源 + ΔDispatch + ΔCombine + Δ路由GMM
```

| 方法 | Dispatch / Combine 的处理 |
| --- | --- |
| 方法1：逻辑通信倍率 | 源耗时 × 逻辑远端份额倍率 × TopK 倍率，再对每次目标 Dispatch、Combine **分别加 14 μs** |
| 方法2：历史通信倍率 | 源耗时 × 对应历史通信比值 × 负载形状倍率；**不再叠加 14 μs** |
| 方法3：通信零增量 | 每个目标层的 Dispatch、Combine 沿用对应源层类型均值：倍率固定为1，附加耗时为0；仅路由专家 GMM 使用既有 TopK 倍率 |

方法3沿用方法1/2的目标层数、层类型、参考层、层间衔接和U口径，只取消每层通信的额外增长。其层区间公式为 `W目标 = W源 + GMM源 × (TopK目标 / TopK源 − 1)`。**这不表示整轮通信总量与源模型相等**：目标层数增加时，目标层中的原有通信仍随层数计入。该方法是通信零增量的对照假设，不是目标网络的实测结论，也不把全部FFN或激活成本乘TopK倍率。

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

U 保留下一主模型 KV/输入准备、主/草稿 metadata 的子阶段；metadata 本体与前后的映射准备分开。原始 Task Type 保留 AI_CPU 等执行类型，busy 不等于 AICORE 有效算力。详见[结果读取说明](docs/usage.md#6-如何核对结果)。

CSA、HCA、SWA 保留**层内十阶段展开**以及独立的完整参考层源实测 Top8、PMU、bound 和 Roofline。删除的是各阶段下面的逐算子贡献、互斥覆盖和算子模型增减账本；层内阶段计算、JSON、CSV和表图继续保留。层识别、整层均值、Dispatch/Combine与路由GMM外推、整轮五项、共同源rank及U15保持原口径。

新报告在 `projection.layer_mean_methods[].operator_hotspots[]` 按层类型保存源热点，按原生名称、类型、形状、dtype、stream和Task Type区分指纹。调用次数与kernel累计按同一rank的参考层和保留轮次取均值，单次均值/P90反映实际相交调用在完整层边界内的耗时；并发累计不是互斥周期成本，不乘目标层数或外推倍率。正文保留完整指纹、输入形状、stream和原CSV行号；长文本使用换行或续行。

已有JSON可直接重绘层内十阶段，并复用已保存的热点及PMU；不再绘制目标逐算子贡献账本。旧热点标明原阶段内统计范围。旧热点不会被冒充为新的完整层统计；需要完整层Top8时，用新版核心重新处理原CSV即可，不要求重新采样。证据缺失继续显示未知。详见[Top算子的读取口径](docs/usage.md#top算子的读取口径)。

“HCPre与Q/KV、压缩/索引准备”另提供业务细分表和横向图：HCPre、输入归一化/聚合/量化、Q低秩及展开、Q末处理、窗口KV投影/归一化/RoPE/写入、Attention压缩、CSA Indexer Query/Key缓存/索引选择等。源业务项可能并行，不相加成父项；未知归属不补0，融合kernel不虚拆。新接口为 `attention_prepare_breakdowns`。函数识别在新增标准库模块 `ep128_attention_prepare.py`；同步更新脚本时需将该模块放在主脚本同目录。

三个方法页还提供“下一主模型 KV 与输入准备”和“下一主 metadata 及输入映射准备”的额外细分：表格、横向柱状图和源码函数入口对应。细项按真实设备区间拆成单算子覆盖、具名并发组合和无记录间隙，合计等于原父段；函数入口是带版本依据的源码映射，不是 Python 函数实测耗时。

两段的逐 step 源表集中在“源采样基线”，方法页保留折线和源数据链接；使用本 case 在外推前选定的共同源 rank，三种方法展示相同源轮次证据，父阶段汇总并列均值、P50、P90和最大值。这里的 step 是保留周期及其原始 round_id，不冒充 trace 中的 `ProfilerStep`；不会逐点改选其他 rank。未识别 step 保留空值和原因，折线断开，不补0。

三个方法页中，CSA、HCA、SWA 的源实测 Top8 热点后展示 PMU、流水候选和独立的 Roofline 定量表，覆盖其中入选的 Attention 与 MoE 算子。MTE1、Fixpipe 已列入主表；AIC 与 AIV 按各自适用的竞争流水检查完整性，缺项降低置信度。原始指标按源 kernel **完整调用 Duration 加权**，不乘目标层数或外推倍率；跨阶段裁剪不能测得局部 PMU。高活动比只是候选，MTE2 高不等于 HBM 饱和，`cube_utilization` 不是 MFU，`1−VEC` 不是 C/V 空泡。

本次核对了公开 NPU 性能 Skills 的实际代码与硬件范围，独立实现证据分级和显式 Roofline；没有安装外部 Skill 或引入新依赖。检索清单见[公开 Skill 调研](docs/operator-bound-skill-survey.md)，判据、指标、硬件依据模板和限制见[算子 bound 说明](docs/operator-bound.md)。没有准确源逻辑 NPU、同计算精度峰值、有效 FLOPs 和同层级 bytes 时，定量模型留空；不以 GMM 容量行、packed 权重形状或默认 FP8 倍率补齐。

已有 JSON 可直接更新判据，不重读九个 case 的 CSV：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report_7/model_architecture_report.json --refresh-bound --out "ep128_report_7/专业bound.xlsx"
```

另存 `专业bound.bound.json`、Excel 和 HTML，原报告不变，共为10个 sheet。若旧报告未保存逐调用诊断，不能恢复联合覆盖/短调用检查，对应置信度降低；未采的 PMU 继续未知。可用 `--bound-template bound_profile.local.json` 导出全空硬件/工作量模板，核实填写后以 `--bound-profile bound_profile.local.json` 生成定量结果；详见上述说明。需要新流量列时才补充原采样导出并重新统计，重绘不会制造计数器。

两组 U 子阶段仍沿用源实测、保留一次，不乘目标层数。删除算子贡献下钻不影响层内阶段、这两组细分及逐step折线；旧JSON缺少U证据时，重绘不能补出实测内容。

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
| `最终结果.csv` | 三种方法的周期、TPOT、单卡吞吐与可用状态 |
| `model_architecture_report.json` | 核心结果、配置、源基线、逐 rank 汇总及证据索引，支持离线重绘 |
| `10T模型推理估算.xlsx` / 同名 `.html` | 总览、源基线、时间账本、三方法估算、公式假设与复核记录 |
| `layers.csv` / `components.csv` | 源层与组件实测 |
| `rank_estimates.csv` / `round_estimates.csv` | 逐 rank 与逐轮外推 |
| `boundary_accounting.csv` / `compute_evidence.csv` | 边界计账与路由 GMM 证据 |
| `step_residual_stages.csv` / `step_residual_substages.csv` | U15 父阶段与 Markov、下一主输入及 metadata 子阶段汇总 |
| `step_residual_focus_steps.csv` / `step_residual_focus_summary.csv` | 两个下一主准备阶段的逐 rank、逐 HC 周期变化及互斥细分汇总 |
| `step_residual_details/` | 各 rank 的原始设备明细、精确阶段边界、来源链、两段关注区间的逐步与分段证据；包含层内十阶段明细 |

Excel 的 BS 折线取 `config.bs` 数值排序；S 折线取配置投机步长排序。同方法、同指标、可比条件下的不同固定参数曲线合并展示，其他声明条件不同仍分组。某条曲线有重复横坐标时保留原行并停止绘制该条歧义曲线；缺值留空断线，不补0。单点可在合并图中显示为标记，整个条件组只有一个横坐标时仅保留表格。顶部索引列出全部采样组合，避免把单独分组误认为遗漏。

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
