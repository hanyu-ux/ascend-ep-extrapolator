# 使用说明

## 环境

使用 Python 3.10 或更新版本。核心统计、Excel 写入和 HTML 导出均只依赖标准库，不需要安装 requirements，也不需要在运行统计的机器上安装 PyTorch、CANN 或连接 NPU。

日常使用需要根目录的主脚本、两个报表模块、新增Attention准备识别模块和配置文件；使用方法2时，还须保留随仓提供的 `data/communication-reference.csv`。另行准备自己的采样数据，四个Python文件应保持在同一目录：

```text
profile_ep128.py
build_ep128_leadership_excel.py
ep128_xlsx_writer.py
ep128_attention_prepare.py
profile_ep128_cases.example.json
data/communication-reference.csv
```

`tests/` 是维护代码时运行的回归测试，`tools/` 是可选检查工具，`docs/` 保存说明与 PPT，`examples/` 保存合成数据示例。它们均不参与默认估算流程。

`tools/validate_ep128_capture.py` 和 `tools/profile_ep128_comm_reference.py` 可按需独立复核采样或通信表，不是运行时依赖。

## 1. 准备配置和数据

复制 `profile_ep128_cases.example.json` 为 `cases.local.json`，只编辑副本。确认以下内容：

- `sample_model` 与真实采样的层数、隐藏维度、专家数、TopK、量化及压缩比例一致。
- `target_model.main` 和 `target_model.draft` 分别描述目标主、草稿模型；当前实现核对的源结构是 8 主层、3 草稿层、每轮一次草稿 forward。
- case 的 `bs` 是每个 DP 副本的请求数。名称中的数字不参与推断。
- `base_ep`、`target_ep`、DP、物理卡数、每卡参与芯片数必须分别核对。不要自动把 EP、DP、rank 数和物理卡数当作同一个数。
- `expected_ranks` 及可选的 `expected_rank_names` 对应本次采样全局逻辑 rank，不能反复选取同一个 CSV 冒充不同 rank。
- 每个 case 单独填写 `num_speculative_tokens` 和 `acceptance`（小数 r），按新公式 `SAL=(1+S)×r` 计算。也可直接填写 `synthetic_acceptance_length`（兼容 `synthetic_acception_length`）；多种声明须一致，均为配置假设。
- `trim_rounds` 默认首尾各裁剪 2 轮。输入必须有足够的完整 HC 周期。

示例配置中的 `data/bs8_s7` 等路径是占位。可以递归指定整个 case 的目录，也可以用显式 `inputs` 避免 rank 归属含糊：

```json
{
  "name": "bs8-s7",
  "bs": 8,
  "num_speculative_tokens": 7,
  "acceptance": 0.65,
  "inputs": [
    {"rank": "dp0", "path": "data/bs8_s7/dp0/kernel_details.csv"},
    {"rank": "dp1", "path": "data/bs8_s7/dp1/kernel_details.csv"}
  ]
}
```

这只演示两项输入，实际数量应与本次源采样相符。JSON 中的相对路径（包括 `communication_reference`）均相对于 cases JSON 所在目录。命令行 `--comm-reference` 的相对路径则相对于运行命令时的工作目录。使用 `inputs` 时移除该 case 原来的 `path`，避免同时指定两种输入方式。

同次、同 rank 的 `trace_view.json` 可放在 CSV 相邻目录，或在对应 input 中配置 `trace_path`。缺 trace 时保留设备统计和缺失状态；不要借用其他轮次或其他 rank 的 trace。

### 多组草稿长度与接受长度

同一份 `cases.local.json` 可同时包含不同 BS、不同 `num_speculative_tokens`/`draft_tokens`、不同 `synthetic_acceptance_length` 的独立采样；每个 case 的名称须唯一，路径只包含该次采样的各 rank。工具不会从目录名推断 S，也不会用一份实测时间生成另一种 S 的采样。

每组 case 可分别填写 S 和接受率 r，统一使用 **`SAL=(1+S)×r`**。r 可随 case 变化；以下固定新口径60%的示例用于核对算术：

| num_speculative_tokens / S | acceptance / r | SAL / Neff（含bonus） |
| --- | --- | --- |
| 3 | 0.6 | 2.4 |
| 5 | 0.6 | 3.6 |
| 7 | 0.6 | 4.8 |
| 9 | 0.6 | 6.0 |

只填 `acceptance` 即使用新口径，规范化结果写入 `acceptance_semantics=mean_output_fraction_including_bonus`。S=7、r=0.65 对应原 SAL=5.2；S=7、r=0.8 对应 SAL=6.4。接受率不强制固定，也不从 profiling 反推出实测接受率。

直接填写 SAL 仍受支持，已含 bonus，不再加1。`num_speculative_tokens` 是 `draft_tokens` 的别名；`synthetic_acception_length` 是 SAL 的兼容拼写；同层同义字段冲突会报错。case 显式只填 SAL 时清除继承的接受率与旧固定输出量；同时填 SAL/Neff/r 必须一致。默认 r 对不同 S 重新算 SAL，SAL 路径自动生成 token_basis，显式 Neff 仍需来源说明。

有效输出仍要求 `1≤SAL≤1+S`，因此 `1/(1+S)≤r≤1`（S=0 时 r=1），不额外加 bonus 或钳制结果。新分析中仍生效的旧语义 `mean_accepted_draft_fraction` 会被拒绝并提示迁移；保持旧 SAL 时用 `r新=SAL/(1+S)`。旧 JSON 重绘保留历史值及旧公式标识，不自动转换为新测量。

复用同一采样只改 r 时，脚本只重算有效输出量、吞吐及TPOT，每轮耗时及五项分解保持原估算；这是固定耗时敏感性分析。真实接受率变化对 KV/输入准备、上下文和调度的耗时影响尚未建模。

完整混合配置片段见 [README](../README.md)。用原主入口一次运行全部 cases，不需要单独的 scenarios 模式；同一 Excel 的配置对照表和各方法说明会列出每组 BS、S、SAL/Neff。无接受率输入时不展示该列；旧报告若明确声明了有效接受率，仍保留其说明。

## 2. 核对输入清单

```bash
python -u profile_ep128.py cases.local.json --list-inputs
```

该命令只发现输入，不解析 CSV。可用少量文件检查导出字段：

```bash
python profile_ep128.py --inventory data/bs8_s7/dp0/kernel_details.csv
```

实际运行参数以 `python profile_ep128.py --help` 为准。

## 3. 统计并生成 Excel

```bash
python -u profile_ep128.py cases.local.json --out ep128_report --excel
```

Windows 可以使用 `py -3 -u` 替换 `python -u`。日志会显示进度、所用文件、版本和具体失败原因。U 阶段诊断会读取同次设备记录；有 trace 时还会扫描 Host 事件并核对来源，因此耗时可能明显高于仅解析 CSV。

多个 case 默认按 case 粒度使用最多2个独立进程；单个 case 直接运行，其内部的 rank 仍顺序处理。需要调整 case 并行数时使用：

```bash
python -u profile_ep128.py cases.local.json --out ep128_report --excel --workers 4
```

`--workers 1` 是串行对照。自动预算受任务数、CPU 及内存约束：探测到可用内存不足4 GiB时降为1个进程；探测不到内存时最多2个。显式设置的正整数只按任务数封顶，低内存时会提示，但不会自动改小。`--parallel case --workers 9` 可以让9个有效 case 一起运行，每个 case 内的 rank 顺序分析。

单个 case 内的 rank 也可并行，用另一种粒度：

```bash
python -u profile_ep128.py cases.local.json --out ep128_report --excel --parallel rank --workers 4
```

此时所有 case 共用一个最多4个进程的池，每份 rank 是独立任务；不会创建“case进程 × rank进程”的嵌套池。case 的 rank 按输入顺序连续提交；单个 case 也可以使用多个进程。rank 分析复用完整 case 的模型、拓扑及覆盖配置，所有 rank 完成后才按原顺序汇总该 case。因此均值相同时选中的 rank、原方法1/2的公式、完整覆盖检查和失败顺序不变。一个 rank 的读取或测量失败、跨 rank CSV 内容重复，仍会使整个 case 失败；并行中已产生的失败 case 诊断文件可能保留，不计入成功结果。

每个 case 完成测量后立即计算方法1/方法2，释放临时算子数据，避免积压到所有 case 都读完。并行子进程将结果写入私有临时目录，主进程读取后删除；大结果不经进程管道复制。rank 模式先暂存各 rank 的 prepared 证据，一个 case 齐备后才加载、汇总，因此临时盘也要有足够空间。各 case/rank 的明细文件路径独立，最终表格仍按配置顺序排列；某 case 失败会保留原因，不自动重新跑一次。所有 rank、逐轮证据、PMU 缺失处理和外推公式均保留。

大采样的单个进程可能使用数 GiB 内存，证据 CSV 也可能比原始文件大；进程数增加不保证等比例提速，尤其是共享慢盘时。日志为 case 与 rank 加标识，并显示测量、估算、导出阶段。不同命令应使用不同 `--out`，避免互相覆盖。已在运行的旧进程会继续使用原代码；这些参数适用于之后启动的命令。

核心输出为 `ep128_report/model_architecture_report.json`、`最终结果.csv`，以及逐 rank、逐轮、层区间、组件、U 阶段等证据 CSV。开启 `--excel` 时另生成 `10T模型推理估算.xlsx` 和同名 HTML。

## 4. 方法2的真实通信参考

原始历史通信参考表已随仓提供为 [`data/communication-reference.csv`](../data/communication-reference.csv)，默认模板已指向它。将配置副本放在仓库根目录时，无需另行准备此表；若将配置移到其他目录，应调整 JSON 中的相对路径或使用绝对路径。从仓库根目录运行时，也可以用命令行显式指定：

```bash
python -u profile_ep128.py cases.local.json --comm-reference data/communication-reference.csv --out ep128_report --excel
```

方法2目前核对的外推范围是 EP16 → EP128，历史表需要同 BS 下 EP32、EP64、EP128 的 send、return 数值。可另附 total 列，存在时须与 send + return 一致。

随仓原表包含 BS `1、2、4、8、16、32、48、64、96、128、192、256`，脚本按 case 的 BS 精确匹配，不自动插值。使用前应核对历史表与当前采样的可比性；需要替换时，通过 JSON 的 `communication_reference` 或命令行 `--comm-reference` 指定经过核对的参考表。

`examples/communication-reference.synthetic.csv` 是另一份全新合成数据，数值仅用于格式演示与回归测试。它不能替代 `data/communication-reference.csv`，不应与真实采样混用来形成性能结论。

没有真实表、BS 不匹配或拟合无效时，保留方法2不可用的原因，方法1仍可计算；不要用任意倍率补齐结果。

## 5. 已有 JSON 的离线重绘

```bash
python -u build_ep128_leadership_excel.py --report ep128_report/model_architecture_report.json --out "ep128_report/10T模型推理估算.xlsx"
```

每个case的表格均就近标注 `BS8 · S9` 这样的参数标识：Excel显示在首列表头的第二行，HTML显示在表题中并支持滚动时保持表题与表头可见，图表目录中的表格条目也包含标识。标识读取该case配置中的BS和S，缺失时显示“未声明”，不从case名称猜测。跨case汇总和通信参考表保留原有范围，不标成某个单独case。已有报告JSON可直接重绘，无需重新采样或重新分析CSV。

重绘只读指定 JSON，不再读取原 CSV、trace 或 cases，也不会自动扫描新增 BS。使用绝对路径可减少误读其他目录旧报告的可能。`schema_version=2` 不因新增 BS 而改变。

新版原始统计默认生成方法1、方法2和方法3。方法3将每层 Dispatch/Combine 的通信倍率设为1、附加耗时设为0，路由专家GMM仍按现有TopK倍率修正；目标层数、参考层、层间衔接和U不改。新增层仍包含原有通信，因此不能将该假设表述为整轮通信总量固定。

已有JSON可以离线补方法3并重绘，无需重读CSV或trace：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report_9/model_architecture_report.json --add-method3 --out "ep128_report_9/三方法估算.xlsx"
```

另存 `.method3.json`、Excel和HTML，原JSON不变。方法3只使用已保存的源层类型均值、源rank和闭合账本；缺少必要证据时明确显示不可用。旧报告若方法1/2尚未使用共同源rank，应同时加 `--align-source-rank`，先对齐再补方法3；该显式对齐可能改变旧报告方法1/2的选中rank及其结果。单纯重绘不会自动把旧方法结果改成方法3。

本次展示更新只需重绘：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report_7/model_architecture_report.json --out "ep128_report_7/合并趋势与目录.xlsx"
```

本次同步更新 `profile_ep128.py` 和生成器，写入器沿用 `20261111.1.internal_links`（更旧写入器需一并更新）。总览的每个指标按方法分别合并曲线，BS 横轴区分不同 S/Neff，S 横轴区分不同固定 BS。方法1/2/3页也包含两种横轴的吞吐、TPOT、完整周期和五项耗时图；完整周期（ITL代理）是独立折线，五项分量不代替总时长。不同单位不混轴，不可比模型/拓扑/覆盖或采样契约继续隔离，缺点断线。

选中 rank 的逐轮阶段数据表移到“源采样基线”，同 case/rank 去重；方法页只保留该部分的折线和源数据链接，原父阶段均值、细项与函数对照不动。旧报告的不同方法若选中不同 rank，保留各自真实源表，不能为合并展示而改 rank。

总览、源页及方法页顶部可按 case 跳转，第4行有固定“本页完整图表目录”入口；完整目录列出图和分析表并提供返回链接。HTML 目录支持按 case/方法/图表名搜索，跳转到折叠的详情会自动展开。目录只改变位置与链接，单元格值、图缓存及原报告不改。

新版统计先用各 rank 外推前的完整源周期求保留轮次均值，选最大者作为同一 case 三种方法共用的 rank；均值并列按源输入顺序取第一个。选 rank 时包含主0、主/草稿、衔接和 U，主0只从后续参考层均值中排除。三种方法仍各自计算并保留全部 rank 的外推结果，最终卡片、五项、U阶段和源热点固定取共同 rank；不保证该 rank 也是外推后最慢者。最大值只限已采 rank。

旧 JSON 直接重绘会保留所存数据并标明旧口径。要使用已存的逐 rank 外推结果切换到共同源 rank，无需重读九个 case 的 CSV：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report/model_architecture_report.json --align-source-rank --out "ep128_report/共同源rank.xlsx"
```

生成 `共同源rank.xlsx`、同名 HTML 和 `共同源rank.source_rank.json`，原 JSON 不变；原有 CSV 导出仍属于旧报告。共同rank的周期、吞吐、TPOT、五项和整层分类取自各方法自己的 `per_rank` 行；U/逐 step 从相同 rank 的实测诊断取值。旧报告未保存新 rank 的 Top8/PMU 时显示缺证据，不沿用旧 rank 明细，也不将另一方法的目标增量复制过来。缺少共同 rank 的完整结果时停止并报具体缺项，不只改标签。若需要补齐缺失证据，再用新版核心处理原 CSV，不要求重新采样。

新结果在 `projection.rank_selection` 和各方法的 `rank_selection` 中保存选择口径、源均值、平局规则和 rank 覆盖；共同策略为 `max_of_captured_rank_mean_source_cycle`。不可用方法保持不可用，并用 `source_rank` 记录共同源参考，不伪填目标结果。

如果只想先看配置版式、还没有采样数据，可执行：

```bash
python build_ep128_leadership_excel.py --cases profile_ep128_cases.example.json --preview --out "reports/配置版式预览.xlsx"
```

预览中的性能指标保持待估算，不是外推结果。

## 6. 如何核对结果

1. 检查 case 的 `config.bs`、Neff、采样背景和 rank 覆盖。
2. 检查每种方法的状态和所选 rank。新版按外推前源整轮平均耗时选一个 rank，同一 case 的方法1/2/3共用；方法不可用时不伪填目标 rank。
3. 五项分解应闭合到目标周期；U 父阶段是 U 的展开，不再加到周期上。
4. 主0仅从参考均值排除；目标主层仍完整计数。
5. TPOT = 所选 rank 的目标平均周期 / Neff，不再除 BS 或 DP。
6. U 子阶段缺锚点时保留 `UNRESOLVED`；缺失不是零。
7. 源周期和软件回归闭合不能证明目标外推准确。新增层衔接、U 平稳、固定通信附加值和历史倍率迁移都是需要目标实测检验的假设。

不同 BS 的折线按数值排序；每条曲线固定 S/Neff，同方法、同指标的可比曲线合并展示。模型、拓扑及其他采样条件不同仍分组。某条曲线出现重复 BS 时保留原行并停止绘制该条歧义曲线；缺值断线，不补0。单点曲线可在合并图中显示为标记，整个条件组只有一个横坐标时只显示表格。总览开头列出全部采样组合，日志中可核对实际读入的 JSON。

投机步长 S 对比也按数值排序，读取配置中的 `draft_tokens`（兼容 `num_speculative_tokens`），不从 case 名称猜值；S=0 是合法档位。每条曲线固定 BS，同方法、同指标的可比曲线合并展示。模型、拓扑、采样条件和 rank 覆盖等非轴条件不同仍分组。各 S 沿用自己的配置 SAL/Neff，并在表中列出；没有明确固定接受率时，这是不同 S 配置方案的对比，不能只归因于 S。某条曲线出现重复 S 时保留原行并停止绘制该条歧义曲线，不取平均或任意一条。

| 位置 | 新增或保留的对比 |
| --- | --- |
| 总览 | 外推方法1/2/3的单卡吞吐、TPOT、完整周期，分别随 BS 和 S 变化 |
| 源采样基线 | 源单卡吞吐、TPOT、完整周期和整轮五项耗时，分别随 BS 和 S 变化 |
| 方法1估算、方法2估算、方法3估算 | 各方法单卡吞吐、TPOT、完整周期和整轮五项耗时随 BS 和 S 变化；原细节表及横向柱状图保留 |

完整周期明确标为 **ITL代理（ms/轮）**，只表示每轮输出间隔，非实测逐 token ITL。源 TPOT 使用源周期/配置 Neff，源单卡吞吐直接保留核心 aggregate 的原值。源五项从 aggregate 所选源 rank 的全部保留轮次、整数纳秒层边界计算，包含主0，逐轮和均值均须闭合；不借用目标分解，旧报告也不借用方法页中可能不同的 rank。旧 JSON 缺测量边界时，已有吞吐/周期仍可展示，缺证据的五项留空并说明原因。所有图可通过上面的离线重绘命令生成，不需原 CSV；本次同步更新核心与生成器，保持主脚本、生成器、XLSX写入模块及新增Attention准备识别模块在同目录。

### 方法页中的保留视图

“方法1估算”、“方法2估算”和“方法3估算”三个sheet中，每个case继续展示整轮五项、U15和下一主准备阶段，并提供CSA、HCA、SWA各自的源实测Top8、PMU及bound。保留这三类层的层内十阶段展开；仅删除阶段下面的逐算子贡献下钻，包括算子覆盖账本、并行组合与gap明细，不再为这些逐算子账本执行互斥区间扫描。

整层识别、同类参考层均值及原外推公式保留：Dispatch/Combine与路由GMM只按既有父模型增加对应增量，激活和其他层成本不整体乘倍率。默认主0仍只排除参考，完整源周期与目标层数不因此缩减。三个方法共用外推前按源整轮均值选定的rank，各方法的通信假设及五项账本不变。

U中的“下一主模型KV与输入准备”和“下一主metadata及输入映射准备”继续各自提供表格、横向柱状图、函数入口与逐step折线。这些源实测子阶段作为U的展开保留一次，不乘目标层数。缺失或不闭合的证据标为不可用，未知不补0。

### Attention准备段的业务细分

十阶段第一项仍以首HCPre.start到Attention.start的真实区间为父项。方法1/2/3另显示 `attention_prepare_breakdowns` 的源业务细分，不乘目标层数或通信/TopK倍率；同case三方法复用同一源rank和同一份细分。细分只解释已有父项，不增加整层、整轮、TPOT或吞吐成本。

细项包含HCPre、输入归一化/可选AllGather/可确认的共享输入量化、Q低秩投影/Q展开/Q最终处理、窗口KV投影/归一化/RoPE/写入、Attention压缩、CSA Indexer Query/Key缓存/索引选择。各行提供函数入口、调用数均值、kernel累计、记录并集、首尾跨度及样本覆盖。Q/KV/压缩/Indexer可能重叠，因此分组并集、累计和跨度均不可相加成父耗时。父项保留记录并集和CSV未覆盖量；未覆盖不直接等同于CPU或同步等待。

只有全部同类参考层×保留轮次均能确认归属，才给出该行全样本均值。部分确认保留known_samples/total_samples、未知原因和空值，不把缺证据补为0。重复MatMul/RmsNorm不按第N次调用或固定stream号猜归属；Compressor内部与评分/TopK融合kernel不虚拆内部时长。缺少必要shape/scope证据的记录进入“未归属记录”。SWA的压缩/Indexer、HCA的Indexer按结构标为不适用。旧JSON没有这份细分时，离线重绘明确标未采集；用新版重新处理原CSV可尝试识别，不需先重采样，但CSV缺失的关联不会凭空恢复。

新增模块 `ep128_attention_prepare.py` 仅用Python标准库，与 `profile_ep128.py` 放在同目录。

### Top算子的读取口径

新报告使用 `projection.layer_mean_methods[].operator_hotspots[]` 保存每个 `role+attention` 的完整参考层热点及其 `kernels`。Top8以源 `kernel_sum_ms` 排序，不再依赖层内10阶段锚点或目标贡献账本。

- **参考范围**：取本方法共同源rank的同类参考层及保留轮次，按完整层首HCPre至末HCPost的边界裁剪任务；主0按原参考策略排除。保留任务的完整原生名称、类型、输入输出形状、dtype、stream和Task Type，不凭名称猜测Q/KV业务。
- **调用与耗时**：相交调用次数、kernel累计先在每轮对全部参考层取均值，再对保留轮次取均值。已完整读取的样本中指纹未出现，按确定的0参与均值；证据缺失保持未知。单次均值/P90取实际观察到的层内裁剪调用分布，不是逐轮P90或整个周期P90。
- **解释范围**：kernel累计允许并发重叠，不能相加成完整层或整轮墙钟，也不等于目标模型的可节省时间。Top8保持源实测，不乘目标层数、通信或TopK倍率；融合kernel内部不按比例拆时。
- **来源与排版**：保留原始CSV行号、rank、轮次与层号，完整指纹和长形状在Excel／HTML中换行或续行。续行不是额外调用或统计项，Top8之后的PMU表与上方热点一一对应。

旧JSON可以直接重绘：生成器保留旧十阶段展开，忽略逐算子目标贡献账本，读取已有 `target_stage_breakdowns[].stage_rows[].operator_breakdown.kernels` 作为兼容热点，明确标注“旧阶段内统计范围”。重绘不重建阶段、不读取CSV，也不会把旧的阶段裁剪均值当成完整层均值；其热点排序和取值依旧数据保留。要得到新的完整层Top8，用新版核心重新处理原CSV，不需重新采样。旧报告没有所选rank热点时仍显示缺证据，不借用其他rank。

```bash
python build_ep128_leadership_excel.py --report ep128_report/model_architecture_report.json --out "ep128_report/10T模型推理估算.xlsx"
```

### U15 与输入、metadata 子阶段

`step_residual_stages.csv` 继续保留原来的 U1 8 段、U2 7 段父阶段；`step_residual_substages.csv` 和每 rank 阶段明细保存 `level=1` 子阶段，`parent_stage_key` 指向父段。原有逐位置 Markov 子阶段继续保留。

全部匹配时，“轮次剩余区间耗时构成”仍展示15阶段。若某个窗口全部轮次均为 `UNRESOLVED`，但单一未识别记录的轮次数、累计纳秒与窗口总量一致，则用该“未识别整窗”总耗时与另一侧已匹配父阶段共同展示；例如一个整窗加7段时，图中是8项，明确标注部分识别。每项互斥且总和须同时等于两窗口总量、诊断 U 和本方法所选 rank 的 U；不完整或矛盾的证据仍不能绘图。不把未识别窗口猜成8个阶段，不将未知值补零，也不把 Markov 等子段重复加入父项。

若原因包含 `KvCompressEpilog`，检查同一行的 `failed=`：它区分 context 行数/hidden、跨层 context 形状一致性、投影输出、cache 布局、payload、slots 和 dtype 等校验。原因同时带出原始 CSV 行号及观测形状、类型。一维 UINT8 缓存、BF16 payload、INT32 slots 签名保持严格校验；仅有融合 padding 导致的 context 裁剪不再要求投影行数与融合总行数完全相等，但各草稿层必须处理同一 context 形状。更新核心后须使用原 CSV 重新生成 JSON，再导出 Excel；只重绘旧 JSON 不会重新识别阶段。

阶段汇总至多保留3条有界原因样例，避免大量不同 CSV 行号撑大 Excel 单元格；完整原因仍保存在逐轮阶段明细 CSV 中。

“missing final hidden-width Norm before vocabulary projection”表示未能建立最终 Norm 到词表投影的边界，不证明模型没有 Norm。Norm 可以先处理全部或填充 token，再通过 Index/Gather 抽取投影所需行。脚本仅接受形状和时序完整的已知抽行链；否则继续显示未知，并附投影激活形状及可见 Norm 候选，便于核对原始 CSV。

下一主模型 `main_inputs` 可细分为：

| 子阶段 | 原生任务锚点 | 缺失处理 |
| --- | --- | --- |
| KV block 清零 | `_zero_kv_blocks_kernel` | 无 block 需清理时可自然缺席，允许多个清理任务 |
| 应用待写入项 | `_apply_write_kernel` | 无待写入项时可自然缺席 |
| 展开请求与 token 映射 | `_expand_idx_mapping_kernel` | 必需 |
| 准备位置及序列长度 | `_prepare_pos_seq_lens_kernel` | 必需 |
| 拼接采样与草稿 token | `_combine_sampled_and_draft_tokens_kernel` | 必需 |
| 聚合 KV 块表 | `_gather_block_tables_kernel` | 必需 |
| 计算 KV slot 映射 | `_compute_slot_mappings_kernel` | 必需 |

锚点须满足顺序、唯一性及父区间边界。缺少必需锚点或顺序冲突时，保留父阶段总耗时，细分标为 `UNRESOLVED`，不能自行补 0 或借其他轮次边界。

`main_metadata`、`draft_metadata` 按实际设备任务的 start/end 提取 Attention metadata 和 Indexer metadata **本体**，并把首个本体之前、每个本体之后至下一个本体/父段结尾的区间保留为“映射与准备”。重复本体带各自序号，不假设一次构建必定只有一个 metadata 任务。映射准备中可能包括 RoPE 查表、序列处理、slot 格式化和 Cast/Copy；这些宽标签不等于完整 Python 模块边界。

`recorded_busy_ms` 合并已记录设备任务区间，`kernel_sum_ms` 求 task duration 和，二者不混用。AI_CPU metadata 同样进入设备 busy；busy 高不代表 AICORE 有效算力高。`no_recorded_ms` 是 CSV 任务没有覆盖的区间，不自动等于可消除的空闲。有同次已对齐 trace 时，`trace_dma_union_ms` 记录阶段内 DMA 并集，`trace_dma_in_csv_uncovered_ms` 只计算其填补 CSV 未覆盖的部分，`uncovered_after_trace_dma_ms` 保留补 DMA 后仍未覆盖的区间；最后一列仍不自动等于 idle。Host 来源缺失和 graph 共享来源继续保留未知状态。

**U 子阶段只展开父阶段，所有子段含在原来的 U 中。** 不能把父段和子段重复相加，不能将 U 中的工作移入层区间后宣称周期已经缩短。

### 下一主准备阶段的细项与逐 step 变化

方法1、方法2和方法3在原U15父阶段表图之后，追加 `main_inputs`、`main_metadata` 的细项、函数对应和逐 step 变化，保留原有15父阶段、level=1子阶段及估算公式。`measurements[].step_residuals.focus_diagnostics` 保存各 rank 的新增证据。

- 细项采用原始整数纳秒区间，在父段内合并所有可见 stream 的时间边界，分为单算子覆盖、具名并发组合、无设备记录间隙。各项互斥且覆盖父段，完整证据缺失时不编造函数归属。
- 函数表给出设备入口、对应 Python 包装入口、相对源码路径、版本依据和原始 CSV 行号。它表示静态源码对应，不表示 trace 已测得该 Python 函数的入口到返回耗时。间隙没有足够来源证据时保留未知，不归成某个 CPU 函数。
- 逐 step 图沿用本 case 外推前源整轮均值最大的共同 rank，三种方法不再分别改选；同一张图中两段、所有点都属于这个 rank。分类保留原始 HC `round_id`，另列保留序号；不是 `ProfilerStep` 或墙钟采样序号。方法页保留折线，逐轮源表集中到“源采样基线”，点击“源数据”即可跳转；同 case/rank 不重复保存。
- 父阶段汇总中的均值、P50、P90、最大值及匹配数量说明波动。源表中的缺失 step 留空并标原因，折线断开；不能用0代替，也不借用别的 rank 或前后 step。
- 细分表的分母是对应父段，细项相加等于父段；不得再把细项加到 U、完整周期、TPOT 或吞吐中。

`step_residual_focus_steps.csv` 导出所有已采集 rank 的父段逐步数据；`step_residual_focus_summary.csv` 导出细项均值和函数证据。每 rank 的关注区间明细 CSV 保留精确起止和源任务行。旧报告没有该结构时显示“未采集”，需要重新处理原 CSV；不要求为了软件更新重新采样。

### Top8 热点的 PMU 与 bound

每类层的源实测Top8按核心保存的完整层热点排序；旧JSON则保留原阶段内统计口径及排序，在各方法页追加对应的 PMU 表、方向判断和高级指标详情。PMU 仅描述源采样中的完整原生 kernel 调用，不是目标 10T 估算的硬件指标。

| 字段 | 含义 |
| --- | --- |
| `aic_mac_ratio` | AIC 矩阵乘加流水活跃周期占比 |
| `aiv_vec_ratio` | AIV 向量流水活跃周期占比 |
| `aic/aiv_scalar_ratio` | 对应核的标量、地址和控制处理流水活跃占比 |
| `aic/aiv_mte2_ratio` | GM向核内供数的流水活动；cache下不等同HBM物理读取 |
| `aic/aiv_mte3_ratio` | 逻辑写出或片内搬运，通路依架构；不能一概解释为写GM |
| `aic_mte1_ratio` / `aic_fixpipe_ratio` | 片内输入搬运和结果搬运流水活跃占比 |
| `aicore_time` / `aiv_time` | PMU 按分核周期推导的核内时间，不能与任务 Duration 的差值直接等同等待 |
| `Block Num` / `Mix Block Num` | 算子启动分核配置，不是实测吞吐利用率 |
| `icache_miss_rate` / `cube_utilization` | 指令缓存未命中和导出的 Cube 指标；后者不等同 FLOPs/硬件峰值定义的 MFU |

汇总使用每次原生调用的完整 Duration 加权，保留逐指标已知样本数、总样本数和已知 Duration 覆盖率。未知、非法、冲突或未导出值不补0；观测到的0继续保留。一个原生调用与多个阶段相交时，引用的是同次完整 PMU，不能当作各阶段独立采集或把这些指标相加。

流水可以同时工作，因此比例不能堆叠成100%或相加为总时间。MIX 算子联合考虑 AIC/AIV，两侧比例分母不同；向量算子的 AIC 零值不能判作低效。高计算占用、高搬运占用、标量突出、通信/同步复合任务、短任务以及核内时间与任务跨度差异分别处理。表中给出的 bound 是保守方向和置信度，不证明带宽饱和、因果关键路径或优化收益。确证需要真实有效行数、硬件精度/频率、实际访存和依赖信息。

输入 CSV 中已有 PMU 列即可使用；字段缺失时继续生成其他结果并明确提示。只有补采硬件 PMU 才能获得原采样未记录的指标，不能靠重绘补齐。

新版主表直接列 MTE1、Fixpipe，并按 AIC/AIV 的竞争流水、逐项及同次联合覆盖检查置信度；未知不默认 compute-bound。另有独立 Roofline 表，严格要求显式源逻辑NPU、匹配计算dtype的峰值、固定有效工作量、同存储层级的bytes。规则与空模板填写方法见[算子bound说明](operator-bound.md)，公开来源和许可见[Skill调研](operator-bound-skill-survey.md)。

已有报告可以只更新判据，不重读CSV：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report_7/model_architecture_report.json --refresh-bound --out "ep128_report_7/专业bound.xlsx"
```

另存 `.bound.json`、Excel和HTML，共保留10个sheet。未存逐调用诊断的旧JSON无法恢复联合覆盖/短任务检查，明确降级；指标值和时间账本保留。`--bound-template bound_profile.local.json` 生成全空硬件/工作量模板，核实后使用 `--bound-profile bound_profile.local.json`；不需要定量模型时无需填写。参数可与 `--align-source-rank` 合用；先按已存数据对齐，再更新bound，缺证据仍不借用其他rank。

### CSA、HCA、SWA整层统计与旧输出

整层HCPre→末HCPost边界、层类型、组件实测、同类参考层均值和Top8继续保留；核对时固定case、rank和周期，并区分整层wall与单个Attention／GMM kernel时长。具体采样还要核对主／草稿token数、padding、图模式和Task Type。

新版核心保留 `measurements[].layer_diagnostics.layer_summary` 整层统计、`stage_summary`、`swa_diagnostics`、方法级 `target_stage_breakdowns` 及层内十阶段CSV（`layer_stage_summary.csv`、`swa_stage_summary.csv`、`*-layer-stages.csv`、`*-swa-stages.csv`）。新版不生成各阶段的 `operator_breakdown`，Top8单独保存在 `operator_hotspots`。`layers.csv`、`components.csv`、`layer_means.csv`、五项账本、U15及其输入／metadata明细继续导出。旧输出目录已有的历史文件不作为新结果，应以本次JSON及导出日志为准；可用新的 `--out` 目录避免混读。

## 7. 运行回归测试

日常统计无需执行测试。维护代码时，从仓库根目录运行：

```bash
python -B -S -m unittest discover -s tests -p "test_ep128*.py" -v
```

`-S` 用于验证标准库环境。依赖未归档私有采样的 8 项测试会明确跳过；可选 openpyxl 读取检查在未安装该包时跳过。测试使用的合成数据不能作为目标硬件实测证据。

后续修改统一从新分支发起 PR，完成相关回归与证据复核再合并；仓库所有者已授权验证通过后直接合并。每次 PR 保留变更范围、验证命令、结果与尚未验证的性能假设。

## 8. 可选独立检查工具

从仓库根目录执行：

```bash
python tools/validate_ep128_capture.py cases.local.json --stats-dir ep128_report
python tools/profile_ep128_comm_reference.py data/communication-reference.csv --out communication_reference_report
```

前者独立核对原始采样及已导出的账本，后者检查指定通信表并导出观测比值。它们不会被主估算脚本自动调用，也不是运行前的必需步骤。其他参数见各工具的 `--help`。
