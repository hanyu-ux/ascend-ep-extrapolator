# 使用说明

## 环境

使用 Python 3.10 或更新版本。核心统计、Excel 写入和 HTML 导出均只依赖标准库，不需要安装 requirements，也不需要在运行统计的机器上安装 PyTorch、CANN 或连接 NPU。

日常使用需要根目录的三个运行脚本和配置文件；使用方法2时，还须保留随仓提供的 `data/communication-reference.csv`。另行准备自己的采样数据，三个脚本应保持在同一目录：

```text
profile_ep128.py
build_ep128_leadership_excel.py
ep128_xlsx_writer.py
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
- 每个 case 单独填写 `num_speculative_tokens` 和 `synthetic_acceptance_length`（兼容 `synthetic_acception_length`），不需要接受率。长度已含 bonus，是配置假设；旧 Neff/接受率配置仍兼容。
- `trim_rounds` 默认首尾各裁剪 2 轮。输入必须有足够的完整 HC 周期。

示例配置中的 `data/bs8_s7` 等路径是占位。可以递归指定整个 case 的目录，也可以用显式 `inputs` 避免 rank 归属含糊：

```json
{
  "name": "bs8-s7",
  "bs": 8,
  "num_speculative_tokens": 7,
  "synthetic_acceptance_length": 5.2,
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

推荐直接在每组 case 填写 S 与 SAL，不在 defaults 中填写接受率或统一输出长度。以下数值对应之前60%的示例，但工具不会强制这个比例：

| num_speculative_tokens / S | synthetic_acceptance_length / Neff（含一个bonus） |
| --- | --- |
| 3 | 2.8 |
| 5 | 4.0 |
| 7 | 5.2 |
| 9 | 6.4 |

SAL 已包含 bonus，直接作为 Neff 使用，不再加1。框架采样配置须与本组实际数据一致；分析配置不能改变真实采样条件，也不会从 profiling 反推出实测接受率。

`num_speculative_tokens` 是 `draft_tokens` 的别名；`synthetic_acception_length` 是 `synthetic_acceptance_length` 的兼容拼写。defaults 和 case 各自先校验同义字段，再由 case 覆盖 defaults；同层两种写法不等会报错。case 显式填写 SAL 且没有自己声明接受率时，清除继承的旧接受率策略及固定输出长度；不受 defaults 中旧60%值干扰。同 case 自己另外声明的 Neff/接受率仍必须一致。SAL 路径自动生成 token_basis，直接填这两个参数即可；旧显式 Neff 路径仍需来源说明。

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

此时所有 case 共用一个最多4个进程的池，每份 rank 是独立任务；不会创建“case进程 × rank进程”的嵌套池。case 的 rank 按输入顺序连续提交；单个 case 也可以使用多个进程。rank 分析复用完整 case 的模型、拓扑及覆盖配置，所有 rank 完成后才按原顺序汇总该 case。因此均值相同时选中的 rank、两种方法的公式、完整覆盖检查和失败顺序不变。一个 rank 的读取或测量失败、跨 rank CSV 内容重复，仍会使整个 case 失败；并行中已产生的失败 case 诊断文件可能保留，不计入成功结果。

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

重绘只读指定 JSON，不再读取原 CSV、trace 或 cases，也不会自动扫描新增 BS。使用绝对路径可减少误读其他目录旧报告的可能。`schema_version=2` 不因新增 BS 而改变。

新版统计先用各 rank 外推前的完整源周期求保留轮次均值，选最大者作为同一 case 两种方法共用的 rank；均值并列按源输入顺序取第一个。选 rank 时包含主0、主/草稿、衔接和 U，主0只从后续参考层均值中排除。两种方法仍各自计算并保留全部 rank 的外推结果，最终卡片、五项、阶段和源诊断固定取共同 rank；不保证该 rank 也是外推后最慢者。最大值只限已采 rank。

旧 JSON 直接重绘会保留所存数据并标明旧口径。要使用已存的逐 rank 外推结果切换到共同源 rank，无需重读九个 case 的 CSV：

```bash
python -u build_ep128_leadership_excel.py --report ep128_report/model_architecture_report.json --align-source-rank --out "ep128_report/共同源rank.xlsx"
```

生成 `共同源rank.xlsx`、同名 HTML 和 `共同源rank.source_rank.json`，原 JSON 不变；原有 CSV 导出仍属于旧报告。共同 rank 的周期、吞吐、TPOT、五项、分类和阶段取自各方法自己的 `per_rank` 行；U/逐 step 从相同 rank 的实测诊断取值。旧报告未保存新 rank 的 Top8/PMU 时显示缺证据，不沿用旧 rank 明细，也不将另一方法的目标增量复制过来。缺少共同 rank 的完整结果时停止并报具体缺项，不只改标签。若需要补齐缺失证据，再用新版核心处理原 CSV，不要求重新采样。

新结果在 `projection.rank_selection` 和各方法的 `rank_selection` 中保存选择口径、源均值、平局规则和 rank 覆盖；共同策略为 `max_of_captured_rank_mean_source_cycle`。不可用方法保持不可用，并用 `source_rank` 记录共同源参考，不伪填目标结果。

如果只想先看配置版式、还没有采样数据，可执行：

```bash
python build_ep128_leadership_excel.py --cases profile_ep128_cases.example.json --preview --out "reports/配置版式预览.xlsx"
```

预览中的性能指标保持待估算，不是外推结果。

## 6. 如何核对结果

1. 检查 case 的 `config.bs`、Neff、采样背景和 rank 覆盖。
2. 检查每种方法的状态和所选 rank。新版按外推前源整轮平均耗时选一个 rank，同一 case 的方法1/2共用；方法不可用时不伪填目标 rank。
3. 五项分解应闭合到目标周期；U 父阶段是 U 的展开，不再加到周期上。
4. 主0仅从参考均值排除；目标主层仍完整计数。
5. TPOT = 所选 rank 的目标平均周期 / Neff，不再除 BS 或 DP。
6. U 子阶段缺锚点时保留 `UNRESOLVED`；缺失不是零。
7. 源周期和软件回归闭合不能证明目标外推准确。新增层衔接、U 平稳、固定通信附加值和历史倍率迁移都是需要目标实测检验的假设。

不同 BS 的折线按数值排序。非 BS 条件不同会分组；重复 BS 不合并，缺值断线，单点组只显示表格。总览开头列出全部采样组合，日志中可核对实际读入的 JSON。

固定 BS 的投机步长 S 对比也按数值排序，读取配置中的 `draft_tokens`（兼容 `num_speculative_tokens`），不从 case 名称猜值；S=0 是合法档位。模型、拓扑、采样条件和 rank 覆盖等非轴条件不同会分组。各 S 沿用自己的配置 SAL/Neff，并在表中列出；没有明确固定接受率时，这是不同 S 配置方案的对比，不能只归因于 S。重复 S 保留原行并抑制该组折线，不取平均或任意一条。

| 位置 | 新增或保留的对比 |
| --- | --- |
| 总览 | 外推方法1/2的单卡吞吐、TPOT、完整周期，分别随 BS 和 S 变化 |
| 源采样基线 | 源单卡吞吐、TPOT、完整周期和整轮五项耗时，分别随 BS 和 S 变化 |
| 方法1估算、方法2估算 | 各方法整轮五项耗时随 BS 和 S 变化；原细节表及横向柱状图保留 |

完整周期明确标为 **ITL代理（ms/轮）**，只表示每轮输出间隔，非实测逐 token ITL。源 TPOT 使用源周期/配置 Neff，源单卡吞吐直接保留核心 aggregate 的原值。源五项从 aggregate 所选源 rank 的全部保留轮次、整数纳秒层边界计算，包含主0，逐轮和均值均须闭合；不借用目标方法选中的 rank 或目标分解。旧 JSON 缺测量边界时，已有吞吐/周期仍可展示，缺证据的五项留空并说明原因。所有图可通过上面的离线重绘命令生成，不需原 CSV；需同步更新核心与生成器两个脚本，写入器仍放在同目录。

### 方法页中的阶段展开

“方法1估算”和“方法2估算”两个 sheet 中，每个 case 的“轮次剩余区间耗时构成”图下，依次展示五组独立表格和原生可编辑横向柱状图：

1. 主模型 CSA 的目标层内 10 阶段估算。
2. 主模型 HCA 的目标层内 10 阶段估算。
3. 草稿模型 SWA 的目标层内 10 阶段估算。
4. 下一主模型 KV 与输入准备子阶段。
5. 下一主 metadata 及输入映射准备子阶段。

前三组使用当前方法 `selected_rank`、保留轮次和原估算相同的参考层：先在每轮内取同类型参考层均值，再按保留轮次取均值，默认主0不参加参考。表格同时给出源单层均值、目标单层均值、目标同类全部层合计以及占完整目标周期的比例；主图采用目标同类全部层合计，便于比较目标模型的耗时构成。两种方法使用各自的通信倍率，共用在外推前按源整轮均值选定的 rank，不混用其他 rank 的诊断。

Dispatch 阶段只追加核心的 Dispatch 增量，Combine 阶段只追加 Combine 增量；“专家计算及衔接”整段只追加其中路由 GMM 的增量，激活及衔接时间不整体乘倍率。其他阶段沿用源参考均值。目标 10 段之和必须等于目标单层耗时，乘同类型目标层数后等于该类型层区间合计；这些费用已经包含在完整周期内，不再次追加。占周期比例的分母包含其他类型层、层间衔接和 U，因此单个类型的比例之和通常小于 100%。

后两组 U 子阶段仍按该方法所选 rank 的源实测展示，作为保留一次的 U 的展开，不套目标层数。源采样的逐层 CSA/HCA/SWA 诊断继续保存在“层与组件实测”和“时间账本”中。缺锚点、子阶段不完整或与父层不能闭合时，只把对应类型的目标细分标为不可用，父目标估算仍保留；未知值不补 0。旧 JSON 缺少 `target_stage_breakdowns` 时须用新版核心重新处理采样，仅重绘不能生成目标阶段估算。

### 算子下钻的读取口径

CSA、HCA、SWA 各自的10阶段图后，增加两组独立表格与横向柱状图；HTML预览使用同样的数据和口径。

- **目标覆盖与外推增减账本**：阶段内各流任务按整数纳秒起止边界切成互斥区间。串行项显示原生算子，并行项列出具体成员；任务间隔保留前后记录位置。源覆盖按同类目标层数累计，Dispatch/Combine/GMM的父模型增减单独列出，减少项保留负号。全部子项与原10阶段及同类层总成本闭合，不再加到周期或TPOT。
- **源实测算子热点**：按原生名称、类型、输入输出形状、dtype、stream和Task Type保留不同指纹。展示每参考层每轮平均相交调用次数、阶段内执行累计、段内单次均值/P90及代表性原始行号/轮次/层号。跨阶段任务分别裁剪，相交次数不能当成整个周期唯一调用次数直接相加；并发的执行累计也不能当成周期耗时相加。

可见预览保留Top项，目标账本另列其余覆盖、间隔及正负模型增量合计；这些折叠合计包含多项，不作为单个热点排名。完整条目在 JSON 的 `projection.layer_mean_methods[].target_stage_breakdowns[].stage_rows[].operator_breakdown` 中。只有最终所选rank的顶层分组附算子证据，逐轮和其他rank的原10阶段结构保持轻量。

每个预览项按完整父阶段、算子类型、输入形状和stream分行显示，并行项逐个列出成员。来源列保留完整原生名称、输出形状和dtype。Excel表格与柱状图、HTML图表按文本长度增加高度，不再用省略号截断名称；超高单元格使用续行，续行不是新的统计项。更新生成器后，可用已有算子证据的JSON重新导出：

```bash
python build_ep128_leadership_excel.py --report ep128_report/model_architecture_report.json --out "ep128_report/10T模型推理估算.xlsx"
```

这次名称显示更新不需要重新采样或运行核心统计；原有Top项数量、排序和耗时值保持不变。

两种方法各自继承父项的选中rank、参考层、保留轮次及权重：先在每轮取同类参考层均值，再对该rank的全部保留轮次取均值。指纹在某个已完整解析的样本中未出现时按确定的0参与全样本均值；证据缺失保持不可用，不补0。主模型第0层仍不进入参考样本，目标层数不因排除参考层减少。单次P90描述已观察到的段内调用分布，不是整轮P90，也不用于相加闭合。

读图时先看目标贡献，再结合相交次数、单次耗时、并行成员、任务间隔和原始行号确定检查对象。当前算子视图只提供覆盖和执行证据，关键路径与优化收益保持未验证：长耗时或独占覆盖不能证明整轮正在等待该任务；通信及GMM增量仍是父模型中全部暴露的估算假设。无任务记录间隔不等于CPU时间，融合kernel内部不按比例猜分。旧JSON或某组证据不完整时只禁用对应新图，保留原有父结果与阶段图。

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

方法1和方法2在原 U 15父阶段表图之后，追加 `main_inputs`、`main_metadata` 的细项、函数对应和逐 step 变化，保留原有15父阶段、level=1子阶段及估算公式。`measurements[].step_residuals.focus_diagnostics` 保存各 rank 的新增证据。

- 细项采用原始整数纳秒区间，在父段内合并所有可见 stream 的时间边界，分为单算子覆盖、具名并发组合、无设备记录间隙。各项互斥且覆盖父段，完整证据缺失时不编造函数归属。
- 函数表给出设备入口、对应 Python 包装入口、相对源码路径、版本依据和原始 CSV 行号。它表示静态源码对应，不表示 trace 已测得该 Python 函数的入口到返回耗时。间隙没有足够来源证据时保留未知，不归成某个 CPU 函数。
- 逐 step 图沿用本 case 外推前源整轮均值最大的共同 rank，两种方法不再分别改选；同一张图中两段、所有点都属于这个 rank。分类保留原始 HC `round_id`，另列保留序号；不是 `ProfilerStep` 或墙钟采样序号。
- 表中均值、P50、P90、最大值及匹配数量说明波动。缺失 step 留空并标原因，折线断开；不能用0代替，也不借用别的 rank 或前后 step。
- 细分表的分母是对应父段，细项相加等于父段；不得再把细项加到 U、完整周期、TPOT 或吞吐中。

`step_residual_focus_steps.csv` 导出所有已采集 rank 的父段逐步数据；`step_residual_focus_summary.csv` 导出细项均值和函数证据。每 rank 的关注区间明细 CSV 保留精确起止和源任务行。旧报告没有该结构时显示“未采集”，需要重新处理原 CSV；不要求为了软件更新重新采样。

### Top8 热点的 PMU 与 bound

每类层的源实测 Top8 热点保持原排序，在两方法页追加对应的 PMU 表、方向判断和高级指标详情。PMU 仅描述源采样中的完整原生 kernel 调用，不是目标 10T 估算的硬件指标。

| 字段 | 含义 |
| --- | --- |
| `aic_mac_ratio` | AIC 矩阵乘加流水活跃周期占比 |
| `aiv_vec_ratio` | AIV 向量流水活跃周期占比 |
| `aic/aiv_scalar_ratio` | 对应核的标量、地址和控制处理流水活跃占比 |
| `aic/aiv_mte2_ratio` | 从外部存储层级读入数据的搬运流水活跃占比 |
| `aic/aiv_mte3_ratio` | 向外部存储层级写出数据的搬运流水活跃占比 |
| `aic_mte1_ratio` / `aic_fixpipe_ratio` | 片内输入搬运和结果搬运流水活跃占比 |
| `aicore_time` / `aiv_time` | PMU 按分核周期推导的核内时间，不能与任务 Duration 的差值直接等同等待 |
| `Block Num` / `Mix Block Num` | 算子启动分核配置，不是实测吞吐利用率 |
| `icache_miss_rate` / `cube_utilization` | 指令缓存未命中和导出的 Cube 指标；后者不等同 FLOPs/硬件峰值定义的 MFU |

汇总使用每次原生调用的完整 Duration 加权，保留逐指标已知样本数、总样本数和已知 Duration 覆盖率。未知、非法、冲突或未导出值不补0；观测到的0继续保留。一个原生调用与多个阶段相交时，引用的是同次完整 PMU，不能当作各阶段独立采集或把这些指标相加。

流水可以同时工作，因此比例不能堆叠成100%或相加为总时间。MIX 算子联合考虑 AIC/AIV，两侧比例分母不同；向量算子的 AIC 零值不能判作低效。高计算占用、高搬运占用、标量突出、通信/同步复合任务、短任务以及核内时间与任务跨度差异分别处理。表中给出的 bound 是保守方向和置信度，不证明带宽饱和、因果关键路径或优化收益。确证需要真实有效行数、硬件精度/频率、实际访存和依赖信息。

输入 CSV 中已有 PMU 列即可使用；字段缺失时继续生成其他结果并明确提示。只有补采硬件 PMU 才能获得原采样未记录的指标，不能靠重绘补齐。

### CSA、HCA、SWA 10 段与同 rank 对照

报告 JSON 的 `measurements[].layer_diagnostics` 保存 CSA、HCA、SWA 的源层内阶段诊断；`layer_stage_summary.csv` 给出每 case、每 rank、每层的汇总，`*-layer-stages.csv` 保留逐轮 stage 边界与状态，位于 `step_residual_details/` 相应 case/rank 的证据目录。原有 `swa_stage_summary.csv`、`*-swa-stages.csv` 和 SWA 诊断字段继续保留兼容。源诊断读取旧 JSON 时，若只有 SWA 证据，就只展示可用的 SWA，主模型 CSA/HCA 保持不可用；方法页的目标阶段图另需新版核心提供 `target_stage_breakdowns`，不从这些源诊断反推。

| stage key | 含义 |
| --- | --- |
| `attention_prepare` | Attention HCPre、Q/KV 投影与 cache 准备；主模型 CSA/HCA 还含压缩或索引准备 |
| `attention` | Attention 内核 |
| `attention_output` | Attention 输出投影与整理 |
| `attention_post` | Attention HCPost 与层内衔接 |
| `moe_prepare` | FFN HCPre、共享/路由准备至 Dispatch 前 |
| `dispatch` | 专家 Dispatch |
| `experts` | 路由专家计算及其间隔 |
| `combine` | 专家 Combine |
| `ffn_output` | FFN 输出投影与整理 |
| `ffn_post` | FFN HCPost |

10 段按原层首 HCPre → 末 HCPost 的连续锚点分割，合计闭合到该层的实测 wall。它们不额外加入外推层均值或周期；不支持模板保留未识别状态。对照时先固定 case、rank 和周期，再区分“整层 wall”与“attention 内核”，并核对主、草稿实际输入 token 数、padding、图模式与 Task Type。具体采样的分析结果和优化方案仅保存在本地。

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
