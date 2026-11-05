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
- `effective_tokens_per_round` 与 `token_basis` 明确有效新增 token 数的来源。示例 5.2 已包括 bonus，是假设，不是已核验接受率。
- `trim_rounds` 默认首尾各裁剪 2 轮。输入必须有足够的完整 HC 周期。

示例配置中的 `data/bs8_s7` 等路径是占位。可以递归指定整个 case 的目录，也可以用显式 `inputs` 避免 rank 归属含糊：

```json
{
  "name": "bs8-s7",
  "bs": 8,
  "draft_tokens": 7,
  "inputs": [
    {"rank": "dp0", "path": "data/bs8_s7/dp0/kernel_details.csv"},
    {"rank": "dp1", "path": "data/bs8_s7/dp1/kernel_details.csv"}
  ]
}
```

这只演示两项输入，实际数量应与本次源采样相符。JSON 中的相对路径（包括 `communication_reference`）均相对于 cases JSON 所在目录。命令行 `--comm-reference` 的相对路径则相对于运行命令时的工作目录。使用 `inputs` 时移除该 case 原来的 `path`，避免同时指定两种输入方式。

同次、同 rank 的 `trace_view.json` 可放在 CSV 相邻目录，或在对应 input 中配置 `trace_path`。缺 trace 时保留设备统计和缺失状态；不要借用其他轮次或其他 rank 的 trace。

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

如果只想先看配置版式、还没有采样数据，可执行：

```bash
python build_ep128_leadership_excel.py --cases profile_ep128_cases.example.json --preview --out "reports/配置版式预览.xlsx"
```

预览中的性能指标保持待估算，不是外推结果。

## 6. 如何核对结果

1. 检查 case 的 `config.bs`、Neff、采样背景和 rank 覆盖。
2. 检查每种方法的状态和所选 rank。方法1、2可能选择不同 rank。
3. 五项分解应闭合到目标周期；U 父阶段是 U 的展开，不再加到周期上。
4. 主0仅从参考均值排除；目标主层仍完整计数。
5. TPOT = 所选 rank 的目标平均周期 / Neff，不再除 BS 或 DP。
6. U 子阶段缺锚点时保留 `UNRESOLVED`；缺失不是零。
7. 源周期和软件回归闭合不能证明目标外推准确。新增层衔接、U 平稳、固定通信附加值和历史倍率迁移都是需要目标实测检验的假设。

不同 BS 的折线按数值排序。非 BS 条件不同会分组；重复 BS 不合并，缺值断线，单点组只显示表格。总览开头列出全部采样组合，日志中可核对实际读入的 JSON。

### 方法页中的阶段展开

“方法1估算”和“方法2估算”两个 sheet 中，每个 case 的“轮次剩余区间耗时构成”图下，依次展示五组独立表格和原生可编辑横向柱状图：

1. 主模型 CSA 的目标层内 10 阶段估算。
2. 主模型 HCA 的目标层内 10 阶段估算。
3. 草稿模型 SWA 的目标层内 10 阶段估算。
4. 下一主模型 KV 与输入准备子阶段。
5. 下一主 metadata 及输入映射准备子阶段。

前三组使用当前方法 `selected_rank`、保留轮次和原估算相同的参考层：先在每轮内取同类型参考层均值，再按保留轮次取均值，主0不参加参考。表格同时给出源单层均值、目标单层均值、目标同类全部层合计以及占完整目标周期的比例；主图采用目标同类全部层合计，便于比较目标模型的耗时构成。两种方法独立使用各自的通信倍率和所选 rank，不混用其他 rank 的诊断。

Dispatch 阶段只追加核心的 Dispatch 增量，Combine 阶段只追加 Combine 增量；“专家计算及衔接”整段只追加其中路由 GMM 的增量，激活及衔接时间不整体乘倍率。其他阶段沿用源参考均值。目标 10 段之和必须等于目标单层耗时，乘同类型目标层数后等于该类型层区间合计；这些费用已经包含在完整周期内，不再次追加。占周期比例的分母包含其他类型层、层间衔接和 U，因此单个类型的比例之和通常小于 100%。

后两组 U 子阶段仍按该方法所选 rank 的源实测展示，作为保留一次的 U 的展开，不套目标层数。源采样的逐层 CSA/HCA/SWA 诊断继续保存在“层与组件实测”和“时间账本”中。缺锚点、子阶段不完整或与父层不能闭合时，只把对应类型的目标细分标为不可用，父目标估算仍保留；未知值不补 0。旧 JSON 缺少 `target_stage_breakdowns` 时须用新版核心重新处理采样，仅重绘不能生成目标阶段估算。

### U15 与输入、metadata 子阶段

`step_residual_stages.csv` 继续保留原来的 U1 8 段、U2 7 段父阶段；`step_residual_substages.csv` 和每 rank 阶段明细保存 `level=1` 子阶段，`parent_stage_key` 指向父段。原有逐位置 Markov 子阶段继续保留。

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
