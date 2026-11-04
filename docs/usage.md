# 使用说明

## 环境

使用 Python 3.10 或更新版本。核心统计、Excel 写入和 HTML 导出均只依赖标准库，不需要安装 requirements，也不需要在运行统计的机器上安装 PyTorch、CANN 或连接 NPU。

日常使用只需要根目录的三个运行脚本和配置文件，并准备自己的采样数据；三个脚本应保持在同一目录：

```text
profile_ep128.py
build_ep128_leadership_excel.py
ep128_xlsx_writer.py
profile_ep128_cases.example.json
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

这只演示两项输入，实际数量应与本次源采样相符。所有相对路径均相对于 cases JSON 所在目录。使用 `inputs` 时移除该 case 原来的 `path`，避免同时指定两种输入方式。

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

默认模板指向 `data/communication-reference.csv`；也可以在运行时覆盖：

```bash
python -u profile_ep128.py cases.local.json --comm-reference data/communication-reference.csv --out ep128_report --excel
```

方法2目前核对的外推范围是 EP16 → EP128，历史表需要同 BS 下 EP32、EP64、EP128 的 send、return 数值。可另附 total 列，存在时须与 send + return 一致。

仓库中的 `examples/communication-reference.synthetic.csv` 是全新合成数据，数值仅用于格式演示与回归测试。它不包含原历史实测表，不应与真实采样混用来形成性能结论。

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

## 7. 运行回归测试

日常统计无需执行测试。维护代码时，从仓库根目录运行：

```bash
python -B -S -m unittest discover -s tests -p "test_ep128*.py" -v
```

`-S` 用于验证标准库环境。依赖未归档私有采样的 8 项测试会明确跳过；可选 openpyxl 读取检查在未安装该包时跳过。测试使用的合成数据不能作为目标硬件实测证据。

## 8. 可选独立检查工具

从仓库根目录执行：

```bash
python tools/validate_ep128_capture.py cases.local.json --stats-dir ep128_report
python tools/profile_ep128_comm_reference.py data/communication-reference.csv --out communication_reference_report
```

前者独立核对原始采样及已导出的账本，后者检查指定通信表并导出观测比值。它们不会被主估算脚本自动调用，也不是运行前的必需步骤。其他参数见各工具的 `--help`。
