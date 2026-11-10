# 公开 NPU 算子 bound 分析 Skills 调查

调查日期：2026-11-10。目标：为本仓库离线 Top8 算子分析选取可独立实现、能回溯证据的规则。本次没有安装 Skill、执行外部脚本、采集设备数据或复制外部实现。

这是一次非穷尽检索：搜索公开 GitHub、厂商文档及 Skill 索引，再回到原仓库核对默认分支 commit、实际 `SKILL.md`、相关脚本和许可证。索引站只用于发现入口。以下“已核验”指静态阅读文件；不代表在本机 NPU 上验证可运行，也不代表厂商认证了我们的推论。仓库自称“官方 bundle”不等于所有收录内容都由硬件厂商维护。

## 可直接采取的结论

现有公开方案没有一个可以无条件替代我们的多 rank、多 step、Attention/MoE Top8 分析。最值得吸收的是：Ascend msopprof 的**按架构、按存储路径建模**，AWS Neuron 的**实际活动时间→实际工作量→理想工作量**证据链，以及 AMD AIE 的**内核性能与全图数据流分开验证**。Skill 名称、MAC 忙碌率、MFU 或一个固定阈值均不能单独证明瓶颈。

建议自行实现小型规则层，保留出处和适用范围；不要在分析脚本中自动安装、调用或嵌入这些第三方 Skill。许可不明、文档和代码矛盾、依赖专有工具的内容只作调查线索。

## 固定快照与许可

下列 SHA 是本次读取的默认分支快照；版本更新后需要重新核对。许可按实际 `LICENSE`/文件头记录，不以 GitHub API 的 `NOASSERTION` 等同于“无许可证”。外部同步目录仍须逐文件核对上游许可。

| 公开仓库 | 固定 commit | 本次核对的许可 | 性质 |
|---|---|---|---|
| ascend-ai-coding/awesome-ascend-skills | [`75ec89ce9d401be8ac95980e1c3f42f8d4315057`](https://github.com/ascend-ai-coding/awesome-ascend-skills/commit/75ec89ce9d401be8ac95980e1c3f42f8d4315057) | [README 内 MIT 文本](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/README.md#许可证)；external 多处 `license: UNKNOWN`，个别代码另标 CANN Open Software License 2.0 | 社区集合、本地 Skill 与外部同步内容混合 |
| Ascend/agent-skills | [`155ac37bd169ddb89479af528297cfb2237400aa`](https://github.com/Ascend/agent-skills/commit/155ac37bd169ddb89479af528297cfb2237400aa) | [MulanPSL-2.0](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/LICENSE) | Ascend 组织公开 Skill |
| huaweicloud/huaweicloud-skills | [`1547b98b48ece13ba364eebe74a47ba55669923f`](https://github.com/huaweicloud/huaweicloud-skills/commit/1547b98b48ece13ba364eebe74a47ba55669923f) | [MIT](https://github.com/huaweicloud/huaweicloud-skills/blob/1547b98b48ece13ba364eebe74a47ba55669923f/LICENSE)（仍需注意文件级来源） | Huawei Cloud 公开 Skill |
| Ascend/msopprof | [`3d18a9fa93d348f56a3877e2b73f43c34abd1586`](https://github.com/Ascend/msopprof/commit/3d18a9fa93d348f56a3877e2b73f43c34abd1586) | [代码 MulanPSL-2.0](https://github.com/Ascend/msopprof/blob/3d18a9fa93d348f56a3877e2b73f43c34abd1586/LICENSE)；[文档 CC BY 4.0](https://github.com/Ascend/msopprof/blob/3d18a9fa93d348f56a3877e2b73f43c34abd1586/docs/LICENSE) | 实际 profiling 工具，不能仅因有指南而叫 Skill |
| fla-org/flash-linear-attention | [`b8da4873fa0834849b64938af9f89bc35a347d90`](https://github.com/fla-org/flash-linear-attention/commit/b8da4873fa0834849b64938af9f89bc35a347d90) | [MIT](https://github.com/fla-org/flash-linear-attention/blob/b8da4873fa0834849b64938af9f89bc35a347d90/LICENSE) | 算子项目内 Ascend Skill |
| xLLM-AI/xllm-workflow | [`af230ee3182aec8e2214ee0e5501d9be6aea4238`](https://github.com/xLLM-AI/xllm-workflow/commit/af230ee3182aec8e2214ee0e5501d9be6aea4238) | 此快照未找到仓库级 LICENSE，不能据“公开可读”直接复制 | 项目工作流 Skill |
| aws-neuron/neuron-agentic-development | [`e63853627271bcb98d941c242641e698d210d5a0`](https://github.com/aws-neuron/neuron-agentic-development/commit/e63853627271bcb98d941c242641e698d210d5a0) | [Apache-2.0](https://github.com/aws-neuron/neuron-agentic-development/blob/e63853627271bcb98d941c242641e698d210d5a0/LICENSE.txt)；README 页脚的 All Rights Reserved 不是完整许可文本 | AWS Neuron 专用 Skill 与分析脚本 |
| Xilinx/mlir-aie | [`cbfc9921dab673d58409a542c08d14c30042b683`](https://github.com/Xilinx/mlir-aie/commit/cbfc9921dab673d58409a542c08d14c30042b683) | [Apache-2.0 WITH LLVM-exception](https://github.com/Xilinx/mlir-aie/blob/cbfc9921dab673d58409a542c08d14c30042b683/LICENSE)；Skill 文件也有 SPDX | AMD AIE2/AIE2P NPU Skill |
| DeepLink-org/DeepEval-Skills | [`5c4e1e97df9781ab36d8b851db2cb560718a01ca`](https://github.com/DeepLink-org/DeepEval-Skills/commit/5c4e1e97df9781ab36d8b851db2cb560718a01ca) | [BSD-3-Clause](https://github.com/DeepLink-org/DeepEval-Skills/blob/5c4e1e97df9781ab36d8b851db2cb560718a01ca/LICENSE) | 多硬件基准评测 Skill |
| qualcomm/qai-appbuilder | [`5f82738194cbf1352ab443bd0c665e8516377381`](https://github.com/qualcomm/qai-appbuilder/commit/5f82738194cbf1352ab443bd0c665e8516377381) | [BSD-3-Clause](https://github.com/qualcomm/qai-appbuilder/blob/5f82738194cbf1352ab443bd0c665e8516377381/LICENSE)，文件含明确 SPDX | Qualcomm HTP/SNPE/QNN 工具及 Skill |
| pytorch/executorch | [`5af315ae94fd71c4753e8ce5e68fe1279e32a7eb`](https://github.com/pytorch/executorch/commit/5af315ae94fd71c4753e8ce5e68fe1279e32a7eb) | [BSD License 文本](https://github.com/pytorch/executorch/blob/5af315ae94fd71c4753e8ce5e68fe1279e32a7eb/LICENSE) | 框架内 Qualcomm 后端 Skill |
| intel/skills | [`7a7dc387348a2442ffe00ca603f1fee6c4ffd192`](https://github.com/intel/skills/commit/7a7dc387348a2442ffe00ca603f1fee6c4ffd192) | [Apache-2.0](https://github.com/intel/skills/blob/7a7dc387348a2442ffe00ca603f1fee6c4ffd192/LICENSE) | 当前相关 profiling 入口主要是 Intel GPU/XPU |
| open-edge-platform/skills | [`933402013d45aeb8952c80ed66282f9c77bcf1c7`](https://github.com/open-edge-platform/skills/commit/933402013d45aeb8952c80ed66282f9c77bcf1c7) | [Apache-2.0](https://github.com/open-edge-platform/skills/blob/933402013d45aeb8952c80ed66282f9c77bcf1c7/LICENSE) | 包含 Intel NPU 的应用/流水线 Skill |

### 1. awesome-ascend-skills：确有分析脚本，但“roofline”范围有限

[profiling-analysis-computing](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/skills/profiling/profiling-analysis/profiling-analysis-computing/SKILL.md) 有高耗时筛选、指令占比透视、shape 解析等实际脚本；[profiling-analysis-roofline](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/skills/profiling/profiling-analysis/profiling-analysis-roofline/SKILL.md) 对应可读的 [kernel_roofline_compute_check.py](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/skills/profiling/profiling-analysis/profiling-analysis-roofline/scripts/kernel_roofline_compute_check.py)。后者计算 FLOPs、估算张量字节数，以计算/带宽下界比较实际耗时，能识别一部分 MatMul、量化 GEMM、GroupedMatmul、稀疏 Attention、KV 更新类型；未知类型保留未建模。

局限需要写进我们的设计：

- 默认入口先筛 `aic_mac_ratio >= 0.80`，因此不能覆盖低 MAC 的搬运、Vector、Scalar 瓶颈，更不是全 Top8 的通用判定器。
- 主要字节量来自 shape 与 dtype 的估算，不是实测 HBM 事务；不能区分片内复用、缓存命中、重复装载和布局转换。
- `_finalize_grouped` 明确拿不到 `group_list`，使用容量 M 和活跃专家读权重的范围假设；随后存在 `ceiling` 修正。它没有恢复真实每专家行数，因此不能把该估计当成精确 GMM MFU/roofline。
- 对若干 norm/quant/permute 类型直接使用“搬运”近似、将 FLOPs 设为 0；低精度峰值采用可配置倍率（默认 2），未知 dtype 有默认字节值。这些都是模型假设，不应沿用成硬件事实。

可采用：输入列别名、未知类型保留、显式 GB/s/TB/s 单位、理论量与实测量分别显示。应独立补齐：有效行数、量化 scale/布局、Attention 有效 Q/KV 长度、dtype 对应的真实峰值以及每个估算的上下界。

### 2. awesome 的 external：真实脚本与文字指导要分开

[AscendC optim 同步副本](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/external/gitcode-ascend/ascendc-operator-performance-optim/SKILL.md) 标记原仓为 `gitcode.com/Ascend/agent-skills`，同步 commit 为 `1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6`；不能当作 GitHub Ascend 仓当前 HEAD。

[PyPTO perf-analyzer](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/external/cannbot/ops/pypto-op-perf-tune/perf-analyzer/SKILL.md) 标记同步自 `cann/cannbot-skills`、commit `ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda`。其 [analyze_perf.py](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/external/cannbot/ops/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py) 实际解析 PyPTO `bubble_analysis.log`，按核区分执行、调度等待、前驱等待，计算核间负载差异；这比用 `1 - VEC ratio` 定义空泡更有根据。但它依赖 PyPTO 的日志契约，不能从普通 `kernel_details.csv` 自动还原这些等待。

[tune-incore](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/external/cannbot/ops/pypto-op-perf-tune/tune-incore/SKILL.md) 关注单 task 指令、tile 和特殊 shape，[tune-swimlane](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/75ec89ce9d401be8ac95980e1c3f42f8d4315057/external/cannbot/ops/pypto-op-perf-tune/tune-swimlane/SKILL.md) 关注泳道、任务依赖、stitch 和调度。它们是面向 PyPTO 的调优流程，不是已实现的跨算子 PMU 规则引擎。三个文件均标 `license: UNKNOWN`，`analyze_perf.py` 文件头另指向 CANN Open Software License 2.0；不从集合 README 的 MIT 推导整份 external 可直接复制。

可采用：将“核内指令限制”“任务依赖等待”“核间不均衡”作为不同问题；等待摘要必须标明原始日志及统计分母。星级/固定阈值评分只可作为可解释的提示。

### 3. Ascend/agent-skills：开发优化闭环，不是仅凭 CSV 自动断根因

[ascendc-operator-performance-optim](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/skills/ascendc-operator-performance-optim/SKILL.md) 按 tiling、搬运、API、内存层级、流水审查源码，要求保存基线、精度验证并用同 case 比较。配套 [performance-eval](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/skills/ascendc-operator-performance-eval/SKILL.md) 有 torch_npu profiler 示例和 case/report 格式；[Triton optim](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/skills/triton-operator-performance-optim/SKILL.md) 有 UB、Cube、tiling 等路径。

其有用之处是“证据→一次改动→正确性→重测”的流程。诸如 blockDim、16KB 搬运、512B 对齐和 double buffer 的建议均需匹配架构与 kernel 数据流；引用的收益数字是示例，不能写成通用承诺。源码审查流程依赖自定义 kernel 源码，不能把分析者没有看到的闭源算子内部原因填为已确认。

### 4. huaweicloud-skills：当前 upstream 有诊断器，但默认规则不能照搬

[MFU Skill](https://github.com/huaweicloud/huaweicloud-skills/blob/1547b98b48ece13ba364eebe74a47ba55669923f/skills/ai/modelarts/huawei-cloud-ascend-op-mfu-calculator/SKILL.md) 当前快照包含真实 [bottleneck_diagnoser.py](https://github.com/huaweicloud/huaweicloud-skills/blob/1547b98b48ece13ba364eebe74a47ba55669923f/skills/ai/modelarts/huawei-cloud-ascend-op-mfu-calculator/scripts/bottleneck_diagnoser.py)。它读 CSV 的时长、FLOPs、读写 bytes、CPU/AICPU 时间，允许传峰值，输出理由与建议。

静态审查发现重要限制：`classify_operator` 在缺少主导信号时默认标 `compute-bound`，在计算/带宽利用率都低时也标 `compute-bound`；通信按名称模式归类。这些结果不能当成因果结论。其 [MFU methodology](https://github.com/huaweicloud/huaweicloud-skills/blob/1547b98b48ece13ba364eebe74a47ba55669923f/skills/ai/modelarts/huawei-cloud-ascend-op-mfu-calculator/references/mfu-calculation-methodology.md) 的 batched 示例把 batch=32 重复乘了一次，说明公式、示例和硬件表也必须独立验算。MFU 低本身不能区分供数、尾块、同步、带宽或启动延迟。

[AscendC optim](https://github.com/huaweicloud/huaweicloud-skills/blob/1547b98b48ece13ba364eebe74a47ba55669923f/skills/ai/modelarts/huawei-cloud-ascendc-operator-performance-optim/SKILL.md) 包含 device/simulator wrapper 与流水/Scalar/搬运参考，适合作为进一步实验清单。[msopprof Skill](https://github.com/huaweicloud/huaweicloud-skills/blob/1547b98b48ece13ba364eebe74a47ba55669923f/skills/ai/modelarts/huawei-cloud-msot-msopprof-operator-profiler/SKILL.md) 主要组织真实工具采集；[profiler DB Skill](https://github.com/huaweicloud/huaweicloud-skills/blob/1547b98b48ece13ba364eebe74a47ba55669923f/skills/ai/modelarts/huawei-cloud-ascend-profiler-db-explorer/SKILL.md) 提供 DB/schema 查询方法。二者不等于另一个独立的自动 bound 分析引擎；README/Skill 中未有可核验实现支持的自动诊断和收益描述不作采纳依据。

可采用：保留原始 bytes/FLOPs/峰值与理由。必须更改的思想：数据缺失输出 `unknown`，名称只能给算子类别，不能把空缺默认填为 compute-bound。

### 5. Ascend/msopprof：本次最有针对性的 Ascend 技术依据

[用户指南](https://github.com/Ascend/msopprof/blob/3d18a9fa93d348f56a3877e2b73f43c34abd1586/docs/en/user_guide/msopprof_user_guide.md) 与实际 [roofline.cpp](https://github.com/Ascend/msopprof/blob/3d18a9fa93d348f56a3877e2b73f43c34abd1586/csrc/op_profiling/profiling/device/data_visualize/roofline.cpp) 都已核验。代码区分 `RoofLineOf910B`、`RoofLineOfA5`、`RoofLineOf310P`；按计算单元、存储单元、搬运流水和具体存储路径构造多个 roofline，而不只看总 HBM 带宽。

重要细节：

- MTE1 必须纳入分析；它可能限制片内供数，不能被 MTE2/3 替代。MTE 活动率高也不等于 HBM 带宽打满。
- `RoofLineOfA5::CubePipeLine` 中 MTE2 取 GM→L1 流量，MTE3 取 UB→L1 流量，MTE1 取 L1 相关读量；910B 的 PMU 推导不同。因此不能跨架构统一把所有 MTE3 标作“写回 GM”。FIXP 也应按实际路径解读。
- `GenerateAdvice` 使用 roofline 使用率 80% 和最大流水活动率 80% 分支给 compute/memory/latency 提示。它是该工具有前提的规则，不是“80% 即根因已证明”的定理。
- 功能随芯片不同：指南注明 A3 某些峰值带宽比例不展示，950 的 SIMD Vector 存储视图有限；指令时间线、MemoryDetail、replay 模式也有各自限制。不能把最新源码支持等同于内网已安装版本支持。

可采用：架构适配器、路径/峰值/计数器绑定、无适配时降为描述性活动倾向。真实带宽或具体路径证据不足时不输出“HBM-bound 已确认”。

### 6. FLA 与 xLLM：贴近实际项目，但不能过度解读

FLA 的 [fla-ascend-performance](https://github.com/fla-org/flash-linear-attention/blob/b8da4873fa0834849b64938af9f89bc35a347d90/.agents/skills/fla-ascend-performance/SKILL.md) 及 [analyze_profile.py](https://github.com/fla-org/flash-linear-attention/blob/b8da4873fa0834849b64938af9f89bc35a347d90/.agents/skills/fla-ascend-performance/scripts/analyze_profile.py) 确实能读取 Cube、Vector、Scalar、MTE1/2/3、Fixpipe、UB、cache 列。它明确强调流水活动不等于瓶颈；PipeUtilization 与 MemoryUB 分次采集，保持 workload 不变。分析器只是筛选和展示，不自动证明 bound；递归取首个 CSV 以及 filter 未命中回退 TopN，需要我们的脚本改为严格源归属和显式缺失。

xLLM 的 [profiler](https://github.com/xLLM-AI/xllm-workflow/blob/af230ee3182aec8e2214ee0e5501d9be6aea4238/skills/xllm-npu-profiler/SKILL.md) 有实际 CSV/重叠汇总 [脚本](https://github.com/xLLM-AI/xllm-workflow/blob/af230ee3182aec8e2214ee0e5501d9be6aea4238/skills/xllm-npu-profiler/scripts/analyze_xllm_npu_profile.py)，[compute-simulation](https://github.com/xLLM-AI/xllm-workflow/blob/af230ee3182aec8e2214ee0e5501d9be6aea4238/skills/xllm-npu-compute-simulation/SKILL.md) 是带硬件静态表的 FLOPs/what-if 流程。它把未校准估算标方向性判断，这是合理边界；其 [硬件表](https://github.com/xLLM-AI/xllm-workflow/blob/af230ee3182aec8e2214ee0e5501d9be6aea4238/reference/knowledge/npu-hardware-specs.json) 含按服务器规格推导的 per-NPU estimate，不能直接作为我们的逻辑 device 峰值，尤其需区分物理卡、die 与 rank。该仓未核实到仓库级 LICENSE。

## 其他真正 NPU：可借鉴的方法与不能迁移的部分

| 平台与已核验 Skill | 实际能力 | 本项目适用边界 |
|---|---|---|
| AWS Trainium/Inferentia：[neuron-nki-profile-querying](https://github.com/aws-neuron/neuron-agentic-development/blob/e63853627271bcb98d941c242641e698d210d5a0/skills/neuron-nki-profile-querying/SKILL.md)、[framework-profile-analysis](https://github.com/aws-neuron/neuron-agentic-development/blob/e63853627271bcb98d941c242641e698d210d5a0/skills/neuron-framework-profile-analysis-vllm-neuron/SKILL.md) | neuron-explorer 的 Summary、ActiveTime、Instruction、DMA 等真实 trace/parquet/SQL；分析引擎活动、传输 bytes、重复加载、TensorE 工作量 | 是强证据链参考；使用 Neuron 工具/schema，不能把 TensorE/DMA/峰值直接映射为 Ascend MAC/MTE |
| AWS：[hardware-entitlement](https://github.com/aws-neuron/neuron-agentic-development/blob/e63853627271bcb98d941c242641e698d210d5a0/skills/neuron-hardware-entitlement/SKILL.md) | 有 [operator_entitlement.py](https://github.com/aws-neuron/neuron-agentic-development/blob/e63853627271bcb98d941c242641e698d210d5a0/skills/neuron-hardware-entitlement/scripts/operator_entitlement.py)；按并行配置和模型计算算力/HBM 下界、组件边界与实际驻留 HBM 归属 | certified scope 明确只有 TE/HBM，VE/SC 未建模；理想融合/驻留、组件串行假设不等于当前执行时间 |
| AMD Ryzen/AIE NPU：[aie-kernel-opt](https://github.com/Xilinx/mlir-aie/blob/cbfc9921dab673d58409a542c08d14c30042b683/skills/aie-kernel-opt/SKILL.md)、[aie-dataflow-opt](https://github.com/Xilinx/mlir-aie/blob/cbfc9921dab673d58409a542c08d14c30042b683/skills/aie-dataflow-opt/SKILL.md) | AIE2/AIE2P C++/Peano 编译报告、II/资源/寄存器/栈限制、核 trace cycles 与全图 DMA/dispatch 分开验证，引用仓库实际构建和性能测试工具 | 可借鉴“静态候选≠实测加速”、同 workload A/B、全图限制与内核限制分开；其频率、端口、II、平台案例阈值不可搬到 Ascend |
| Cambricon MLU：[cambricon-nlp-operator](https://github.com/DeepLink-org/DeepEval-Skills/blob/5c4e1e97df9781ab36d8b851db2cb560718a01ca/skills/Cambricon/nlp/cambricon-nlp-operator/SKILL.md) | MLU590/CNNL/torch_mlu 的 GEMM/Conv/长尾/Transformer Block 基准，有 [runner](https://github.com/DeepLink-org/DeepEval-Skills/blob/5c4e1e97df9781ab36d8b851db2cb560718a01ca/skills/Cambricon/nlp/cambricon-nlp-operator/scripts/run_longtail.py)、case 契约、当前 run token 与结果校验 | 确是真 MLU Skill；主要证明准确性和延迟，没有核验到 MAC/内存分层 roofline 的 PMU 分类引擎 |
| Qualcomm HTP：[qai-runner-skill / aipc-toolkit](https://github.com/qualcomm/qai-appbuilder/blob/5f82738194cbf1352ab443bd0c665e8516377381/tools/skills/qai-runner-skill/SKILL.md) | [profiling 指南](https://github.com/qualcomm/qai-appbuilder/blob/5f82738194cbf1352ab443bd0c665e8516377381/tools/skills/qai-runner-skill/references/profiling.md) 用 QAIRT/QNN/SNPE detailed/optrace、HTP reader、逐层 cycles；有 [JSON→trace 转换脚本](https://github.com/qualcomm/qai-appbuilder/blob/5f82738194cbf1352ab443bd0c665e8516377381/tools/skills/qai-runner-skill/scripts/convert_qnn_to_trace.py) | 工具链专属，不是跨 NPU bound 判定器。转换脚本人工顺序累加层时间，且有默认总 cycles/time；不得用其重建时间线证明真实 overlap/关键路径。指南声称该脚本未随仓库提供，但当前 tree 实际存在，文档有漂移 |
| Qualcomm HTP/eNPU：[ExecuTorch qualcomm](https://github.com/pytorch/executorch/blob/5af315ae94fd71c4753e8ce5e68fe1279e32a7eb/.claude/skills/qualcomm/SKILL.md) | QNN 后端构建、算子/模型接入及逐层数值调试 | profiling 路由指向 `profiling.md`，并明确 `file TBD`；该快照该文件确实缺失，不能算已交付 kernel-bound Skill |
| Intel NPU：[DLStreamer coding-agent](https://github.com/open-edge-platform/skills/blob/933402013d45aeb8952c80ed66282f9c77bcf1c7/.agents/skills/dlstreamer-coding-agent/SKILL.md) | 明确可选择 Intel GPU/NPU/CPU，生成部署及推理流水，要求核对模型/元素是否支持 NPU | 是真实 NPU 应用工作流，没有核验到核内计算/搬运计数器 bound 引擎 |

AWS 的 [performance-bounds](https://github.com/aws-neuron/neuron-agentic-development/blob/e63853627271bcb98d941c242641e698d210d5a0/skills/neuron-nki-profile-querying/references/performance-bounds.md) 对本项目尤其有借鉴价值：先区分当前活动时间、按实际工作量估计的理想时间、去除冗余工作后的理想时间，再检查未受优化影响的其他引擎和关键路径。文档明确该组公式当前假设**单 NeuronCore**，且间隙不等于可直接节省的端到端时长；供数改善可能同时缩短 TensorE 活动时间，不能把各项时间当独立可加。

## 排除项与检索缺口

- [intel/skills 的 unitrace](https://github.com/intel/skills/blob/7a7dc387348a2442ffe00ca603f1fee6c4ffd192/skills/xpu-profile-unitrace/SKILL.md) 指向 `pti-gpu`、SYCL/Level Zero 和多 GPU；[openvino-gpu-run](https://github.com/intel/skills/blob/7a7dc387348a2442ffe00ca603f1fee6c4ffd192/skills/openvino-gpu-run/SKILL.md) 也明确 GPU。不能因为含 XPU/OpenVINO 词就列为 Intel NPU 核级分析能力。
- Gaudi/Habana：本次查过公开 Skill 搜索及 [Gaudi Profiling Workflow](https://docs.habana.ai/en/latest/Profiling/Profiling_Workflow.html)。可核验的是厂商 profiling 文档；未核实到直接适配 Gaudi 核级 bound 的公开 `SKILL.md`。同名 `gaudi` 的 3D/Blender Skill 已排除。未找到不意味着不存在。
- QNN 也会命中“Qualitative Neural Network”提示词技能，MLU/算力规划会命中泛 GPU 部署指南；均不作为目标硬件 bound 方案。
- 本次没有逐个审计所有 fork、GitCode 独有分支、闭源包或商业市场 Skill；external 的上游与同步版可能不一致。没有本机安装的 Skill 不应在报告中写成已调用。

## 对现有分析脚本的接入规则

以下为独立实现建议，是否已接入以本仓实际代码和测试为准。

| 规则 | 需要的证据 | 输出方式 |
|---|---|---|
| 先判断数据能回答什么 | 硬件/SoC、CANN/tool 版本、PMU set、字段单位、计数器分母、采集来源 | 明确 `observed`、`heuristic`、`model_estimate`、`insufficient`，不混写 |
| PMU 展示完整且按架构适配 | AIC/AIV MAC/VEC/Scalar、MTE1/2/3、Fixpipe、核内时间，区分不支持/缺采集/实测零 | 缺失不填 0；MTE1/Fixpipe 等关键路径缺失要影响覆盖率和置信度 |
| 活动倾向与物理 bound 分开 | 单条/聚合流水比仅能说明忙碌；真实带宽/冲突/指令等待才能加强定位 | “矩阵活动高”“片内搬运候选”等有条件结论，不直接宣称已确认 HBM-bound |
| FLOPs 必须匹配语义 | GEMM 的 M/N/K、有效 batch；GMM 各专家有效行；Attention 有效 Q/KV、head、window/sparsity；量化 dtype/布局 | 可算才算；容量 shape 只能是容量估计，未知不伪造有效 token/路由 |
| 字节量分实测和下界 | HBM/L2/L1/UB/L0 路径 bytes；或可解释的张量最小 I/O | 分列 `measured_bytes` / `estimated_min_bytes`；不可拿 tensor bytes 等同于实际 HBM 事务 |
| 峰值有来源和范围 | 对应 dtype、实际核数/频率、单核/逻辑 device/物理卡、带宽层级 | 不把低精度一律设为 FP16 的 2 倍；架构不明不套 msopprof/AWS 常数 |
| roofline 是带假设的下界 | 有效工作量及同范围峰值/带宽 | `max(compute_floor, memory_floor)` 只用于允许重叠的同一组件；跨有依赖的组件按真实图约束处理 |
| 不用补数推出空泡 | 同一时窗的活动区间、核/worker ID、同步/前驱等待证据 | 禁止 `100%-VEC` 当 C/V 空泡；禁止 MAC、MTE、VEC 活动率直接相加或用差值当重叠 |
| 源采样要保持归属 | case/rank/step/op 类型、shape、stream/task 身份、PMU采集轮次及覆盖次数 | 只在可对齐的调用间汇总；重复 run/不同 rank 不混合，分次采集不冒充同次时间线 |
| 低利用率或资料矛盾 | 实际缺项、模型异常、估算效率超合理范围、内部未建模工作 | 保留 `unknown`/模型不适配及原因，不能默认 compute-bound 或删掉异常行 |
| 优化建议须可证伪 | 预期哪个 bytes/冲突/活动/时长应变化；同 workload 正确性与复测 | 单变量实验清单；分析建议不写成已实施收益，合成/重建时间线不可提供真实 overlap 证据 |

推荐实施顺序：先补齐元数据/覆盖率/架构标注与诚实的候选判断；再对有可靠 shape 的 GEMM 做有效工作量计算；有真实流量、路径带宽和架构适配后才增加更强 roofline。这样可以复用社区的方法，同时保留我们已有多 case、多 rank、多 step、Attention/MoE 的数据归属和展示设计。
