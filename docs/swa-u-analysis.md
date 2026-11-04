# SWA 与 U 阶段：A3 回放分析及优化验证方案

这次修改细化离线分析，**没有修改推理框架、算子或模型输出，也没有取得已经验证的运行时加速**。CSA、HCA 的实现优化由其他分析任务负责；这里保留同 rank、同周期的对照，分析 SWA 和主、草稿之间的 U。

## 1. 数据范围与结论强度

回放来自两组独立的 A3 小模型采样，分别记为样本 A、B。每组包含 BS=4、8，各有 dp0、dp1，共 8 个采样/rank 组合；每个 rank 首尾各裁剪 2 个完整 HC 周期，共保留 944 个周期。它们是两个独立服务的采样，不能合并成一个更大的 EP 组。

源模型为 8 主层、3 草稿层，隐藏维度 1024、MoE 中间维度 512、16 个 query heads、head_dim=512、SWA 窗口 128、草稿候选长度 S=7；源 MoE TopK=16。每轮草稿 backbone 只有一次 forward。**这不是内网目标完整模型或 A5/EP128 的性能复现。** 下文具体 U 数值只取样本 A、BS4、dp0 的 68 个保留周期，不能与其他采样组的绝对时间混算。

按证据强度区分：

- **实测事实**：设备任务时间、Task Type、形状、流、层边界、周期内的守恒和重复序列。
- **源码支持的候选解释**：相应实现存在重复 metadata 构建路径、非因果可见 slot 构造和 graph padding。
- **待验证优化**：融合小算子、复用 metadata、调整图桶等是否缩短关键路径，必须通过相同运行条件的 A/B 采样确认。

采样中的 Python stack、模块标记和逐请求 ProfilerStep 不完整；一些 graph task 没有独立 Host 来源。设备锚点细分不等于证明了完整 Python 函数边界，Host 同时段覆盖也不等于本阶段独有 CPU 开销。

## 2. SWA 慢的是整层，还是 attention 内核

以下均为 **每层均值，单位 ms**。HCA 使用主1/3/5/7，CSA 使用主2/4/6，SWA 使用草稿0/1/2；只排除参考主0，保留其他层的 hash/normal 差异。每行单独比较同次采样、同 rank 的结果，不将不同 rank 的最大值拼接。

| 样本 | BS | rank | HCA 整层 | CSA 整层 | SWA 整层 | HCA attention | CSA attention | SWA attention |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 4 | dp0 | 4.327 | 5.547 | 3.870 | 0.203 | 0.189 | 0.490 |
| A | 4 | dp1 | 4.339 | 5.573 | 3.811 | 0.221 | 0.195 | 0.455 |
| A | 8 | dp0 | 4.106 | 5.510 | 3.916 | 0.218 | 0.258 | 0.526 |
| A | 8 | dp1 | 4.148 | 5.506 | 3.993 | 0.176 | 0.267 | 0.565 |
| B | 4 | dp0 | 0.614 | 0.803 | 0.808 | 0.035 | 0.050 | 0.131 |
| B | 4 | dp1 | 0.616 | 0.800 | 0.753 | 0.035 | 0.050 | 0.132 |
| B | 8 | dp0 | 0.704 | 0.911 | 0.871 | 0.038 | 0.070 | 0.179 |
| B | 8 | dp1 | 0.703 | 0.909 | 1.092 | 0.038 | 0.070 | 0.180 |

**8 个组合的 SWA attention 内核均比同组 HCA、CSA 更重；SWA 整层并非普遍最慢。** 944 个同周期对照中，314 个周期的 SWA 层均值同时超过 HCA、CSA。整层还包含 Q/KV 投影、HC、路由、专家计算、Dispatch、Combine 和间隔，attention 的排序不会自动成为整层排序。

以样本 A、BS4、dp0 为例，SWA 每层的 attention 内核约 0.490 ms、Dispatch 约 1.136 ms、Combine 约 0.367 ms。只优化 attention 有明确成本范围，不能把整层 3.870 ms 都算成可消除的 attention 开销。内网“SWA 整层明显更慢”的具体原因尚未由这些 A3 数据证明。

### 可见范围与 padding

DSpark 的非因果草稿 attention 让当前 query 看到尾部历史窗口和当前整个草稿块，**不是对完整历史进行双向 attention，也不是把 attention 再计算一遍**。实现用显式 slot 索引表达可见集合；可见集合、元数据 tiling 参数和实际 attention 参数必须一致。[官方 DSA 实现](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v1.py#L428-L527)

SWA、CSA、HCA 均包含原始窗口 attention；压缩比例不是 SWA 与其他两类耗时差异的充分解释。需要继续核对显式索引路径、tiling、query 长度、KV 布局、上下文长度和实际芯片上的算子实现。

A3 的 BS4 主层输入为 32 个 token，草稿为 35；BS8 主层为 64，样本 A 草稿为 56，样本 B 草稿为 70。源码中的 `sample_from_anchor=True` 使用 S 个草稿 query，False 才使用 S+1；BS4 的 35-token 草稿形状与 7-query 的图桶填充相符。不能把 35/32 的差异解释成实测有效输出 token 数，也不能认为所有 BS、采样组都有相同 padding 比例。可见索引宽度按 128 对齐，窗口 128、S=7 会产生 256 列存储；无效列是 -1，256 不等于实际可见 256 个 KV。

### PMU 与层内间隔提供的线索

样本 B、dp0 的 attention 内核 PMU 均值如下，三类 `Block Num` 均为 24：

| BS | 内核类型 | `aic_mte2_ratio` | `aic_mac_ratio` |
| --- | --- | ---: | ---: |
| 4 | HCA | 0.224 | 0.038 |
| 4 | CSA | 0.328 | 0.076 |
| 4 | SWA | 0.768 | 0.011 |
| 8 | HCA | 0.359 | 0.066 |
| 8 | CSA | 0.437 | 0.109 |
| 8 | SWA | 0.867 | 0.014 |

这些流水线活动指标支持“显式原始 slot 访问、数据搬运成本值得优先检查”的**中等置信度候选**；它们不是互斥 wall 份额，不等于实际带宽，更不能直接算 MFU。需要固定 Q/KV、mask、indices、tiling 与算子版本的微基准，才能判断是否确实由该路径造成。[昇腾官方对 MTE2 指标及数据搬运优化的说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_10009.html)

另以整数 ns 重扫 8 份 CSV，在排除主0的保留 HCA/CSA/SWA 层内，最大连续 CSV 未覆盖间隔为 **8.979 μs**，未发现 ≥100 μs 的大洞。已观测到的 SWA attention 更重，不能主要归因为这些层内存在大段 kernel 发射空洞；此结论不覆盖 U，也不表示硬件有效工作达到 100%。

## 3. U 中哪些成本值得先做

U 保留原来的 8+7 个父阶段。下表是样本 A、BS4、dp0 的逐轮均值，单位 ms。`busy union` 是所有已记录设备任务区间的并集；包含 AI_CPU、向量、计算和通信任务。它不是 AICORE 有效算力，也不是 MFU。`未记录` 只表示这些记录未覆盖的区间，不能在缺完整证据时全部叫作真实硬件空闲。

| U 父阶段 | wall | busy union | 未记录 |
| --- | ---: | ---: | ---: |
| 末主输出归并与 HC Head | 0.123 | 0.117 | 0.006 |
| 主最终 Norm、词表投影及采样前整理 | 0.166 | 0.147 | 0.019 |
| 拒绝采样与采样状态更新 | 0.115 | 0.114 | 0.001 |
| 草稿上下文特征融合 | 0.052 | 0.051 | 0.001 |
| DFlash 输入准备 | 0.052 | 0.052 | 0.000 |
| 草稿 Context KV 投影与写入 | 0.940 | 0.932 | 0.008 |
| 草稿 metadata 与输入映射准备 | 1.608 | 1.560 | 0.048 |
| 草稿 Embedding 与 HC 展开 | 0.025 | 0.023 | 0.001 |
| 草稿输出归并与 HC Head | 0.154 | 0.149 | 0.005 |
| 草稿最终 Norm 与词表投影 | 0.065 | 0.065 | 0.000 |
| 逐位置 Markov 偏置与选 token | 0.309 | 0.304 | 0.006 |
| 草稿结果整理与轮次交接 | 0.099 | 0.043 | 0.057 |
| 下一主模型 KV 与输入准备 | 0.146 | 0.111 | 0.036 |
| 下一主 metadata 与输入映射准备 | 1.125 | 1.091 | 0.033 |
| 下一主 Embedding 与 HC 展开 | 0.027 | 0.026 | 0.001 |

U 合计 **5.005770294 ms**，其中 CSV 未覆盖约 **0.221646 ms**；同次已对齐 trace 的 DMA 又覆盖其中 **0.124098 ms**，剩余约 **0.097549 ms**，仍不能全部认定为空闲。两个 metadata 父段合计约 **2.733 ms，占 U 的 54.6%**。所以缩短 U 应同时减少真实 metadata/映射任务和等待，不能只寻找 Host 空泡。

| 阶段 | CSV 未覆盖 μs | 其中 trace DMA 覆盖 μs | 补 DMA 后仍未覆盖 μs |
| --- | ---: | ---: | ---: |
| 下一主 KV 与输入准备 | 35.534 | 26.753 | 8.781 |
| 下一主 metadata | 33.392 | 26.041 | 7.351 |
| 草稿结果交接 | 56.620 | 43.030 | 13.589 |
| 草稿 metadata | 47.852 | 22.338 | 25.515 |

DMA 仅计算与同阶段 CSV 未覆盖区间的交集；不重复加入已有 busy，也不把 MODEL_WAIT、EVENT_RESET 等控制事件当成计算。补充 trace DMA 只改进诊断覆盖，不改变 U、层区间或周期。

### 下一主模型 KV 与输入准备

| 子任务 | wall 均值 ms | 识别条件 |
| --- | ---: | --- |
| KV block 清零 | 0.068 | `_zero_kv_blocks_kernel`，允许没有需清理的 block |
| 应用待写入 KV 块表 | 0.035 | `_apply_write_kernel`，允许没有待写入项 |
| 展开请求与 token 映射 | 0.013 | `_expand_idx_mapping_kernel` |
| 准备位置与序列长度 | 0.003 | `_prepare_pos_seq_lens_kernel` |
| 拼接采样与草稿 token | 0.003 | `_combine_sampled_and_draft_tokens_kernel` |
| 聚合 KV 块表 | 0.012 | `_gather_block_tables_kernel` |
| 计算 KV slot 映射 | 0.011 | `_compute_slot_mappings_kernel` |

这些是相邻锚点之间的连续区间，包含段内间隔；不是上述 kernel 的 duration 单独求和。后 5 项合计约 0.043 ms；清零及待写入两项连同对应间隔约 0.103 ms。优先核对 block 分配、复用及写入批次，避免先把全部精力花在最后几个微秒小 kernel 上。无须执行的可选任务可以自然缺席；缺少必需锚点则保留未识别状态。

### metadata：AI_CPU 本体与映射准备分开

下一主 metadata 每轮 133 个任务，其中 3 个 `SparseAttnSharedkvMetadata` 与 1 个 `QuantLightningIndexerV2Metadata` 在该样本的 **Task Type 均为 AI_CPU、Block Num=0**，本体时长合计约 **0.504 ms**；其余映射、准备及间隔约 **0.621 ms**。这些 AI_CPU 本体与后继 AIV 串行，在这段上明显暴露；这不能写成 1.091 ms 的 AICORE 计算。

前部可见 RoPE 查表的 Broadcast/Cast/Gather，之后穿插序列长度、attention/indexer metadata，再有 block/offset slot 格式化的 Ge/Fill/Div/Where/Remainder/Stack 和输入 Cast/Copy。草稿 metadata 每轮 258 个任务，含 2 个 AI_CPU attention metadata 本体，合计约 **0.206 ms**，以及多组显式 SWA 可见索引构造。拆分用每个 metadata 任务的实际 start/end 保留本体，前后空隙单列“映射与准备”；不会凭任务所处位置把整段命名为某个 Python 模块。

## 4. 源码解释与版本界限

以下函数/行号来自归档源码或官方固定版本；它们用于解释候选实现机制。A3 逐任务计时证据独立成立，但没有每个采样进程完整安装树的逐文件运行时证明，**不能把所有源码参考都称为与采样完全匹配**。内网版本更不能由本地文件名推定。

| 证据对象 | 可核对内容 | SHA-256 |
| --- | --- | --- |
| `swa_a3_summary.json`，本地离线分析产物 | 8 组合、944 周期、24 类层汇总 | `fb9e423d3f4526692fb867804ad41f62f832beaaee012d9d1057223c424719c5` |
| `u_evidence.json`，本地离线分析产物 | 68 周期 U15、子段、代表周期任务 | `4e553cb0b6029b17f724b868635089ac52933d149e31501a95472209bfe2f103` |
| `swa_a3_pmu.json`，本地派生摘要 | 分 rank、层类型的流水线活动指标 | `48cb156de2129abbaad8b459ccc7b4a43d69ac1d7e3bcbe7f54c8a8ae6465d17` |
| `validation_summary.json`，新脚本真实数据回放 | 68 周期、U 子阶段闭合及对齐 trace DMA | `fd99356457650997ec17f526563e472a180ee3ae5572f812178227a197adc6e7` |
| `current-image-graph-evidence.json`，归档源码证据容器 | 下列源码转储及各自校验值 | `a2bb8843d798ce735a106de92421a3ef033825c4d0aa1423df433ebe8a48cbd4` |
| `current-image-graph-evidence.json` 中的 Ascend v2 DSpark speculator | `build_draft_attn_metadatas`，291–324；`propose`，410–478 | `3580c1f6d5add803bbfb97f54aa2474a491165e21cfc1aab6d7e367508e205ba` |
| 同一归档中的 Ascend DFlash ACL graph manager | `run_fullgraph`，94–105 | `dc4ee25fee88ef1f82f24a64743904a0eda2d191c756e234136cfbb3b1ad1ecb` |
| 同一归档中的 upstream DFlash speculator | `propose`，450–473 | `f9c100f42f9326e48eb84e89281897e859ba761943f54d593523555ccd98dc63` |
| 同一归档中的 upstream DSpark speculator | query 数，47–55；`_generate_draft`，253–274 | `dcf4abdedbd2bddd0d76be198ccf5b4e467c46f645038c35fcab821491a35b03` |
| 已归档的官方 v0.30.0 `attention/dsa_v1.py` | 显式可见 slot，428–527；builder，846–1010、1085–1170 | `555c7589a6af0b790d4da15fba99c15ba70064404420c01baf2af27dd4d79ef2` |

原始采样、源码转储和含部署信息的 JSON 不随公开仓库发布；上面的摘要是脱敏汇总及审计标识，不能替代自己运行时的源文件与数据。

### 候选 1：FULL replay 之前重复构建草稿 metadata

归档 upstream DFlash `propose` 先调用 `_build_uniform_attn_metadata`，随后在 FULL 路径进入 `run_fullgraph`；Ascend 的 graph manager 又调用 DSpark `build_draft_attn_metadatas`，后者再次进入 `_build_uniform_attn_metadata`。A3 中两批 RoPE/可见索引准备及 2 个 metadata 本体与该路径相符。

这是优先验证的重复工作候选。应让 FULL 路径复用同一步已建 metadata，或明确只建一次满足 capture 形状的 metadata。**不能直接删掉第一次调用**：它还承担 builder 状态刷新；固定地址、pad request、query_start_loc、SAS metadata buffer 和异步 graph update 都要保持正确。两次构造也可能存在不同输入，须逐字段比较才能判定哪些可以复用。

旧 v1 proposer 的“DSpark eager only”注释不适用于这条 v2 FULL graph 路径；不能据此建议把当前草稿“首次改成入图”。

### 候选 2：融合显式可见 slot 及 slot 格式化

`build_dspark_swa_indices` 用多次张量操作构造每请求网格，再 Gather 块表并 repeat_interleave 到 query 维。可以评估一个直接写入持久输出 buffer 的融合 kernel，或先共享同一步不随 cache group 改变的 query 长度与位置网格，再分别查询各组块表。不同 cache group 的物理块号、block size 可能不同，不能把同一份物理 slot 输出无条件共享。

当前官方实现已经使用共享请求 preamble、固定 metadata buffer，并提供 `formatted_slot_mapping` 入口。优化前先检查运行版本是否已有这些能力，避免重新实现相同缓存。A3 的 block/offset 格式化与 A5 某些 flat-slot 路径不同，不直接照搬 A3 的算子数量。[官方 builder](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v1.py#L846-L1170)、[官方 KV 布局实现](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_attn_kv_plan.py#L170-L178)

## 5. 优化顺序与预算上界

下表“上界”只是该样本中被处理区间的成本范围；真实实现必须保留必要工作，实际收益会更小，也可能没有收益。相互重叠的预算不能相加；若优化把 U 的任务搬入层内，必须看完整周期是否下降。

| 优先级 | 作用范围与候选 | 此样本预算范围 | 主要正确性风险 | 后续验证 |
| --- | --- | --- | --- | --- |
| P0 | 统一同 rank、BS、S、层与 attention-only 计时口径 | 测量改进，不计加速 | 错把 U、padding 或层内通信当 attention | 离线核对边界、形状、同周期对照 |
| P1 | 草稿 FULL 路径只构建一次正确 metadata | 不超过草稿 metadata 的 1.608 ms；不能预先按一半计收益 | builder 状态、静态地址、图更新与 padding | 先做调用/字段对照，再做设备 A/B |
| P1 | SWA 索引与 slot 格式化融合、复用组无关中间量 | 只在两个 metadata 父段 2.733 ms 内取对应映射成本；与上一项重叠 | -1、空 query、页边界、各组块表、异步覆盖 | CPU 参考验证后做 kernel 正确性和 A/B |
| P1 | 减少主 metadata AI_CPU 本体开销或与独立工作重叠 | 当前主 metadata 本体约 0.504 ms | tiling 依赖当前长度；跨流过早消费 | 算子版本/依赖检查、跨流事件与端到端 A/B |
| P2 | KV block 清理/待写入去重与批处理 | 下一主输入准备合计 0.146 ms，其中后 5 项约 0.043 ms | 未清理状态污染、请求换序、页复用 | preemption、复用、完成/新增请求覆盖 |
| P2 | Context KV 投影/写入融合或减少中间复制 | Context KV 父段 0.940 ms | 每层投影权重不同，不能直接复用 KV；RoPE/格式需一致 | 输出与 cache 等价验证，真实层宽 A/B |
| P2 | 更合适的草稿图桶、attention tiling/显式索引路径 | 参考 SWA attention 每层 0.455–0.565 ms（样本 A） | graph 固定地址、DP 对齐、可见集合及数值误差 | 固定形状 kernel 与完整周期同时比较 |
| P3 | Markov、Head、采样与交接继续排查 | Markov 父段约 0.309 ms；交接约 0.099 ms | Markov 左到右依赖、随机数与拒绝采样语义 | 相同 seed 与接受率/输出评测；不能随意并行化全部位置 |

其余 U 同样保留，不以量小为由补零或删除。拒绝采样、上下文融合、Embedding、HC Head 的必要计算不能消失。把所有 U 都消除的反事实上界才是约 5.006 ms；它不是可实现收益预测。若 Neff 固定，TPOT 的变化为 `ΔR / Neff`；修改草稿长度后 Neff 也可能改变，必须重新测量。

## 6. 验证与验收

离线已经能验证：阶段锚点顺序、父子区间守恒、记录 task 类型、SWA/主层输入形状、同周期对照及源码候选重复调用。新脚本对上述 68 周期回放得到 **1,020 个 U 父阶段、1,904 个子阶段均 MATCHED，272 个父子窗口按整数 ns 闭合**；每轮 28 个子阶段包括 7 个 Markov 和 21 个本次重点子阶段。源周期仍为 **55.968415 ms**，U 仍为 **5.005770294 ms**。代表周期 581 个 U 任务的名称、时间戳和 Duration 已独立对回原 CSV；重点输入/metadata 的来源链也逐项复核。离线不能测出 kernel 融合、入图调整或跨流重叠后的真实性能。

运行时优化应另建 PR，保持一个主要变量，按以下顺序验收：

1. 固定模型/权重、芯片、软件与算子版本、TP/DP/EP、BS、S、上下文长度、graph 配置和 profiling 设置；记录原始文件校验值。
2. 使用参考实现检查生成 slot：短/长上下文、0 长 padding query、页边界、不同 block size、多 cache group、请求换序、-1 slot、容量边界；验证持久地址和异步 buffer 生命周期。
3. 验证输出、KV 内容、主模型确认结果；涉及 Markov/采样时增加相同 seed、分布、有效接受长度和模型评测。不能只对比吞吐。
4. 使用相同请求轨迹交替运行基线和候选，区分冷启动与稳定轮次；比较各 rank 的 R、U15、子阶段均值/中位数/P90、设备 busy union、未记录区间和 kernel sum。需要完整 runtime trace 才能确认跨流等待是否真的减少。
5. 检查周期下降、TPOT 与有效输出长度是否同时符合预期；不能把工作从 U 搬到层内、将不同 rank 最大项相加或减少采样覆盖当作加速。

内网复核至少需要同次、同 rank 的原始设备任务和对应 trace、实际源/草稿配置、S 与有效接受长度、软件/算子版本及 graph 模式。公开仓库继续只保存通用脚本、合成回归和脱敏分析。

## 7. 如何读取这次新增结果

常规统计命令不变，详见[使用说明](usage.md)。`step_residual_stages.csv` 保留 U15；`step_residual_substages.csv` 增加下一主输入准备及主/草稿 metadata 的 level=1 子阶段，Markov 子阶段继续保留。

`swa_stage_summary.csv` 给出逐 case、逐 rank、逐草稿层的 10 段统计；每 rank 的 `*-swa-stages.csv` 保留逐轮边界。每段始终含在原 SWA 层区间内，U 子段始终含在父 U 内，**均不另加到估算周期**。不支持的模板、缺锚点或不完整覆盖保持 `UNRESOLVED`/不可用状态；未知不补零。跨 rank 汇总和外推仍使用原有规则，不把这些诊断段重新拼成一个虚构周期。
