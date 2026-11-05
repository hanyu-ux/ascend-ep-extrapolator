# 纯实测 Host profiling

`profile_host.py` 独立分析 Ascend 导出的 `trace_view.json`，用于梳理可见 CPU 调用、运行时 API、同步、拷贝、入队/出队，以及同一 trace 中的设备任务。它直接统计采样结果，不计算 EP 外推，不需要 HC、草稿模型、接受长度或通信参考 CSV。

Step5、GQA+DSA、950 和单主模型可以作为场景信息填写。这些信息是用户声明，不会自动证明已识别模型层结构、已验证硬件性能或发现客户瓶颈。工具不会把已有 V4 的 CSA/HCA/SWA 阶段映射直接套到其他模型。

## 使用

需要 Python 3.10 或更新版本；分析和 HTML/CSV/JSON 输出只用标准库。加 `--excel` 时使用仓库自带 XLSX writer，无需安装 Office 或第三方 Excel 库。

```bash
python profile_host.py /path/to/trace_view.json \
  --out outputs/step5-host \
  --model Step5 --hardware 950 --main-only --excel
```

Windows 示例：

```powershell
python profile_host.py "D:\profiling\trace_view.json" --out "D:\reports\step5-host" --model Step5 --hardware 950 --main-only --excel
```

`--main-only` 记录“本场景没有草稿模型”的声明，不会自动验证模型拓扑。没有该选项时，工具同样不要求存在草稿模型。

默认分析整个 capture。只看一个已知时间窗口时，`--start-us` 和 `--end-us` 必须一起提供，采用原始 trace 时间轴上的绝对微秒值；它们始终以微秒为单位。

```bash
python profile_host.py /path/to/trace_view.json \
  --out outputs/one-window --start-us 123456000 --end-us 123466000 --excel
```

输入 `ts`/`dur` 默认按微秒解析。只有确认导出时间单位不同才使用 `--time-unit ns` 或 `--time-unit ms`；修改这个选项不会改变窗口参数的微秒单位。

输入时间先用 Decimal 读取。合法亚纳秒时间不会丢弃：起点及原始起点加时长得到的终点，分别按 `ROUND_HALF_EVEN` 量化至最近整数纳秒；每个端点误差至多 0.5 ns，单次时长误差至多 1 ns。量化数量及实际最大端点误差保留在 `source.time_quantization`，同时展示诊断；这不表示采样具有纳秒级物理准确度。负时长、非有限数值和越界时间仍视为非法。显式窗口参数要求精确到整数纳秒。

工具按事件类别和进程信息识别 Host/设备。自定义 trace 缺少这些标记时，可明确指定进程 PID；每个选项可重复使用。PID 必须来自该 trace 的 `pid` 字段，不是按进程名字猜测的系统 PID。

```bash
python profile_host.py /path/to/trace_view.json \
  --out outputs/selected-worker --host-pid 123 --device-pid 456 --excel
```

第一版输入为 JSON trace，支持 `traceEvents` 包装或顶层事件数组，以及完整时长 `X` 事件和按 `(pid, tid)` 配对的 `B`/`E` 事件。尚不直接读取 profiler DB、`kernel_details.csv` 或压缩 trace，也不拼接不同采样的时钟。损坏或截断 JSON 会报错；非法时间值、缺失配对、未支持的事件类型会在诊断中记录。

## 如何阅读结果

先看 HTML 开头的**主时间账本**，或 Excel 的**主时间账本** sheet：每个所选设备 PID 都有一份完整窗口，表格和横向柱状图显示各类互斥区间的耗时及占比。图表引用表格单元格，可在 Excel 编辑。设备 PID 不同，或采样不同，分母也各自独立，不能把它们相加。

账本按事件起止边界切分时间轴，不依赖 decode 周期，也不要求存在草稿模型。同一时刻只进入一个主表项目：

1. 有设备记录：只有一个设备语义类别时归入该类别；多个类别同时存在时归入“设备多类别重叠”。Host 调用即使同时存在，也不再次计入。
2. 没有设备记录但有 Host 调用：只有一个 Host 类别时归入该类别；多个类别同时存在时归入“Host 多类别重叠”。
3. 双方均无记录：归入未观测区间。包括窗口边界，不把这些时间静默丢弃或归成 CPU 忙时。

每份账本保留 `total_ns`、`closed` 和 `closure_error_ns`。完整窗口大于零时，各主项占比相加为 100%，各主项整数纳秒之和必须等于窗口总时长，闭合误差为 0 ns。显示精度可能使逐项四舍五入后的百分比略有偏差，精确核对使用 JSON / CSV 的整数纳秒。

例如，100 ms 窗口可以拆成 Attention 单独存在 30 ms、通信单独存在 10 ms、两者重叠 20 ms、仅 Host 可见 20 ms、双方均无记录 20 ms，主表恰好 100 ms。不能把 Attention 原始合计 50 ms、通信原始合计 30 ms 再加到这张表上。

**并行与重叠明细** sheet / HTML 折叠表展开重叠父项内具体的类别组合。这些组合是父项的子集，不能再次加入主表。记录覆盖状态的四项（Host 与设备同时可见、仅设备、仅 Host、双方无记录）也是同一窗口的另一视角，不是另一组可叠加的耗时。

类别由已记录的事件名称、类别等规则识别，未识别事件保留为“其他”。闭合说明时间账本没有重复或遗漏，**不说明业务阶段已完整识别**。通用矩阵算子不能仅凭名称断言属于 QKV 或输出投影，未采集的 metadata 构造、调度等业务阶段也不会凭空补出。设备优先是计时展示规则，不是“设备导致了这段等待”的因果判断。Host 和设备记录都可能含等待，因此这些占比不是硬件利用率。

再看原有 Host 热点、API 与同步、线程等辅助表，用于定位哪些调用覆盖了这些时间；它们保留 inclusive / self / union 原口径，不能与主账本重复相加。没有 `time_ledgers` 的旧 JSON 仍能按原格式导出。

JSON 保留统计口径、时间窗口、输入指纹、覆盖情况及诊断；CSV 便于筛选和二次分析，HTML 便于浏览。可选 XLSX 不挂在外推报告的“方法1/方法2”中。

| 文件 | 内容 |
| --- | --- |
| `host_profile_report.json` / `.html` / `.xlsx` | 结构化报告、浏览页、可选 Excel |
| `time_ledger.csv` | 每设备完整窗口的全部互斥主项、占比与闭合信息 |
| `time_ledger_details.csv` | 全部单类别及并行组合明细，含所属主表父项 |
| `time_ledger_coverage.csv` | 同一窗口四种 Host / 设备记录覆盖状态 |
| `threads.csv` / `host_hotspots.csv` | 按线程轨道聚合、按事件聚合的 wall 统计 |
| `host_api.csv` | API、同步、拷贝与队列事件 |
| `steps.csv` | 实际记录的步骤 annotation 区间 |
| `device_summary.csv` / `device_gaps.csv` | 各设备进程的记录覆盖与无记录区间 |
| `source_links.csv` | 设备任务与 Host 提交的来源关联及证据状态 |

主时间账本完整展示全部主项，不受 50 行预览限制。HTML 和 XLSX 的重叠组合最多逐项预览 50 行，超出的组合按所属重叠父项分别追加“其余已识别重叠组合”合计行，使用未显示组合的整数纳秒之和，保持可见子表与父项对账。这些合计行只是展示汇总，不代表未知，也不写入完整 JSON / CSV。原有辅助明细表最多预览 50 行。进程轨道清单及类别识别结果也会展示，完整清单见 JSON 的 `inventory`。self 不可用的行不会以零值混入 self 排名。

| 指标 | 具体含义 | 使用限制 |
| --- | --- | --- |
| Inclusive wall | 单次调用起止区间，聚合时相加 | 包含子调用和等待，嵌套调用相加会重复计时 |
| Self wall | 减去同一进程、同一线程中可确认嵌套的子区间 | 仍是墙钟时间，可能包含同步、阻塞、抢占；不是 OS CPU 执行时间 |
| Host wall union | 所选 Host 调用区间的时间并集 | 多线程重叠只计一次，不等于 CPU 利用率 |
| 各线程 union 之和 | 先在线程内取并集，再对线程求和 | 可大于整个窗口时长，不能直接加到设备耗时上 |
| Device recorded union | 同一设备进程内各 stream 任务区间的并集 | 表示已记录任务的覆盖，AICPU 和 DMA 另列 |
| No recorded device interval | 窗口内未被该设备进程已记录任务覆盖的区间 | 不代表整张卡空闲，更不直接证明 CPU 供给不足 |
| Queue / submit delay | 有明确关联证据时，列出队列与设备执行的间隔 | 是延迟，不能作为 CPU busy 或与区间耗时相加 |

出现交叉、重复或相同起止时间且无法确定包含关系的 Host 区间时，相关 self 值显示不可用。某聚合行只要含不可确认的调用，就不把剩余有效 self 的和伪装成完整 self 总量。窗口分析先保留完整嵌套关系再裁剪，避免把窗口外开始的子调用误记为父调用 self。

跨进程相同 `tid`、connection ID 或 correlation ID 不应自动视为同一线程或同一来源。CANN 与 Python 的合成 PID 也可能表示同一条物理线程，因此报告的 `(pid, tid)` 是 trace 线程轨道，轨道数不能当作 OS 线程数，跨轨道时长相加可能重复。来源关联只在记录支持且无歧义时给出；共享图或缺失关联保留未知。同步 API 耗时长可能是等待设备完成，不应直接归因为 CPU 算力不足。

Host 与设备空段重叠只说明两者发生在同一时间。确认 CPU 供给瓶颈还需要检查明确的提交关联、任务队列、设备上其他 stream 的工作，以及同步关系；不能只按时间重叠确定因果。队列/来源明细和设备空段明细为有上限的样本，统计总数与覆盖统计采用完整所选范围，截断信息在 `detail_limits` 中保留；不能用明细行数代替全量计数。

报告中的 CPU running time 和 CPU utilization 在没有 OS 调度证据时保留不可用。AICPU 属于设备侧处理器，不计作服务器 Host CPU。工具只覆盖本次采样可见的进程；前端、调度器、tokenizer 或其他 worker 如果没有被采集，就不会凭空出现在结果中。

## 采样建议和当前边界

优先提供一次完整导出的 `trace_view.json`，保留进程/线程元数据、CPU 调用和 CANN/runtime API；同时采集设备任务才能查看两侧覆盖和间隔。后续针对 Step5 的阶段归因，需要同次采样的实际算子/调用记录，运行框架及版本、模型 config，以及能对应到源码或用户 annotation 的阶段标记。

可以先用低开销 CPU/API 采样定位可见热点，再对小窗口增加调用栈或阶段 annotation。调用栈、shape 和 Python 跟踪可能增加采集开销，应与未开启这些选项的基线比较；不同开关下的时长不要直接当成优化收益。第一版接受已包含调用栈的 trace，不沿用 V4 外推入口的 `with_stack=false` 限制。

只有 trace 已记录的真实 `ProfilerStep#N` / `Iteration#N` annotation 才能作为迭代证据。没有步骤标记时只报告 capture 或显式窗口，不能从相似 kernel 自动推断请求边界、生成 token 数或 TPOT。没有草稿模型时，也不引入投机解码的 U1/U2、S、接受长度等口径。

实现通过合成事件测试并可用现有 A3/V4 trace 回归验证读取和统计能力。A3 回归不等于 Step5/950 验证；在获得客户采样前，不能承诺 CPU 热点排序、层内拆解精度或优化收益。

## 测试

```bash
python -B -S -m unittest discover -s tests -p 'test_ep128_host_profile.py' -v
python -B -S -m unittest discover -s tests -p 'test_ep128_host_ledger*.py' -v
```

测试数据均为手工构造的事件，覆盖嵌套 self 计算、进程/线程隔离、窗口裁剪、交叉/重复区间、B/E 配对、时间单位、缺失覆盖和非法输入。客户 trace、详细分析和优化方案应留在本地，不随代码提交。
