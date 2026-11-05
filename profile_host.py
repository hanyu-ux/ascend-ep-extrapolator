#!/usr/bin/env python3
"""Model-independent, measured Host timeline analysis (standard library only).

Chrome/Ascend trace endpoints are decoded as Decimal and quantized to integer ns.
Raw events spill to a temporary SQLite database; no model/draft/HC rules apply.
"""
import argparse
from array import array
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from functools import lru_cache
import heapq
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

from profile_ep128 import residual_args, residual_iter_trace_events


HOST_VERSION = "20261105.3.exclusive_ledger"
DETAIL_LIMIT = 1000
HOST_KINDS = ("cpu", "python", "annotation", "runtime", "sync", "copy", "enqueue", "dequeue")
CPU_KINDS = ("cpu", "python", "annotation")
SEMANTICS = [
    "本报告仅统计所选 trace 中可见的实测区间，不执行层数、EP 或 token 外推。",
    "trace 起止时间以 Decimal 读取，再分别按 ROUND_HALF_EVEN 量化至整数纳秒；每个端点误差至多 0.5 ns，单次时长误差至多 1 ns，量化数量在诊断中保留。",
    "Host inclusive 是调用跨度；self 是同一 pid/tid 轨道内扣除嵌套子调用后的跨度，均非 OS 实际运行时间。",
    "调用跨度可包含等待、抢占和同步；无系统调度采样时，CPU 忙时与利用率不可用。",
    "线程轨道覆盖合计不是物理线程 CPU 工作量：CANN 与 Python 可能使用不同合成 pid 表示同一 OS 线程。",
    "设备记录覆盖包含计算、通信、AICPU、DMA 和等待类设备任务；无记录区间不等于硬件空闲。",
    "Host 与设备无记录区间的重叠仅是时间证据，不证明 Host 导致设备等待。",
    "API、队列等待、提交至执行延迟、Host self 与设备覆盖不可相加为总耗时。",
    "没有步骤标注时只报告采样/选定窗口；ProfilerStep 标注也不自动等于请求或生成 token。",
    "模型、芯片与无草稿设置为用户声明；未自动验证 Step5 架构、950 兼容性或优化收益。",
    "worker trace 不保证覆盖独立的调度器、分词器和 API 服务进程，也不推断缺失 Python 模块。",
    "时间账本以同一采样/选定窗口为分母，按半开区间逐纳秒闭合；主表各项互斥，合计严格等于窗口跨度。",
    "时间账本先区分有设备记录与无设备记录，再在各区间列出同时可见的语义类别；多类别重叠单列，不按任意优先级分给某个算子。",
    "有设备记录期间的 Host 调用不另计入主表；Host 与设备的四种覆盖状态另表闭合，两张表不可相加。",
    "无设备记录时的 Host 调用范围可能包含同步、通信等待和抢占，不表示 CPU 实际运行；双方均无已分类记录的时间明确保留。",
    "设备类别由可见算子名/任务类型保守识别；普通矩阵乘不推断属于 QKV、输出投影或专家，Attention 不自动命名为 SWA。",
    "每个设备 pid 单独建立完整窗口账本，均关联所选 Host 轨道；不同设备账本不能相加为同一采样的墙钟时间。",
]


# Category names are evidence labels, not inferred model stages.  In particular,
# ordinary GEMM cannot establish whether work belongs to QKV, O projection or an
# expert.  Overlapping *different* categories have their own ledger bucket.
_LEDGER_CLASSES = {
    "device_qkv_rope_cache": "显式 QKV / RoPE / KV cache 算子记录",
    "device_attention": "普通/融合 Attention 记录（不推断 SWA）",
    "device_sparse_attention": "稀疏 Attention 记录",
    "device_sparse_indexer": "稀疏索引器记录",
    "device_indexer_metadata": "索引器/稀疏 Attention 设备元数据记录",
    "device_moe_route": "显式 MoE 路由/分发/归并记录",
    "device_expert_matmul": "分组矩阵乘记录（专家候选）",
    "device_matmul": "其他矩阵乘记录（归属未确认）",
    "device_fused_matmul_communication": "融合矩阵乘与通信记录（不可再拆）",
    "device_norm_activation": "归一化/激活算子记录（含同名融合算子）",
    "device_dma": "设备 DMA / memcpy / memset 记录",
    "device_control": "设备同步/事件/通知/控制记录",
    "device_communication": "设备通信记录",
    "device_aicpu_other": "其他 AICPU 记录（非服务器 CPU）",
    "device_other": "其他/未细分设备记录",
    "host_only_communication": "无设备记录：Host 通信调用范围",
    "host_only_sync": "无设备记录：Host 同步调用范围",
    "host_only_copy": "无设备记录：Host 拷贝/转换调用范围",
    "host_only_queue": "无设备记录：Host 队列调用范围",
    "host_only_runtime": "无设备记录：运行时 API 调用范围",
    "host_only_other": "无设备记录：其他 Host 调用范围",
    "device_overlap": "多类设备记录重叠（另列组合）",
    "host_overlap": "无设备记录：多类 Host 调用范围重叠",
    "unobserved": "Host 与设备均无已分类记录",
}


@lru_cache(maxsize=16384)
def _ledger_class(name, tasktype, kind):
    """Conservative category of one already classified Host/device event."""
    lower = name.lower()
    compact = re.sub(r"[^a-z0-9]", "", lower)
    task = re.sub(r"[^a-z0-9]", "", tasktype.lower())
    if kind in ("device", "aicpu", "dma"):
        # MC2 can expose the same fused operation as AI_CPU and communication
        # records.  Classify its explicit name first, so those duplicate views
        # do not fabricate a cross-category overlap.
        if ("matmul" in compact or "gemm" in compact) and any(
                token in compact for token in ("reducescatter", "allreduce", "allgather")):
            return "device_fused_matmul_communication"
        if any(token in compact for token in ("unionsparsemeta", "sparsemeta", "indexermeta",
                                              "csaprefillupdate", "appendcurrentregion", "decodesummarylayout",
                                              "genericblocksparseattentionmetadata")):
            return "device_indexer_metadata"
        if "indexer" in compact:
            return "device_sparse_indexer"
        if any(token in compact for token in ("sparseattention", "blocksparseattention")):
            return "device_sparse_attention"
        if any(token in compact for token in ("flashattention", "fainfer", "fusedinferattention",
                                              "fusedattention", "attentionscore", "pagedattention")):
            return "device_attention"
        if any(token in compact for token in ("qkv", "rotary", "rope", "reshapeandcache", "kvcache")):
            return "device_qkv_rope_cache"
        if "moe" in compact and any(token in compact for token in
                ("gating", "routing", "route", "dispatch", "combine", "tokenperm", "distribute")):
            return "device_moe_route"
        if any(token in compact for token in ("groupedmatmul", "groupedgemm", "groupgemm")) or compact.startswith("gmm"):
            return "device_expert_matmul"
        if "matmul" in compact or "gemm" in compact:
            return "device_matmul"
        if any(token in compact for token in ("rmsnorm", "layernorm", "swiglu", "gelu", "silu", "relu")):
            return "device_norm_activation"
        if kind == "dma" or any(token in compact + task for token in ("memcpy", "memset", "dma")):
            return "device_dma"
        if any(token in compact + task for token in
               ("synchronize", "streamwait", "event", "notify", "streamswitch", "streamactive", "labelgoto", "labelset", "conditionsqe")):
            return "device_control"
        if "communication" in task or any(token in compact for token in
                ("hccl", "hcom", "allreduce", "allgather", "reducescatter", "alltoall", "broadcast")) or re.search(
                    r"(?:^|[^a-z])(?:send|recv|receive)(?:$|[^a-z])", lower):
            return "device_communication"
        return "device_aicpu_other" if kind == "aicpu" else "device_other"
    if any(token in lower for token in ("gloo", "c10d", "hccl", "hcom")):
        return "host_only_communication"
    if kind == "sync":
        return "host_only_sync"
    if kind == "copy" or re.match(r"^aten::(?:copy_|to|_to_copy)(?:$|\.)", lower):
        return "host_only_copy"
    if kind in ("enqueue", "dequeue") or "queue" in lower:
        return "host_only_queue"
    return "host_only_runtime" if kind == "runtime" else "host_only_other"


def _exclusive_time_ledgers(db, start, end, device_pids):
    """Partition raw classified event intervals; never use capped detail rows.

    An endpoint sweep maintains multiplicities, so nested or duplicated records
    of one category cover time once.  Different concurrent categories retain
    their exact set, rather than being assigned by an arbitrary priority.
    """
    total = end - start
    if total <= 0:
        raise ValueError("exclusive ledger requires a positive window")
    host_sql = ",".join("?" * len(HOST_KINDS))
    all_kinds = (*HOST_KINDS, "device", "aicpu", "dma")
    kind_sql = ",".join("?" * len(all_kinds))
    db.execute("CREATE INDEX IF NOT EXISTS ledger_order ON events(start,end)")
    ledgers = []
    for pid in device_pids or (None,):
        primary, details, states = {}, {}, {}
        active = {"host": Counter(), "device": Counter()}
        endings, cursor = [], start
        previous = {"primary": None, "details": None, "state": None}
        state_labels = {"both": "Host 与设备均有记录", "device_only": "仅设备有记录",
                        "host_only": "仅 Host 有记录（含等待）", "neither": "双方均无已分类记录"}

        def record(container, key, duration, group, **fields):
            row = container.setdefault(key, {"key": key, "duration_ns": 0, "segment_count": 0, **fields})
            row["duration_ns"] += duration
            if previous[group] != key:
                row["segment_count"] += 1
            previous[group] = key

        def account(left, right):
            if right <= left:
                return
            host_keys, device_keys = tuple(sorted(active["host"])), tuple(sorted(active["device"]))
            if device_keys:
                components = device_keys
                key = device_keys[0] if len(device_keys) == 1 else "device_overlap"
            elif host_keys:
                components = host_keys
                key = host_keys[0] if len(host_keys) == 1 else "host_overlap"
            else:
                components, key = (), "unobserved"
            duration = right - left
            record(primary, key, duration, "primary", label=_LEDGER_CLASSES[key])
            detail_key = key + ("|" + "+".join(components) if components else "")
            detail_label = " + ".join(_LEDGER_CLASSES[item] for item in components) if len(components) > 1 else _LEDGER_CLASSES[key]
            record(details, detail_key, duration, "details", parent_key=key,
                   label=detail_label, components=list(components))
            state = "both" if host_keys and device_keys else "host_only" if host_keys else "device_only" if device_keys else "neither"
            record(states, state, duration, "state", label=state_labels[state])

        def advance(stamp):
            nonlocal cursor
            while endings and endings[0][0] <= stamp:
                stop = endings[0][0]
                account(cursor, stop)
                cursor = stop
                while endings and endings[0][0] == stop:
                    _, side, category = heapq.heappop(endings)
                    active[side][category] -= 1
                    if not active[side][category]:
                        del active[side][category]
            account(cursor, stamp)
            cursor = stamp

        sql = ("SELECT name,tasktype,kind,start,end FROM events WHERE kind IN (" + kind_sql + ") "
               "AND end>? AND start<? AND (kind IN (" + host_sql + ") OR pid=?) ORDER BY start,end,id")
        args = (*all_kinds, start, end, *HOST_KINDS, pid)
        for name, tasktype, kind, left, right in db.execute(sql, args):
            left, right = max(start, left), min(end, right)
            if right <= left:
                continue
            advance(left)
            side = "host" if kind in HOST_KINDS else "device"
            category = _ledger_class(name, tasktype, kind)
            active[side][category] += 1
            heapq.heappush(endings, (right, side, category))
        advance(end)
        if endings or active["host"] or active["device"]:
            raise ValueError("exclusive ledger has unfinished active intervals")

        parent_sums = Counter()
        for row in details.values():
            parent_sums[row["parent_key"]] += row["duration_ns"]
        errors = [sum(rows["duration_ns"] for rows in source.values()) - total
                  for source in (primary, details, states)]
        errors.extend(parent_sums[key] - row["duration_ns"] for key, row in primary.items())
        if any(errors):
            raise ValueError("exclusive timeline ledger failed exact integer-ns closure: " + repr(errors))

        def output(rows):
            return [{**row, "duration_ms": row["duration_ns"] / 1e6,
                     "share": row["duration_ns"] / total} for row in rows]

        state_rows = [states.get(key, {"key": key, "label": label, "duration_ns": 0, "segment_count": 0})
                      for key, label in state_labels.items()]
        ledgers.append({"device_pid": pid, "start_ns": start, "end_ns": end,
            "total_ns": total, "total_ms": total / 1e6, "closed": True, "closure_error_ns": 0,
            "breakdown": output(sorted(primary.values(), key=lambda row: (-row["duration_ns"], row["key"]))),
            "details": output(sorted(details.values(), key=lambda row: (row["parent_key"], -row["duration_ns"], row["key"]))),
            "coverage_states": output(state_rows),
            "classifications": [{"key": key, "label": label,
                "side": "device" if key.startswith("device_") else "host" if key.startswith("host_") else "unknown"}
                for key, label in _LEDGER_CLASSES.items()],
            "semantics": ["所有占比均除以同一完整窗口，主表/明细/四态表分别闭合，三者不可相加。",
                "同类别并行/重复记录合并覆盖；不同类别并行单列组合，不推断资源忙时或业务因果。",
                "无设备记录的 Host 调用跨度不等于 CPU 实际运行，未观测时间不等于硬件空闲。"]})
    return ledgers


def _scaled_ns(value, factor=1000):
    if isinstance(value, bool) or value is None:
        raise ValueError("timestamp/duration must be a finite number")
    try:
        number = Decimal(str(value)) * factor
        if not number.is_finite() or abs(number) >= 2 ** 63:
            raise ValueError("timestamp/duration is nonfinite or outside int64")
        return number
    except InvalidOperation as error:
        raise ValueError("invalid timestamp/duration") from error


def _ns(value, factor=1000):
    """Explicit CLI windows retain strict, exact integer-ns coordinates."""
    number = _scaled_ns(value, factor)
    if number != number.to_integral_value():
        raise ValueError("explicit timestamp has sub-ns precision")
    return int(number)


def _identity(value):
    return str(value) if value is not None else ""


def _connection(value):
    value = _identity(value).strip()
    return None if value in ("", "-1", "18446744073709551615") else value


def _device_process(name):
    return bool(re.search(r"^ascend[ ._]hardware\b|^device(?:\b|_)|^(?:npu|gpu)[ :\d]", name, re.I))


def _classify(name, category, process, tasktype, pid, host_pids, device_pids):
    cat, lower = category.lower(), name.lower()
    tokens = set(re.split(r"[,;\s]+", cat))
    device_hint = bool(tokens.intersection(("kernel", "aicore", "aicpu", "device")))
    host_hint = bool(tokens.intersection(("cpu_op", "python_function", "user_annotation", "runtime", "cann", "cann_api")))
    if pid in device_pids:
        device = True
    elif pid in host_pids:
        device = False
    elif _device_process(process) or device_hint:
        if host_hint:
            return "unknown"
        device = True
    elif host_hint or re.search(r"^python(?:\b|$)|^cann(?:\b|$)|^runtime(?:\b|$)", process, re.I):
        device = False
    else:
        return "unknown"
    if device:
        if "aicpu" in cat or "AI_CPU" in tasktype.upper() or "AICPU" in tasktype.upper():
            return "aicpu"
        if re.search(r"dma|memcpy|memset", tasktype, re.I) or re.search(r"memcpy|memset", lower):
            return "dma"
        return "device"
    if "enqueue" in lower:
        return "enqueue"
    if "dequeue" in lower:
        return "dequeue"
    if re.search(r"synchronize|streamwaitevent|eventwait|notify_wait|synchronize", lower):
        return "sync"
    if re.search(r"memcpy|memset|hosttodevice|devicetohost", lower):
        return "copy"
    if "python_function" in tokens:
        return "python"
    if "user_annotation" in tokens:
        return "annotation"
    if "cpu_op" in tokens:
        return "cpu"
    return "runtime" if tokens.intersection(("runtime", "cann", "cann_api")) or re.search(r"^cann(?:\b|$)|^runtime(?:\b|$)", process, re.I) else "cpu"


class _Union:
    """Union length for intervals fed in ascending start order."""
    def __init__(self):
        self.end, self.total = None, 0

    def add(self, start, end):
        if end > start:
            self.total += max(0, end - max(start, self.end if self.end is not None else start))
            self.end = max(end, self.end if self.end is not None else end)


def _union(spans):
    end, result = None, []
    for start, stop in sorted(spans):
        if stop <= start:
            continue
        if end is None or start > end:
            result.append([start, stop])
        else:
            result[-1][1] = max(end, stop)
        end = result[-1][1]
    return result


def _overlap(spans, start, end):
    # Disjoint merged intervals; prefix search avoids scanning earlier trace.
    starts, ends, prefix = spans
    left, right = bisect_right(ends, start), bisect_right(starts, end - 1)
    if right <= left:
        return 0
    total = prefix[right] - prefix[left]
    return total - max(0, start - starts[left]) - max(0, ends[right - 1] - end)


def _coverage(spans):
    merged = _union(spans)
    starts, ends, prefix = array("q"), array("q"), array("q", [0])
    for start, end in merged:
        starts.append(start); ends.append(end); prefix.append(prefix[-1] + end - start)
    return starts, ends, prefix


def _percentile(values, fraction):
    values = sorted(values)
    point = (len(values) - 1) * fraction
    left = int(point)
    return (values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (point - left)) / 1e6


def _ingest(db, path, factor, diagnostics, progress):
    db.executescript("""
        CREATE TABLE events(id INTEGER PRIMARY KEY,pid TEXT,tid TEXT,name TEXT,cat TEXT,
            start INTEGER,end INTEGER,kind TEXT,conn TEXT,corr TEXT,tasktype TEXT,
            self_ns INTEGER,self_status TEXT);
        CREATE TABLE edges(id INTEGER,pid TEXT,tid TEXT,name TEXT,cat TEXT,stamp INTEGER,
            ph TEXT,conn TEXT,corr TEXT,tasktype TEXT);
        CREATE TABLE flows(key TEXT,ph TEXT,pid TEXT,tid TEXT,stamp INTEGER);
    """)
    processes, threads, scan, phases = {}, {}, {}, Counter()
    batch, edge_batch, flow_batch = [], [], []
    max_rounding_error = Decimal(0)

    def flush():
        db.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        db.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?)", edge_batch)
        db.executemany("INSERT INTO flows VALUES (?,?,?,?,?)", flow_batch)
        batch.clear(); edge_batch.clear(); flow_batch.clear()

    for number, event in enumerate(residual_iter_trace_events(path, stats=scan), 1):
        phase = _identity(event.get("ph"))
        phases[phase] += 1
        pid, tid = _identity(event.get("pid")), _identity(event.get("tid"))
        name, cat = str(event.get("name", "")), str(event.get("cat", ""))
        args = residual_args(event)
        if phase == "M":
            if name == "process_name":
                processes[pid] = str(args.get("name", ""))
            elif name == "thread_name":
                threads[pid, tid] = str(args.get("name", ""))
            continue
        if phase not in ("X", "B", "E", "s", "f", "t"):
            continue
        try:
            if not pid or not tid:
                raise ValueError("pid/tid missing")
            raw_start = _scaled_ns(event.get("ts"), factor)
            raw_duration = _scaled_ns(event.get("dur"), factor) if phase == "X" else Decimal(0)
            raw_end = raw_start + raw_duration
            # Validate the raw duration before rounding: a negative fraction
            # must not become a legitimate zero-duration event.
            if raw_duration < 0 or abs(raw_end) >= 2 ** 63:
                raise ValueError("invalid duration/end")
            start = int(raw_start.to_integral_value(rounding=ROUND_HALF_EVEN))
            end = int(raw_end.to_integral_value(rounding=ROUND_HALF_EVEN))
            if abs(start) >= 2 ** 63 or abs(end) >= 2 ** 63:
                raise ValueError("rounded timestamp outside int64")
        except (ValueError, ArithmeticError):
            diagnostics["INVALID_EVENT"] += 1
            continue
        error = max(abs(raw_start - start), abs(raw_end - end))
        if error:
            diagnostics["SUB_NS_QUANTIZATION"] += 1
            max_rounding_error = max(max_rounding_error, error)
        conn = _connection(args.get("connectionid"))
        corr = _connection(args.get("correlationid"))
        task = str(args.get("tasktype", ""))
        if phase == "X":
            batch.append((number, pid, tid, name, cat, start, end, None, conn, corr, task, None, None))
        elif phase in ("B", "E"):
            edge_batch.append((number, pid, tid, name, cat, start, phase, conn, corr, task))
        else:
            identity = event.get("id2", event.get("id"))
            if identity is not None:
                # Chrome id2.local is process-local; id2.global and plain id
                # flow tokens may cross processes. Never bridge by tid alone.
                local = pid if isinstance(identity, dict) and "local" in identity else None
                key = json.dumps([cat, name, event.get("scope"), local, identity], sort_keys=True, default=str)
                flow_batch.append((key, phase, pid, tid, start))
        if len(batch) + len(edge_batch) + len(flow_batch) >= 2000:
            flush()
        if number % 50000 == 0 and progress:
            progress(f"读取 trace：{number:,} 个事件")
    flush()
    db.execute("CREATE INDEX edges_order ON edges(pid,tid,stamp,id)")
    pending, last_thread = [], None
    for identity, pid, tid, name, cat, stamp, ph, conn, corr, task in db.execute(
            "SELECT * FROM edges ORDER BY pid,tid,stamp,id"):
        if (pid, tid) != last_thread:
            diagnostics["UNMATCHED_BEGIN"] += len(pending)
            pending, last_thread = [], (pid, tid)
        if ph == "B":
            pending.append((identity, pid, tid, name, cat, stamp, conn, corr, task))
        elif not pending:
            diagnostics["UNMATCHED_END"] += 1
        else:
            row = pending.pop()
            if (name and name != row[3]) or (cat and cat != row[4]):
                diagnostics["MISMATCHED_BEGIN_END"] += 1
                continue
            batch.append((*row[:6], stamp, None, *row[6:], None, None))
            if len(batch) >= 2000:
                flush()
    diagnostics["UNMATCHED_BEGIN"] += len(pending)
    flush()
    scan["phase_counts"] = dict(phases)
    scan["time_quantization"] = {"unit": "ns", "rounding": "ROUND_HALF_EVEN",
        "quantized_events": diagnostics["SUB_NS_QUANTIZATION"],
        "max_endpoint_rounding_error_ns": float(max_rounding_error)}
    scan["unsupported_phase_events"] = sum(count for ph, count in phases.items()
        if ph not in ("M", "X", "B", "E", "s", "f", "t"))
    diagnostics["UNSUPPORTED_PHASE"] += scan["unsupported_phase_events"]
    return processes, threads, scan


def _self_accounting(db, start, end, processes, thread_names, diagnostics):
    """Construct the full same-track tree before clipping to the report window."""
    threads, hotspots, updates = [], [], []
    query = "SELECT id,name,cat,start,end,kind FROM events WHERE pid=? AND tid=? AND kind IN (%s) ORDER BY start,end DESC,id" % ",".join("?" * len(HOST_KINDS))
    tracks = db.execute("SELECT DISTINCT pid,tid FROM events WHERE kind IN (%s) ORDER BY pid,tid" % ",".join("?" * len(HOST_KINDS)), HOST_KINDS).fetchall()
    for pid, tid in tracks:
        groups, stack, wall = {}, [], _Union()
        selected_calls, inclusive, self_sum, invalid = 0, 0, 0, 0

        def finish(row):
            nonlocal selected_calls, inclusive, self_sum, invalid
            left, right = max(start, row["start"]), min(end, row["end"])
            valid = row["valid"]
            own = row["end"] - row["start"] - sum(b - a for a, b in _union(row["children"])) if valid else None
            updates.append((own, "VALID" if valid else "AMBIGUOUS", row["id"]))
            if len(updates) >= 2000:
                db.executemany("UPDATE events SET self_ns=?,self_status=? WHERE id=?", updates)
                updates.clear()
            if right < left or (right == left and not start <= row["start"] < end):
                return
            clipped = max(0, right - left)
            child = sum(max(0, min(right, b) - max(left, a)) for a, b in _union(row["children"]))
            own_clip = clipped - child if valid else None
            selected_calls += 1; inclusive += clipped
            invalid += not valid
            self_sum += own_clip or 0
            key = row["kind"], row["name"], row["cat"]
            group = groups.setdefault(key, {"values": [], "self_ns": 0, "invalid": 0})
            group["values"].append(clipped)
            group["self_ns"] += own_clip or 0
            group["invalid"] += not valid

        for identity, name, cat, left, right, kind in db.execute(query, (pid, tid, *HOST_KINDS)):
            while stack and stack[-1]["end"] <= left:
                finish(stack.pop())
            row = {"id": identity, "name": name, "cat": cat, "start": left, "end": right,
                   "kind": kind, "children": [], "valid": True}
            if stack and (right > stack[-1]["end"] or (left, right) == (stack[-1]["start"], stack[-1]["end"])):
                row["valid"] = False
                for ancestor in stack:
                    ancestor["valid"] = False
                while stack and right > stack[-1]["end"]:
                    finish(stack.pop())
            if stack:
                stack[-1]["children"].append((left, right))
            stack.append(row)
            wall.add(max(start, left), min(end, right))
        while stack:
            finish(stack.pop())
        if not selected_calls:
            continue
        common = {"pid": pid, "tid": tid, "process": processes.get(pid, ""), "thread": thread_names.get((pid, tid), "")}
        threads.append({**common, "calls": selected_calls, "inclusive_sum_ms": inclusive / 1e6,
            "self_sum_ms": None if invalid else self_sum / 1e6, "self_unavailable_calls": invalid,
            "wall_union_ms": wall.total / 1e6})
        for (kind, name, cat), group in groups.items():
            values = group["values"]
            hotspots.append({**common, "kind": kind, "name": name, "category": cat, "calls": len(values),
                "inclusive_sum_ms": sum(values) / 1e6,
                "self_sum_ms": None if group["invalid"] else group["self_ns"] / 1e6,
                "self_unavailable_calls": group["invalid"], "mean_ms": sum(values) / len(values) / 1e6,
                "p50_ms": _percentile(values, .5), "p95_ms": _percentile(values, .95), "max_ms": max(values) / 1e6,
                "status": "AMBIGUOUS_SELF" if group["invalid"] else "MEASURED"})
        diagnostics["AMBIGUOUS_SELF"] += invalid
    db.executemany("UPDATE events SET self_ns=?,self_status=? WHERE id=?", updates)
    return sorted(threads, key=lambda row: -row["wall_union_ms"]), sorted(hotspots, key=lambda row: -row["inclusive_sum_ms"])


def _device_accounting(db, start, end, processes, coverages, host_available):
    devices, largest = [], []
    total_gaps = 0
    for (pid,) in db.execute("SELECT DISTINCT pid FROM events WHERE kind IN ('device','aicpu','dma') ORDER BY pid").fetchall():
        union, aicpu, dma = _Union(), _Union(), _Union()
        count, duration_sum, gap_host, gap_count = 0, 0, 0, 0
        cursor, previous = start, None

        def gap(left, right, following):
            nonlocal gap_host, gap_count, total_gaps
            if right <= left:
                return
            gap_count += 1; total_gaps += 1
            host = _overlap(coverages["host"], left, right)
            gap_host += host
            row = {"device_pid": pid, "start_ns": left, "end_ns": right, "duration_ms": (right - left) / 1e6,
                "kind": "whole_window" if previous is None and following is None else "leading" if previous is None else "trailing" if following is None else "internal",
                "previous_name": previous[0] if previous else None, "previous_tid": previous[1] if previous else None,
                "next_name": following[0] if following else None, "next_tid": following[1] if following else None,
                "host_overlap_ms": host / 1e6 if host_available else None,
                "sync_overlap_ms": _overlap(coverages["sync"], left, right) / 1e6 if host_available else None,
                "copy_overlap_ms": _overlap(coverages["copy"], left, right) / 1e6 if host_available else None,
                "api_overlap_ms": _overlap(coverages["api"], left, right) / 1e6 if host_available else None,
                "evidence": "HOST_OVERLAP_ONLY" if host else "INSUFFICIENT_HOST_EVIDENCE"}
            entry = (right - left, total_gaps, row)
            if len(largest) < DETAIL_LIMIT:
                heapq.heappush(largest, entry)
            elif entry[:2] > largest[0][:2]:
                heapq.heapreplace(largest, entry)

        for name, tid, left, right, kind in db.execute("SELECT name,tid,start,end,kind FROM events WHERE pid=? AND kind IN ('device','aicpu','dma') AND end>? AND start<? ORDER BY start,end DESC,id", (pid, start, end)):
            left, right = max(start, left), min(end, right)
            count += 1; duration_sum += right - left
            if right <= left:
                continue
            if left > cursor:
                gap(cursor, left, (name, tid))
            if right > cursor:
                cursor, previous = right, (name, tid)
            union.add(left, right)
            if kind == "aicpu":
                aicpu.add(left, right)
            if kind == "dma":
                dma.add(left, right)
        if not count:
            continue
        gap(cursor, end, None)
        devices.append({"pid": pid, "process": processes.get(pid, ""), "calls": count,
            "recorded_union_ms": union.total / 1e6, "recorded_sum_ms": duration_sum / 1e6,
            "no_recorded_ms": (end - start - union.total) / 1e6, "gap_count": gap_count,
            "aicpu_union_ms": aicpu.total / 1e6, "dma_union_ms": dma.total / 1e6,
            "host_overlap_in_gaps_ms": gap_host / 1e6 if host_available else None})
    return devices, [entry[2] for entry in sorted(largest, reverse=True)], total_gaps


class _CallIndex:
    def __init__(self, db, kinds):
        self.tracks = {}
        sql = "SELECT id,pid,tid,start,end FROM events WHERE kind IN (%s) ORDER BY pid,tid,start,end DESC,id" % ",".join("?" * len(kinds))
        for identity, pid, tid, start, end in db.execute(sql, kinds):
            vectors = self.tracks.setdefault((pid, tid), tuple(array("q") for _ in range(4)))
            for vector, value in zip(vectors, (start, end, max(end, vectors[2][-1] if vectors[2] else end), identity)):
                vector.append(value)

    def containing(self, pid, tid, stamp):
        vectors = self.tracks.get((pid, tid))
        if not vectors:
            return []
        starts, ends, prefix, identities = vectors
        return [identities[i] for i in range(bisect_right(prefix, stamp), bisect_right(starts, stamp)) if ends[i] > stamp]


def _source_links(db, start, end, progress):
    """Explicit IDs/flow endpoints only; never infer an edge from a shared tid."""
    db.executescript("""
        CREATE INDEX conn_events ON events(conn,kind);
        CREATE INDEX corr_events ON events(pid,corr,kind);
        CREATE INDEX flow_endpoint ON flows(ph,pid,tid,stamp);
        CREATE INDEX flow_token ON flows(key,ph);
    """)
    cpus, dequeues = _CallIndex(db, CPU_KINDS), _CallIndex(db, ("dequeue",))
    enqueues = _CallIndex(db, ("enqueue",))
    columns = ("id", "pid", "tid", "name", "start", "end", "kind", "conn", "corr", "self_status")

    @lru_cache(maxsize=8192)
    def event(identity):
        return dict(zip(columns, db.execute("SELECT id,pid,tid,name,start,end,kind,conn,corr,self_status FROM events WHERE id=?", (identity,)).fetchone()))

    @lru_cache(maxsize=8192)
    def candidates(conn):
        if conn is None:
            return (), 0, 0
        nodes = tuple(row[0] for row in db.execute("SELECT id FROM events WHERE conn=? AND kind IN ('runtime','sync','copy')", (conn,)))
        groups, tasks = db.execute("SELECT count(DISTINCT pid),count(*) FROM events WHERE conn=? AND kind IN ('device','aicpu','dma')", (conn,)).fetchone()
        return nodes, groups, tasks

    @lru_cache(maxsize=8192)
    def flow_sources(key):
        return db.execute("SELECT pid,tid,stamp FROM flows WHERE key=? AND ph='s'", (key,)).fetchall()

    def cpu_at(pid, tid, stamp):
        calls = sorted((event(i) for i in cpus.containing(pid, tid, stamp)), key=lambda r: (r["start"], -r["end"]))
        if not calls or any(row["self_status"] != "VALID" for row in calls):
            return None
        return calls[-1]

    counts, rows, total = Counter(), [], 0
    query = "SELECT id,pid,tid,name,start,conn FROM events WHERE kind IN ('device','aicpu','dma') AND end>? AND start<? ORDER BY start,id"
    for identity, pid, tid, name, stamp, conn in db.execute(query, (start, end)):
        total += 1
        row = {"device_event_id": identity, "device_pid": pid, "device_tid": tid,
            "device_name": name, "start_ns": stamp, "api_event_id": None, "api_pid": None,
            "api_tid": None, "api_name": None, "cpu_event_id": None, "cpu_name": None,
            "enqueue_event_id": None, "dequeue_event_id": None, "queue_wait_ms": None,
            "submit_to_device_ms": None, "status": "UNLINKED", "reason": "no unique explicit source"}
        nodes, device_groups, device_tasks = candidates(conn)
        reasons, source_points, cpu_candidates, origin_flows = [], [], {}, []
        ambiguous = False
        for (key,) in db.execute("SELECT DISTINCT key FROM flows WHERE ph='f' AND pid=? AND tid=? AND stamp=?", (pid, tid, stamp)):
            origins = flow_sources(key)
            if len(origins) == 1:
                source_points.append(origins[0])
                origin_flows.append((key, origins[0]))
            elif origins:
                ambiguous = True; reasons.append("reused flow start ID")
        matched = [event(i) for i in nodes]
        if len(matched) > 1:
            matched = [node for node in matched if any(node["pid"] == p and node["tid"] == t and node["start"] <= s < node["end"] for p, t, s in source_points)]
            if len(matched) != 1:
                ambiguous = True; reasons.append("reused connection ID")
        if device_groups > 1:
            matched = [node for node in matched if any(node["pid"] == p and node["tid"] == t and node["start"] <= s < node["end"] for p, t, s in source_points)]
            if len(matched) != 1:
                ambiguous = True; reasons.append("connection ID spans device processes without explicit API flow")
        node = matched[0] if len(matched) == 1 else None
        if node and node["start"] > stamp:
            ambiguous = True; reasons.append("API starts after device task")
            node = None
        graph_named = bool(node and re.search(r"graph|aclmdl.*execute", node["name"], re.I))
        graph_shared = bool(node and (device_tasks > 1 or graph_named))
        direct_cpus = {}
        for key, (p, t, s) in origin_flows:
            # HostToDevice starts inside an API are not per-kernel CPU
            # origins. A graph's one-to-many flow is not such an origin either.
            inside_api = node and (p, t) == (node["pid"], node["tid"]) and node["start"] <= s < node["end"]
            unique_finish = db.execute("SELECT count(*) FROM flows WHERE key=? AND ph='f'", (key,)).fetchone()[0] == 1
            if s <= stamp and not inside_api and (not graph_shared or unique_finish):
                cpu = cpu_at(p, t, s)
                if cpu:
                    direct_cpus[cpu["id"]] = cpu
        if node:
            row.update(api_event_id=node["id"], api_pid=node["pid"], api_tid=node["tid"], api_name=node["name"],
                submit_to_device_ms=(stamp - node["start"]) / 1e6)
            source_points.append((node["pid"], node["tid"], node["start"]))
        deq_candidates = {}
        for p, t, s in source_points:
            if s > stamp:
                ambiguous = True; reasons.append("flow starts after device task")
                continue
            cpu = cpu_at(p, t, s)
            if cpu:
                cpu_candidates[cpu["id"]] = cpu
            for i in dequeues.containing(p, t, s):
                deq_candidates[i] = event(i)
        if len(deq_candidates) == 1:
            deq = next(iter(deq_candidates.values()))
            # Cross-process queue edges need an explicit flow. The correlation
            # fallback is strictly scoped to the dequeue process, never tid.
            flowed_enqs = set()
            for (key,) in db.execute("SELECT DISTINCT key FROM flows WHERE ph='f' AND pid=? AND tid=? AND stamp=?", (deq["pid"], deq["tid"], deq["start"])):
                if json.loads(key)[0] != "async_task_queue":
                    continue
                origins = flow_sources(key)
                if len(origins) != 1:
                    ambiguous = True; reasons.append("reused queue flow ID")
                    continue
                p, t, s = origins[0]
                if s > deq["start"]:
                    ambiguous = True; reasons.append("queue flow starts after dequeue")
                    continue
                flowed_enqs.update(enqueues.containing(p, t, s))
            enqs = [(i,) for i in flowed_enqs]
            if not enqs and deq["corr"] is not None:
                enqs = db.execute("SELECT id FROM events WHERE pid=? AND corr=? AND kind='enqueue'", (deq["pid"], deq["corr"])).fetchall()
            if len(enqs) == 1:
                enq = event(enqs[0][0])
                if flowed_enqs and enq["corr"] is not None and deq["corr"] is not None and enq["corr"] != deq["corr"]:
                    ambiguous = True; reasons.append("queue flow contradicts correlation ID")
                elif enq["start"] <= deq["start"]:
                    row.update(enqueue_event_id=enq["id"], dequeue_event_id=deq["id"])
                    if enq["end"] <= deq["start"]:
                        row["queue_wait_ms"] = (deq["start"] - enq["end"]) / 1e6
                    else:
                        reasons.append("queue intervals overlap: wait unavailable")
                    cpu = cpu_at(enq["pid"], enq["tid"], enq["start"])
                    if cpu:
                        cpu_candidates[cpu["id"]] = cpu
                else:
                    ambiguous = True; reasons.append("enqueue starts after dequeue")
            elif len(enqs) > 1:
                ambiguous = True; reasons.append("reused queue correlation ID within process")
        elif len(deq_candidates) > 1:
            ambiguous = True; reasons.append("multiple containing dequeue calls")
        if graph_shared:
            # A graph launch is shared by many kernels. Only a per-task flow
            # can identify a CPU origin; an enclosing launch scope cannot.
            cpu_candidates = direct_cpus
            for field in ("enqueue_event_id", "dequeue_event_id", "queue_wait_ms", "submit_to_device_ms"):
                row[field] = None
            reasons.append("shared graph/API launch: no per-kernel launch latency")
        if len(cpu_candidates) == 1:
            cpu = next(iter(cpu_candidates.values()))
            row.update(cpu_event_id=cpu["id"], cpu_name=cpu["name"])
        elif len(cpu_candidates) > 1:
            ambiguous = True; reasons.append("conflicting CPU origins")
        row["status"] = ("AMBIGUOUS" if ambiguous else
            "CPU_ONLY" if graph_shared and row["cpu_event_id"] else
            ("GRAPH_SHARED_NO_ORIGIN" if graph_named else "SHARED_API_NO_ORIGIN") if graph_shared else
            "LINKED" if row["cpu_event_id"] and node else
            "CPU_ONLY" if row["cpu_event_id"] else "API_ONLY" if node else "UNLINKED")
        if ambiguous:
            # Never publish a guessed origin or latency as a measured link.
            for field in ("api_event_id", "api_pid", "api_tid", "api_name", "cpu_event_id", "cpu_name", "enqueue_event_id", "dequeue_event_id", "queue_wait_ms", "submit_to_device_ms"):
                row[field] = None
        row["reason"] = "; ".join(reasons) if reasons else "explicit trace identity/flow" if row["status"] != "UNLINKED" else row["reason"]
        counts[row["status"]] += 1
        if len(rows) < DETAIL_LIMIT:
            rows.append(row)
        if total % 25000 == 0 and progress:
            progress(f"关联 Host 来源：{total:,} 个设备任务")
    return rows, dict(counts), total


def analyze_trace(path, *, start_us=None, end_us=None, host_pids=None, device_pids=None,
                  time_unit="us", model="", hardware="", main_only=False, progress=None):
    """Return measured timeline accounting; caller writes chosen report formats."""
    factors = {"us": 1000, "ns": 1, "ms": 1000000}
    if time_unit not in factors:
        raise ValueError("time_unit must be us, ns or ms")
    if (start_us is None) != (end_us is None):
        raise ValueError("start_us and end_us must be provided together")
    requested_hosts = {_identity(p) for p in host_pids or ()}
    requested_devices = {_identity(p) for p in device_pids or ()}
    if requested_hosts & requested_devices:
        raise ValueError("one pid cannot be both Host and device")
    path = Path(path).resolve()
    diagnostics = Counter()
    with tempfile.TemporaryDirectory(prefix="ascend-host-") as temporary:
        with closing(sqlite3.connect(str(Path(temporary) / "events.sqlite"))) as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("PRAGMA temp_store=FILE")
            processes, thread_names, scan = _ingest(db, path, factors[time_unit], diagnostics, progress)
            updates = []
            process_kinds = defaultdict(set)
            for identity, pid, name, cat, tasktype in db.execute("SELECT id,pid,name,cat,tasktype FROM events"):
                kind = _classify(name, cat, processes.get(pid, ""), tasktype, pid, requested_hosts, requested_devices)
                if requested_hosts and kind in HOST_KINDS and pid not in requested_hosts:
                    kind = "excluded"
                if requested_devices and kind in ("device", "aicpu", "dma") and pid not in requested_devices:
                    kind = "excluded"
                updates.append((kind, identity)); process_kinds[pid].add(kind)
                if len(updates) >= 2000:
                    db.executemany("UPDATE events SET kind=? WHERE id=?", updates); updates.clear()
            db.executemany("UPDATE events SET kind=? WHERE id=?", updates)
            db.executescript("CREATE INDEX track_time ON events(pid,tid,start,end); CREATE INDEX kind_time ON events(kind,start,end);")
            present_pids = set(process_kinds)
            if (requested_hosts | requested_devices) - present_pids:
                raise ValueError("requested pid has no complete duration events: " + ", ".join(sorted((requested_hosts | requested_devices) - present_pids)))
            bounds = db.execute("SELECT min(start),max(end) FROM events WHERE kind NOT IN ('unknown','excluded')").fetchone()
            if bounds[0] is None:
                raise ValueError("no classified timed Host/device events; inspect metadata or specify --host-pid/--device-pid")
            start, end = (_ns(start_us), _ns(end_us)) if start_us is not None else bounds
            if end <= start:
                raise ValueError("analysis window must have a positive duration")
            if end <= bounds[0] or start >= bounds[1]:
                raise ValueError("analysis window does not intersect classified events")
            threads, hotspots = _self_accounting(db, start, end, processes, thread_names, diagnostics)
            def coverage(kinds, pid=None):
                sql = "SELECT start,end FROM events WHERE kind IN (%s) AND end>? AND start<?" % ",".join("?" * len(kinds))
                args = (*kinds, start, end)
                if pid is not None:
                    sql += " AND pid=?"; args = (*args, pid)
                return _coverage((max(start, a), min(end, b)) for a, b in db.execute(sql + " ORDER BY start,end", args))
            coverages = {"host": coverage(HOST_KINDS), "sync": coverage(("sync",)), "copy": coverage(("copy",)),
                         "api": coverage(("runtime", "sync", "copy"))}
            devices, gaps, gap_total = _device_accounting(db, start, end, processes, coverages, bool(threads))
            device_cover = coverage(("device", "aicpu", "dma"), devices[0]["pid"]) if len(devices) == 1 else None
            if progress:
                progress("建立互斥完整时间账本（保留重叠与未观测区间）")
            time_ledgers = _exclusive_time_ledgers(db, start, end, [row["pid"] for row in devices])
            host_union_ns = coverages["host"][2][-1]
            for ledger in time_ledgers:
                state_ns = {row["key"]: row["duration_ns"] for row in ledger["coverage_states"]}
                if state_ns["both"] + state_ns["host_only"] != host_union_ns:
                    raise ValueError("exclusive ledger Host union differs from existing coverage")
                recorded_union_ns = ((device_cover if device_cover is not None else
                    coverage(("device", "aicpu", "dma"), ledger["device_pid"]))[2][-1]
                    if ledger["device_pid"] is not None else 0)
                if state_ns["both"] + state_ns["device_only"] != recorded_union_ns:
                    raise ValueError("exclusive ledger device union differs from existing coverage")
            links, source_counts, link_total = _source_links(db, start, end, progress)
            steps = []
            for identity, pid, tid, name, left, right in db.execute("SELECT id,pid,tid,name,start,end FROM events WHERE kind='annotation' AND end>? AND start<? ORDER BY start,id", (start, end)):
                if not re.match(r"^(?:ProfilerStep|Iteration)#\d+(?:\b|$)", name):
                    continue
                left, right = max(start, left), min(end, right)
                recorded = _overlap(device_cover, left, right) / 1e6 if device_cover is not None else None
                steps.append({"event_id": identity, "name": name, "pid": pid, "tid": tid,
                    "start_ns": left, "end_ns": right, "wall_ms": (right - left) / 1e6,
                    "host_wall_union_ms": _overlap(coverages["host"], left, right) / 1e6,
                    "device_recorded_union_ms": recorded,
                    "device_gap_ms": (right - left) / 1e6 - recorded if recorded is not None else None})
            unknown = db.execute("SELECT count(*) FROM events WHERE kind='unknown'").fetchone()[0]
            diagnostics["UNCLASSIFIED_EVENT"] += unknown
            self_unavailable = sum(row["self_unavailable_calls"] for row in threads)
            host_count = sum(row["calls"] for row in threads)
            if not host_count:
                diagnostics["NO_HOST_EVENTS"] += 1
            if not devices:
                diagnostics["NO_DEVICE_EVENTS"] += 1
            if not steps:
                diagnostics["NO_STEP_ANNOTATION"] += 1
            summary = {"host_events": host_count, "host_threads": len(threads),
                "host_wall_union_ms": coverages["host"][2][-1] / 1e6 if host_count else None,
                "host_thread_union_sum_ms": sum(row["wall_union_ms"] for row in threads) if host_count else None,
                "host_inclusive_sum_ms": sum(row["inclusive_sum_ms"] for row in threads) if host_count else None,
                "host_self_sum_ms": None if self_unavailable or not host_count else sum(row["self_sum_ms"] for row in threads),
                "self_unavailable_events": self_unavailable, "device_events": link_total,
                "cpu_running_ms": None, "cpu_utilization_percent": None}
            coverage_status = {"host_status": "PARTIAL" if self_unavailable else "VISIBLE_EVENTS_ONLY" if host_count else "UNAVAILABLE",
                "device_status": "RECORDED_TASKS_ONLY" if devices else "UNAVAILABLE",
                "source_status": "COMPLETE" if link_total and source_counts.get("LINKED", 0) == link_total else "PARTIAL" if source_counts.get("LINKED", 0) + source_counts.get("CPU_ONLY", 0) + source_counts.get("API_ONLY", 0) else "UNAVAILABLE",
                "source_status_counts": source_counts}
            return {"schema_version": 1, "tool_version": HOST_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": {"path": str(path), **scan},
                "context": {"model": model, "hardware": hardware, "main_only": True if main_only else None, "architecture_status": "USER_DECLARED_UNVERIFIED"},
                "scope": {"start_ns": start, "end_ns": end, "wall_ms": (end - start) / 1e6,
                    "mode": "explicit" if start_us is not None else "capture",
                    "host_pids": sorted({row["pid"] for row in threads}), "device_pids": [row["pid"] for row in devices], "time_unit": time_unit},
                "summary": summary,
                "inventory": [{"pid": pid, "name": processes.get(pid, ""), "kind": ",".join(sorted(kinds))} for pid, kinds in sorted(process_kinds.items())],
                "threads": threads, "hotspots": hotspots,
                "apis": [row for row in hotspots if row["kind"] not in CPU_KINDS],
                "steps": steps, "devices": devices, "gaps": gaps, "links": links,
                "time_ledgers": time_ledgers,
                "coverage": coverage_status,
                "detail_limits": {"gaps": DETAIL_LIMIT, "links": DETAIL_LIMIT, "gap_rows_total": gap_total, "link_rows_total": link_total,
                    "gaps_truncated": gap_total > DETAIL_LIMIT, "links_truncated": link_total > DETAIL_LIMIT},
                "diagnostics": [{"code": code, "count": count, "message": _DIAGNOSTICS.get(code, code)} for code, count in sorted(diagnostics.items()) if count],
                "semantics": SEMANTICS}


_DIAGNOSTICS = {
    "SUB_NS_QUANTIZATION": "合法亚纳秒事件已按最近整数纳秒保留；每个端点量化误差至多 0.5 ns，单次时长误差至多 1 ns，未作为非法事件丢弃。",
    "INVALID_EVENT": "跳过缺失 pid/tid、非有限或非法时间的事件。",
    "UNMATCHED_BEGIN": "B 事件缺少 E，未伪造结束时间。", "UNMATCHED_END": "E 事件缺少 B，未伪造开始时间。",
    "MISMATCHED_BEGIN_END": "B/E 名称或类别不匹配，未纳入调用统计。",
    "UNSUPPORTED_PHASE": "存在非时长/flow/metadata 事件，未作为调用耗时。",
    "AMBIGUOUS_SELF": "同轨道调用相交或同边界重复，相关 self 不可用。",
    "UNCLASSIFIED_EVENT": "事件缺少可靠 Host/设备分类证据，未计入耗时；可显式指定 pid。",
    "NO_HOST_EVENTS": "所选窗口无可见 Host 调用，不能从设备数据推算 CPU 耗时。",
    "NO_DEVICE_EVENTS": "所选窗口无设备时长事件，设备覆盖与来源不可用。",
    "NO_STEP_ANNOTATION": "未见 ProfilerStep/Iteration 标注，不推断请求或生成轮次。",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ascend 通用 Host 实测分析（无外推；无需草稿或 HC 标记）")
    parser.add_argument("trace", type=Path, help="同次采样的 trace_view.json / Chrome trace JSON")
    parser.add_argument("--out", type=Path, default=Path("host_report"))
    parser.add_argument("--host-pid", action="append", help="限定 Host 进程轨道，可重复；未知分类时也作为覆盖设置")
    parser.add_argument("--device-pid", action="append", help="限定设备进程轨道，可重复")
    parser.add_argument("--start-us", help="分析窗口开始，trace 时钟的绝对微秒值")
    parser.add_argument("--end-us", help="分析窗口结束，须与 --start-us 一起使用")
    parser.add_argument("--time-unit", choices=("us", "ns", "ms"), default="us", help="输入 ts/dur 单位，Chrome trace 默认 us；不取 displayTimeUnit")
    parser.add_argument("--model", default="", help="用户声明的模型名，不作为结构证据")
    parser.add_argument("--hardware", default="", help="用户声明的芯片/机器型号")
    parser.add_argument("--main-only", action="store_true", help="记录用户声明：没有草稿模型")
    parser.add_argument("--excel", action="store_true", help="另导出带原生条形图的 XLSX")
    args = parser.parse_args(argv)
    try:
        report = analyze_trace(args.trace, start_us=args.start_us, end_us=args.end_us,
            host_pids=args.host_pid, device_pids=args.device_pid, time_unit=args.time_unit,
            model=args.model, hardware=args.hardware, main_only=args.main_only,
            progress=lambda value: print(value, file=sys.stderr, flush=True))
        from host_profile_report import write_reports
        artifacts = write_reports(report, args.out, excel=args.excel)
    except (ValueError, OSError, sqlite3.Error) as error:
        print("Host 分析失败：" + str(error), file=sys.stderr)
        return 2
    print("Host 实测报告：" + artifacts["html"])
    if args.excel:
        print("Excel：" + artifacts["xlsx"])
    print("所选窗口 %.3f ms；Host %s，设备 %s；CPU 实际忙时/利用率未采集。" % (
        report["scope"]["wall_ms"], report["coverage"]["host_status"], report["coverage"]["device_status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
