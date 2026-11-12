"""Conservative business-stage accounting inside one Attention-prepare window.

This module uses only the standard library and never estimates a target model.
It accepts native timed records (objects or dictionaries), clips them to the
supplied parent window, and attributes a record at most once.  Generic MatMul,
quantization and cache-scatter names are deliberately insufficient evidence.
No per-operator contribution ledger is produced.
"""

from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
import re


_DSA = "vllm_ascend/attention/dsa_v1.py"
_MODEL = "vllm_ascend/models/deepseek_v4/model.py"
_INDEXER = "vllm_ascend/models/deepseek_v4/indexer.py"
_DEVICE = "vllm_ascend/device/device_op.py"
_COMPRESSOR = "vllm_ascend/models/deepseek_v4/compressor.py"

# These are business groups, not an exclusive timeline partition. Their busy
# unions can overlap on different streams, so summing them is not wall time.
_GROUPS = (
    ("hc_pre", "Attention HCPre", (_MODEL + ":DeepseekV4DecoderLayer.hc_pre / forward",)),
    ("input_norm", "输入归一化", (_MODEL + ":DeepseekV4DecoderLayer.forward：input_layernorm",)),
    ("input_allgather", "输入AllGather（可选）", (_MODEL + ":DeepseekV4DecoderLayer.forward：sp_all_gather",)),
    ("input_quant", "共享输入量化（需明确证据）", (_DSA + ":_mla_prolog_single_stream / _mla_prolog_multistream",)),
    ("q_low_rank", "Q低秩投影", (_DSA + ":_mla_prolog_single_stream / _mla_prolog_multistream：wq_a",)),
    ("q_expand", "Q中间归一化、量化与展开", (_DSA + ":_mla_prolog_single_stream / _mla_prolog_multistream：q_norm / wq_b",)),
    ("q_final", "Q最终归一化与RoPE", (_DEVICE + ":DeviceOperator.apply_dsa_q_rms", _DSA + ":Q的inplace_partial_rotary_mul调用")),
    ("window_kv_projection", "窗口KV投影", (_DSA + ":_mla_prolog_single_stream / _mla_prolog_multistream：wkv",)),
    ("window_kv_norm", "窗口KV归一化", (_DSA + ":_mla_prolog_single_stream / _mla_prolog_multistream：kv_norm",)),
    ("window_kv_rope", "窗口KV RoPE", (_DSA + ":KV的inplace_partial_rotary_mul调用",)),
    ("window_kv_cache", "窗口KV写入缓存", (_DSA + ":_mla_prolog_single_stream / _mla_prolog_multistream中的缓存写入",)),
    ("attention_compress", "Attention压缩KV与缓存写入", (_DSA + ":_maybe_update_compressed_caches_and_select_topk", _COMPRESSOR + ":Compressor.forward")),
    ("indexer_query", "Indexer Query投影、RoPE与变换", (_INDEXER + ":DeepseekV4Indexer.forward及其prolog",)),
    ("indexer_key_cache", "Indexer Key与缓存准备（含融合Query量化）", (_INDEXER + ":AscendIndexerOps.quantize_key_and_update_cache", _DEVICE + ":DeviceOperator.indexer_quant_scatter")),
    ("indexer_select", "Indexer权重、Query量化与评分TopK", (_INDEXER + ":AscendIndexerOps.select_topk / DeepseekV4Indexer.forward",)),
    ("unassigned", "未确认归属的准备段记录", ()),
)

_NOTE = ("各组按相交记录裁剪到父窗口；kernel_sum是记录时长累计，busy_union是组内区间并集，"
         "span是组内最早开始到最晚结束。不同组可并行，三者均不能直接相加作为父耗时。"
         "recorded_busy仅表示导出记录覆盖，并非硬件忙碌率；uncovered仅为无这些记录覆盖的时间，"
         "不能直接称为CPU耗时、等待或设备空闲。融合kernel整体归组，不虚拆内部耗时。"
         "未见记录的适用组标UNKNOWN，不把缺失观测解释为确定零耗时。")


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _ns(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("nanosecond bounds must be exact integers")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
        raise ValueError("nanosecond bounds must be exact integers")
    return int(value)


def _union(intervals):
    total = 0
    right = None
    for a, b in sorted(intervals):
        if right is None or a >= right:
            total += b - a
            right = b
        elif b > right:
            total += b - right
            right = b
    return total


def _text(task):
    fields = ("scope", "function", "function_name", "module", "module_name",
              "python_scope", "call_stack", "name", "op_type")
    return " ".join(str(_get(task, key, "") or "") for key in fields).lower()


def _has(text, *tokens):
    # A dotted Python scope, a slash path and a profiler underscore prefix are
    # treated alike; wq_a must not match wq_ab or another unrelated identifier.
    return any(re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", text)
               for token in tokens)


def _shapes(value):
    if isinstance(value, (list, tuple)):
        if all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            return [tuple(value)]
        return [tuple(x) if isinstance(x, (list, tuple)) else () for x in value]
    return _string_shapes(str(value or ""))


@lru_cache(maxsize=4096)
def _string_shapes(value):
    result = []
    for field in value.strip('"').split(";"):
        field = field.strip().strip("[]() ")
        if not field or field.upper() in ("N/A", "NULL", "NONE"):
            result.append(())
        elif re.fullmatch(r"\s*-?\d+(?:\s*,\s*-?\d+)*\s*", field):
            result.append(tuple(int(x.strip()) for x in field.split(",")))
        else:
            result.append(())
    return tuple(result)


def _dimension(model, *keys):
    maps = [model] if isinstance(model, Mapping) else []
    if maps:
        maps += [model.get(name) for name in ("attention_prepare", "hf_config", "attention_config")
                 if isinstance(model.get(name), Mapping)]
    for source in maps:
        for key in keys:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def _dimensions(model):
    head = _dimension(model, "head_dim", "qk_head_dim")
    if head is None:
        nope = _dimension(model, "qk_nope_head_dim")
        rope = _dimension(model, "qk_rope_head_dim")
        if nope and rope:
            head = nope + rope
    return {"hidden": _dimension(model, "hidden_size"),
            "q_rank": _dimension(model, "q_lora_rank"),
            "head": head,
            "index_head": _dimension(model, "index_head_dim"),
            "local_heads": _dimension(model, "local_num_attention_heads", "n_local_heads"),
            "local_index_heads": _dimension(model, "local_index_heads", "index_n_local_heads")}


def _fused_compressor(text):
    return "compressor" in text and not any(x in text for x in ("metadata", "statecache", "state_cache"))


@lru_cache(maxsize=8192)
def _explicit_candidates(text):
    """Only scope-like names that identify business semantics, not stream IDs."""
    found = set()
    if re.search(r"(?:^|[^a-z0-9])hc_?pre(?:_?v\d+)?(?:$|[^a-z0-9])", text):
        found.add("hc_pre")
    if _has(text, "input_layernorm", "input_layer_norm", "attention_input_norm"):
        found.add("input_norm")
    if _has(text, "input_allgather", "input_all_gather", "attention_input_allgather"):
        found.add("input_allgather")
    if _has(text, "shared_hs_quant", "share_hs_quant", "hidden_states_quant", "shared_input_quant"):
        found.add("input_quant")

    # Indexer has its own wq_b and compressor. Enclosing scopes must be checked
    # before the same leaf names in the main Attention prolog.
    indexer_scope = _has(text, "indexer", "deepseekv4indexer")
    compressor_scope = _fused_compressor(text)
    if indexer_scope:
        if _has(text, "compressor", "k_cache", "key_cache", "quantize_key_and_update_cache",
                "update_scale_cache", "indexer_key"):
            found.add("indexer_key_cache")
        elif _has(text, "weights_proj", "select_topk", "quantize_query", "indexer_select"):
            found.add("indexer_select")
        elif _has(text, "wq_b", "cv_wq_b", "q_rope", "q_hadamard", "indexer_query"):
            found.add("indexer_query")
    elif not compressor_scope:
        if _has(text, "wq_a", "cv_wq_a", "q_a_matmul", "q_low_rank"):
            found.add("q_low_rank")
        if _has(text, "q_norm_without_weight", "apply_dsa_q_rms", "q_rms", "q_rope", "query_rope"):
            found.add("q_final")
        elif _has(text, "q_norm", "wq_b", "cv_wq_b", "q_b_quant", "q_b_matmul", "q_expand"):
            found.add("q_expand")
        if _has(text, "wkv", "cv_wkv", "kv_matmul", "window_kv_projection"):
            found.add("window_kv_projection")
        if _has(text, "kv_norm", "window_kv_norm"):
            found.add("window_kv_norm")
        if _has(text, "kv_rope", "window_kv_rope"):
            found.add("window_kv_rope")
        if _has(text, "window_kv_cache", "swa_kv_cache", "swa_cache_write"):
            found.add("window_kv_cache")

    if _has(text, "compute_attention_compressed_kv", "scatter_attention_compressed_kv",
            "attention_compressor", "attention_compress"):
        found.add("attention_compress")
    compact = re.sub(r"[^a-z0-9]", "", text)
    if any(token in compact for token in ("quantlightningindexer", "lightningindexer", "indexertopk", "sparseindexer")):
        found.add("indexer_select")
    if any(token in compact for token in ("indexerquantscatter", "indexerscatterscalepart3")):
        found.add("indexer_key_cache")
    return frozenset(found)


def _shape_candidates(task, dims):
    text = _text(task)
    inputs, outputs = _shapes(_get(task, "input_shapes")), _shapes(_get(task, "output_shapes"))
    x = inputs[0] if inputs else ()
    y = outputs[0] if outputs else ()
    result = set()
    hidden, rank, head = dims["hidden"], dims["q_rank"], dims["head"]
    if ("matmul" in text or "gemm" in text) and len(inputs) > 1 and x and y:
        weight = inputs[1]
        candidates = (("q_low_rank", hidden, rank),
                      ("window_kv_projection", hidden, head),
                      ("q_expand", rank, dims["local_heads"] * head if dims["local_heads"] and head else None),
                      ("indexer_query", rank, dims["local_index_heads"] * dims["index_head"]
                       if dims["local_index_heads"] and dims["index_head"] else None))
        for key, src, dst in candidates:
            if src and dst and x[-1] == src and y[-1] == dst and weight in ((src, dst), (dst, src)):
                result.add(key)
    if "rmsnorm" in re.sub(r"[^a-z0-9]", "", text) and len(inputs) > 1 and x:
        weight = inputs[1]
        # A full exported input + weight shape is required. An unweighted
        # per-head Q norm must not be mistaken for the weighted KV norm.
        candidates = (("input_norm", hidden), ("q_expand", rank), ("window_kv_norm", head))
        for key, dim in candidates:
            if dim and weight == (dim,) and x[-1] == dim and len(x) == 2:
                result.add(key)
    if "rotary" in text and len(x) >= 3:
        # Both branches invoke the same RoPE operation. Only an explicit local
        # head count >1 separates the Query head axis from singleton KV.
        heads = dims["local_heads"]
        if head and heads and heads > 1 and x[-1] == head:
            if x[-2] == heads:
                result.add("q_final")
            if x[-2] == 1:
                result.add("window_kv_rope")
        ih, index_heads = dims["index_head"], dims["local_index_heads"]
        if ih and index_heads and x[-1] == ih and x[-2] == index_heads:
            result.add("indexer_query")
    if _fused_compressor(text):
        # Compressor.forward passes its FP32 norm weight as input 5. This
        # distinguishes the Attention and Indexer compressor only when the
        # configured head dimensions differ; output shape alone is weaker.
        norm_shape = inputs[5] if len(inputs) > 5 else ()
        for key, dim in (("attention_compress", head), ("indexer_key_cache", dims["index_head"])):
            if dim and norm_shape == (dim,):
                result.add(key)
    return result


def _not_applicable(key, attention):
    return (attention == "SWA" and key in ("attention_compress", "indexer_query", "indexer_key_cache", "indexer_select")
            or attention == "HCA" and key.startswith("indexer_"))


def analyze_attention_prepare(tasks, start_ns, end_ns, *, model=None, attention=None, scope=None):
    """Account for visible business work inside ``[start_ns, end_ns)``.

    ``tasks`` may contain ``ATask`` instances or equivalent dictionaries.
    Optional explicit function/module scope fields improve attribution; normal
    kernel CSVs often omit them. Optional model dimensions are never inferred
    from fixed product defaults. The function does not mutate any argument.

    An unobserved applicable group has ``status=UNKNOWN`` and null timing
    values. Structural exclusions have ``status=NOT_APPLICABLE`` and zeros.
    Unexpected observed records contradicting such an exclusion remain in
    ``unassigned`` with an explanation rather than silently disappearing.
    """
    left, right = _ns(start_ns), _ns(end_ns)
    if right <= left:
        raise ValueError("Attention-prepare parent window must have positive width")
    attention = str(attention or "").upper()
    dims = _dimensions(model)
    records, invalid_count = [], 0
    for task in tasks:
        try:
            a, b = max(left, _ns(_get(task, "start_ns"))), min(right, _ns(_get(task, "end_ns")))
        except (TypeError, ValueError):
            invalid_count += 1
            continue
        if b > a:
            records.append((task, a, b, _text(task)))
    fused_count = sum(_fused_compressor(text) for _, _, _, text in records)
    intervals, reasons = defaultdict(list), defaultdict(set)
    for task, a, b, text in records:
        candidates = _explicit_candidates(text)
        proof = "明确函数或scope标签"
        if not candidates:
            candidates = _shape_candidates(task, dims)
            proof = "实际输入/权重/输出shape与显式模型维度匹配"
        if not candidates and attention == "HCA" and fused_count == 1 and _fused_compressor(text):
            candidates = {"attention_compress"}
            proof = "HCA准备窗口唯一Compressor融合记录；不拆内部流水"
        if len(candidates) == 1:
            key = next(iter(candidates))
            if _not_applicable(key, attention):
                key, proof = "unassigned", "记录语义与声明层类型不一致，需要核对窗口/层类型"
        else:
            key = "unassigned"
            proof = ("多个业务归属具有相同证据：" + ", ".join(sorted(candidates)) if candidates
                     else "名称/shape不足以唯一确认业务归属；不按stream编号或MatMul顺序猜测")
        intervals[key].append((a, b))
        reasons[key].add(proof)

    rows = []
    for key, label, functions in _GROUPS:
        observed = intervals[key]
        row = {"key": key, "label": label, "kernel_count": len(observed),
               "function_entries": list(functions)}
        if observed:
            row.update(status="MATCHED" if key != "unassigned" else "UNASSIGNED",
                       reason="；".join(sorted(reasons[key])),
                       kernel_sum_ns=sum(b - a for a, b in observed),
                       busy_union_ns=_union(observed), span_ns=max(b for _, b in observed) - min(a for a, _ in observed))
        elif key == "unassigned":
            # Unlike an absent business branch, this bookkeeping bucket is
            # exhaustive over the valid intersecting records we just read.
            row.update(status="MATCHED", reason="已读取的有效相交记录中没有未归属项",
                       kernel_sum_ns=0, busy_union_ns=0, span_ns=0)
        elif _not_applicable(key, attention):
            row.update(status="NOT_APPLICABLE", reason=attention + "结构不包含该分支",
                       kernel_sum_ns=0, busy_union_ns=0, span_ns=0)
        else:
            row.update(status="UNKNOWN", reason="父窗口未取得可唯一归属的记录；不代表实际零耗时",
                       kernel_sum_ns=None, busy_union_ns=None, span_ns=None)
        rows.append(row)

    all_intervals = [(a, b) for _, a, b, _ in records]
    assigned = [interval for key, spans in intervals.items() if key != "unassigned" for interval in spans]
    unassigned = intervals["unassigned"]
    recorded_busy = _union(all_intervals)
    assigned_sum = sum(b - a for a, b in assigned)
    unassigned_sum = sum(b - a for a, b in unassigned)
    count = len(records)
    status = "UNAVAILABLE" if not count else "PARTIAL" if unassigned or invalid_count else "MATCHED"
    reason = ("父窗口没有相交的有效定时记录" if not count else
              "存在不能唯一归属的记录；已确认部分仍保留" if unassigned else "已确认记录均已归属；未观测业务组仍标UNKNOWN")
    if invalid_count:
        reason += "；跳过" + str(invalid_count) + "条无有效整数时间戳的记录"
    return {"status": status, "reason": reason,
            "scope": dict(scope) if isinstance(scope, Mapping) else {}, "attention": attention or None,
            "start_ns": str(left), "end_ns": str(right), "parent_wall_ns": right - left,
            "kernel_count": count, "invalid_record_count": invalid_count,
            "kernel_sum_ns": assigned_sum + unassigned_sum,
            "recorded_busy_ns": recorded_busy, "uncovered_ns": right - left - recorded_busy,
            "assigned_kernel_count": len(assigned), "unassigned_kernel_count": len(unassigned),
            "assigned_kernel_sum_ns": assigned_sum, "unassigned_kernel_sum_ns": unassigned_sum,
            "assigned_busy_ns": _union(assigned), "unassigned_busy_ns": _union(unassigned),
            "cross_group_overlap_ns": sum(_union(x) for x in intervals.values()) - recorded_busy,
            "rows": rows, "note": _NOTE,
            "function_entries": sorted({entry for _, _, refs in _GROUPS for entry in refs})}
