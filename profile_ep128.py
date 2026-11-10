#!/usr/bin/env python3
"""Offline HC layer means, two communication estimates and per-card throughput.

Standard library only. Measure eight main and three draft layers from native
kernel CSVs, exclude main0 from reference means, then select the maximum rank
mean cycle. The renderer consumes these results without estimating again.
"""
import argparse
from array import array
import bisect
import collections
import codecs
from contextlib import contextmanager, nullcontext
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import sqlite3
import sys
import tempfile
import time

SCHEMA_VERSION = 2
PROFILE_VERSION = "20261110.2.parallel_analysis"
METHOD1_EXTRA_US = 14.0
ESTIMATE_XLSX = "10T模型推理估算.xlsx"

A_PRE = re.compile(r"(?:^|_)HcPre(?:V\d+)?(?:_|$)", re.I)
A_POST = re.compile(r"(?:^|_)HcPost(?:V\d+)?(?:_|$)", re.I)
A_DISPATCH = re.compile(r"MoeDistributeDispatch", re.I)
A_NATIVE_DISPATCH = re.compile(r"(?:aclnn)?MoeDistributeDispatch(?:V([123]))?", re.I)
A_COMBINE = re.compile(r"MoeDistributeCombine", re.I)
A_INDEXER = re.compile(r"LightningIndexer|IndexerTopK|SparseIndexer", re.I)



@dataclass(slots=True)
class ATask:
    row_index: int
    start_ns: int
    end_ns: int
    wait_ns: int
    name: str
    op_type: str
    device_id: str | None
    model_id: str | None
    stream_id: str | None
    task_id: str | None
    context_id: str | None
    batch_id: str | None
    subtask_id: str | None
    input_shapes: str
    output_shapes: str
    input_dtypes: str
    output_dtypes: str
    task_type: str | None
    op_state: str | None
    row_indices: list = field(default_factory=list)
    wait_available: bool = True
    pmu: dict = field(default_factory=dict)

    @property
    def label(self):
        return self.op_type + " " + self.name



def a_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def a_id(value):
    if value is None or str(value).strip() in ("", "N/A", "None", "null"):
        return None
    value = str(value).strip()
    return str(int(value)) if re.fullmatch(r"[+-]?\d+", value) else value



def a_alias(row, names, label, normalize=None):
    entries = []
    for name in names:
        value = row.get(name)
        if value is None or str(value).strip() in ("", "N/A", "None", "null"):
            continue
        entries.append((name, normalize(value) if normalize else str(value).strip()))
    if entries and any(value != entries[0][1] for _, value in entries):
        raise ValueError("conflicting CSV aliases for " + label + ": " + str(entries))
    return entries[0][1] if entries else None



def a_ns(value, label="timestamp", missing=False):
    # Trace JSON already supplies exact Decimal timestamps. Retain the number
    # instead of formatting and reparsing it for every timed event. Other
    # types keep the original conversion and Decimal-context rounding rules.
    decimal_value = type(value) is Decimal
    if not decimal_value and (value is None or str(value).strip() in ("", "N/A", "None")):
        if missing:
            return None
        raise ValueError(label + " is missing")
    try:
        number = value if decimal_value else Decimal(str(value).strip())
        if not number.is_finite():
            raise ValueError(label + " must be finite")
        return int((number * 1000).to_integral_value())
    except (ArithmeticError, ValueError) as error:
        raise ValueError(label + " must be finite microseconds: " + str(value)) from error



def a_exact_ns(value, label):
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(label + " must be an exact integer nanosecond string or integer")


_A_PMU_FIELDS = {
    "aicore_time_us": "us", "aiv_time_us": "us",
    "aic_total_cycles": "cycles", "aiv_total_cycles": "cycles",
    "aic_icache_miss_rate": "ratio", "aiv_icache_miss_rate": "ratio",
    "cube_utilization": "ratio", "block_num": "count", "mix_block_num": "count",
}
for _pmu_core, _pmu_pipes in (("aic", ("mac", "vec", "scalar", "mte1", "mte2", "mte3", "fixpipe")),
                              ("aiv", ("mac", "vec", "scalar", "mte1", "mte2", "mte3", "fixpipe"))):
    for _pmu_pipe in _pmu_pipes:
        _A_PMU_FIELDS[_pmu_core + "_" + _pmu_pipe + "_ratio"] = "ratio"
        _A_PMU_FIELDS[_pmu_core + "_" + _pmu_pipe + "_time_us"] = "us"
del _pmu_core, _pmu_pipes, _pmu_pipe


def _a_pmu_header(header):
    """Resolve optional hardware counters without changing timed-row parsing."""
    raw = str(header).lower().replace("μ", "u").replace("µ", "u").replace("％", "%")
    normalized = re.sub(r"[^a-z0-9]", "", raw)
    if normalized in ("aictime", "aictimeus", "aictimemicroseconds"):
        return "aicore_time_us", 1.
    for field, unit in _A_PMU_FIELDS.items():
        base = re.sub(r"[^a-z0-9]", "", field[:-3] if unit == "us" else field)
        if unit == "us":
            if normalized in (base, base + "us", base + "microseconds"):
                return field, 1.
            if normalized == base + "ns":
                return field, .001
            if normalized == base + "ms":
                return field, 1000.
        elif normalized in (base, base + "percent", base + "percentage", base + "pct"):
            return field, .01 if unit == "ratio" and ("%" in raw or normalized != base or field == "cube_utilization") else 1.
    return None


class _ANativePmu(dict):
    """Parser-validated native counters, with no extra serialized fields.

    A capture is read completely (including duplicate conflicts) before its
    tasks enter the immutable indexed analysis. Reuse those validated values
    instead of copying/revalidating every counter in every clipped stage.
    Manually constructed task dictionaries still take the validation path.
    """
    __slots__ = ()


@lru_cache(maxsize=8192)
def _a_pmu_value(raw, unit, factor):
    """Pure bounded cache for recurring zero/missing/ratio CSV strings."""
    text = raw.strip()
    if text.lower() in ("", "n/a", "none", "null", "--", "-"):
        return None, False
    text = text.replace("％", "%")
    percent = text.endswith("%")
    try:
        if percent and unit != "ratio":
            raise ValueError("percent suffix is only valid for optional PMU ratios")
        value = float(text[:-1].strip() if percent else text)
        value *= .01 if percent else factor
        if (not math.isfinite(value) or value < 0 or unit == "ratio" and value > 1
                or unit in ("count", "cycles") and not value.is_integer()):
            raise ValueError("invalid optional PMU value")
        return value, False
    except (ValueError, TypeError, OverflowError):
        return None, True


def _a_read_pmu(row, columns):
    values, invalid = {}, set()
    for header, field, factor in columns:
        raw = row.get(header)
        if raw is None:
            continue
        value, invalid_value = _a_pmu_value(raw if isinstance(raw, str) else str(raw),
                                           _A_PMU_FIELDS[field], factor)
        if invalid_value:
            invalid.add(field)
        elif value is not None:
            if field in values and not math.isclose(values[field], value, rel_tol=1e-9, abs_tol=1e-12):
                invalid.add(field)
            else:
                values[field] = value
    for field in invalid:
        values.pop(field, None)
    if not values and not invalid:
        return {}
    return _ANativePmu(metrics=values, invalid_fields=sorted(invalid),
                      core_type=str(row.get("Accelerator Core") or row.get("Task Type") or "").strip())


def _a_merge_pmu(left, right):
    """Duplicate native events remain one event, even with optional counters."""
    metrics = dict(left.get("metrics", {}))
    invalid = set(left.get("invalid_fields", ())) | set(right.get("invalid_fields", ()))
    for field, value in right.get("metrics", {}).items():
        if field in metrics and not math.isclose(metrics[field], value, rel_tol=1e-9, abs_tol=1e-12):
            invalid.add(field)
        else:
            metrics[field] = value
    for field in invalid:
        metrics.pop(field, None)
    constructor = _ANativePmu if isinstance(left, _ANativePmu) and isinstance(right, _ANativePmu) else dict
    return constructor(metrics=metrics, invalid_fields=sorted(invalid),
                       core_type=left.get("core_type") or right.get("core_type") or "")



def read_capture(path, rank):
    """Keep original row references and metadata; deduplicate identical events."""
    path = Path(path).resolve()
    before = path.stat()
    tasks, seen, devices, duplicates, warnings = [], {}, set(), 0, []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        normalized_headers = [str(name).strip() for name in reader.fieldnames or []]
        if len(set(normalized_headers)) != len(normalized_headers):
            raise ValueError(str(path) + ": duplicate CSV header")
        reader.fieldnames = normalized_headers
        headers = set(normalized_headers)
        pmu_columns = [(header, *resolved) for header in normalized_headers
                       if (resolved := _a_pmu_header(header)) is not None]
        for alternatives in (("Name", "Op Name"), ("Start Time(us)", "Start Time (us)"),
                             ("Duration(us)", "Duration (us)")):
            if not headers.intersection(alternatives):
                raise ValueError(str(path) + ": missing CSV column " + alternatives[0])
        for index, row in enumerate(reader, 1):
            if None in row:
                raise ValueError(str(path) + f": extra fields in data record {index}")
            stamp = a_alias(row, ("Start Time(us)", "Start Time (us)"), "start",
                            lambda value: a_ns(value, "start"))
            if stamp is None:
                warnings.append(f"data record {index}: no device start; not a timed task")
                continue
            duration = a_alias(row, ("Duration(us)", "Duration (us)"), "duration",
                               lambda value: a_ns(value, "duration"))
            if duration is None or duration < 0:
                raise ValueError(str(path) + f": invalid duration in data record {index}")
            name = a_alias(row, ("Name", "Op Name"), "name")
            if not name:
                raise ValueError(str(path) + f": missing name in data record {index}")
            metadata = {}
            for key, aliases in {
                "device_id": ("Device_id", "Device ID", "Device Id", "device_id"),
                "model_id": ("Model ID", "Model Id", "model_id"),
                "stream_id": ("Stream ID", "Stream Id", "stream_id"),
                "task_id": ("Task ID", "Task Id", "task_id"),
                "context_id": ("Context ID", "Context Id", "context_id"),
                "batch_id": ("Batch ID", "Batch Id", "batch_id"),
                "subtask_id": ("Subtask ID", "Subtask Id", "SubTask ID", "subtask_id"),
            }.items():
                metadata[key] = a_alias(row, aliases, key, a_id)
            if metadata["device_id"] is not None:
                devices.add(metadata["device_id"])
            wait = a_alias(row, ("Wait Time(us)", "Wait Time (us)"), "wait",
                           lambda value: a_ns(value, "wait"))
            if wait is not None and wait < 0:
                raise ValueError("negative Wait Time in data record " + str(index))
            values = dict(start_ns=stamp, end_ns=stamp + duration, wait_ns=wait or 0, wait_available=wait is not None,
                          name=name, op_type=str(row.get("Type") or "").strip(), **metadata,
                          input_shapes=str(row.get("Input Shapes") or ""),
                          output_shapes=str(row.get("Output Shapes") or ""),
                          input_dtypes=str(row.get("Input Data Types") or ""),
                          output_dtypes=str(row.get("Output Data Types") or ""),
                          task_type=a_alias(row, ("Task Type", "Accelerator Core"), "task_type")
                          if not (row.get("Task Type") and row.get("Accelerator Core"))
                          else str(row["Task Type"]).strip(),
                          op_state=a_alias(row, ("OP State", "Op State"), "op_state"))
            key = tuple(values.items())
            pmu = _a_read_pmu(row, pmu_columns)
            if key in seen:
                duplicates += 1
                seen[key].row_indices.append(index)
                seen[key].pmu = _a_merge_pmu(seen[key].pmu, pmu)
            else:
                task = ATask(row_index=index, row_indices=[index], pmu=pmu, **values)
                seen[key] = task
                tasks.append(task)
    if len(devices) > 1:
        raise ValueError(str(path) + ": multiple device IDs " + str(sorted(devices)))
    if not tasks or not any(task.end_ns > task.start_ns for task in tasks):
        raise ValueError(str(path) + ": no timed device tasks")
    digest = a_sha(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(str(path) + ": CSV changed while being read; wait for export to complete")
    return {"rank": str(rank), "path": str(path), "sha256": digest,
            "tasks": sorted(tasks, key=lambda task: (task.start_ns, task.end_ns, task.row_index)),
            "device_id": next(iter(devices)) if devices else None, "duplicate_rows": duplicates,
            "parse_warnings": warnings}



def a_union(intervals):
    output = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if output and left <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], right))
        else:
            output.append((left, right))
    return output



def a_stats(tasks, start, end):
    clipped = [(max(start, task.start_ns), min(end, task.end_ns))
               for task in tasks if task.start_ns < end and task.end_ns > start]
    return {"wall_ms": (max((b for _, b in clipped), default=start) -
                        min((a for a, _ in clipped), default=start)) / 1e6,
            "busy_union_ms": sum(b - a for a, b in a_union(clipped)) / 1e6,
            "kernel_sum_ms": sum(b - a for a, b in clipped) / 1e6,
            "calls": len(clipped)}



def a_schedule(model, case):
    main = model.get("num_hidden_layers")
    if (type(main) is not int or main < 1 or main != 8
            or case.get("draft_layers", 3) != 3 or case.get("draft_passes", 1) != 1):
        raise ValueError("V2 source contract requires exactly 8 main layers and 3 draft layers in one pass")
    ratios = model.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) < main:
        raise ValueError("source compress_ratios must cover configured main layers")
    ratios = ratios[:main] + (ratios[main:main+3] if len(ratios) >= main+3 else [0, 0, 0])
    if any(type(value) is not int or value not in (0, 4, 128) for value in ratios):
        raise ValueError("unsupported source attention ratio")
    count = model.get("num_hash_layers")
    if type(count) is not int or not 0 <= count <= main:
        raise ValueError("num_hash_layers must be an integer in [0,main depth]")
    return [("main" if i < main else "draft", i if i < main else i - main,
             {0: "SWA", 4: "CSA", 128: "HCA"}[value], "hash" if i < count else "normal")
            for i, value in enumerate(ratios)]



def a_hc_cycles(tasks, model, case):
    schedule = a_schedule(model, case)
    starts = [task.start_ns for task in tasks]
    pre = [task for task in tasks if A_PRE.search(task.name) or A_PRE.search(task.op_type)]
    dispatch = [task for task in tasks if A_DISPATCH.search(task.label)]
    ds = [task.start_ns for task in dispatch]
    blocks = []
    for first, ffn, following in zip(pre, pre[1:], pre[2:]):
        if bisect.bisect_left(ds, ffn.start_ns) != bisect.bisect_left(ds, first.start_ns):
            continue
        if bisect.bisect_left(ds, following.start_ns) - bisect.bisect_left(ds, ffn.start_ns) != 1:
            continue
        selected = tasks[bisect.bisect_left(starts, first.start_ns):bisect.bisect_left(starts, following.start_ns)]
        combines = [task for task in selected if A_COMBINE.search(task.label)]
        attn_posts = [task for task in selected if task.start_ns < ffn.start_ns and
                      (A_POST.search(task.name) or A_POST.search(task.op_type))]
        ffn_posts = [task for task in selected if task.start_ns >= ffn.start_ns and
                     (A_POST.search(task.name) or A_POST.search(task.op_type))]
        if not combines or not attn_posts or not ffn_posts:
            continue
        if len(combines) != 1 or len(attn_posts) != 1 or len(ffn_posts) != 1:
            raise ValueError("ambiguous HC/communication markers; no complete-layer claim was made")
        routed = next(task for task in selected if ffn.start_ns <= task.start_ns < following.start_ns
                      and A_DISPATCH.search(task.label))
        chain = (first, attn_posts[0], ffn, routed, combines[0], ffn_posts[0], following)
        if any(left.end_ns > right.start_ns for left, right in zip(chain, chain[1:])):
            raise ValueError("invalid HC dependency order: overlapping or reversed source markers")
        blocks.append({"start_ns": first.start_ns, "end_ns": ffn_posts[0].end_ns,
                       "next_start_ns": following.start_ns,
                       "csa": any(task.start_ns < ffn.start_ns and A_INDEXER.search(task.label)
                                  and "metadata" not in task.label.lower() for task in selected)})
    size = len(schedule)
    phase = case.get("phase_offset")
    if phase is not None:
        if type(phase) is not int or not 0 <= phase < size:
            raise ValueError(f"phase_offset must be a verified integer slot in [0,{size})")
        if any(block["csa"] for block in blocks) and any(
                block["csa"] != (schedule[i % size][2] == "CSA") for i, block in enumerate(blocks[phase:])):
            raise ValueError("phase_offset contradicts observed CSA order")
    else:
        candidates = []
        for candidate in range(size):
            count = (len(blocks) - candidate) // size
            if count >= 2 and all(blocks[candidate + i]["csa"] == (schedule[i % size][2] == "CSA")
                                  for i in range(count * size)):
                candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError("cannot uniquely align source CSA period; require verified phase_offset, never guessed")
        phase = candidates[0]
    result = []
    for index in range(phase, len(blocks) - size, size):
        selected = blocks[index:index + size]
        if any(left["next_start_ns"] != right["start_ns"] for left, right in zip(selected, selected[1:])):
            continue
        if selected[-1]["next_start_ns"] != blocks[index + size]["start_ns"]:
            continue
        start, end = selected[0]["start_ns"], blocks[index + size]["start_ns"]
        inside = tasks[bisect.bisect_left(starts, start):bisect.bisect_left(starts, end)]
        if sum(bool(A_DISPATCH.search(task.label)) for task in inside) != size or sum(
                bool(A_COMBINE.search(task.label)) for task in inside) != size:
            raise ValueError(f"source HC cycle does not contain exactly {size} Dispatch and {size} Combine markers")
        result.append({"round_id": "hc-" + str(len(result) + 1), "start": start, "end": end,
                       "hc_blocks": list(zip(selected, schedule))})
    return result



def a_dims(value):
    return [[int(number) for number in re.findall(r"-?\d+", part)] for part in value.split(";")]



def a_dispatch_inputs(task):
    """Resolve supported device CSV layouts without searching arbitrary slots.

    The observed V3 graph export may prepend a one-dimensional INT32 input
    before x/expert_ids. That layout requires aligned dtype evidence; the
    Python API's argument positions alone do not describe this device row.
    """
    unknown = {'layout': 'unknown', 'x_shape': None, 'expert_ids_shape': None,
               'x_index': None, 'expert_ids_index': None, 'reason': None}
    schemas = [match.group(1) or '1' for value in (task.op_type, task.name)
               if (match := A_NATIVE_DISPATCH.fullmatch(value.strip()))]
    if not schemas or len(set(schemas)) != 1:
        return {**unknown, 'reason': 'native Dispatch schema is missing or conflicting'}
    shapes = a_dims(task.input_shapes)
    raw_dtypes = task.input_dtypes.strip()
    dtypes = [re.sub(r'[^A-Z0-9]', '', value.upper())
              for value in raw_dtypes.split(';')] if raw_dtypes else []
    if dtypes and len(dtypes) != len(shapes):
        return {**unknown, 'reason': 'shape/dtype input slots are not aligned'}

    def matrix(shape):
        return len(shape) == 2 and all(value > 0 for value in shape)

    def int32(value):
        return value in ('INT32', 'DTINT32', 'TORCHINT32')

    def activation(value):
        return value in ('FLOAT', 'FLOAT32', 'DTFLOAT', 'DTFLOAT32', 'FP32',
                         'BF16', 'DTBF16', 'BFLOAT16', 'DTBFLOAT16', 'TORCHBFLOAT16',
                         'FLOAT16', 'DTFLOAT16', 'FP16', 'HALF', 'TORCHFLOAT16',
                         'INT8', 'DTINT8', 'TORCHINT8',
                         'FLOAT8E4M3FN', 'DTFLOAT8E4M3FN', 'FLOAT8E5M2', 'DTFLOAT8E5M2')

    def resolved(layout, x_index, ids_index):
        return {'layout': layout, 'x_shape': shapes[x_index],
                'expert_ids_shape': shapes[ids_index], 'x_index': x_index,
                'expert_ids_index': ids_index, 'reason': None}

    if (len(shapes) >= 2 and matrix(shapes[0]) and matrix(shapes[1])
            and (not dtypes or activation(dtypes[0]) and int32(dtypes[1]))):
        return resolved('standard', 0, 1)
    if (schemas[0] == '3' and len(shapes) >= 3 and dtypes
            and len(shapes[0]) == 1 and shapes[0][0] > 0 and int32(dtypes[0])
            and matrix(shapes[1]) and activation(dtypes[1])
            and matrix(shapes[2]) and int32(dtypes[2])):
        return resolved('leading_int32_context', 1, 2)
    return {**unknown, 'reason': 'unsupported Dispatch activation/route-ID input layout'}



class ATimeIndex:
    """All-stream clipped queries including tasks already active at the boundary."""
    def __init__(self, tasks):
        self.tasks = tasks
        self.starts = [task.start_ns for task in tasks]
        self.prefix_ends = []
        latest = -(1 << 100)
        for task in tasks:
            latest = max(latest, task.end_ns)
            self.prefix_ends.append(latest)

    def overlapping(self, start, end):
        left = bisect.bisect_right(self.prefix_ends, start)
        right = bisect.bisect_left(self.starts, end)
        return [task for task in self.tasks[left:right] if task.end_ns > start]



def a_source_shape_error(task, model, reason, capture=None):
    """Preserve the evidence behind a shape rejection; never infer a new K."""
    location = ''
    if capture is not None:
        location = str(capture.get('rank', 'unknown_rank')) + ' ' + str(capture.get('path', '')) + ': '
    return ValueError(location + reason + ' in CSV data record ' + str(task.row_index)
        + '; configured hidden_size=' + str(model.get('hidden_size'))
        + ', num_experts_per_tok=' + str(model.get('num_experts_per_tok'))
        + '; Name=' + repr(task.name) + '; Type=' + repr(task.op_type)
        + '; Input Shapes=' + repr(task.input_shapes)
        + '; Input Data Types=' + repr(task.input_dtypes)
        + '; Output Shapes=' + repr(task.output_shapes))



def a_validate_source_shapes(tasks, model, capture=None):
    """Validate resolved Dispatch inputs; retain opaque layouts as diagnostics."""
    hidden, topk = model.get("hidden_size"), model.get("num_experts_per_tok")
    warnings, unknown_layouts = [], set()
    for task in tasks:
        if not A_DISPATCH.search(task.label):
            continue
        inputs = a_dispatch_inputs(task)
        if inputs['layout'] == 'unknown':
            key = (task.op_type, task.input_shapes, task.input_dtypes, inputs['reason'])
            if key not in unknown_layouts:
                unknown_layouts.add(key)
                warnings.append('DISPATCH_INPUT_LAYOUT_UNVERIFIED: data record '
                    + str(task.row_index) + '; ' + str(inputs['reason'])
                    + '; Name=' + repr(task.name) + '; Type=' + repr(task.op_type)
                    + '; Input Shapes=' + repr(task.input_shapes)
                    + '; Input Data Types=' + repr(task.input_dtypes))
            continue
        x_shape, ids_shape = inputs['x_shape'], inputs['expert_ids_shape']
        if hidden is not None and x_shape[1] != hidden:
            raise a_source_shape_error(task, model,
                'Dispatch input hidden size contradicts sample_model'
                + f' (x_input_index={inputs["x_index"]}, observed_H={x_shape[1]}, configured_H={hidden})', capture)
        if topk is not None and ids_shape[1] != topk:
            raise a_source_shape_error(task, model,
                'Dispatch routing-ID TopK contradicts sample_model'
                + f' (expert_ids_input_index={inputs["expert_ids_index"]}, observed_K={ids_shape[1]}, configured_K={topk})', capture)
        if x_shape[0] != ids_shape[0]:
            raise a_source_shape_error(task, model,
                'Dispatch activation/routing row counts contradict each other'
                + f' (x_rows={x_shape[0]}, expert_ids_rows={ids_shape[0]})', capture)
    return warnings



def measure_capture(capture, model, case, base=None):
    """Keep exact HC bounds and coverage totals, without per-task ownership ledgers."""
    tasks = capture['tasks']
    warnings = list(capture.get('parse_warnings', []))
    warnings.extend(a_validate_source_shapes(tasks, model, capture))
    cycles = a_hc_cycles(tasks, model, case)
    trim = positive_int(case.get('trim_rounds', 2), 'trim_rounds', zero=True)
    bs = positive_int(case.get('bs'), 'bs')
    effective = finite_number(case.get('effective_tokens_per_round'),
                              'effective_tokens_per_round', minimum=1,
                              maximum=1 + positive_int(case.get('draft_tokens'), 'draft_tokens', zero=True))
    if len(cycles) <= 2 * trim:
        raise ValueError(f'only {len(cycles)} complete source cycles; cannot trim {trim} from each end')
    selected = cycles[trim:len(cycles) - trim] if trim else cycles
    index, rounds = ATimeIndex(tasks), []
    for cycle in selected:
        start, end = cycle['start'], cycle['end']
        duration = (end - start) / 1e6
        layers = []
        for bounds, (role, number, attention, router) in cycle['hc_blocks']:
            first, last = bounds['start_ns'], bounds['end_ns']
            inside = tasks[bisect.bisect_left(index.starts, first):bisect.bisect_left(index.starts, last)]
            stats = a_stats(index.overlapping(first, last), first, last)
            components = {component + '_ms': math.fsum(task.end_ns - task.start_ns for task in inside
                          if pattern.search(task.label)) / 1e6
                          for component, pattern in (('dispatch', A_DISPATCH), ('combine', A_COMBINE))}
            layers.append({'role': role, 'layer': number, 'attention': attention, 'router': router,
                           'class_key': role + '_' + attention + '_' + router,
                           'start_ns': str(first), 'end_ns': str(last), 'wall_ms': (last - first) / 1e6,
                           'busy_union_ms': stats['busy_union_ms'], 'kernel_sum_ms': stats['kernel_sum_ms'],
                           **components, 'scope': 'HC_marker_interval'})
        busy = a_stats(index.overlapping(start, end), start, end)['busy_union_ms']
        totals = {role: math.fsum(layer['wall_ms'] for layer in layers if layer['role'] == role)
                  for role in ('main', 'draft')}
        outside = duration - math.fsum(totals.values())
        if outside < -1e-8 or busy > duration + 1e-8:
            raise ValueError('source HC intervals or recorded task union exceed the cycle')
        ledger_rows = [{'key': role + '_HC', 'label': role + ' HC标记区间', 'duration_ms': value}
                       for role, value in totals.items()]
        ledger_rows.append({'key': 'outside_HC', 'label': 'HC区间之外（含层间衔接、切换及尾部）',
                            'duration_ms': max(0., outside)})
        ledger = {'rows': ledger_rows, 'duration_ms': duration,
                  'category_total_ms': math.fsum(row['duration_ms'] for row in ledger_rows),
                  'recorded_busy_ms': busy, 'no_recorded_ms': max(0., duration - busy),
                  'closed': True, 'scope': 'HC_marker_partition; coverage totals are not extra costs'}
        rounds.append({'round_id': cycle['round_id'], 'start_ns': str(start), 'end_ns': str(end),
                       'duration_ms': duration, 'scope': 'HC_cycle_not_verified_request_iteration',
                       'active_bs': bs, 'effective_tokens_per_round': effective,
                       'output_tokens': bs * effective,
                       'token_basis': 'CONDITIONAL_configured_BS_times_Neff_not_measured_accepted_tokens',
                       'layers': layers, 'ledger': ledger})
    return {'rank': capture['rank'], 'path': capture['path'], 'sha256': capture['sha256'],
            'device_id': capture['device_id'], 'duplicate_rows': capture['duplicate_rows'],
            'rounds': rounds, 'warnings': sorted(set(warnings)),
            'capture_reasons': ['HC_CYCLE_ONLY: device marker timing, configured BS and Neff']}



def b_number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None



def b_positive_int(value):
    return type(value) is int and value > 0



def b_topology(case):
    """Keep physical cards, logical NPUs, DP request groups and EP distinct.

    chips_per_card declares the uniformly participating logical NPUs per card,
    not the silicon's maximum chip capacity. Using both chips contributes two
    capture ranks while remaining one card in the throughput denominator.
    Omitted chips_per_card preserves the
    historical one-logical-NPU-per-card contract; no topology is inferred from EP.
    """
    topology, errors = {}, []
    for prefix, ep_field in (('source', 'base_ep'), ('target', 'target_ep')):
        cards, chips = case.get(prefix + '_devices'), case.get(prefix + '_chips_per_card', 1)
        dp, ep = case.get(prefix + '_dp'), case.get(ep_field)
        if not b_positive_int(chips):
            errors.append(prefix + '_chips_per_card必须为正整数')
        budget = cards * chips if b_positive_int(cards) and b_positive_int(chips) else None
        topology[prefix] = {'physical_cards': cards, 'chips_per_card': chips,
                            'logical_devices': budget, 'dp_request_groups': dp, 'expert_parallel': ep}
        if budget is not None:
            if b_positive_int(dp) and dp > budget:
                errors.append(prefix + ' DP不能超过logical NPU预算（physical cards×chips_per_card）')
            if b_positive_int(ep) and ep > budget:
                errors.append(ep_field + '不能大于' + prefix + ' logical NPU预算（physical cards×chips_per_card）')
    source_budget = topology['source']['logical_devices']
    if source_budget is not None and case.get('expected_ranks') != source_budget:
        errors.append('expected_ranks必须等于源logical NPU数（physical cards×source_chips_per_card），不能遗漏芯片rank')
    return topology, errors



def b_unique(values):
    return list(dict.fromkeys(str(value) for value in values if value))



def b_schedule(model, case, target_model):
    errors, source, target = [], [], []
    kinds = {0: 'SWA', 4: 'CSA', 128: 'HCA'}
    count, hashes = model.get('num_hidden_layers'), model.get('num_hash_layers')
    drafts = case.get('draft_layers')
    if count != 8 or drafts != 3 or case.get('draft_passes') != 1:
        errors.append('V2仅支持源8主层＋3草稿层、草稿单趟前向')
    if type(hashes) is not int or hashes < 0 or not b_positive_int(count) or hashes > count:
        errors.append('源num_hash_layers/num_hidden_layers缺失或无效')
    ratios = model.get('compress_ratios')
    if (not isinstance(ratios, list) or not b_positive_int(count) or not b_positive_int(drafts)
            or len(ratios) < count + drafts):
        errors.append('源compress_ratios必须覆盖8主＋3草稿层')
    elif not errors:
        for role, n, offset in (('main', count, 0), ('draft', drafts, count)):
            for i, ratio in enumerate(ratios[offset:offset+n]):
                if type(ratio) is not int or ratio not in kinds:
                    errors.append(f'源{role}.{i}不支持的attention ratio={ratio}')
                    continue
                router = 'hash' if role == 'main' and i < hashes else 'normal'
                source.append((role, i, kinds[ratio], router))
    if not isinstance(target_model, dict):
        errors.append('缺少显式target_model')
        target_model = {}
    for role in ('main', 'draft'):
        entry = target_model.get(role)
        if not isinstance(entry, dict):
            errors.append(f'缺少目标{role}配置')
            continue
        n, dest_ratios = entry.get('num_hidden_layers'), entry.get('compress_ratios')
        dest_hashes = entry.get('num_hash_layers', hashes) if role == 'main' else 0
        if (not b_positive_int(n) or not isinstance(dest_ratios, list) or len(dest_ratios) < n
                or type(dest_hashes) is not int or dest_hashes < 0 or dest_hashes > n):
            errors.append(f'目标{role}层数/hash/compress_ratios无效')
            continue
        for i, ratio in enumerate(dest_ratios[:n]):
            if type(ratio) is not int or ratio not in kinds:
                errors.append(f'目标{role}.{i}不支持的attention ratio={ratio}')
                continue
            target.append((role, i, kinds[ratio], 'hash' if i < dest_hashes else 'normal'))
    for role in ('main', 'draft'):
        old = [x for x in source if x[0] == role]
        new = [x for x in target if x[0] == role]
        if old and (len(new) < len(old) or new[:len(old)] != old):
            errors.append(f'目标{role}必须保留源层分类前缀，不能替换或减少源层')
        if role == 'draft' and len(new) != len(old):
            errors.append('V2不支持增加或减少草稿层')
        if role == 'main' and any(x[3] != 'normal' for x in new[len(old):]):
            errors.append('仅支持在主模型尾部追加normal路由层')
    return source, target, b_unique(errors)



def b_shapes(model, target_model, case):
    errors, shapes = [], {}
    ep0, ep1 = case.get('base_ep'), case.get('target_ep')
    if not b_positive_int(ep0) or not b_positive_int(ep1) or ep1 < ep0:
        errors.append('base_ep/target_ep必须为有效且非递减的正整数')
    keys = ('hidden_size', 'moe_intermediate_size', 'n_routed_experts', 'num_experts_per_tok')
    src = {k: model.get(k) for k in keys}
    if not all(b_positive_int(v) for v in src.values()):
        errors.append('源H/M/E/K必须为已知正整数')
    if not errors and (src['n_routed_experts'] % ep0 or src['num_experts_per_tok'] > src['n_routed_experts']):
        errors.append('源专家数不能整除EP，或源TopK大于专家数')
    source_valid = not errors
    for role in ('main', 'draft'):
        entry = target_model.get(role) if isinstance(target_model, dict) else None
        dst = {k: entry.get(k) for k in keys} if isinstance(entry, dict) else {k: None for k in keys}
        local0 = src['n_routed_experts'] // ep0 if source_valid else None
        local1 = (dst['n_routed_experts'] // ep1 if b_positive_int(ep1)
                  and b_positive_int(dst['n_routed_experts']) and dst['n_routed_experts'] % ep1 == 0 else None)
        if not all(b_positive_int(v) for v in dst.values()) or local1 is None:
            errors.append(f'目标{role} H/M/E/K或本地专家数未知/无效')
        else:
            if dst['num_experts_per_tok'] > dst['n_routed_experts']:
                errors.append(f'目标{role} TopK超过专家数')
            if any(dst[k] != src[k] for k in ('hidden_size', 'moe_intermediate_size')) or local0 != local1:
                errors.append(f'目标{role}改变H/M或本地专家数，V2不支持')
            for k in ('quantization', 'quantization_config', 'expert_dtype', 'torch_dtype'):
                if k in entry and entry[k] != model.get(k):
                    errors.append(f'目标{role}改变{k}，完整层source形状不能直接沿用')
        shapes[role] = {'source': src.copy(), 'target': dst, 'local0': local0, 'local1': local1}
    return shapes, b_unique(errors)



def b_comm(method_id, model, shapes, case, projection, reference_rows):
    """Historical reference ratios for method2; method1 is constructed directly."""
    if method_id != 'method2':
        raise ValueError('b_comm only fits the historical method2 reference')
    errors, assumptions = [], []
    info = {'id': method_id, 'multipliers': {}, 'extra_ms_per_call': 0.}
    ep0, ep1 = case.get('base_ep'), case.get('target_ep')
    config = {'fit': 'log2_ols_ep16_extrapolation', 'reference_bs': case.get('bs')}
    info['model'] = 'log2_ols_ep16_extrapolation_with_measured_ep128_endpoint'
    fit_name = config.get('fit')
    if fit_name not in ('log2_ols_ep16_extrapolation', 'linear_ep16_to_ep128'):
        errors.append('方法2需要显式log2_ols_ep16_extrapolation模型')
    if (ep0, ep1) != (16, 128):
        errors.append('方法2参考转移仅支持EP16→EP128')
    requested = config.get('reference_bs')
    if requested is None:
        requested = case.get('bs')
        info['selection_basis'] = 'configured_case_bs_for_reference_diagnostic_only'
    else:
        info['selection_basis'] = 'explicit_reference_bs_declaration'
    info['fit_selection_basis'] = ('explicit_log2_OLS_declaration' if fit_name in
        ('log2_ols_ep16_extrapolation', 'linear_ep16_to_ep128') else
        'fixed_log2_OLS_reference_diagnostic_only_not_authorized_transfer')
    if requested != case.get('bs'):
        errors.append('方法2 reference_bs必须匹配每DP rank请求数')
    rows = [r for r in (reference_rows or []) if isinstance(r, dict) and b_number(r.get('bs')) == b_number(requested)]
    if len(rows) != 1:
        errors.append('方法2缺少唯一匹配BS的原始通信表记录')
    fits = {}
    for component, suffix in (('dispatch', 'send'), ('combine', 'return')):
        values = [b_number(rows[0].get(f'ep{ep}_{suffix}')) for ep in (32, 64, 128)] if len(rows) == 1 else [None]*3
        if any(v is None or v <= 0 for v in values):
            errors.append(f'方法2{component}参考值缺失或非正')
            fits[component] = None
            continue
        slope = (values[2]-values[0])/2
        intercept = statistics.fmean(values)-slope
        fitted16 = intercept-slope
        in_domain = [intercept+slope*x for x in (0, 1, 2)]
        if (b_number(fitted16) is None or fitted16 <= 0
                or any(b_number(v) is None or v <= 0 for v in in_domain)):
            errors.append(f'方法2{component} EP16外推或EP32/64/128域内拟合非正/非有限')
            fits[component] = None
            continue
        fits[component] = {'observed_ep32': values[0], 'observed_ep64': values[1],
                           'observed_ep128': values[2], 'intercept': intercept, 'slope': slope,
                           'fitted_ep16': fitted16, 'ep128_over_fitted_ep16': b_number(values[2]/fitted16),
                           'fitted_ep32': in_domain[0], 'fitted_ep64': in_domain[1],
                           'fitted_ep128': in_domain[2],
                           'residuals_observed_minus_fitted': [v-f for v, f in zip(values, in_domain)],
                           'fit_axis': 'log2(EP/32)',
                           'units': 'original_reference_units_unknown_cancel_in_ratio',
                           'formula': 'x=log2(EP/32); b=(t128-t32)/2; a=mean(t32,t64,t128)-b; t16=a-b; ratio=t128/t16'}
    for op, fit in fits.items():
        if fit is not None and fit['ep128_over_fitted_ep16'] is None:
            errors.append(f'方法2{op}外推比值非有限数')
            fits[op] = None
    info['reference_bs'], info['fits'] = requested, fits
    for role, shape in shapes.items():
        src, dst = shape['source'], shape['target']
        kh = (dst['num_experts_per_tok']*dst['hidden_size']/(src['num_experts_per_tok']*src['hidden_size'])
              if all(b_positive_int(x.get(k)) for x in (src, dst)
                     for k in ('num_experts_per_tok', 'hidden_size')) else None)
        info['multipliers'][role] = {op: b_number(fits[op]['ep128_over_fitted_ep16']*kh)
                                    if fits.get(op) is not None and kh is not None else None
                                    for op in ('dispatch', 'combine')}
        info['multipliers'][role]['payload_shape_ratio'] = kh
    assumptions.append('方法2仅用未知共同单位相消后的参考比值；EP16是三点log2 OLS外推，EP128端取参考实测值；不另加方法1的14us')
    info['reasons'] = b_unique(errors)
    return info, info['reasons'], assumptions



def positive_int(value, name, *, optional=False, zero=False):
    if value is None and optional:
        return None
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f'{name} must be an integer >= {0 if zero else 1}')
    return value



def finite_number(value, name, *, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number (not bool)')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} is outside [{minimum}, {maximum}]')
    return value



def referenced_object(value, base, name):
    if isinstance(value, str):
        path = Path(value).expanduser()
        value = json.loads((path if path.is_absolute() else base / path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object or a JSON file path')
    return value



def normalize_case(defaults, item, base):
    def token_layer(values, label):
        if not isinstance(values, dict):
            raise ValueError(label + ' must be an object')
        result = dict(values)
        for field in ('draft_tokens', 'num_speculative_tokens'):
            if field in result:
                positive_int(result[field], label + '.' + field, zero=True)
        if 'num_speculative_tokens' in result:
            if 'draft_tokens' in result and result['draft_tokens'] != result['num_speculative_tokens']:
                raise ValueError(label + ': draft_tokens conflicts with num_speculative_tokens')
            result['draft_tokens'] = result.pop('num_speculative_tokens')
        for field in ('synthetic_acceptance_length', 'synthetic_acception_length'):
            if field in result:
                finite_number(result[field], label + '.' + field, minimum=1)
        if 'synthetic_acception_length' in result:
            if ('synthetic_acceptance_length' in result and not math.isclose(
                    result['synthetic_acceptance_length'], result['synthetic_acception_length'], abs_tol=1e-9)):
                raise ValueError(label + ': synthetic_acceptance_length conflicts with synthetic_acception_length')
            result['synthetic_acceptance_length'] = result.pop('synthetic_acception_length')
        return result

    # Normalize aliases separately so an explicit case S can override the
    # other spelling in defaults without silently inheriting the old length.
    defaults, item = token_layer(defaults, 'defaults'), token_layer(item, 'case')
    if any(field in item for field in ('effective_tokens_per_round', 'synthetic_acceptance_length',
                                       'acceptance', 'acceptance_semantics')):
        # A case's explicit output policy replaces inherited fixed lengths.
        # Keep a declared default rate as a consistency constraint unless the
        # case overrides it; explicit Neff may still inherit its explanation.
        defaults.pop('effective_tokens_per_round', None)
        defaults.pop('synthetic_acceptance_length', None)
    if 'synthetic_acceptance_length' in item and not any(
            field in item for field in ('acceptance', 'acceptance_semantics')):
        # A case's explicit synthetic length is its complete output policy;
        # an older default acceptance-rate policy must not constrain it.
        defaults.pop('acceptance', None)
        defaults.pop('acceptance_semantics', None)
    case = {**defaults, **item}
    if not isinstance(case.get('name'), str) or not case['name'].strip():
        raise ValueError('Each case needs a nonempty name')
    for key in ('bs', 'draft_tokens'):
        positive_int(case.get(key), key, zero=key == 'draft_tokens')
    if case.get('with_stack') is not False:
        raise ValueError('Declare with_stack=false for the timing capture; do not mix enabled-stack captures')
    excluded = case.get('exclude_target_layers', [0])
    if excluded != [0] or any(type(value) is not int for value in excluded):
        raise ValueError('层均值外推固定使用exclude_target_layers=[0]；只排除参考样本，不减少目标层数')
    case['exclude_target_layers'] = [0]
    if case.get('spec_method', 'dspark') != 'dspark':
        raise ValueError('V2 currently supports the reviewed DSpark 8-main + 3-draft scheme only')
    case['draft_layers'] = positive_int(case.get('draft_layers', 3), 'draft_layers')
    case['draft_passes'] = positive_int(case.get('draft_passes', 1), 'draft_passes')
    if (case['draft_layers'], case['draft_passes']) != (3, 1):
        raise ValueError('The reviewed capture contract is three draft layers, one draft pass')
    case['trim_rounds'] = positive_int(case.get('trim_rounds', 2), 'trim_rounds', zero=True)
    rate_effective = None
    if case.get('acceptance_semantics') == 'mean_accepted_draft_fraction':
        rate = finite_number(case.get('acceptance'), 'acceptance', minimum=0, maximum=1)
        rate_effective = float(Decimal(1) + Decimal(case['draft_tokens']) * Decimal(str(rate)))
    if 'effective_tokens_per_round' in case:
        effective = finite_number(case['effective_tokens_per_round'], 'effective_tokens_per_round', minimum=1, maximum=1 + case['draft_tokens'])
        if not isinstance(case.get('token_basis'), str) or not case['token_basis'].strip():
            raise ValueError('effective_tokens_per_round requires token_basis explaining configured/observed and whether bonus is included')
    elif 'synthetic_acceptance_length' in case:
        effective = finite_number(case['synthetic_acceptance_length'], 'synthetic_acceptance_length', minimum=1, maximum=1 + case['draft_tokens'])
        case['token_basis'] = 'Configured synthetic mean new tokens per request per round, including the bonus token; not measured output'
    elif rate_effective is not None:
        effective = rate_effective
        case['token_basis'] = 'Configured 1+S*r, r=mean accepted draft count/S; includes one bonus token; not measured output'
    else:
        raise ValueError('Set effective_tokens_per_round plus token_basis; acceptance alone is ambiguous')
    if 'synthetic_acceptance_length' in case and not math.isclose(effective, finite_number(
            case['synthetic_acceptance_length'], 'synthetic_acceptance_length',
            minimum=1, maximum=1 + case['draft_tokens']), abs_tol=1e-9):
        raise ValueError('effective_tokens_per_round conflicts with synthetic_acceptance_length')
    if rate_effective is not None and not math.isclose(effective, rate_effective, abs_tol=1e-9):
        raise ValueError('effective output length conflicts with acceptance: mean_accepted_draft_fraction '
                         + 'requires 1 + draft_tokens * acceptance = ' + str(rate_effective)
                         + ' (including one bonus token), observed ' + str(effective))
    case['effective_tokens_per_round'] = effective
    for key in ('base_ep', 'expected_ranks'):
        positive_int(case.get(key), key)
    case['target_ep'] = positive_int(case.get('target_ep'), 'target_ep')
    for key in ('source_dp', 'source_devices', 'target_dp', 'target_devices'):
        case[key] = positive_int(case.get(key), key, optional=True)
    for prefix in ('source', 'target'):
        chip_key = prefix + '_chips_per_card'
        case[chip_key] = positive_int(case.get(chip_key, 1), chip_key)
        cards = case[prefix + '_devices']
        logical = cards * case[chip_key] if cards is not None else None
        case[prefix + '_logical_devices'] = logical
        if case[prefix + '_dp'] is not None and logical is not None and case[prefix + '_dp'] > logical:
            raise ValueError(prefix + '_dp cannot exceed physical cards * chips_per_card (logical NPUs)')
    case['ranks_per_node'] = positive_int(case.get('ranks_per_node', 8), 'ranks_per_node')
    case['traffic_scope'] = case.get('traffic_scope', 'interhost')
    if case['traffic_scope'] not in ('interhost', 'remote', 'all_remote', 'intranode'):
        raise ValueError('traffic_scope must be interhost, all_remote or intranode')
    for obsolete in ('data_fraction', 'extra_us', 'measurement_only', 'ownership_file', 'projection', 'layer_mean_exclude_main_layers'):
        case.pop(obsolete, None)

    return case



def validate_models(model, target):
    main = positive_int(model.get('num_hidden_layers'), 'sample_model.num_hidden_layers')
    if main != 8:
        raise ValueError('Keep sample_model.num_hidden_layers=8: only the captured eight main layers are parsed')
    for key in ('hidden_size', 'moe_intermediate_size', 'n_routed_experts', 'num_experts_per_tok'):
        positive_int(model.get(key), 'sample_model.' + key)
    hash_layers = positive_int(model.get('num_hash_layers'), 'sample_model.num_hash_layers', zero=True)
    if hash_layers > main:
        raise ValueError('sample_model.num_hash_layers exceeds main depth')
    ratios = model.get('compress_ratios')
    if not isinstance(ratios, list) or len(ratios) < main+3 or any(type(x) is not int or x not in (0, 4, 128) for x in ratios[:main+3]):
        raise ValueError('sample_model.compress_ratios must cover captured 8 main + 3 draft slots using 0/4/128')
    if model['num_experts_per_tok'] > model['n_routed_experts']:
        raise ValueError('sample TopK exceeds routed expert count')
    if target is not None:
        for role in ('main', 'draft'):
            entry = target.get(role)
            if not isinstance(entry, dict):
                raise ValueError('target_model.' + role + ' must be an explicit object')
            for key in ('num_hidden_layers', 'hidden_size', 'moe_intermediate_size', 'n_routed_experts', 'num_experts_per_tok'):
                positive_int(entry.get(key), 'target_model.' + role + '.' + key)
            ratios = entry.get('compress_ratios')
            if not isinstance(ratios, list) or len(ratios) != entry['num_hidden_layers'] or any(type(x) is not int or x not in (0, 4, 128) for x in ratios):
                raise ValueError('Destination role needs its exact compress_ratios sequence')
            positive_int(entry.get('num_hash_layers'), 'target_model.' + role + '.num_hash_layers', zero=True)
            if role == 'draft' and entry['num_hash_layers'] != 0:
                raise ValueError('Reviewed DSpark draft routing is normal, not a main-model hash layer')
            if entry['num_experts_per_tok'] > entry['n_routed_experts'] or entry['num_hash_layers'] > entry['num_hidden_layers']:
                raise ValueError('Invalid destination TopK/hash depth')



def rank_name(path, index):
    for part in reversed(str(path).replace('\\', '/').split('/')[:-1]):
        match = re.search(r'(?:^|[_-])(?:dp|rank)[_-]?(\d+)(?=$|[_-])', part, re.I)
        if match:
            return 'dp' + str(int(match[1]))
    return 'trace' + str(index)



def find_files(source, base, progress=None):
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = base / path
    if progress is not None:
        progress('  查找 kernel_details.csv：' + str(path))
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise ValueError('Profile path not found: ' + str(path))
    files = sorted({p.resolve() for p in path.rglob('kernel_details.csv')})
    if not files:
        raise ValueError('No parsed kernel_details.csv in ' + str(path))
    return files



def case_inputs(case, base, progress=None):
    if case.get('inputs') is not None:
        inputs = case['inputs']
        if not isinstance(inputs, list) or not inputs:
            raise ValueError('inputs must be a nonempty [{path,rank}] list')
        selected = []
        for item in inputs:
            if not isinstance(item, dict) or not isinstance(item.get('rank'), str) or not item['rank']:
                raise ValueError('Each explicit input needs path and a unique global rank name')
            files = find_files(item.get('path', ''), base, progress=progress)
            if len(files) != 1:
                raise ValueError('Explicit input path must resolve to exactly one CSV')
            selected.append((files[0], item['rank']))
    else:
        paths = case.get('paths', [case['path']] if case.get('path') else [])
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
            raise ValueError('Provide path, paths, or explicit inputs')
        files = sorted({p for source in paths for p in find_files(source, base, progress=progress)})
        selected = [(p, rank_name(p, i)) for i, p in enumerate(files)]
    paths = [p for p, _ in selected]
    names = [r for _, r in selected]
    duplicates = [name for name, count in collections.Counter(names).items() if count > 1]
    if duplicates or len(set(paths)) != len(paths):
        raise ValueError('Duplicate global rank/capture. Select one capture per global rank: ' + ', '.join(duplicates))
    return paths, names



def list_inputs(config, base):
    failed, total, incomplete = 0, 0, 0
    for index, item in enumerate(config['cases'], 1):
        name = item.get('name', 'unnamed') if isinstance(item, dict) else 'invalid'
        print(f'case [{index}/{len(config["cases"])}]: {name}；开始扫描输入路径', flush=True)
        try:
            if not isinstance(item, dict):
                raise ValueError('Each case must be an object')
            case = {**config.get('defaults', {}), **item}
            files, names = case_inputs(case, base, progress=lambda message: print(message, flush=True))
            total += len(files)
            print('kernel_details.csv:', len(files), flush=True)
            for path, rank in zip(files, names):
                print(rank + '\t' + str(path), flush=True)
            coverage = rank_coverage_details(names, case)
            print(f'配置期望rank：{len(coverage["expected"])}；实际发现CSV：{len(files)}；'
                  f'已识别全局rank：{len(coverage["expected"]) - len(coverage["missing"])}', flush=True)
            print('Expected global-rank coverage:', coverage['complete'], flush=True)
            if coverage['missing']:
                print('缺少全局rank：' + ', '.join(coverage['missing']), flush=True)
            if coverage['unexpected']:
                print('非预期/未识别全局rank：' + ', '.join(coverage['unexpected']), flush=True)
            if not coverage['complete']:
                incomplete += 1
                print('覆盖不完整：仅找到当前配置所选路径中的CSV；请核对另一机同次采样的目录和解析输出。', flush=True)
        except (ValueError, OSError, KeyError, TypeError) as error:
            failed += 1
            print('ERROR: ' + str(error), flush=True)
    print(f'输入清单完成：{len(config["cases"])} 个case，已列出 {total} 份CSV，'
          f'失败 {failed} 个case；覆盖不完整 {incomplete} 个case。', flush=True)
    return int(failed > 0)



def rank_coverage_details(names, case):
    count = positive_int(case.get('expected_ranks'), 'expected_ranks')
    expected = case.get('expected_rank_names', ['dp' + str(i) for i in range(count)])
    if not isinstance(expected, list) or len(expected) != count or any(not isinstance(x, str) or not x for x in expected) or len(set(expected)) != len(expected):
        raise ValueError('expected_rank_names must list every expected global rank exactly once')
    observed = list(dict.fromkeys(names))
    present, wanted = set(observed), set(expected)
    missing = [name for name in expected if name not in present]
    unexpected = [name for name in observed if name not in wanted]
    return {'expected': expected, 'observed': observed, 'missing': missing,
            'unexpected': unexpected, 'complete': not missing and not unexpected and len(names) == len(expected)}



def rank_coverage(names, case):
    return rank_coverage_details(names, case)['complete']



def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    p = (len(values) - 1) * q
    i, j = math.floor(p), math.ceil(p)
    return values[i] + (values[j] - values[i]) * (p - i)



def _ledger_value(ledger, key):
    value = ledger.get(key) if isinstance(ledger, dict) else None
    return value if type(value) in (int, float) and math.isfinite(value) else None



def summarize_source(measurements, case):
    per_rank, classes = [], collections.defaultdict(list)
    for capture in measurements:
        rounds = capture['rounds']
        if not rounds:
            raise ValueError('No retained rounds for ' + capture['rank'])
        durations = [r['duration_ms'] for r in rounds]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in durations):
            raise ValueError('Every retained source cycle must have a positive finite duration')
        total_ms = sum(durations)
        outputs = [r['output_tokens'] for r in rounds]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in outputs):
            raise ValueError('Source output counts must be nonnegative finite numbers')
        conditional_rate = 1000 * sum(outputs) / total_ms
        busy = [_ledger_value(r.get('ledger', {}), 'recorded_busy_ms') for r in rounds]
        uncovered = [_ledger_value(r.get('ledger', {}), 'no_recorded_ms') for r in rounds]
        for row in rounds:
            for layer in row.get('layers', []):
                classes[(layer['class_key'], layer['scope'])].append(layer)
        # This is a local DP-request-group output rate. System per-device rates
        # below additionally account for the declared topology.
        dp, devices = case.get('source_dp'), case.get('source_devices')
        device_rate = conditional_rate * dp / devices if dp is not None and devices is not None else None
        per_rank.append({'rank': capture['rank'], 'rounds': len(rounds),
                         'cycle_ms_mean': statistics.fmean(durations),
                         'cycle_ms_p50': quantile(durations, .5), 'cycle_ms_p95': quantile(durations, .95),
                         'total_cycle_ms': total_ms, 'total_output_tokens': sum(outputs),
                         'recorded_busy_ms_mean': statistics.fmean(busy) if all(v is not None for v in busy) else None,
                         'no_recorded_ms_mean': statistics.fmean(uncovered) if all(v is not None for v in uncovered) else None,
                         'conditional_per_rank_tokens_s': conditional_rate,
                         'conditional_per_device_tokens_s': device_rate,
                         'token_basis': sorted({r['token_basis'] for r in rounds}),
                         'scope': sorted({r['scope'] for r in rounds}),
                         'consecutive_decode_verified': all(r.get('consecutive_decode') is True for r in rounds)})
    selected = max(per_rank, key=lambda r: r['cycle_ms_mean'])
    dp, devices = case.get('source_dp'), case.get('source_devices')
    global_rate = selected['conditional_per_rank_tokens_s'] * dp if dp is not None else None
    per_device = global_rate / devices if global_rate is not None and devices is not None else None
    class_stats = []
    for (key, scope), layers in sorted(classes.items()):
        row = {'class_key': key, 'scope': scope, 'samples': len(layers)}
        for metric in ('wall_ms', 'busy_union_ms', 'dispatch_ms', 'combine_ms', 'router_ms', 'routed_mlp_ms'):
            values = [l.get(metric) for l in layers]
            row[metric + '_mean'] = statistics.fmean(values) if all(type(v) in (int, float) and math.isfinite(v) for v in values) else None
        class_stats.append(row)
    topology, topology_errors = b_topology(case)
    source_topology_errors = [reason for reason in topology_errors
                              if reason.startswith(('source', 'base_ep', 'expected_ranks'))]
    source_budget = topology['source']['logical_devices']
    complete = (case['_rank_coverage_complete'] is True and not source_topology_errors
                and source_budget is not None and len(measurements) == source_budget)
    return {'status': 'SOURCE_CONDITIONAL_RATE', 'per_rank': per_rank, 'class_stats': class_stats,
            'topology': topology['source'],
            'topology_reasons': source_topology_errors,
            'aggregate': {'selected_rank': selected['rank'], 'cycle_ms': selected['cycle_ms_mean'],
                          'per_device_tokens_s': per_device, 'global_tokens_s': global_rate,
                          'source_physical_cards': devices,
                          'source_chips_per_card': topology['source']['chips_per_card'],
                          'source_logical_devices': topology['source']['logical_devices'],
                          'scope': '按源设备周期和有效输出数换算的条件系统单卡均摊吞吐；非客户端实测吞吐',
                          'aggregation': 'max_of_rank_mean_proxy',
                          'captured_ranks': len(measurements), 'expected_ranks': case['expected_ranks'],
                          'rank_coverage_complete': complete,
                          'declared_rank_names_complete': case['_rank_coverage_complete'],
                          'formula': 'Q_device=(1000*sum(output_tokens)/sum(cycle_ms))*source_dp/source_devices; selected rank=max(mean cycle)'}}



def read_reference(path):
    if path is None:
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Communication reference has empty or duplicate headers')
        if 'bs' not in reader.fieldnames:
            raise ValueError('Communication reference needs bs column')
        return list(reader)



def analyze_case(case, model, target, base, reference_rows=None):
    files, names = case_inputs(case, base)
    case = {**case, '_rank_coverage_complete': rank_coverage(names, case)}
    residual_dir = case.pop('_step_residual_dir', None)
    captures, measurements, capture_hashes, prepared = [], [], {}, {}
    schedule, _, errors = b_schedule(model, case, target)
    shapes, shape_errors = b_shapes(model, target, case)
    source_errors = errors + shape_errors
    required = {role: shape['target']['num_experts_per_tok'] != shape['source']['num_experts_per_tok']
                for role, shape in shapes.items()} if not source_errors else {}
    excluded = set(case.get('exclude_target_layers', [0]))
    for rank_number, (path, rank) in enumerate(zip(files, names), 1):
        rank_started = stage_started = time.perf_counter()
        for info in path.parent.parent.glob('profiler_info*.json'):
            common = json.loads(info.read_text(encoding='utf-8-sig')).get('config', {}).get('common_config', {})
            if common.get('with_stack') is True or common.get('with_modules') is True:
                raise ValueError(f'{info}: enabled stack/modules capture conflicts with timing baseline')
        print(f'  读取 {rank}（{rank_number}/{len(names)}）: {path}', flush=True)
        capture = read_capture(path, rank)
        print(f'  {rank} CSV/PMU读取完成：{len(capture["tasks"])}条任务，'
              f'{time.perf_counter() - stage_started:.2f}s；开始HC周期识别', flush=True)
        stage_started = time.perf_counter()
        if capture['sha256'] in capture_hashes:
            raise ValueError('Identical CSV content was selected as different global ranks: '
                             + capture_hashes[capture['sha256']] + ' and ' + rank)
        capture_hashes[capture['sha256']] = rank
        measured = measure_capture(capture, model, case, base)
        print(f'  {rank} HC周期识别完成：{len(measured["rounds"])}轮，'
              f'{time.perf_counter() - stage_started:.2f}s；开始层内阶段统计', flush=True)
        trace_path = next((item.get('trace_path') for item in case.get('inputs', [])
                           if item.get('rank') == rank), None)
        if trace_path is not None:
            trace_path = Path(trace_path).expanduser()
            if not trace_path.is_absolute():
                trace_path = Path(base) / trace_path
        try:
            # The CLI always supplies its output directory. Direct API callers
            # still receive base measurements without writing beside inputs or
            # returning references to deleted temporary detail files.
            with (nullcontext(residual_dir) if residual_dir is not None else
                  tempfile.TemporaryDirectory(prefix='ep128-residual-')) as detail_dir:
                stage_started = time.perf_counter()
                try:
                    measured['layer_diagnostics'], measured['swa_diagnostics'] = layer_analyze_capture(
                        capture, measured, detail_dir, with_swa_compat=True)
                except (OSError, ValueError, TypeError, ArithmeticError) as error:
                    for field in ('layer_diagnostics', 'swa_diagnostics'):
                        measured[field] = {'status': 'ERROR', 'reason': str(error),
                                           'layer_summary': [], 'stage_summary': [], 'stages_csv': None}
                print(f'  {rank} 层内阶段统计结束：{time.perf_counter() - stage_started:.2f}s；'
                      '开始HC剩余区间/Host来源统计', flush=True)
                stage_started = time.perf_counter()
                measured['step_residuals'] = residual_analyze_capture(
                    capture, measured, detail_dir, trace_path=trace_path, case=case,
                    progress=lambda message: print('    ' + message, flush=True))
                print(f'  {rank} HC剩余区间/Host来源统计结束：'
                      f'{time.perf_counter() - stage_started:.2f}s', flush=True)
        except (OSError, ValueError, TypeError, ArithmeticError) as error:
            measured['step_residuals'] = {'status': 'ERROR', 'windows': [], 'summary': [],
                                          'trace': {'status': 'ERROR', 'reason': str(error)}}
            measured['warnings'].append('HC_RESIDUAL_DIAGNOSTIC_FAILED: ' + str(error))
        finally:
            if residual_dir is None:
                for field in ('device_tasks_csv', 'host_events_csv', 'stages_csv', 'source_links_csv'):
                    measured['step_residuals'][field] = None
                focus = measured['step_residuals'].get('focus_diagnostics')
                if isinstance(focus, dict):
                    for field in ('segments_csv', 'steps_csv'):
                        focus[field] = None
                for field in ('layer_diagnostics', 'swa_diagnostics'):
                    if isinstance(measured.get(field), dict):
                        measured[field]['stages_csv'] = None
        measurements.append(measured)
        captures.append({key: capture.get(key) for key in ('rank', 'path', 'sha256', 'device_id', 'duplicate_rows')})
        if not source_errors:
            print(f'  {rank} 开始逐轮算子/PMU汇总', flush=True)
            stage_started = time.perf_counter()
            try:
                starts = [task.start_ns for task in capture['tasks']]
                time_index = ATimeIndex(capture['tasks'])
                prepared[rank] = [_lm_source_round(cycle, capture['tasks'], starts, model, case,
                                                  schedule, excluded, required, time_index=time_index,
                                                  rank=rank) for cycle in measured['rounds']]
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                source_errors.append(str(error))
            print(f'  {rank} 逐轮算子/PMU汇总结束：{time.perf_counter() - stage_started:.2f}s', flush=True)
        print(f'  {rank} 分析结束：共{time.perf_counter() - rank_started:.2f}s', flush=True)
    reasons = [message for item in measurements for message in item.get('warnings', [])]
    if not case['_rank_coverage_complete']:
        reasons.append(f'RANK_COVERAGE: captured {len(names)}/{case["expected_ranks"]}; only captured ranks are estimated')
    return {'name': case['name'], 'config': case, 'source_model': model, 'target_model': target,
            'captures': captures, 'measurements': measurements, 'source': summarize_source(measurements, case),
            'projection': {}, 'review': {'reasons': sorted(set(reasons)), 'findings': [
                'All source layers are measured; main0 is excluded only from reference means.',
                'Source HC walls include existing communication and compute; add only modeled deltas.',
                'Recorded task union excludes duplicate overlap and does not add Wait to wall time.',
                'Same-role boundaries and the cross-role/tail residual are retained once.',
                'Per-card throughput divides configured system output by physical cards.']},
            '_prepared_layer_means': prepared, '_layer_mean_source_errors': source_errors}



@contextmanager
def atomic_writer(path, encoding='utf-8'):
    """Close temporary handles before replace/cleanup, including on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding=encoding, newline='', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)



# HC-residual base measurements; all times are observations, never extra costs.
def residual_ns(value):
    return a_exact_ns(value, "residual boundary")


@contextmanager
def residual_csv(path, fields):
    with atomic_writer(path, encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        yield writer


RESIDUAL_VERSION = PROFILE_VERSION


RESIDUAL_CHUNK_BYTES = 256 * 1024


RESIDUAL_MAX_EVENT_CHARS = 16 * 1024 * 1024


RESIDUAL_ANCHOR_LIMIT = 512


RESIDUAL_MIN_ANCHORS = 3


RESIDUAL_CLOCK_TOLERANCE_NS = 1000


RESIDUAL_SCOPE = "HC_cycle_residual_not_ProfilerStep_or_verified_request_iteration"


RESIDUAL_COMM = re.compile(r"MoeDistribute(Dispatch|Combine)(?:V\d+)?", re.I)


def residual_complement(left, right, intervals):
    result, cursor = [], left
    for start, end in intervals:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < right:
        result.append((cursor, right))
    return result


def residual_windows(measured):
    windows, previous = [], None
    for cycle in measured.get("rounds", []):
        start, end = residual_ns(cycle["start_ns"]), residual_ns(cycle["end_ns"])
        if end <= start or previous is not None and start < previous:
            raise ValueError("residual HC cycles overlap or have nonpositive duration")
        previous = end
        layers = sorted(cycle["layers"], key=lambda layer: residual_ns(layer["start_ns"]))
        roles = [layer.get("role") for layer in layers]
        main = [layer for layer in layers if layer.get("role") == "main"]
        draft = [layer for layer in layers if layer.get("role") == "draft"]
        if not main or not draft or roles != ["main"] * len(main) + ["draft"] * len(draft):
            raise ValueError("residual windows require contiguous main then draft HC chains")
        cursor = start
        for layer in layers:
            a, b = residual_ns(layer["start_ns"]), residual_ns(layer["end_ns"])
            if not cursor <= a < b <= end:
                raise ValueError("residual HC layers overlap or cross the cycle")
            cursor = b
        first = residual_ns(main[0]["start_ns"])
        if first != start:
            raise ValueError("HC cycle must begin at the first main HCPre; prefix cannot be silently omitted")
        main_end, draft_start = residual_ns(main[-1]["end_ns"]), residual_ns(draft[0]["start_ns"])
        draft_end = residual_ns(draft[-1]["end_ns"])
        residual = end - start - (main_end - first) - (draft_end - draft_start)
        pairs = (("main_to_draft", "主模型结束至草稿开始", main_end, draft_start),
                 ("draft_to_nextmain", "草稿结束至下一轮主模型开始", draft_end, end))
        if sum(b - a for _, _, a, b in pairs) != residual:
            raise ValueError("residual window durations do not close to the measured HC cycle")
        if "duration_ms" in cycle and abs(float(cycle["duration_ms"]) - (end - start) / 1e6) > 1e-8:
            raise ValueError("HC cycle duration contradicts residual window timestamps")
        for key, label, left, right in pairs:
            windows.append({"round_id": cycle.get("round_id"), "window_key": key, "label": label,
                            "start_ns": str(left), "end_ns": str(right), "duration_ms": (right - left) / 1e6,
                            "cycle_residual_ms": residual / 1e6, "closed": True, "scope": RESIDUAL_SCOPE})
    if not windows:
        raise ValueError("no retained HC cycles for residual analysis")
    return windows


RESIDUAL_DEVICE_FIELDS = ("rank", "round_id", "window_key", "row_index", "row_indices", "name", "op_type",
                  "device_id", "model_id", "stream_id", "task_id", "context_id", "batch_id", "subtask_id",
                  "task_type", "op_state", "input_shapes", "output_shapes", "input_dtypes", "output_dtypes",
                  "start_ns", "end_ns", "duration_ms", "clip_start_ns", "clip_end_ns",
                  "clip_duration_ms", "wait_ms_diagnostic")


RESIDUAL_HOST_FIELDS = ("rank", "round_id", "window_key", "trace_event_index", "name", "category", "process_name",
                "pid", "tid", "trace_start_ns", "aligned_start_ns", "aligned_end_ns", "duration_ms",
                "clip_start_ns", "clip_end_ns", "clip_duration_ms", "sync_visible", "copy_visible",
                "graph_replay_visible", "args_json")


RESIDUAL_HOST_NUMBERS = ("host_span_union_ms", "host_in_device_no_recorded_ms", "host_event_count",
                 "host_sync_events", "host_copy_events", "host_graph_replay_events")

RESIDUAL_DMA_NUMBERS = ("trace_dma_union_ms", "trace_dma_in_csv_uncovered_ms", "uncovered_after_trace_dma_ms")

RESIDUAL_SOURCE_COUNTS = ("source_device_tasks", "source_cpu_tasks", "source_api_tasks", "source_queue_tasks",
                          "source_graph_tasks", "source_ambiguous_tasks")
RESIDUAL_SOURCE_NUMBERS = ("source_cpu_union_ms", "source_cpu_span_ms", "source_api_union_ms",
                           "source_queue_wait_union_ms", "source_submit_to_device_ms")
RESIDUAL_SOURCE_SCOPE = ("按同次trace显式flow、connection_id和correlation_id回查设备任务的Host来源，允许提交早于U；"
    "CPU为可见调用区间union，非CPU忙时或完整Python模块；队列等待和提交至执行是延迟，"
    "各列及跨阶段不可相加到U。共享图且无来源的任务不分摊Host时间。")
RESIDUAL_SOURCE_FIELDS = ("row_index", "name", "device_start_ns", "device_end_ns", "connection_id", "status",
    "cpu_event_index", "cpu_name", "cpu_start_ns", "cpu_end_ns", "cpu_leaf_event_index", "cpu_leaf_name",
    "cpu_ancestors_json", "enqueue_event_index", "enqueue_start_ns", "enqueue_end_ns", "dequeue_event_index",
    "dequeue_start_ns", "dequeue_end_ns", "correlation_id", "node_event_index", "node_start_ns", "node_end_ns",
    "queue_wait_ns", "submit_to_device_ns", "source_before_window_ns", "reason", "flow_ids_json")


def residual_device_rows(capture, windows, index, path):
    uncovered = []
    with residual_csv(path, RESIDUAL_DEVICE_FIELDS) as writer:
        for window in windows:
            left, right = residual_ns(window["start_ns"]), residual_ns(window["end_ns"])
            tasks = index.overlapping(left, right)
            clips = []
            for task in tasks:
                a, b = max(left, task.start_ns), min(right, task.end_ns)
                if b <= a:
                    continue
                clips.append((a, b))
                record = {key: getattr(task, key, None) for key in RESIDUAL_DEVICE_FIELDS
                          if key not in ("row_indices", "start_ns", "end_ns")}
                record.update(rank=capture.get("rank"), round_id=window["round_id"], window_key=window["window_key"],
                              row_indices=json.dumps(getattr(task, "row_indices", [getattr(task, "row_index", None)])),
                              start_ns=str(task.start_ns), end_ns=str(task.end_ns),
                              duration_ms=(task.end_ns - task.start_ns) / 1e6,
                              clip_start_ns=str(a), clip_end_ns=str(b), clip_duration_ms=(b - a) / 1e6,
                              wait_ms_diagnostic=getattr(task, "wait_ns", 0) / 1e6)
                writer.writerow(record)
            busy = a_union(clips)
            busy_ns = sum(b - a for a, b in busy)
            window.update(recorded_busy_ms=busy_ns / 1e6, no_recorded_ms=(right - left - busy_ns) / 1e6,
                          kernel_sum_ms=sum(b - a for a, b in clips) / 1e6, task_count=len(clips))
            window.update(dict.fromkeys(RESIDUAL_HOST_NUMBERS))
            uncovered.append(residual_complement(left, right, busy))
    return uncovered


class residual_JSONStream:
    """Decode one trace event at a time; bound each individual JSON value."""
    def __init__(self, path, chunk_size):
        self.stream = Path(path).open("rb")
        self.chunk_size = chunk_size
        self.buffer, self.position, self.eof = "", 0, False
        self.utf8 = codecs.getincrementaldecoder("utf-8-sig")()
        self.decoder = json.JSONDecoder(parse_float=Decimal,
                                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        self.digest, self.byte_count, self.max_buffer = hashlib.sha256(), 0, 0

    def close(self):
        self.stream.close()

    def fill(self):
        if self.eof:
            return False
        chunk = self.stream.read(self.chunk_size)
        self.byte_count += len(chunk)
        self.digest.update(chunk)
        self.eof = not chunk
        self.buffer = self.buffer[self.position:] + self.utf8.decode(chunk, final=self.eof)
        self.position = 0
        self.max_buffer = max(self.max_buffer, len(self.buffer))
        return bool(chunk)

    def char(self):
        while self.position >= len(self.buffer):
            if not self.fill():
                return ""
        return self.buffer[self.position]

    def whitespace(self):
        while self.char() and self.char().isspace():
            self.position += 1

    def expect(self, value):
        self.whitespace()
        if self.char() != value:
            raise ValueError("invalid or truncated trace JSON; expected " + repr(value))
        self.position += 1

    def value(self):
        self.whitespace()
        while True:
            try:
                value, stop = self.decoder.raw_decode(self.buffer, self.position)
                if stop - self.position > RESIDUAL_MAX_EVENT_CHARS:
                    raise ValueError("individual trace JSON value exceeds bounded parser limit")
                self.position = stop
                return value
            except json.JSONDecodeError as error:
                if len(self.buffer) - self.position > RESIDUAL_MAX_EVENT_CHARS:
                    raise ValueError("individual trace JSON value exceeds bounded parser limit") from error
                if not self.fill():
                    raise ValueError("invalid or truncated trace JSON event") from error

    def skip_value(self):
        # Outer wrapper fields such as stackFrames can be large. Skip them
        # lexically rather than materializing another giant JSON object.
        self.whitespace()
        first = self.char()
        if not first:
            raise ValueError("truncated trace JSON wrapper")
        if first not in "[{\"":
            while self.char() and self.char() not in ",]} \r\n\t":
                self.position += 1
            return
        stack, quoted, escaped = [], first == '"', False
        if first in "[{":
            stack.append(first)
        self.position += 1
        while True:
            char = self.char()
            if not char:
                raise ValueError("truncated trace JSON wrapper")
            self.position += 1
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                    if not stack:
                        return
            elif char == '"':
                quoted = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                if not stack or (stack.pop(), char) not in (("[", "]"), ("{", "}")):
                    raise ValueError("mismatched trace JSON wrapper")
                if not stack:
                    return


def residual_iter_trace_events(path, *, chunk_size=RESIDUAL_CHUNK_BYTES, stats=None):
    """Yield dict events from Chrome list/traceEvents JSON without json.load."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("trace chunk_size must be a positive integer")
    reader = residual_JSONStream(path, chunk_size)

    def array():
        reader.expect("[")
        reader.whitespace()
        if reader.char() == "]":
            reader.position += 1
            return
        while True:
            event = reader.value()
            if not isinstance(event, dict):
                raise ValueError("trace array must contain JSON event objects")
            yield event
            reader.whitespace()
            separator = reader.char()
            if separator == "]":
                reader.position += 1
                return
            reader.expect(",")

    try:
        reader.whitespace()
        if reader.char() == "[":
            yield from array()
        else:
            reader.expect("{")
            found = False
            while True:
                reader.whitespace()
                if reader.char() == "}":
                    reader.position += 1
                    break
                key = reader.value()
                if not isinstance(key, str):
                    raise ValueError("trace wrapper key is not a string")
                reader.expect(":")
                if key == "traceEvents":
                    if found:
                        raise ValueError("duplicate traceEvents array")
                    found = True
                    yield from array()
                else:
                    reader.skip_value()
                reader.whitespace()
                if reader.char() == "}":
                    reader.position += 1
                    break
                reader.expect(",")
            if not found:
                raise ValueError("trace JSON has no traceEvents array")
        reader.whitespace()
        if reader.char():
            raise ValueError("trailing data after trace JSON")
        if stats is not None:
            stats.update(sha256=reader.digest.hexdigest(), bytes_read=reader.byte_count,
                         parser_max_buffer_chars=reader.max_buffer)
    finally:
        reader.close()


def residual_id(value):
    return None if value is None else str(value).strip()


@lru_cache(maxsize=512)
def residual_arg_key(key):
    return re.sub(r"[^a-z0-9]", "", key.lower())


def residual_args(event):
    raw = event.get("args")
    return {residual_arg_key(str(key)): value for key, value in raw.items()} if isinstance(raw, dict) else {}


def residual_kind(name):
    match = RESIDUAL_COMM.search(str(name))
    return match.group(0).lower() if match else None


@lru_cache(maxsize=256)
def residual_device_category(category):
    return bool(re.search(r"kernel|device|aicore|aicpu|npu", category))


def residual_device_hint(event, args=None):
    cat = str(event.get("cat", "")).lower()
    if args is None:
        args = residual_args(event)
    return bool(residual_device_category(cat) or
                "tasktype" in args and ("physicstreamid" in args or "streamid" in args))


@lru_cache(maxsize=256)
def residual_device_process(name):
    return bool(re.search(r"ascend.hardware|device|kernel|gpu|npu", name, re.I))


@lru_cache(maxsize=512)
def residual_host_category(category, process):
    return bool(set(re.split(r"[,; ]+", category)).intersection(("cpu_op", "python_function", "user_annotation"))
                or re.search(r"cann|runtime|^python$", process, re.I)
                or category in ("cann", "runtime", "cann_api"))


def residual_host(event, processes, args=None, device_hint=None):
    process = processes.get(residual_id(event.get("pid")), "")
    if device_hint is None:
        device_hint = residual_device_hint(event, args)
    if device_hint or residual_device_process(process):
        return False
    cat = str(event.get("cat", "")).lower()
    return residual_host_category(cat, process)


def residual_anchor_candidates(tasks):
    full, base = {}, {}
    for task in tasks:
        kind = residual_kind(getattr(task, "op_type", "") + " " + task.name)
        duration = task.end_ns - task.start_ns
        if not kind or duration <= 0:
            continue
        key = kind, duration
        base.setdefault(key, []).append(task)
        ids = tuple(residual_id(getattr(task, field, None)) for field in ("stream_id", "task_id", "model_id"))
        if all(value is not None for value in ids):
            full.setdefault(key + ids, []).append(task)
    candidates = {task.start_ns: task for values in list(full.values()) + list(base.values())
                  if len(values) == 1 for task in values}
    values = sorted(candidates.values(), key=lambda task: task.start_ns)
    if len(values) > RESIDUAL_ANCHOR_LIMIT:
        values = [values[i * (len(values) - 1) // (RESIDUAL_ANCHOR_LIMIT - 1)] for i in range(RESIDUAL_ANCHOR_LIMIT)]
    allowed = {task.start_ns for task in values}
    lookup = {}
    for label, groups in (("full", full), ("base", base)):
        for key, matches in groups.items():
            if len(matches) == 1 and matches[0].start_ns in allowed:
                lookup[label, key] = matches[0].start_ns
    return {task.start_ns: task for task in values}, lookup


def residual_progress(progress, stage, count=None):
    if progress is not None:
        progress(stage if count is None else stage + f"：已扫描 {count:,} 个事件")


def residual_trace_scan(path, tasks, progress):
    candidates, lookup = residual_anchor_candidates(tasks)
    matches, processes, scan = {}, {}, {}
    counts = {"events": 0, "x_events": 0, "be_events_unsupported": 0, "invalid_timed_events": 0}
    last = time.monotonic()
    for event in residual_iter_trace_events(path, stats=scan):
        counts["events"] += 1
        phase = event.get("ph")
        if phase == "M" and event.get("name") == "process_name" and event.get("pid") is not None:
            processes[residual_id(event.get("pid"))] = str(residual_args(event).get("name", ""))
        if phase in ("B", "E"):
            counts["be_events_unsupported"] += 1
        if phase == "X":
            counts["x_events"] += 1
            kind = residual_kind(event.get("name", ""))
            if kind:
                try:
                    duration, start = a_ns(event["dur"]), a_ns(event["ts"])
                except (ValueError, KeyError, ArithmeticError):
                    counts["invalid_timed_events"] += 1
                    continue
                args = residual_args(event)
                ids = (residual_id(args.get("physicstreamid", args.get("streamid"))),
                       residual_id(args.get("taskid")), residual_id(args.get("modelid")))
                source = lookup.get(("full", (kind, duration) + ids))
                if source is None:
                    source = lookup.get(("base", (kind, duration)))
                if source is not None:
                    hits = matches.setdefault(source, [])
                    if len(hits) < 9:
                        hits.append((start, residual_id(event.get("pid")), residual_device_hint(event, args)))
        now = time.monotonic()
        if counts["events"] % 50000 == 0 or now - last >= 5:
            residual_progress(progress, "trace元数据与设备时钟锚点", counts["events"])
            last = now
    offsets, anchors = [], []
    for source, hits in matches.items():
        selected = [hit for hit in hits if hit[2] or residual_device_process(processes.get(hit[1], ""))]
        if len(selected) != 1 or len(hits) >= 9:
            continue
        start, pid, _ = selected[0]
        offsets.append(source - start)
        anchors.append({"csv_row_index": getattr(candidates[source], "row_index", None),
                        "csv_start_ns": str(source), "trace_start_ns": str(start), "trace_pid": pid})
    result = {**scan, **counts, "process_names": processes, "anchor_count": len(offsets),
              "required_anchor_count": RESIDUAL_MIN_ANCHORS, "clock_tolerance_ns": RESIDUAL_CLOCK_TOLERANCE_NS,
              "supported_host_phase": "X; B/E durations are not reconstructed"}
    if len(offsets) < RESIDUAL_MIN_ANCHORS:
        return {**result, "status": "CLOCK_UNCONFIRMED", "reason": "fewer than three unique matching device-kernel anchors"}, None
    ordered = sorted(offsets)
    offset = (ordered[len(ordered) // 2] + ordered[(len(ordered) - 1) // 2]) // 2
    residual = max(abs(value - offset) for value in offsets)
    result.update(clock_offset_ns=str(offset), max_clock_residual_ns=residual, anchors=anchors[:8])
    if residual > RESIDUAL_CLOCK_TOLERANCE_NS:
        return {**result, "status": "CLOCK_MISMATCH", "reason": "device anchors do not support one constant trace-to-CSV offset"}, None
    return {**result, "status": "ALIGNED_X_EVENTS"}, offset


@lru_cache(maxsize=4096)
def residual_flags(name):
    return (bool(re.search(r"synchroni[sz]e|(?:stream|event).*wait|wait.*(?:stream|event)", name, re.I)),
            bool(re.search(r"memcpy|memset|copy_|aten::copy|HostToDevice|DeviceToHost", name, re.I)),
            bool(re.search(r"graph.*(?:execute|launch|replay)|(?:execute|launch|replay).*graph|aclmdl(?:RI)?Execute|rtModelExecute", name, re.I)))


def residual_source_rows(connection, tasks, stages, destination, progress):
    """Join explicit trace links on disk; interval indexes keep exact int64 ns.

    A device connection identifies a Node, its worker interval identifies the
    dequeue, and correlation identifies the enqueue. CPU nesting supplies only
    the recorded call chain. An async flow corroborates it, never overrides a
    conflicting queue chain. Duplicate IDs and missing graph starts stay visible.
    """
    connection.executescript("""
        CREATE INDEX source_conn ON source_events(kind,connection_id);
        CREATE INDEX source_corr ON source_events(kind,correlation_id);
        CREATE INDEX flow_end ON source_flows(ph,pid,tid,stamp);
        CREATE INDEX flow_key ON source_flows(flow_key,ph);
        CREATE INDEX device_row ON source_devices(row_index);
        CREATE TABLE resolved(row_index INTEGER PRIMARY KEY,cpu INTEGER,node INTEGER,enq INTEGER,deq INTEGER,
                              wait INTEGER,latency INTEGER,status TEXT);
    """)
    # Four compact integer vectors per thread avoid repeated scans of 217k CPU
    # intervals. Dequeue uses worker tid because CANN has a synthetic process ID.
    intervals = {}
    for identity, kind, pid, tid, start, end in connection.execute(
            "SELECT event_index,kind,pid,tid,start,end FROM source_events WHERE kind IN ('cpu','dequeue') "
            "ORDER BY kind,pid,tid,start,end"):
        key = (kind, pid if kind == "cpu" else None, tid)
        vectors = intervals.setdefault(key, ([], [], [], []))
        vectors[0].append(start); vectors[1].append(end); vectors[3].append(identity)
    for key, vectors in intervals.items():
        ordered = sorted(zip(vectors[0], vectors[1], vectors[3]))
        maximum = -2 ** 63
        compact = tuple(array('q') for _ in range(4))
        for start, end, identity in ordered:
            maximum = max(maximum, end)
            for vector, value in zip(compact, (start, end, maximum, identity)):
                vector.append(value)
        intervals[key] = compact

    @lru_cache(maxsize=4096)
    def event(identity):
        row = connection.execute("SELECT event_index,kind,pid,tid,name,start,end,connection_id,correlation_id "
                                 "FROM source_events WHERE event_index=?", (identity,)).fetchone()
        return dict(zip(("id", "kind", "pid", "tid", "name", "start", "end", "conn", "corr"), row))

    def containing(kind, pid, tid, start, end=None):
        vectors = intervals.get((kind, pid if kind == "cpu" else None, tid))
        if vectors is None:
            return []
        starts, ends, prefix, identities = vectors
        stop, first = bisect.bisect_right(starts, start), bisect.bisect_right(prefix, start)
        return [event(identities[i]) for i in range(first, stop)
                if ends[i] > start and (end is None or ends[i] >= end)]

    def cpu_chain(pid, tid, point):
        rows = sorted(containing("cpu", pid, tid, point), key=lambda row: (row["start"], -row["end"], row["id"]))
        if not rows or any(a["end"] < b["end"] for a, b in zip(rows, rows[1:])):
            return []
        return rows

    @lru_cache(maxsize=8192)
    def flow_starts(key):
        return connection.execute("SELECT pid,tid,stamp FROM source_flows WHERE flow_key=? AND ph='s'", (key,)).fetchall()

    @lru_cache(maxsize=8192)
    def flow_finishes(key):
        return connection.execute("SELECT count(*) FROM source_flows WHERE flow_key=? AND ph='f'", (key,)).fetchone()[0]

    wanted = {task.row_index: task for task in tasks}
    owners = collections.defaultdict(list)
    boundaries = {}
    for stage_id, stage in enumerate(stages):
        key = stage["round_id"], stage["window_key"]
        boundaries[key] = min(boundaries.get(key, residual_ns(stage["start_ns"])), residual_ns(stage["start_ns"]))
        for row_index in stage["_task_rows"]:
            owners[row_index].append(stage_id)
    status_counts = collections.Counter()
    with residual_csv(destination, RESIDUAL_SOURCE_FIELDS) as writer:
        for number, (row_index, task) in enumerate(sorted(wanted.items()), 1):
            record = dict.fromkeys(RESIDUAL_SOURCE_FIELDS)
            record.update(row_index=row_index, name=task.name, device_start_ns=str(task.start_ns),
                          device_end_ns=str(task.end_ns), status="UNLINKED")
            devices = connection.execute("SELECT pid,tid,start,end,connection_id FROM source_devices WHERE row_index=?",
                                         (row_index,)).fetchall()
            cpu, node, enq, deq, chain = None, None, None, None, []
            flows, reasons = [], []
            if len(devices) != 1:
                record.update(status="AMBIGUOUS" if devices else "NO_TRACE_DEVICE",
                              reason="multiple exact trace device matches" if devices else "no exact trace device match")
            else:
                pid, tid, start, end, conn = devices[0]
                record["connection_id"] = conn
                finishes = connection.execute("SELECT cat,flow_key FROM source_flows WHERE ph='f' AND pid=? AND tid=? AND stamp=?",
                                               (pid, tid, start)).fetchall()
                flows = list(dict.fromkeys(key for _, key in finishes))
                nodes = [event(row[0]) for row in connection.execute(
                    "SELECT event_index FROM source_events WHERE kind='node' AND connection_id=?", (conn,))] if conn is not None else []
                # Multiple candidates are accepted only if the explicit flow
                # source uniquely identifies one of their intervals.
                duplicate_nodes = len(nodes) > 1
                if duplicate_nodes:
                    starts = [point for cat, key in finishes if cat == "HostToDevice" for point in flow_starts(key)]
                    nodes = [candidate for candidate in nodes if any(candidate["pid"] == p and candidate["tid"] == t
                              and candidate["start"] <= stamp < candidate["end"] for p, t, stamp in starts)]
                ambiguous = duplicate_nodes and len(nodes) != 1
                if len(nodes) == 1:
                    node = nodes[0]
                    dequeues = containing("dequeue", None, node["tid"], node["start"], node["end"])
                    if len(dequeues) == 1:
                        deq = dequeues[0]
                        enqueues = [event(row[0]) for row in connection.execute(
                            "SELECT event_index FROM source_events WHERE kind='enqueue' AND correlation_id=?", (deq["corr"],))]
                        if len(enqueues) == 1:
                            enq = enqueues[0]
                        elif enqueues:
                            ambiguous = True
                            reasons.append("duplicate enqueue correlation_id")
                    elif dequeues:
                        ambiguous = True
                        reasons.append("multiple dequeue intervals contain Node")
                queue_chain = cpu_chain(enq["pid"], enq["tid"], enq["start"]) if enq else []
                if queue_chain and queue_chain[0]["end"] < enq["end"]:
                    queue_chain = []
                    ambiguous = True
                    reasons.append("CPU source does not contain the enqueue interval")
                if enq and deq:
                    queue_flows = connection.execute("SELECT flow_key FROM source_flows WHERE ph='f' AND cat='async_task_queue' "
                        "AND pid=? AND tid=? AND stamp>=? AND stamp<?", (deq["pid"], deq["tid"], deq["start"], deq["end"])).fetchall()
                    if queue_flows and not any(enq["pid"] == p and enq["tid"] == t and enq["start"] <= stamp < enq["end"]
                                               for (key,) in queue_flows for p, t, stamp in flow_starts(key)):
                        ambiguous = True
                        reasons.append("queue flow disagrees with correlation_id pair")
                candidates = []
                for cat, key in finishes:
                    if cat == "async_npu":
                        for p, t, stamp in flow_starts(key):
                            candidate = cpu_chain(p, t, stamp)
                            if candidate:
                                candidates.append(candidate)
                if queue_chain:
                    matching = [candidate for candidate in candidates if candidate[0]["id"] == queue_chain[0]["id"]]
                    if candidates and not matching:
                        ambiguous = True
                        reasons.append("CPU flow disagrees with explicit queue source")
                    else:
                        chain = queue_chain
                        if len({candidate[0]["id"] for candidate in candidates}) > 1:
                            reasons.append("reused async flow resolved by queue correlation")
                elif len({candidate[0]["id"] for candidate in candidates}) == 1:
                    chain = candidates[0]
                elif candidates:
                    ambiguous = True
                    reasons.append("multiple CPU flow origins without queue corroboration")
                if (node and node["start"] > start or chain and chain[0]["start"] > start
                        or enq and deq and enq["start"] > deq["start"]):
                    ambiguous = True
                    reasons.append("source starts after its linked execution; causal order is inconsistent")
                if ambiguous:
                    record["status"] = "AMBIGUOUS"
                    reasons.append("source chain is not unique")
                    cpu, node, enq, deq, chain = None, None, None, None, []
                else:
                    cpu = chain[0] if chain else None
                    graph = not node and any(cat == "HostToDevice" and not flow_starts(key) and flow_finishes(key) > 1
                                             for cat, key in finishes)
                    record["status"] = ("LINKED" if cpu and node else "CPU_ONLY" if cpu else "API_ONLY" if node else
                                        "GRAPH_SHARED_NO_ORIGIN" if graph else "UNLINKED")
                    if graph:
                        reasons.append("shared HostToDevice flow has no recorded start; graph Host time is not apportioned")
                    if not cpu:
                        reasons.append("no unique recorded CPU source")
                for prefix, value in (("cpu", cpu), ("node", node), ("enqueue", enq), ("dequeue", deq)):
                    if value:
                        record.update({prefix + "_event_index": value["id"], prefix + "_start_ns": str(value["start"]),
                                       prefix + "_end_ns": str(value["end"])})
                if cpu:
                    record.update(cpu_name=cpu["name"], cpu_leaf_event_index=chain[-1]["id"], cpu_leaf_name=chain[-1]["name"],
                        cpu_ancestors_json=json.dumps([{key: value[key] for key in ("id", "name", "start", "end")}
                                                       for value in chain], ensure_ascii=False, separators=(",", ":")))
                if enq and deq:
                    record["correlation_id"] = enq["corr"]
                    if deq["start"] >= enq["end"]:
                        record["queue_wait_ns"] = deq["start"] - enq["end"]
                    else:
                        reasons.append("dequeue starts before enqueue interval ends; wait is unknown")
                if node and start >= node["start"]:
                    record["submit_to_device_ns"] = start - node["start"]
                origin = cpu or node
                if origin and owners[row_index]:
                    boundary = min(boundaries[(stages[i]["round_id"], stages[i]["window_key"])] for i in owners[row_index])
                    record["source_before_window_ns"] = max(0, boundary - origin["start"])
                record["reason"] = "; ".join(reasons) or None
            record["flow_ids_json"] = json.dumps(flows, ensure_ascii=False, separators=(",", ":"))
            writer.writerow(record)
            status_counts[record["status"]] += 1
            connection.execute("INSERT INTO resolved VALUES (?,?,?,?,?,?,?,?)", (row_index,
                cpu["id"] if cpu else None, node["id"] if node else None, enq["id"] if enq else None,
                deq["id"] if deq else None, record["queue_wait_ns"], record["submit_to_device_ns"], record["status"]))
            if number % 4000 == 0:
                residual_progress(progress, "回查设备任务的CPU、队列和CANN来源", number)

    def union_ms(identities):
        spans = [(event(identity)["start"], event(identity)["end"]) for identity in identities]
        return sum(b - a for a, b in a_union(spans)) / 1e6 if spans else None

    for stage in stages:
        rows = [connection.execute("SELECT cpu,node,enq,deq,wait,latency,status FROM resolved WHERE row_index=?", (identity,)).fetchone()
                for identity in stage["_task_rows"]]
        cpus, nodes = {row[0] for row in rows if row[0]}, {row[1] for row in rows if row[1]}
        queues = {(row[2], row[3]) for row in rows if row[2] and row[3] and row[4] is not None}
        latencies = [row[5] for row in rows if row[5] is not None]
        stage.update(source_device_tasks=len(rows), source_cpu_tasks=sum(row[0] is not None for row in rows),
            source_api_tasks=sum(row[1] is not None for row in rows), source_queue_tasks=sum(row[2] is not None and row[3] is not None for row in rows),
            source_graph_tasks=sum(row[6] == "GRAPH_SHARED_NO_ORIGIN" for row in rows),
            source_ambiguous_tasks=sum(row[6] == "AMBIGUOUS" for row in rows),
            source_status="COMPLETE" if rows and all(row[6] == "LINKED" for row in rows) else "PARTIAL" if cpus or nodes else "UNAVAILABLE",
            source_cpu_union_ms=union_ms(cpus), source_api_union_ms=union_ms(nodes),
            source_cpu_span_ms=(max(event(i)["end"] for i in cpus) - min(event(i)["start"] for i in cpus)) / 1e6 if cpus else None,
            source_queue_wait_union_ms=sum(b - a for a, b in a_union([(event(a)["end"], event(b)["start"]) for a, b in queues])) / 1e6 if queues else None,
            source_submit_to_device_ms=statistics.fmean(latencies) / 1e6 if latencies else None)
    return {"source_links_csv": str(destination), "source_status": "COMPLETE" if status_counts["LINKED"] == len(wanted) else
            "PARTIAL" if status_counts["LINKED"] or status_counts["CPU_ONLY"] or status_counts["API_ONLY"] else "UNAVAILABLE",
            "source_scope": RESIDUAL_SOURCE_SCOPE, "source_task_status_counts": dict(status_counts)}


def residual_host_rows(path, capture, windows, uncovered, trace, offset, destination, directory, progress, stages=()):
    starts, ends = [residual_ns(row["start_ns"]) for row in windows], [residual_ns(row["end_ns"]) for row in windows]
    counts = [{key: 0 for key in RESIDUAL_HOST_NUMBERS} for _ in windows]
    by_window = collections.defaultdict(list)
    device_pids = {row["trace_pid"] for row in trace.get("anchors", [])}
    dma_available = len(device_pids) == 1
    for row in list(windows) + list(stages):
        row.update(dict.fromkeys(RESIDUAL_DMA_NUMBERS, 0. if dma_available else None),
                   trace_dma_status="ALIGNED_SINGLE_DEVICE" if dma_available else "DEVICE_AMBIGUOUS")
    for row in stages:
        row.update(dict.fromkeys(RESIDUAL_HOST_NUMBERS, 0))
        by_window[(row["round_id"], row["window_key"])].append(
            (row, residual_ns(row["start_ns"]), residual_ns(row["end_ns"])))

    # Every gap list is already sorted and disjoint. Search only gaps touched
    # by this Host span instead of rescanning the entire window for each span.
    def gap_index(gaps):
        return gaps, [end for _, end in gaps]

    window_gaps = [gap_index(gaps) for gaps in uncovered]
    stage_gaps = {id(row): gap_index(row["_uncovered"]) for row in stages}

    def gap_overlap(left, right, indexed):
        gaps, ends = indexed
        first = bisect.bisect_right(ends, left)
        total = 0
        for index in range(first, len(gaps)):
            start, end = gaps[index]
            if start >= right:
                break
            total += min(right, end) - max(left, start)
        return total
    temporary = tempfile.NamedTemporaryFile(dir=directory, suffix=".sqlite", delete=False)
    database = Path(temporary.name)
    temporary.close()
    connection = sqlite3.connect(database)
    scan, last = {}, time.monotonic()
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("CREATE TABLE clips (window INTEGER, start INTEGER, end INTEGER)")
        connection.execute("CREATE TABLE dma_clips (window INTEGER, start INTEGER, end INTEGER)")
        connection.execute("CREATE TABLE source_events(event_index INTEGER PRIMARY KEY,kind TEXT,pid TEXT,tid TEXT,name TEXT,"
                           "start INTEGER,end INTEGER,connection_id TEXT,correlation_id TEXT)")
        connection.execute("CREATE TABLE source_flows(cat TEXT,flow_key TEXT,ph TEXT,pid TEXT,tid TEXT,stamp INTEGER)")
        connection.execute("CREATE TABLE source_devices(row_index INTEGER,pid TEXT,tid TEXT,start INTEGER,end INTEGER,connection_id TEXT)")
        wanted_ids = {identity for row in stages for identity in row["_task_rows"]}
        wanted = collections.defaultdict(list)
        wanted_tasks = [task for task in capture["tasks"] if task.row_index in wanted_ids]
        for task in wanted_tasks:
            wanted[(task.name, task.start_ns, task.end_ns)].append(task)
        pending, source_events, source_flows, source_devices = [], [], [], []

        def persist_sources():
            for table, values, placeholders in (("source_events", source_events, "?,?,?,?,?,?,?,?,?"),
                    ("source_flows", source_flows, "?,?,?,?,?,?"), ("source_devices", source_devices, "?,?,?,?,?,?")):
                connection.executemany("INSERT INTO " + table + " VALUES (" + placeholders + ")", values)
                values.clear()

        with residual_csv(destination, RESIDUAL_HOST_FIELDS) as writer:
            for number, event in enumerate(residual_iter_trace_events(path, stats=scan), 1):
                now = time.monotonic()
                if number % 50000 == 0 or now - last >= 5:
                    residual_progress(progress, "trace主机事件与U窗口重叠", number)
                    last = now
                phase = event.get("ph")
                pid, tid = residual_id(event.get("pid")), residual_id(event.get("tid"))
                if phase in ("s", "f"):
                    try:
                        stamp = a_ns(event["ts"]) + offset
                    except (ValueError, KeyError, ArithmeticError):
                        continue
                    identity = event.get("id", event.get("id2"))
                    if identity is not None:
                        cat = str(event.get("cat", ""))
                        flow_key = json.dumps([cat, event.get("name"), event.get("scope"), identity],
                                              ensure_ascii=False, default=str, separators=(",", ":"))
                        source_flows.append((cat, flow_key, phase, pid, tid, stamp))
                    if len(source_flows) >= 1000:
                        persist_sources()
                    continue
                if phase != "X":
                    continue
                try:
                    raw, duration = a_ns(event["ts"]), a_ns(event["dur"])
                    if duration <= 0:
                        continue
                except (ValueError, KeyError, ArithmeticError):
                    continue
                left, right = raw + offset, raw + offset + duration
                args, name = residual_args(event), str(event.get("name", ""))
                device_hint = residual_device_hint(event, args)
                if device_hint or residual_device_process(trace["process_names"].get(pid, "")):
                    dma_type = re.sub(r"[^A-Z0-9]", "", str(args.get("tasktype", "")).upper())
                    if dma_available and pid in device_pids and ("DMA" in dma_type or name.upper().startswith("MEMCPY_ASYNC")):
                        first, final = bisect.bisect_right(ends, left), bisect.bisect_left(starts, right)
                        connection.executemany("INSERT INTO dma_clips VALUES (?,?,?)",
                            [(index, max(starts[index], left), min(ends[index], right)) for index in range(first, final)])
                    for task in wanted.get((name, left, right), []):
                        identifiers = ((getattr(task, "stream_id", None), args.get("physicstreamid", args.get("streamid"))),
                                       (getattr(task, "task_id", None), args.get("taskid")), (getattr(task, "model_id", None), args.get("modelid")))
                        if all(a is None or b is None or residual_id(a) == residual_id(b) for a, b in identifiers):
                            source_devices.append((task.row_index, pid, tid, left, right, residual_id(args.get("connectionid"))))
                    if len(source_devices) >= 1000:
                        persist_sources()
                    continue
                if not residual_host(event, trace["process_names"], args, device_hint):
                    continue
                category = str(event.get("cat", ""))
                kind = ("enqueue" if name.startswith("Enqueue@") else "dequeue" if name.startswith("Dequeue@") else
                        "node" if name.startswith("Node@") else "cpu" if set(re.split(r"[,; ]+", category)).intersection(
                            ("cpu_op", "python_function", "user_annotation")) else None)
                if kind:
                    source_events.append((number, kind, pid, tid, name, left, right,
                                          residual_id(args.get("connectionid")), residual_id(args.get("correlationid"))))
                if len(source_events) >= 1000:
                    persist_sources()
                first, final = bisect.bisect_right(ends, left), bisect.bisect_left(starts, right)
                for index in range(first, final):
                    a, b = max(starts[index], left), min(ends[index], right)
                    if b <= a:
                        continue
                    window, name = windows[index], str(event.get("name", ""))
                    sync, copy, graph = residual_flags(name)
                    targets = [counts[index]] + [row for row, start, end in by_window[(window["round_id"], window["window_key"])]
                        if start < b and end > a]
                    for target in targets:
                        target["host_event_count"] += 1
                        for key, value in (("host_sync_events", sync), ("host_copy_events", copy), ("host_graph_replay_events", graph)):
                            target[key] += int(value)
                    pending.append((index, a, b))
                    if len(pending) >= 1000:
                        connection.executemany("INSERT INTO clips VALUES (?,?,?)", pending)
                        pending.clear()
                    writer.writerow({"rank": capture.get("rank"), "round_id": window["round_id"], "window_key": window["window_key"],
                                     "trace_event_index": number, "name": name, "category": str(event.get("cat", "")),
                                     "process_name": trace["process_names"].get(residual_id(event.get("pid")), ""),
                                     "pid": event.get("pid"), "tid": event.get("tid"), "trace_start_ns": str(raw),
                                     "aligned_start_ns": str(left), "aligned_end_ns": str(right), "duration_ms": duration / 1e6,
                                     "clip_start_ns": str(a), "clip_end_ns": str(b), "clip_duration_ms": (b - a) / 1e6,
                                     "sync_visible": sync, "copy_visible": copy, "graph_replay_visible": graph,
                                     "args_json": json.dumps(event.get("args", {}), ensure_ascii=False, default=str, separators=(",", ":"))})
            if scan["sha256"] != trace["sha256"]:
                raise ValueError("trace changed between clock scan and host scan")
            persist_sources()
            connection.executemany("INSERT INTO clips VALUES (?,?,?)", pending)
            connection.commit()
            current, left, right = None, None, None

            def flush():
                if current is not None:
                    counts[current]["host_span_union_ms"] += (right - left) / 1e6
                    counts[current]["host_in_device_no_recorded_ms"] += gap_overlap(left, right, window_gaps[current]) / 1e6
                    window = windows[current]
                    for stage, start, end in by_window[(window["round_id"], window["window_key"])]:
                        a, b = max(left, start), min(right, end)
                        if b > a:
                            stage["host_span_union_ms"] += (b - a) / 1e6
                            stage["host_in_device_no_recorded_ms"] += gap_overlap(a, b, stage_gaps[id(stage)]) / 1e6

            for window, a, b in connection.execute("SELECT window,start,end FROM clips ORDER BY window,start,end"):
                if current != window or a > right:
                    flush()
                    current, left, right = window, a, b
                else:
                    right = max(right, b)
            flush()
            if dma_available:
                current, intervals = None, []

                def flush_dma():
                    if current is None:
                        return
                    union = a_union(intervals)
                    window = windows[current]
                    window["trace_dma_union_ms"] = sum(b - a for a, b in union) / 1e6
                    window["trace_dma_in_csv_uncovered_ms"] = sum(gap_overlap(a, b, window_gaps[current]) for a, b in union) / 1e6
                    for stage, start, end in by_window[(window["round_id"], window["window_key"])]:
                        clipped = [(max(start, a), min(end, b)) for a, b in union if a < end and b > start]
                        stage["trace_dma_union_ms"] = sum(b - a for a, b in clipped) / 1e6
                        stage["trace_dma_in_csv_uncovered_ms"] = sum(gap_overlap(a, b, stage_gaps[id(stage)]) for a, b in clipped) / 1e6

                for window_index, a, b in connection.execute("SELECT window,start,end FROM dma_clips ORDER BY window,start,end"):
                    if current != window_index:
                        flush_dma()
                        current, intervals = window_index, []
                    intervals.append((a, b))
                flush_dma()
                for row in list(windows) + list(stages):
                    row["uncovered_after_trace_dma_ms"] = max(0., row["no_recorded_ms"] - row["trace_dma_in_csv_uncovered_ms"])
        for window, values in zip(windows, counts):
            window.update(values)
        trace["host_window_event_records"] = sum(row["host_event_count"] for row in counts)
        trace["host_overlap_scope"] = "nonadditive host X-event wall spans; synchronization is not CPU busy time; overlap does not prove causality"
        try:
            return residual_source_rows(connection, wanted_tasks, stages, destination.with_name(
                destination.name.replace("-host-events.csv", "-source-links.csv")), progress)
        except (ValueError, ArithmeticError, sqlite3.Error) as error:
            # Causal source diagnostics must not erase independently valid
            # simultaneous Host coverage or the original device stage spans.
            for row in stages:
                row.update(dict.fromkeys(RESIDUAL_SOURCE_COUNTS[1:] + RESIDUAL_SOURCE_NUMBERS), source_status="UNAVAILABLE")
            return {"source_status": "UNAVAILABLE", "source_reason": str(error), "source_links_csv": None,
                    "source_scope": RESIDUAL_SOURCE_SCOPE}
    finally:
        connection.close()
        database.unlink(missing_ok=True)


RESIDUAL_STAGE_BASIS = ("同次设备锚点之间的连续时间跨度，包含段内任务间隔；"
    "按下一阶段首锚点划界，不以算子Duration求和，不等同Python模块调用耗时。"
    "缺失或歧义锚点保留整段未识别；子阶段已包含于父阶段。")


def diagnostic_metrics(tasks, left, right):
    """Clipped wall/busy clocks and locateable evidence; waits are not wall costs."""
    tasks = sorted((task for task in tasks if task.start_ns < right and task.end_ns > left),
                   key=lambda task: (task.start_ns, task.end_ns, task.row_index))
    clips = [(max(left, task.start_ns), min(right, task.end_ns)) for task in tasks]
    union = a_union(clips)
    busy = sum(b - a for a, b in union)
    gaps = residual_complement(left, right, union)
    groups = collections.defaultdict(list)
    for task, (a, b) in zip(tasks, clips):
        groups[(task.name, getattr(task, "op_type", ""), getattr(task, "stream_id", None))].append((task, b - a))

    def evidence(task):
        return {key: getattr(task, key, None) for key in
                ("row_index", "name", "task_type", "stream_id", "input_shapes", "output_shapes")} | {
                    "start_ns": str(task.start_ns), "end_ns": str(task.end_ns),
                    "duration_ms": (task.end_ns - task.start_ns) / 1e6}

    top = []
    for (name, op_type, stream), values in groups.items():
        top.append({"name": name, "op_type": op_type, "stream_id": stream, "calls": len(values),
                    "kernel_sum_ms": sum(duration for _, duration in values) / 1e6,
                    "wait_sum_ms": sum(task.wait_ns for task, _ in values) / 1e6
                        if all(getattr(task, "wait_available", True) for task, _ in values) else None,
                    "example": evidence(values[0][0])})
    top.sort(key=lambda row: (-row["kernel_sum_ms"], row["name"]))
    gap_rows = []
    for a, b in sorted(gaps, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))[:5]:
        previous = max((task for task in tasks if task.end_ns <= a),
                       key=lambda task: task.end_ns, default=None)
        following = min((task for task in tasks if task.start_ns >= b),
                        key=lambda task: task.start_ns, default=None)
        gap_rows.append({"start_ns": str(a), "end_ns": str(b), "duration_ms": (b - a) / 1e6,
                         "previous": evidence(previous) if previous else None,
                         "following": evidence(following) if following else None})
    kernel_sum = sum(b - a for a, b in clips) / 1e6
    wait_sum = sum(task.wait_ns for task in tasks) / 1e6 if all(getattr(task, "wait_available", True) for task in tasks) else None
    return {"recorded_busy_ms": busy / 1e6, "no_recorded_ms": (right - left - busy) / 1e6,
            "kernel_sum_ms": kernel_sum, "task_count": len(tasks), "task_wait_sum_ms": wait_sum,
            "total_cost_ms": kernel_sum + wait_sum if wait_sum is not None else None,
            "wait_scope": "source Wait values of overlapping tasks; nonadditive, may precede this interval; not idle or CPU time",
            "small_task_count": sum(task.end_ns - task.start_ns <= 10000 for task in tasks),
            "small_kernel_sum_ms": sum(b - a for task, (a, b) in zip(tasks, clips)
                                       if task.end_ns - task.start_ns <= 10000) / 1e6,
            "largest_gap_ms": max((b - a for a, b in gaps), default=0) / 1e6,
            "top_kernels": top[:5], "top_gaps": gap_rows}


RESIDUAL_FOCUS_SCOPE = (
    "源采样同rank逐保留step的父段内设备时间覆盖；所有单算子、具名并发组合和无设备记录间隙互斥闭合父段。"
    "step_index为保留样本顺序，round_id保留原HC周期标识；未获得调度器业务step编号。"
    "均值以全部保留step为分母：某组件在已识别step未出现计0；父段缺失保持未知，不按0补齐。"
    "source_function_mapping仅为固定版本源码入口的静态对应，不是实测Python函数耗时；"
    "同名metadata可来自不同KV组/层/窗口，不由出现顺序猜测具体builder。"
    "无设备记录不等于CPU忙时或设备空闲；原15父阶段、level1及周期/TPOT/吞吐保持原口径。")


def residual_focus_source_mapping(task):
    """Auditable static kernel-to-entry references, never inferred CPU timings."""
    ascend = "vllm-ascend@66b5a5eaca3dbf97aa4022e0d6170ba7ba53dc9f"
    upstream = "vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac (peer source reference)"

    def entry(path, function, line, basis, evidence, **extra):
        return {"path": path, "function": function, "line": line,
                "version_basis": basis, "evidence": evidence,
                "measurement_scope": "static source_function_mapping; runtime entry and Python duration unverified", **extra}

    input_path = "vllm/v1/worker/gpu/input_batch.py"
    known = {
        "_expand_idx_mapping_kernel": ("展开请求与token映射", [entry(input_path, "expand_idx_mapping", 634, upstream,
            "exact Triton kernel launch in wrapper; Ascend GPUModelRunner.prepare_inputs calls wrapper")]),
        "_prepare_pos_seq_lens_kernel": ("准备位置与序列长度", [entry(input_path, "prepare_pos_seq_lens", 304, upstream,
            "exact Triton kernel launch in wrapper; Ascend GPUModelRunner.prepare_inputs calls wrapper")]),
        "_combine_sampled_and_draft_tokens_kernel": ("拼接采样与草稿token", [entry(input_path,
            "combine_sampled_and_draft_tokens", 386, upstream,
            "exact Triton kernel launch; writes input_ids and logits_indices")]),
        "_apply_write_kernel": ("应用暂存的状态/块表写入", [
            entry("vllm/v1/worker/gpu/buffer_utils.py", "StagedWriteTensor.apply_write", 174, upstream,
                  "same kernel used for staged tensor writes; tensor ownership cannot be inferred from kernel name"),
            entry("vllm/v1/worker/gpu/buffer_utils.py", "FusedStagedWriter.apply", 225, upstream,
                  "multi-group staged writes also launch this kernel; not necessarily KV payload copying")]),
        "_gather_block_tables_kernel": ("聚合本批请求KV块表", [entry("vllm/v1/worker/gpu/block_table.py",
            "BlockTables.gather_block_tables", 134, upstream, "exact kernel launch with per-group and padded request grid")]),
        "_compute_slot_mappings_kernel": ("计算token到KV槽位映射", [entry("vllm_ascend/worker/v2/block_table.py",
            "AscendBlockTables.compute_slot_mappings", 82, ascend, "Ascend override launches same-named Triton slot kernel")]),
        "_zero_kv_blocks_kernel": ("KV块清零", [
            entry("vllm_ascend/worker/utils.py", "AscendKVBlockZeroer.zero_block_ids", 350, ascend,
                  "Ascend zeroer launches same-named kernel; v2 AscendV2KVBlockZeroer delegates by cache kind"),
            entry("vllm/v1/worker/utils.py", "KVBlockZeroer.zero_block_ids", 192, upstream,
                  "upstream zeroer can also launch this name; exact implementation needs runtime trace/source check")]),
    }
    if task.name in known:
        label, entries = known[task.name]
        return label, "STATIC_KERNEL_NAME_MATCH", entries
    name = str(getattr(task, "op_type", "")) + " " + task.name
    if re.search(r"QuantLightningIndexer(?:V\d+)?Metadata", name, re.I):
        return "Indexer执行计划metadata", "STATIC_KERNEL_FAMILY_MATCH", [entry(
            "vllm_ascend/attention/dsa_v1.py", "AscendDSAMetadataBuilder._build_qli_metadata", 1012, ascend,
            "native metadata op is called by this helper; layer/cache-group identity is not present in kernel name",
            call_chain=["worker/v2/attn_utils.py:build_attn_metadata", "AscendDSAMetadataBuilder.build",
                        "AscendDSAMetadataBuilder.build_req_metadata", "AscendDSAMetadataBuilder._build_qli_metadata"])]
    if re.search(r"SparseAttnSharedkvMetadata|MixedQuantSparseFlashMlaMetadata", name, re.I):
        return "Attention执行计划metadata", "STATIC_KERNEL_FAMILY_MATCH", [entry(
            "vllm_ascend/attention/dsa_v1.py", "AscendDSAMetadataBuilder._build_sas_metadata", 944, ascend,
            "KV plan supplies sparse-attention metadata op; same family serves different ratios, cache groups and windows",
            call_chain=["worker/v2/attn_utils.py:build_attn_metadata", "AscendDSAMetadataBuilder.build",
                        "AscendDSAMetadataBuilder.build_req_metadata", "AscendDSAMetadataBuilder._build_sas_metadata"])]
    # Generic Cast/Copy/Gather/Mul/Div names occur in many functions.  Their
    # position between metadata calls is not a unique Python attribution.
    return str(getattr(task, "op_type", "") or task.name), "UNMAPPED", []


def residual_focus_partition(parent, tasks, operators=None):
    """Sweep half-open device intervals into an exact, all-stream wall partition.

    Returns compact components plus locateable per-segment CSV evidence. A
    kernel crossing the parent boundary is clipped; same-name concurrent calls
    retain multiplicity. No anchor or optional kernel is required.
    """
    operators = operators if operators is not None else {}
    left, right = int(parent["start_ns"]), int(parent["end_ns"])
    if right <= left:
        raise ValueError("focus parent must have a positive interval")
    events, task_ids = collections.defaultdict(list), {}
    for number, task in enumerate(sorted((task for task in tasks if task.start_ns < right
            and task.end_ns > left and task.end_ns > task.start_ns),
            key=lambda task: (task.start_ns, task.end_ns, task.row_index))):
        signature = tuple(str(getattr(task, field, "") or "") for field in
                          ("name", "op_type", "stream_id", "input_shapes", "output_shapes", "task_type"))
        operator_id = "op_" + hashlib.sha256(json.dumps(signature, ensure_ascii=False).encode()).hexdigest()[:16]
        if operator_id not in operators:
            label, mapping_status, mapping = residual_focus_source_mapping(task)
            operators[operator_id] = dict(zip(("name", "op_type", "stream_id", "input_shapes", "output_shapes", "task_type"), signature))
            operators[operator_id].update(label=label, mapping_status=mapping_status,
                source_function_mapping=mapping, example_row_indices=[])
        examples = operators[operator_id]["example_row_indices"]
        if task.row_index not in examples and len(examples) < 3:
            examples.append(task.row_index)
        task_ids[number] = (task, operator_id)
        events[max(left, task.start_ns)].append((1, number))
        events[min(right, task.end_ns)].append((-1, number))
    events[left]
    events[right]
    active, components, segments, cursor = {}, {}, [], left
    for point in sorted(events):
        if point > cursor:
            counts = collections.Counter(operator_id for _, operator_id in active.values())
            signature = tuple(sorted(counts.items()))
            previous = [task_ids[number] for direction, number in events[cursor] if direction < 0] if not active else []
            following = [task_ids[number] for direction, number in events[point] if direction > 0] if not active else []
            previous_ids = sorted({operator_id for _, operator_id in previous})
            following_ids = sorted({operator_id for _, operator_id in following})
            # Distinguish where each unrecorded gap sits. Combining every gap
            # into a single bucket hides the actionable handoff that grew.
            identity = ("active", signature) if active else ("gap", previous_ids, following_ids)
            component_key = "coverage_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
            if component_key not in components:
                kind = "no_device_record" if not active else "single_operator" if len(active) == 1 else "concurrent_operators"
                labels = [operators[key]["label"] + (" ×" + str(count) if count > 1 else "") for key, count in signature]
                gap_label = ((" + ".join(operators[key]["label"] for key in previous_ids) or "父段开始")
                             + " → " + (" + ".join(operators[key]["label"] for key in following_ids) or "父段结束")
                             + "之间无设备记录（来源未知）")
                mappings = []
                for key, _ in signature:
                    for mapping in operators[key]["source_function_mapping"]:
                        if mapping not in mappings:
                            mappings.append(mapping)
                components[component_key] = {"component_key": component_key,
                    "label": gap_label if not active else " + ".join(labels)
                        + ("（并发）" if len(active) > 1 else ""),
                    "kind": kind, "operator_ids": [key for key, _ in signature],
                    "operator_multiplicity": dict(signature), "source_function_mapping": mappings,
                    "previous_operator_ids": previous_ids, "following_operator_ids": following_ids,
                    "mapping_status": "UNMAPPED" if not active or any(operators[key]["mapping_status"] == "UNMAPPED"
                        for key in counts) else "STATIC_SOURCE_MAPPING", "duration_ns": 0, "segment_count": 0}
            component = components[component_key]
            component["duration_ns"] += point - cursor
            component["segment_count"] += 1
            segments.append({"component_key": component_key, "kind": component["kind"], "label": component["label"],
                "start_ns": str(cursor), "end_ns": str(point), "duration_ns": point - cursor,
                "duration_ms": (point - cursor) / 1e6,
                "row_indices": sorted({row for task, _ in active.values()
                    for row in (getattr(task, "row_indices", None) or [task.row_index])}),
                "previous_row_indices": sorted({row for task, _ in previous
                    for row in (getattr(task, "row_indices", None) or [task.row_index])}),
                "following_row_indices": sorted({row for task, _ in following
                    for row in (getattr(task, "row_indices", None) or [task.row_index])}),
                "operator_ids": component["operator_ids"]})
        # End events precede start events at the same timestamp. All events are
        # applied before measuring the next nonempty interval.
        for direction, number in sorted(events[point]):
            if direction < 0:
                active.pop(number, None)
            else:
                active[number] = task_ids[number]
        cursor = point
    if sum(row["duration_ns"] for row in segments) != right - left:
        raise ValueError("focus coverage does not close to parent")
    return {"components": list(components.values()), "segments": segments,
            "closure_error_ns": sum(row["duration_ns"] for row in segments) - (right - left)}


def residual_focus_diagnostics(stages, windows, index, rank, directory, prefix):
    """Preserve source-rank per-step changes and an additive fine breakdown."""
    tail_windows = [window for window in windows if window["window_key"] == "draft_to_nextmain"]
    operators, stage_reports, step_export = {}, [], []
    segment_path = Path(directory) / (prefix + "-focus-segments.csv")
    step_path = Path(directory) / (prefix + "-focus-steps.csv")
    segment_fields = ("rank", "step_index", "round_id", "stage_key", "component_key", "kind", "label",
                      "start_ns", "end_ns", "duration_ns", "duration_ms", "row_indices", "operator_ids",
                      "previous_row_indices", "following_row_indices")
    with residual_csv(segment_path, segment_fields) as writer:
        for stage_key, label in (("main_inputs", "下一主模型KV与输入准备"),
                                 ("main_metadata", "下一主metadata及输入映射准备")):
            parents = [row for row in stages if row.get("level") == 0 and row["stage_key"] == stage_key]
            by_round = collections.defaultdict(list)
            for parent in parents:
                by_round[parent.get("round_id")].append(parent)
            samples, component_catalog, component_values = [], {}, collections.defaultdict(dict)
            for step_number, window in enumerate(tail_windows, 1):
                # Timestamp containment disambiguates repeated or missing IDs.
                candidates = [row for row in by_round[window.get("round_id")] if
                    int(window["start_ns"]) <= int(row["start_ns"]) < int(row["end_ns"]) <= int(window["end_ns"])]
                parent = candidates[0] if len(candidates) == 1 and candidates[0]["status"] == "MATCHED" else None
                reason = None
                if not candidates:
                    reason = "missing " + stage_key + " parent interval in retained step"
                    if window.get("stage_reason"):
                        reason += "; window: " + str(window["stage_reason"])
                elif len(candidates) > 1:
                    reason = "ambiguous " + stage_key + " parent intervals: observed " + str(len(candidates))
                elif parent is None:
                    reason = "parent status=" + str(candidates[0].get("status")) + "; " + str(
                        candidates[0].get("reason") or "no parser reason provided")
                sample = {"step_index": step_number, "round_id": window.get("round_id"),
                          "status": "MATCHED" if parent else "UNAVAILABLE", "reason": reason,
                          "start_ns": None, "end_ns": None,
                          "duration_ms": None, "recorded_busy_ms": None, "no_recorded_ms": None,
                          "kernel_sum_ms": None, "task_count": None, "component_durations_ms": {}}
                if parent is not None:
                    left, right = int(parent["start_ns"]), int(parent["end_ns"])
                    tasks = index.overlapping(left, right)
                    partition = residual_focus_partition(parent, tasks, operators)
                    sample.update({field: parent.get(field) for field in
                        ("start_ns", "end_ns", "duration_ms", "recorded_busy_ms", "no_recorded_ms", "kernel_sum_ms", "task_count")})
                    sample["closure_error_ns"] = partition["closure_error_ns"]
                    for component in partition["components"]:
                        key = component["component_key"]
                        component_catalog[key] = component
                        component_values[key][step_number] = component["duration_ns"] / 1e6
                        sample["component_durations_ms"][key] = component["duration_ns"] / 1e6
                    for segment in partition["segments"]:
                        writer.writerow({"rank": rank, "step_index": step_number, "round_id": window.get("round_id"),
                            "stage_key": stage_key, **{key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                                if isinstance(value, (dict, list)) else value for key, value in segment.items()}})
                samples.append(sample)
                step_export.append({"rank": rank, "stage_key": stage_key, **sample})
            complete = bool(samples) and all(sample["status"] == "MATCHED" for sample in samples)
            summary = []
            parent_total = sum(sample["duration_ms"] for sample in samples if sample["duration_ms"] is not None)
            for key, component in component_catalog.items():
                values = [component_values[key].get(sample["step_index"], 0.) for sample in samples
                          if sample["status"] == "MATCHED"]
                summary.append({field: value for field, value in component.items() if field not in ("duration_ns", "segment_count")} | {
                    "rounds": len(samples), "observed_steps": len(component_values[key]),
                    "duration_ms_mean": statistics.fmean(values) if complete else None,
                    "duration_ms_p50": statistics.median(values) if complete else None,
                    "duration_ms_p90": quantile(values, .9) if complete else None,
                    "duration_ms_max": max(values) if complete else None,
                    "share_of_parent": sum(values) / parent_total if complete and parent_total else None})
            summary.sort(key=lambda row: (-(row["duration_ms_mean"] or 0.), row["label"], row["component_key"]))
            stage_report = {"stage_key": stage_key, "label": label, "rounds": len(samples),
                "observed_steps": sum(sample["status"] == "MATCHED" for sample in samples),
                "status": "MATCHED" if complete else "PARTIAL" if any(sample["status"] == "MATCHED" for sample in samples) else "UNAVAILABLE",
                "per_step": samples, "summary": summary,
                "duration_ms_p50": statistics.median(sample["duration_ms"] for sample in samples) if complete else None,
                "duration_ms_p90": quantile([sample["duration_ms"] for sample in samples], .9) if complete else None,
                "duration_ms_max": max(sample["duration_ms"] for sample in samples) if complete else None}
            for field in ("duration_ms", "recorded_busy_ms", "no_recorded_ms"):
                values = [sample[field] for sample in samples]
                stage_report[field + "_mean"] = statistics.fmean(values) if complete and all(value is not None for value in values) else None
            stage_reports.append(stage_report)
    write_csv(step_path, step_export)
    return {"version": "1", "rank": rank, "scope": RESIDUAL_FOCUS_SCOPE,
            "status": "MATCHED" if stage_reports and all(stage["status"] == "MATCHED" for stage in stage_reports) else "PARTIAL",
            "stages": stage_reports, "operators": operators, "segments_csv": str(segment_path), "steps_csv": str(step_path)}


def residual_focus_substages(parents, tasks):
    """Refine observed input and metadata paths; never invent missing anchors."""
    result = []
    input_names = (("zero_kv", "KV块清零", "_zero_kv_blocks_kernel"),
                   ("apply_write", "应用待写入KV", "_apply_write_kernel"),
                   ("expand_mapping", "展开请求与token映射", "_expand_idx_mapping_kernel"),
                   ("positions", "准备位置与序列长度", "_prepare_pos_seq_lens_kernel"),
                   ("tokens", "拼接采样与草稿token", "_combine_sampled_and_draft_tokens_kernel"),
                   ("block_tables", "聚合KV块表", "_gather_block_tables_kernel"),
                   ("slot_mapping", "计算KV slot映射", "_compute_slot_mappings_kernel"))
    metadata_pattern = re.compile(r"SparseAttnSharedkvMetadata|MixedQuantSparseFlashMlaMetadata|QuantLightningIndexer(?:V\d+)?Metadata", re.I)
    for parent in parents:
        key = parent["stage_key"]
        if key not in ("main_inputs", "main_metadata", "draft_metadata"):
            continue
        left, right = int(parent["start_ns"]), int(parent["end_ns"])
        inside = sorted((task for task in tasks if left <= task.start_ns < right and task.end_ns > task.start_ns),
                        key=lambda task: (task.start_ns, task.end_ns, task.row_index))

        def child(suffix, label, order, a, b, first=None, following=None, status="MATCHED", reason=None):
            return {**parent, "stage_key": key + "_" + suffix, "label": label, "level": 1,
                    "parent_stage_key": key, "stage_order": parent["stage_order"] * 1000 + order,
                    "start_ns": str(a), "end_ns": str(b), "duration_ns": b - a, "duration_ms": (b - a) / 1e6,
                    "status": status, "reason": reason,
                    "start_anchor": {"name": first.name, "row_index": first.row_index, "timestamp_ns": str(a)} if first else parent["start_anchor"],
                    "end_anchor": {"name": following.name, "row_index": following.row_index, "timestamp_ns": str(b)} if following else parent["end_anchor"],
                    "boundary_basis": (first.name + (".end" if a == first.end_ns else ".start") if first else "父阶段.start")
                        + " → " + (following.name + (".end" if b == following.end_ns else ".start") if following else "父阶段.end")
                        + "；子区间已含在父阶段中"}

        try:
            rows = []
            if key == "main_inputs":
                anchors = []
                for number, (suffix, label, name) in enumerate(input_names):
                    found = [task for task in inside if task.name == name]
                    if (number >= 2 and len(found) != 1) or (number == 1 and len(found) > 1):
                        raise ValueError("missing or ambiguous next-main input anchor " + name)
                    if found:
                        if any(task.end_ns > right for task in found):
                            raise ValueError("input anchor exceeds its parent interval: " + name)
                        anchors.append((suffix, label, found))
                if not anchors or anchors[0][2][0].start_ns != left or any(
                        max(task.end_ns for task in a[2]) > b[2][0].start_ns for a, b in zip(anchors, anchors[1:])):
                    raise ValueError("input substage anchors are overlapping or reversed")
                for number, (suffix, label, found) in enumerate(anchors):
                    following = anchors[number + 1][2][0] if number + 1 < len(anchors) else None
                    rows.append(child(suffix, label, number, found[0].start_ns,
                                      following.start_ns if following else right, found[0], following))
            else:
                anchors = [task for task in inside if metadata_pattern.search(str(getattr(task, "op_type", "")) + " " + task.name)]
                if not anchors or any(task.end_ns > right for task in anchors) or any(
                        a.end_ns > b.start_ns for a, b in zip(anchors, anchors[1:])):
                    raise ValueError("missing, overlapping or out-of-bounds metadata execution anchors")
                rows.append(child("mapping_0", "首个metadata前的映射与准备", 0, left, anchors[0].start_ns,
                                  following=anchors[0]))
                for number, task in enumerate(anchors, 1):
                    label = "Indexer metadata本体" if "indexer" in (str(getattr(task, "op_type", "")) + " " + task.name).lower() else "Attention metadata本体"
                    rows.append(child("op_" + str(number), label + " #" + str(number), number * 2 - 1,
                                      task.start_ns, task.end_ns, task, task))
                    following = anchors[number] if number < len(anchors) else None
                    rows.append(child("mapping_" + str(number), "metadata后的映射与准备 #" + str(number), number * 2,
                                      task.end_ns, following.start_ns if following else right, task, following))
            if sum(row["duration_ns"] for row in rows) != right - left or any(
                    row["duration_ns"] < 0 for row in rows):
                raise ValueError("substage partition does not close to its parent")
            result.extend(row for row in rows if row["duration_ns"] > 0)
        except ValueError as error:
            result.append(child("unresolved", parent["label"] + "（细分未识别）", 0, left, right,
                                status="UNRESOLVED", reason=str(error)))
    return result


def residual_stage_plan(window, tasks, case=None):
    """Resolve ordered execution phases from names, shapes and dependency order.

    Supported anchors describe the observed MRV2 DFlash/DSpark path. A different
    path stays explicitly unresolved rather than inheriting labels by position.
    """
    case = case or {}
    left, right = residual_ns(window["start_ns"]), residual_ns(window["end_ns"])
    tasks = sorted((task for task in tasks if task.start_ns < right and task.end_ns > left
                    and task.end_ns > task.start_ns), key=lambda task: (task.start_ns, task.end_ns, task.row_index))

    def named(pattern):
        matcher = re.compile(pattern, re.I)
        return [task for task in tasks if matcher.search(task.name)]

    def one(values, label):
        if len(values) != 1:
            raise ValueError(label + ": expected one anchor, observed " + str(len(values)))
        return values[0]

    def dims(task, field="input_shapes"):
        return [shape for shape in a_dims(str(getattr(task, field, ""))) if shape]

    def dtypes(task, field="input_dtypes"):
        return [re.sub(r"[^A-Z0-9]", "", value.upper())
                for value in str(getattr(task, field, "")).split(";")]

    def integer64(value):
        return value in ("INT64", "DTINT64", "TORCHINT64")

    def norm_tasks():
        # Both native CANN names and explicit exported operator types occur.
        # FusedRmsNorm/MhcRmsNorm already contain RmsNorm; underscore spelling
        # is an additional known RMSNorm name, not a generic "norm" fallback.
        pattern = re.compile(r"Rms_?Norm", re.I)
        return [task for task in tasks if pattern.search(task.name) or pattern.search(task.op_type)]

    def norm_links(matrix):
        """Exact direct shape, or one explicit row-selection shape chain.

        Main forward can normalize all/padded tokens before the runner selects
        logits_indices and calls LMHead.  A shared hidden width alone cannot
        identify that path: require the selector's actual input AND output.
        """
        activation = dims(matrix)[0]
        candidates = norm_tasks()
        direct = [task for task in candidates if task.end_ns <= matrix.start_ns
                  and dims(task) and dims(task)[0] == activation]
        if direct:
            return direct, []
        links = []
        for norm in candidates:
            inputs, outputs = dims(norm), dims(norm, "output_shapes")
            if not (left <= norm.start_ns < norm.end_ns <= matrix.start_ns
                    and inputs and outputs and len(inputs[0]) == len(activation) == 2
                    and outputs[0] == inputs[0] and inputs[0][1] == activation[1]
                    and inputs[0][0] > activation[0] > 0 and activation[1] > 0):
                continue
            for select in tasks:
                family = re.sub(r"[^a-z0-9]", "", select.op_type.lower())
                # IndexCheck only checks indices and has no selected payload.
                # Embedding/Gather is a separate semantic anchor, never a
                # fallback for an unidentified hidden-state selection.
                if (family not in ("index", "gather", "gatherv2")
                        or "embedding" in select.name.lower()
                        or "indexcheck" in select.name.lower()
                        or not norm.end_ns <= select.start_ns < select.end_ns <= matrix.start_ns):
                    continue
                raw_inputs = a_dims(str(getattr(select, "input_shapes", "")))
                indexed = [(number, shape) for number, shape in enumerate(raw_inputs) if shape]
                outputs = dims(select, "output_shapes")
                if not indexed or indexed[0][0] != 0 or indexed[0][1] != inputs[0] or outputs != [activation]:
                    continue
                extra = [shape for _, shape in indexed[1:]]
                rows = [activation[0]]
                if family == "index":
                    # Native Index can expose two small integer metadata
                    # inputs before the index vector (observed 1;rank;M).
                    # Their values are not available in CSV; do not interpret
                    # them as an axis or invent arbitrary multi-input chains.
                    signature = extra == [rows] or extra == [[1], [2], rows]
                else:
                    signature = extra == [rows] or extra == [rows, [1]]
                if not signature:
                    continue
                types = dtypes(select)
                if any(number < len(types) and types[number]
                       and not (integer64(types[number]) or types[number] in ("INT32", "DTINT32", "TORCHINT32"))
                       for number, _ in indexed[1:]):
                    continue
                links.append((norm, select))
        return [], links

    def norm_failure(matrix, links):
        def brief(shapes):
            return "[" + ";".join(",".join(map(str, shape[:4])) + (",..." if len(shape) > 4 else "")
                                   for shape in shapes[:2]) + (";..." if len(shapes) > 2 else "") + "]"
        visible = norm_tasks()
        sample = "; ".join(task.name[:80] + " in=" + brief(dims(task))
                           + " out=" + brief(dims(task, "output_shapes")) for task in visible[-4:]) or "none"
        return ("ambiguous final Norm row-selection chain" if len(links) > 1 else
                "missing final hidden-width Norm before vocabulary projection") + (
                    "; projection activation=" + brief(dims(matrix)[:1])
                    + "; visible Norm candidates=" + str(len(visible)) + " {" + sample + "}"
                    + "; proven row-selection chains=" + str(len(links)))

    def after(task):
        number = tasks.index(task) + 1
        if number == len(tasks):
            raise ValueError("missing task after anchor " + task.name)
        return tasks[number]

    def anchor(task, stamp, boundary):
        return ({"row_index": task.row_index, "name": task.name, "timestamp_ns": str(stamp)} if task else
                {"name": boundary, "timestamp_ns": str(stamp)})

    def record(key, label, order, start, end, first, following, *, level=0, parent=None, status="MATCHED", reason=None):
        return {"round_id": window["round_id"], "window_key": window["window_key"],
                "stage_key": key, "stage_order": order, "label": label, "level": level,
                "parent_stage_key": parent, "start_ns": str(start), "end_ns": str(end),
                "duration_ns": end - start, "duration_ms": (end - start) / 1e6, "status": status,
                "reason": reason, "start_anchor": anchor(first, start, "previous HCPost.end"),
                "end_anchor": anchor(following, end, "next HCPre.start"),
                "boundary_basis": (first.name + ".start" if first else "上一层链末HCPost.end") + " → "
                    + (following.name + ".start" if following else "下一层链首HCPre.start")
                    + "；连续跨度，含段内间隔"}

    try:
        embedding_gather = re.compile(r"aclnnEmbedding.*Gather", re.I)
        embeds = [task for task in tasks if embedding_gather.search(task.name)
                  or task.op_type.casefold() == "embedding"]
        if not embeds:
            raise ValueError("missing model Embedding/Gather anchor")
        embed = embeds[-1]
        table = dims(embed)[0]
        if len(table) != 2:
            raise ValueError("model Embedding table must identify vocabulary and hidden dimensions")
        vocab, hidden = table
        # A draft head can use a smaller vocabulary than the next main model's
        # embedding. Limit the head search to before sampling, then verify the
        # activation/weight/output relation instead of equating those vocabularies.
        if window["window_key"] == "main_to_draft":
            head_limit = one(named(r"^_compute_local_logits_stats_kernel$"), "logits statistics").start_ns
        else:
            earlier_embeds = [task for task in embeds if task is not embed]
            if not earlier_embeds:
                raise ValueError("missing Markov embedding anchors before next-main embedding")
            head_limit = earlier_embeds[0].start_ns
        if window["window_key"] == "main_to_draft":
            projections = [task for task in tasks if task.end_ns <= head_limit
                           and "matmul" in (task.op_type + task.name).lower()
                           and len(dims(task)) >= 2 and dims(task)[1] == [vocab, hidden]
                           and len(dims(task)[0]) == 2 and dims(task)[0][-1] == hidden
                           and dims(task, "output_shapes") == [[dims(task)[0][0], vocab]]]
        else:
            projections = [task for task in tasks if task.end_ns <= head_limit
                           and "matmul" in (task.op_type + task.name).lower()
                           and len(dims(task)) >= 2
                           and len(dims(task)[0]) == len(dims(task)[1]) == 2
                           and all(value > 0 for shape in dims(task)[:2] for value in shape)
                           and dims(task)[0][1] == dims(task)[1][1]
                           and dims(task, "output_shapes") == [[dims(task)[0][0], dims(task)[1][0]]]
                           and any(norm_links(task))]
        projection = one(projections, "vocabulary projection H->V")
        head_vocab = dims(projection, "output_shapes")[0][1]
        norms, links = norm_links(projection)
        if norms:
            norm = norms[-1]  # Preserve the established exact full-shape path.
        elif len(links) == 1:
            norm, _ = links[0]
        else:
            raise ValueError(norm_failure(projection, links))
        children = []
        if window["window_key"] == "main_to_draft":
            sample = one(named(r"^_compute_local_logits_stats_kernel$"), "logits statistics")
            post = one(named(r"^_post_update_kernel$"), "sample-state update")
            sampler = [sample] + [one(named(pattern), pattern) for pattern in
                (r"^_probabilistic_rejection_kernel$", r"^_resample_kernel$", r"^_categorical_finalize_kernel$")] + [post]
            if any(a.end_ns > b.start_ns for a, b in zip(sampler, sampler[1:])):
                raise ValueError("rejection/resample/finalize order conflicts with state update")
            prepares = named(r"^_prepare_dflash_inputs_kernel(?:_ascend)?$")
            scatter_write = re.compile(r"ScatterNdUpdateSk", re.I)
            writes = [task for task in tasks if scatter_write.search(task.name)
                      or task.op_type.casefold() == "kvcompressepilog"]
            if not prepares or not writes or len(writes) != case.get("draft_layers", len(writes)):
                raise ValueError("missing/inconsistent DFlash input or context-KV write anchors")
            # Context-KV preparation has its own Concat calls. Only the
            # post-verification, pre-input-preparation concat->projection chain
            # identifies auxiliary-feature fusion; a second valid chain is ambiguous.
            concats = [task for task in tasks if post.end_ns <= task.start_ns < prepares[0].start_ns
                       and re.fullmatch(r"Concat(?:D|V2)?", task.op_type, re.I)]
            pairs = []
            for candidate in concats:
                inputs, outputs = dims(candidate), dims(candidate, "output_shapes")
                for matrix in tasks:
                    if not (candidate.end_ns <= matrix.start_ns < matrix.end_ns <= prepares[0].start_ns
                            and "matmul" in (matrix.op_type + matrix.name).lower()):
                        continue
                    source, dest = dims(matrix), dims(matrix, "output_shapes")
                    if not (len(source) >= 2 and len(source[0]) == 2
                            and source[1] == [hidden, source[0][1]]
                            and dest == [[source[0][0], hidden]]):
                        continue
                    inferred = (len(inputs) > 1 and all(len(shape) == 2 and all(value > 0 for value in shape)
                                                       for shape in inputs))
                    if outputs and outputs != [source[0]]:
                        continue
                    if inferred and (any(shape[0] != source[0][0] for shape in inputs)
                                     or sum(shape[1] for shape in inputs) != source[0][1]):
                        continue
                    if not outputs and not inferred:
                        continue  # Opaque layouts do not establish this shape link.
                    pairs.append((candidate, matrix))
            concat, fusion = one(pairs, "auxiliary hidden-state concat/projection chain"
                            + " (scoped Concat=" + str(len(concats)) + ")")
            kv_projections = [task for task in tasks if prepares[-1].end_ns <= task.start_ns < writes[-1].start_ns
                              and "matmul" in (task.op_type + task.name).lower()]
            if (len(kv_projections) != len(writes) or any(
                    not a.end_ns <= b.start_ns for a, b in zip(kv_projections, writes))
                    or any(a.end_ns > b.start_ns for a, b in zip(writes, kv_projections[1:]))):
                raise ValueError("context-KV projections/writes do not form the expected sequence")
            # Packed context caches use an epilog instead of Scatter.
            # The proposer crops fused features to num_context before WKV;
            # their token counts need not equal the padded fusion output.
            # Payload, slots and in-place output must still agree with WKV.
            for matrix, write in zip(kv_projections, writes):
                if write.op_type.casefold() != "kvcompressepilog":
                    continue
                source, dest = dims(matrix), dims(matrix, "output_shapes")
                inputs, outputs = dims(write), dims(write, "output_shapes")
                types, output_types = dtypes(write), dtypes(write, "output_dtypes")
                def cache_dtype(value):
                    return "UINT8" if value in ("UINT8", "DTUINT8", "TORCHUINT8") else None
                cache_kind = cache_dtype(types[0]) if len(types) == 3 else None
                cache_shape = inputs[0] if len(inputs) == 3 else []
                fusion_shape = dims(fusion, "output_shapes")
                projection_shape = (len(source) >= 2 and len(source[0]) == len(source[1]) == 2
                                    and source[0][1] == source[1][1] == hidden
                                    and all(value > 0 for shape in source[:2] for value in shape))
                checks = [
                    ("context_rows/hidden", projection_shape and len(fusion_shape) == 1
                     and len(fusion_shape[0]) == 2 and fusion_shape[0][1] == hidden
                     and source[0][0] <= fusion_shape[0][0]),
                    ("shared_context_shape", projection_shape and all(
                     dims(other) and dims(other)[0] == source[0] for other in kv_projections)),
                    ("projection_output", projection_shape and dest == [[source[0][0], source[1][0]]]),
                    ("cache_layout", len(cache_shape) == 1 and cache_shape[0] > 0),
                    ("payload_shape", len(inputs) == 3 and len(dest) == 1 and inputs[1] == dest[0]),
                    ("slots_shape", len(inputs) == 3 and len(dest) == 1 and len(dest[0]) == 2
                     and inputs[2] == [dest[0][0]]),
                    ("cache_output_shape", bool(cache_shape) and outputs == [cache_shape]),
                    ("cache_dtype", cache_kind is not None),
                    ("payload_dtype", len(types) == 3
                     and types[1] in ("BF16", "DTBF16", "BFLOAT16", "DTBFLOAT16", "TORCHBFLOAT16")),
                    ("slots_dtype", len(types) == 3 and types[2] in ("INT32", "DTINT32", "TORCHINT32")),
                    ("cache_output_dtype", cache_kind is not None and len(output_types) == 1
                     and cache_dtype(output_types[0]) == cache_kind),
                ]
                failed = [label for label, valid in checks if not valid]
                if failed:
                    raise ValueError("KvCompressEpilog cache/payload/slots contradict context-KV projection"
                        + "; failed=" + ",".join(failed)
                        + "; fusion output=" + str(fusion_shape)[:160]
                        + "; projection row=" + str(matrix.row_index)
                        + " in=" + str(source)[:160] + " out=" + str(dest)[:160]
                        + "; epilog row=" + str(write.row_index)
                        + " in=" + str(inputs)[:200] + " out=" + str(outputs)[:160]
                        + "; types=" + str(types)[:160] + " -> " + str(output_types)[:100])
            if not projection.end_ns <= sample.start_ns < post.end_ns <= concat.start_ns < prepares[0].start_ns:
                raise ValueError("main output, verification and draft preparation anchors are out of order")
            metadata_start = after(writes[-1])
            boundaries = [None, norm, sample, concat, prepares[0], after(prepares[-1]), metadata_start, embed, None]
            phases = [("main_head", "末主输出归并与HC Head"), ("main_logits", "最终Norm、词表投影及采样前整理"),
                      ("verification", "拒绝采样与采样状态更新"), ("context_fusion", "草稿上下文特征融合"),
                      ("draft_inputs", "DFlash输入准备（按KV组）"), ("context_kv", "草稿Context KV投影与写入"),
                      ("draft_metadata", "草稿metadata及输入映射准备"), ("draft_embedding", "草稿Embedding与HC展开")]
        elif window["window_key"] == "draft_to_nextmain":
            zero = named(r"^_zero_kv_blocks_kernel$")
            slots = one(named(r"^_compute_slot_mappings_kernel$"), "next-main slot mappings")
            preparation = [one(named(pattern), pattern) for pattern in
                (r"^_expand_idx_mapping_kernel$", r"^_prepare_pos_seq_lens_kernel$",
                 r"^_combine_sampled_and_draft_tokens_kernel$", r"^_gather_block_tables_kernel$")] + [slots]
            apply = named(r"^_apply_write_kernel$")
            if (len(apply) > 1 or any(a.end_ns > b.start_ns for a, b in zip(preparation, preparation[1:]))
                    or any(task.end_ns > preparation[0].start_ns for task in zero + apply)
                    or apply and any(task.end_ns > apply[0].start_ns for task in zero)):
                raise ValueError("next-main KV/input preparation anchors are out of order")
            # No blocks to clear and no pending writes legitimately omit these
            # two kernels. Expand remains mandatory and bounds sampling then.
            preparation_start = min(zero + apply + preparation[:1], key=lambda task: task.start_ns)
            markov_tasks = [task for task in tasks if projection.end_ns <= task.start_ns < preparation_start.start_ns]
            gathers = [task for task in embeds if task is not embed and task in markov_tasks
                       and len(dims(task)[0]) == 2 and dims(task)[0][0] == vocab]
            argmaxes = [task for task in markov_tasks
                        if re.search(r"aclnnArgMax.*ArgMaxV2", task.name, re.I)
                        or re.fullmatch(r"ArgMax(?:V2)?", task.op_type, re.I)]
            counts = (" (Embedding=" + str(len(gathers)) + ", ArgMax=" + str(len(argmaxes))
                      + ", configured positions=" + str(case.get("draft_tokens", "not declared")) + ")")
            if any(re.fullmatch(r"_gumbel_sample_kernel", task.name) for task in markov_tasks):
                raise ValueError("Markov Gumbel sampling observed; greedy ArgMax/Cast/ViewCopy anchors do not identify this path" + counts)
            if not gathers or len(gathers) != case.get("draft_tokens", len(gathers)):
                raise ValueError("Markov embedding count contradicts draft positions" + counts)
            copies, lookup_shape = [], None
            for index, gather in enumerate(gathers):
                stop = gathers[index + 1].start_ns if index + 1 < len(gathers) else preparation_start.start_ns
                low = dims(gather)[0][1]
                lookup = dims(gather)
                if len(lookup) < 2 or len(lookup[1]) != 1 or lookup[1][0] <= 0 or low <= 0:
                    raise ValueError("Markov lookup requires a positive one-dimensional token-ID input" + counts)
                shape = [lookup[1][0], low]
                if (dims(gather, "output_shapes") and dims(gather, "output_shapes") != [shape]
                        or lookup_shape is not None and shape != lookup_shape):
                    raise ValueError("Markov lookup rows/rank contradict output or other positions" + counts)
                lookup_shape = shape
                matrices = [task for task in markov_tasks if gather.end_ns <= task.start_ns < task.end_ns <= stop
                            and "matmul" in (task.op_type + task.name).lower()
                            and len(dims(task)) >= 2 and dims(task)[0] == shape
                            and dims(task)[1] == [head_vocab, low]
                            and dims(task, "output_shapes") == [[shape[0], head_vocab]]]
                matrix = one(matrices, "Markov position " + str(index + 1) + " low-rank bias projection" + counts)
                shape = dims(matrix, "output_shapes")[0]
                argmax = one([task for task in argmaxes if matrix.end_ns <= task.start_ns < task.end_ns <= stop
                              and dims(task) and dims(task)[0] == shape
                              and dims(task, "output_shapes") == [[shape[0]]]],
                             "Markov position " + str(index + 1) + " vocabulary ArgMax" + counts)
                suffix = after(argmax)
                if re.search(r"aclnnArgMax.*Cast", suffix.name):
                    copy = after(suffix)
                    valid_suffix = (argmax.end_ns <= suffix.start_ns < suffix.end_ns <= copy.start_ns
                                    < copy.end_ns <= stop)
                else:
                    copy = suffix
                    copy_shapes, copy_outputs = dims(copy), dims(copy, "output_shapes")
                    copy_types, argmax_types = dtypes(copy), dtypes(argmax, "output_dtypes")
                    output_types = dtypes(copy, "output_dtypes")
                    # Native ArgMax can already return INT64. Direct ViewCopy
                    # is proven by the strided-copy source slot and dtypes;
                    # missing Cast with unknown dtype is still unresolved.
                    valid_suffix = (argmax.end_ns <= copy.start_ns < copy.end_ns <= stop
                        and len(argmax_types) == 1 and integer64(argmax_types[0])
                        and len(copy_shapes) == 8 and copy_shapes[4] == [shape[0]]
                        and len(copy_shapes[0]) == 1 and copy_shapes[0][0] >= shape[0]
                        and all(value == [1] for value in copy_shapes[1:4] + copy_shapes[5:])
                        and copy_outputs == [copy_shapes[0]] and len(copy_types) == 8
                        and all(integer64(value) for value in copy_types)
                        and len(output_types) == 1 and integer64(output_types[0]))
                if not re.search(r"aclnnInplaceCopy.*ViewCopy", copy.name) or not valid_suffix:
                    raise ValueError("unrecognized or out-of-order ArgMax token-buffer update suffix" + counts)
                copies.append(copy)
            markov_end = after(copies[-1])
            if not projection.end_ns <= gathers[0].start_ns or not markov_end.end_ns <= preparation_start.start_ns:
                raise ValueError("draft logits, Markov sampling and next-round anchors are out of order")
            metadata_start = after(slots)
            boundaries = [None, norm, gathers[0], markov_end, preparation_start, metadata_start, embed, None]
            phases = [("draft_head", "草稿输出归并与HC Head"), ("draft_logits", "草稿最终Norm与词表投影"),
                      ("markov", "逐位置Markov偏置与选token"), ("draft_results", "草稿结果整理与轮次交接"),
                      ("main_inputs", "下一主模型KV与输入准备"), ("main_metadata", "下一主metadata及输入映射准备"),
                      ("main_embedding", "下一主Embedding与HC展开")]
            for number, gather in enumerate(gathers):
                following = gathers[number + 1] if number + 1 < len(gathers) else markov_end
                children.append(record("markov_" + str(number + 1), "Markov位置" + str(number + 1), number + 1,
                    gather.start_ns, following.start_ns, gather, following, level=1, parent="markov"))
        else:
            raise ValueError("unsupported residual window kind")
        if not any(re.search(r"SparseAttnSharedkvMetadata|MixedQuantSparseFlashMlaMetadata", task.name)
                   and metadata_start.start_ns <= task.start_ns < embed.start_ns for task in tasks):
            raise ValueError("missing attention metadata between preparation and model embedding")
        stamps = [left] + [task.start_ns for task in boundaries[1:-1]] + [right]
        if any(a >= b for a, b in zip(stamps, stamps[1:])):
            raise ValueError("execution-stage boundaries are reversed, empty or outside the U window")
        result = [record(key, label, order, stamps[order - 1], stamps[order], boundaries[order - 1], boundaries[order])
                  for order, (key, label) in enumerate(phases, 1)]
        return result + children + residual_focus_substages(result, tasks)
    except (ValueError, IndexError, AttributeError) as error:
        return [record("unresolved", "未识别过渡阶段（保留原窗口）", 1, left, right, None, None,
                       status="UNRESOLVED", reason=str(error))]


def residual_stage_summary(stages, windows, level):
    totals = collections.Counter(row["window_key"] for row in windows)
    groups = collections.defaultdict(list)
    for row in stages:
        if row["level"] == level:
            groups[(row["window_key"], row["stage_key"])].append(row)
    result = []
    fields = ("duration_ms", "recorded_busy_ms", "no_recorded_ms", "kernel_sum_ms", "task_count",
              "task_wait_sum_ms", "total_cost_ms", "small_task_count", "small_kernel_sum_ms", "largest_gap_ms") + RESIDUAL_HOST_NUMBERS + RESIDUAL_DMA_NUMBERS
    for (key, stage), rows in groups.items():
        complete = len(rows) == totals[key] and all(row["status"] == "MATCHED" for row in rows)
        item = {field: rows[0][field] for field in ("stage_key", "stage_order", "label", "level", "parent_stage_key", "boundary_basis")}
        reasons = list(dict.fromkeys(row["reason"] for row in rows if row.get("reason")))
        # Per-round reasons retain CSV row references. Do not concatenate
        # hundreds of distinct samples into an oversized Excel cell.
        reason = "; ".join(value[:1200] + ("..." if len(value) > 1200 else "") for value in reasons[:3])
        if len(reasons) > 3:
            reason += "; " + str(len(reasons) - 3) + " additional distinct reasons; see per-round stages CSV"
        item.update(window_key=key, rounds=totals[key], matched_windows=sum(row["status"] == "MATCHED" for row in rows),
                    status="MATCHED" if complete else "UNRESOLVED" if stage == "unresolved" else "PARTIAL",
                    duration_ns_total=sum(row["duration_ns"] for row in rows),
                    reason=reason or None)
        for field in fields:
            values = [row.get(field) for row in rows]
            item[field + "_mean"] = statistics.fmean(values) if complete and all(value is not None for value in values) else None
        item["duration_ms_p50"] = statistics.median(row["duration_ms"] for row in rows) if complete else None
        item["duration_ms_p90"] = quantile([row["duration_ms"] for row in rows], .9) if complete else None
        for field in RESIDUAL_SOURCE_COUNTS:
            values = [row[field] for row in rows]
            item[field + "_total"] = sum(values) if all(value is not None for value in values) else None
        for field in RESIDUAL_SOURCE_NUMBERS:
            values = [row[field] for row in rows]
            # A partial source result describes the measured subset. A missing
            # whole window stays unknown rather than silently reducing its cost.
            item[field + "_mean"] = statistics.fmean(values) if len(rows) == totals[key] and all(value is not None for value in values) else None
        statuses = [row["source_status"] for row in rows]
        item["source_status"] = ("COMPLETE" if len(rows) == totals[key] and all(value == "COMPLETE" for value in statuses) else
                                  "PARTIAL" if any(value in ("COMPLETE", "PARTIAL") for value in statuses) else "UNAVAILABLE")
        result.append(item)
    return sorted(result, key=lambda row: (row["window_key"], row["stage_order"]))


def layer_stage_plan(layer, tasks):
    """Ten exclusive anchor spans for main CSA/HCA and draft SWA HC layers.

    Compression and indexer work, when present, remains within attention
    preparation. Neither an indexer nor a compression kernel is a required
    anchor: the observed attention and two HC pairs define this partition.
    """
    attention_type = str(layer.get("attention") or "Attention")
    left, right = int(layer["start_ns"]), int(layer["end_ns"])
    inside = sorted((task for task in tasks if left <= task.start_ns < right and task.end_ns > task.start_ns),
                    key=lambda task: (task.start_ns, task.end_ns, task.row_index))

    def unique(pattern):
        found = [task for task in inside if pattern.search(task.label)]
        if len(found) != 1:
            raise ValueError("expected one " + pattern.pattern + ", observed " + str(len(found)))
        return found[0]

    def point(task, end=False):
        return {"row_index": task.row_index, "name": task.name,
                "timestamp_ns": str(task.end_ns if end else task.start_ns), "edge": "end" if end else "start"}

    try:
        pre = [task for task in inside if A_PRE.search(task.name) or A_PRE.search(task.op_type)]
        post = [task for task in inside if A_POST.search(task.name) or A_POST.search(task.op_type)]
        if len(pre) != 2 or len(post) != 2:
            raise ValueError(attention_type + " requires two ordered HCPre/HCPost pairs")
        attention = unique(re.compile(r"(?<![A-Za-z])(?:SparseAttnSharedkv|MixedQuantSparseFlashMla|SparseFlashAttention)(?![A-Za-z])", re.I))
        dispatch, combine = unique(A_DISPATCH), unique(A_COMBINE)
        chain = [pre[0], attention, post[0], pre[1], dispatch, combine, post[1]]
        if pre[0].start_ns != left or post[1].end_ns != right or any(
                a.end_ns > b.start_ns for a, b in zip(chain, chain[1:])):
            raise ValueError(attention_type + " anchors overlap, are reversed or disagree with the HC layer boundary")
        stamps = [left, attention.start_ns, attention.end_ns, post[0].start_ns, pre[1].start_ns,
                  dispatch.start_ns, dispatch.end_ns, combine.start_ns, combine.end_ns, post[1].start_ns, right]
        anchors = [point(pre[0]), point(attention), point(attention, True), point(post[0]), point(pre[1]),
                   point(dispatch), point(dispatch, True), point(combine), point(combine, True), point(post[1]), point(post[1], True)]
        prepare = {"CSA": "CSA HCPre与Q/KV、压缩/索引准备",
                   "HCA": "HCA HCPre与Q/KV、压缩准备"}.get(attention_type, attention_type + " HCPre与Q/KV准备")
        labels = [("attention_prepare", prepare), ("attention", attention_type + " attention本体"),
                  ("attention_output", attention_type + "输出投影与拷贝"), ("attention_post", "Attention HCPost与衔接"),
                  ("moe_prepare", "FFN HCPre与MoE路由准备"), ("dispatch", "MoE Dispatch"),
                  ("experts", "专家计算与Dispatch/Combine衔接"), ("combine", "MoE Combine"),
                  ("ffn_output", "共享专家与FFN归并"), ("ffn_post", "FFN HCPost收尾")]
        return [{"stage_key": key, "label": label, "stage_order": number + 1, "status": "MATCHED", "reason": None,
                 "start_ns": str(stamps[number]), "end_ns": str(stamps[number + 1]),
                 "duration_ns": stamps[number + 1] - stamps[number], "duration_ms": (stamps[number + 1] - stamps[number]) / 1e6,
                 "start_anchor": anchors[number], "end_anchor": anchors[number + 1],
                 "boundary_basis": "设备锚点连续跨度；含段内间隙，已包含于" + attention_type + "层，不额外计入周期或TPOT"}
                for number, (key, label) in enumerate(labels)]
    except ValueError as error:
        return [{"stage_key": "unresolved", "label": attention_type + "细分未识别（保留层实测总跨度）", "stage_order": 1,
                 "status": "UNRESOLVED", "reason": str(error), "start_ns": str(left), "end_ns": str(right),
                 "duration_ns": right - left, "duration_ms": (right - left) / 1e6,
                 "start_anchor": None, "end_anchor": None, "boundary_basis": "原始HC层边界；未猜测缺失语义锚点"}]


def swa_stage_plan(layer, tasks):
    """Compatibility entry point for the original draft SWA partition."""
    return layer_stage_plan({**layer, "attention": "SWA"}, tasks)


def layer_analyze_capture(capture, measured, detail_dir, *, with_swa_compat=False):
    """Measure every main CSA/HCA and draft SWA layer without re-estimating.

    The optional compatibility mode returns ``(all_layers, draft_swa)`` and
    writes both detail CSVs from the same measured rows. The CLI uses this mode
    so legacy consumers retain their SWA output without a second analysis.
    """
    index = ATimeIndex(capture["tasks"])
    groups, stages, stage_groups = collections.defaultdict(list), [], collections.defaultdict(list)
    for cycle in measured["rounds"]:
        for layer in cycle["layers"]:
            left, right = int(layer["start_ns"]), int(layer["end_ns"])
            tasks = index.overlapping(left, right)
            metrics = diagnostic_metrics(tasks, left, right)
            groups[(layer["role"], layer["layer"], layer["attention"])].append({"wall_ms": (right - left) / 1e6, **metrics})
            if (layer["role"], layer["attention"]) not in (("main", "CSA"), ("main", "HCA"), ("draft", "SWA")):
                continue
            rows = layer_stage_plan(layer, tasks)
            if sum(row["duration_ns"] for row in rows) != right - left:
                raise ValueError("Layer substages do not close to the original layer interval")
            for row in rows:
                a, b = int(row["start_ns"]), int(row["end_ns"])
                row.update(round_id=cycle["round_id"], role=layer["role"], layer=layer["layer"], attention=layer["attention"],
                           **diagnostic_metrics(index.overlapping(a, b), a, b))
                stages.append(row)
                stage_groups[(layer["role"], layer["layer"], layer["attention"], row["stage_key"])].append(row)
    layer_summary = []
    for (role, layer, attention), rows in groups.items():
        item = {"role": role, "layer": layer, "attention": attention, "rounds": len(rows),
                "reference_eligible": role != "main" or layer != 0,
                "wall_ms_p50": statistics.median(row["wall_ms"] for row in rows),
                "wall_ms_p90": quantile([row["wall_ms"] for row in rows], .9)}
        for field in ("wall_ms", "recorded_busy_ms", "no_recorded_ms", "kernel_sum_ms", "total_cost_ms"):
            values = [row[field] for row in rows]
            item[field + "_mean"] = statistics.fmean(values) if all(value is not None for value in values) else None
        layer_summary.append(item)
    summary = []
    for (role, layer, attention_type, key), rows in stage_groups.items():
        complete = len(rows) == len(measured["rounds"]) and all(row["status"] == "MATCHED" for row in rows)
        item = {field: rows[0][field] for field in ("role", "layer", "attention", "stage_key", "label", "stage_order")}
        item.update(rounds=len(measured["rounds"]), matched_windows=sum(row["status"] == "MATCHED" for row in rows),
                    status="MATCHED" if complete else "UNRESOLVED" if key == "unresolved" else "PARTIAL",
                    reason="; ".join(dict.fromkeys(row["reason"] for row in rows if row.get("reason"))) or None)
        for field in ("duration_ms", "recorded_busy_ms", "no_recorded_ms", "kernel_sum_ms", "total_cost_ms"):
            values = [row[field] for row in rows]
            item[field + "_mean"] = statistics.fmean(values) if complete and all(value is not None for value in values) else None
        item["duration_ms_p50"] = statistics.median(row["duration_ms"] for row in rows) if complete else None
        item["duration_ms_p90"] = quantile([row["duration_ms"] for row in rows], .9) if complete else None
        summary.append(item)
    rank, source = str(capture["rank"]), str(capture.get("path", ""))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rank).strip("._")[:60] or "rank"
    suffix = hashlib.sha256((rank + "\0" + source).encode()).hexdigest()[:10]
    destination = Path(detail_dir) / (safe + "-" + suffix + "-layer-stages.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_csv(destination, [{"rank": rank, **row} for row in stages])
    diagnostic = {"status": "MATCHED" if summary and all(row["status"] == "MATCHED" for row in summary) else "PARTIAL",
            "scope": "源采样同rank逐层HCPre→末HCPost；CSA/HCA/SWA子段已含于原层，wait和kernel_sum非互斥成本；不同token桶不可等同工作量",
            "stages_csv": str(destination), "layer_summary": layer_summary,
            "stage_summary": sorted(summary, key=lambda row: (row["role"], row["layer"], row["attention"], row["stage_order"]))}
    if not with_swa_compat:
        return diagnostic
    swa_summary = [row for row in diagnostic["stage_summary"] if row["role"] == "draft" and row["attention"] == "SWA"]
    swa_destination = Path(detail_dir) / (safe + "-" + suffix + "-swa-stages.csv")
    write_csv(swa_destination, [{"rank": rank, **row} for row in stages
                                if row["role"] == "draft" and row["attention"] == "SWA"])
    swa = {"status": "MATCHED" if swa_summary and all(row["status"] == "MATCHED" for row in swa_summary) else "PARTIAL",
           "scope": "源采样同rank逐层HCPre→末HCPost；SWA子段已含于原层，wait和kernel_sum非互斥成本；不同token桶不可等同工作量",
           "stages_csv": str(swa_destination), "layer_summary": layer_summary, "stage_summary": swa_summary}
    return diagnostic, swa


def swa_analyze_capture(capture, measured, detail_dir):
    """Compatibility report retaining only draft SWA stage rows and CSV."""
    _, swa = layer_analyze_capture(capture, measured, detail_dir, with_swa_compat=True)
    return swa


def residual_analyze_capture(capture, measured, detail_dir, trace_path=None, progress=None, case=None):
    """Measure ordered execution-stage spans inside the original HC residuals.

    Invalid HC boundaries raise ValueError. Missing, malformed or unaligned
    optional host traces produce DEVICE_ONLY and unknown host numeric fields.
    """
    directory = Path(detail_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rank = str(capture.get("rank", measured.get("rank", "rank")))
    source_path = str(capture.get("path", ""))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rank).strip("._")[:60] or "rank"
    suffix = hashlib.sha256((rank + "\0" + source_path).encode()).hexdigest()[:10]
    prefix = safe + "-" + suffix
    device = directory / (prefix + "-device-tasks.csv")
    host = directory / (prefix + "-host-events.csv")
    residual_progress(progress, "统计HC周期主草稿切换和轮次尾部")
    windows = residual_windows(measured)
    index = ATimeIndex(sorted(capture["tasks"], key=lambda task: (task.start_ns, task.end_ns)))
    uncovered = residual_device_rows(capture, windows, index, device)
    stages = []
    for window in windows:
        rows = residual_stage_plan(window, index.overlapping(residual_ns(window["start_ns"]), residual_ns(window["end_ns"])), case)
        parents = [row for row in rows if row["level"] == 0]
        if (sum(row["duration_ns"] for row in parents) != residual_ns(window["end_ns"]) - residual_ns(window["start_ns"])
                or any(a["end_ns"] != b["start_ns"] for a, b in zip(parents, parents[1:]))):
            raise ValueError("execution stages do not continuously close to the original U window")
        window.update(stage_closed=True, stage_status=parents[0]["status"],
                      stage_reason=parents[0].get("reason") if parents[0]["status"] != "MATCHED" else None)
        for row in rows:
            left, right = residual_ns(row["start_ns"]), residual_ns(row["end_ns"])
            tasks = index.overlapping(left, right)
            clips = [(max(left, task.start_ns), min(right, task.end_ns)) for task in tasks]
            union = a_union(clips)
            busy_ns = sum(b - a for a, b in union)
            row.update(recorded_busy_ms=busy_ns / 1e6, no_recorded_ms=(right - left - busy_ns) / 1e6,
                       kernel_sum_ms=sum(b - a for a, b in clips) / 1e6, task_count=len(clips),
                       _uncovered=residual_complement(left, right, union), _task_rows=[task.row_index for task in tasks],
                       source_status="UNAVAILABLE", **dict.fromkeys(RESIDUAL_HOST_NUMBERS + RESIDUAL_SOURCE_COUNTS + RESIDUAL_SOURCE_NUMBERS))
            row.update(diagnostic_metrics(tasks, left, right))
            row["source_device_tasks"] = len(tasks)
        stages.extend(rows)
    path = Path(trace_path) if trace_path is not None else Path(source_path).parent / "trace_view.json"
    path = path.resolve()
    trace = {"status": "NOT_FOUND", "path": str(path), "sha256": None}
    host_path = None
    sources = {"source_links_csv": None, "source_status": "UNAVAILABLE", "source_scope": RESIDUAL_SOURCE_SCOPE}
    if path.suffix == ".gz":
        trace.update(status="UNSUPPORTED_GZIP", reason="provide the uncompressed trace_view.json")
    else:
        try:
            before = path.stat()
            residual_progress(progress, "扫描trace元数据与设备时钟锚点")
            scanned, offset = residual_trace_scan(path, capture["tasks"], progress)
            trace.update(scanned)
            if offset is not None:
                residual_progress(progress, "流式导出U窗口内主机事件")
                sources = residual_host_rows(path, capture, windows, uncovered, trace, offset, host, directory, progress, stages)
                host_path = str(host)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("trace changed during residual analysis")
        except FileNotFoundError as error:
            trace.update(status="NOT_FOUND", reason=str(error))
            host_path = None
            for row in windows + stages:
                row.update(dict.fromkeys(RESIDUAL_HOST_NUMBERS))
        except (OSError, ValueError, ArithmeticError, UnicodeError, sqlite3.Error) as error:
            trace.update(status="UNAVAILABLE", reason=str(error))
            host_path = None
            for row in windows + stages:
                row.update(dict.fromkeys(RESIDUAL_HOST_NUMBERS))
    if host_path is None:
        sources = {"source_links_csv": None, "source_status": "UNAVAILABLE", "source_scope": RESIDUAL_SOURCE_SCOPE}
        for row in stages:
            row.update(dict.fromkeys(RESIDUAL_SOURCE_COUNTS[1:] + RESIDUAL_SOURCE_NUMBERS), source_status="UNAVAILABLE")
        for row in windows + stages:
            row.update(dict.fromkeys(RESIDUAL_DMA_NUMBERS), trace_dma_status="UNAVAILABLE")
    stage_path = directory / (prefix + "-stages.csv")
    focus = residual_focus_diagnostics(stages, windows, index, rank, directory, prefix)
    residual_progress(progress, "按有序执行阶段导出过渡耗时，核对逐窗口闭合")
    fields = tuple(key for key in stages[0] if not key.startswith("_"))
    with residual_csv(stage_path, ("rank", *fields)) as writer:
        for row in stages:
            row.pop("_uncovered", None)
            row.pop("_task_rows", None)
            writer.writerow({"rank": rank, **{key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}})
    summary = []
    for key in ("main_to_draft", "draft_to_nextmain"):
        rows = [window for window in windows if window["window_key"] == key]
        record = {"window_key": key, "label": rows[0]["label"], "rounds": len(rows), "scope": RESIDUAL_SCOPE}
        for field in ("duration_ms", "recorded_busy_ms", "no_recorded_ms", "kernel_sum_ms", "task_count") + RESIDUAL_HOST_NUMBERS + RESIDUAL_DMA_NUMBERS:
            values = [row[field] for row in rows]
            record[field + "_mean"] = statistics.fmean(values) if all(value is not None for value in values) else None
        summary.append(record)
    return {"version": RESIDUAL_VERSION, "status": "DEVICE_AND_HOST" if host_path is not None else "DEVICE_ONLY", "scope": RESIDUAL_SCOPE,
            "rank": rank, "capture_path": source_path, "capture_sha256": capture.get("sha256"),
            "windows": windows, "summary": summary, "device_tasks_csv": str(device), "host_events_csv": host_path,
            "stages_csv": str(stage_path), "stage_summary": residual_stage_summary(stages, windows, 0),
            "substage_summary": residual_stage_summary(stages, windows, 1), "stage_basis": RESIDUAL_STAGE_BASIS,
            "focus_diagnostics": focus,
            **sources,
            "trace": trace, "residual_ms_mean": statistics.fmean(windows[index]["cycle_residual_ms"] for index in range(0, len(windows), 2))}



def atomic_text(path, text):
    with atomic_writer(path) as stream:
        stream.write(text)



def atomic_json(path, value, progress=None):
    """Stream a complete core report without constructing its full JSON string."""
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, allow_nan=False)
    with atomic_writer(path) as stream:
        chunks, size, last = [], 0, time.monotonic()
        for chunk in encoder.iterencode(value):
            chunks.append(chunk)
            size += len(chunk)
            if size < 256 * 1024:
                continue
            stream.write(''.join(chunks))
            chunks, size = [], 0
            now = time.monotonic()
            if progress is not None and now - last >= 5:
                stream.flush()
                progress(f'写入 {Path(path).name}：已写 {stream.tell() / 1024 / 1024:.1f} MiB')
                last = now
        stream.write(''.join(chunks))
        stream.write('\n')



def write_csv(path, rows, progress=None):
    """Write incrementally; a callable supplies two repeatable row scans."""
    if callable(rows):
        row_factory = rows
    else:
        if not isinstance(rows, (list, tuple)):
            rows = tuple(rows)
        row_factory = lambda: iter(rows)
    fields, last = {}, time.monotonic()
    for index, row in enumerate(row_factory(), 1):
        fields.update(dict.fromkeys(row))
        now = time.monotonic()
        if progress is not None and now - last >= 5:
            progress(f'写入 {Path(path).name}：扫描字段，已检查 {index:,} 行')
            last = now
    with atomic_writer(path, encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields) or ['status'])
        writer.writeheader()
        for index, row in enumerate(row_factory(), 1):
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})
            now = time.monotonic()
            if progress is not None and now - last >= 5:
                progress(f'写入 {Path(path).name}：已写 {index:,} 行')
                last = now



def displayed_methods(case):
    return (case.get('projection') or {}).get('layer_mean_methods', [])


def result_value(method, field):
    if method.get('status') != LAYER_MEAN_STATUS or method.get('reasons'):
        return None
    value = method.get(field)
    return value if type(value) in (int, float) and math.isfinite(value) else None


def print_case_result(result):
    for method in displayed_methods(result):
        cycle, rate = result_value(method, 'cycle_ms'), result_value(method, 'per_device_tokens_s')
        text = (f'{rate:.6f} token/s/卡；周期={cycle:.6f}ms；rank={method["selected_rank"]}'
                if rate is not None else '不可计算：' + ' | '.join(method.get('reasons', [])))
        print('  ' + method['name'] + ': ' + text, flush=True)


def result_rows(report):
    rows = []
    for case in report['cases']:
        config, target = case.get('config') or {}, case.get('target_model') or {}
        for method in displayed_methods(case):
            rows.append({'采样组合': case['name'], '每DP请求BS': config.get('bs'),
                         '草稿token数S': config.get('draft_tokens'),
                         '有效输出Neff(含bonus)': config.get('effective_tokens_per_round'),
                         '配置平均草稿接受比例': (config.get('acceptance') if config.get('acceptance_semantics')
                                           == 'mean_accepted_draft_fraction' else None),
                         '目标EP': config.get('target_ep'),
                         '数据范围': '目标' + str((target.get('main') or {}).get('num_hidden_layers', '未知'))
                                    + '主＋' + str((target.get('draft') or {}).get('num_hidden_layers', '未知')) + '草稿条件估算',
                         '方法': method['name'], '周期(ms)': result_value(method, 'cycle_ms'),
                         '单卡吞吐(token/s/卡)': result_value(method, 'per_device_tokens_s'),
                         'TPOT(ms)': result_value(method, 'tpot_ms'), '源参考rank': method.get('selected_rank'),
                         '状态': '条件估算' if result_value(method, 'cycle_ms') is not None else '缺少实际输入',
                         '口径与原因': ' | '.join(method.get('reasons', [])) or
                         '剔除主0参考；逐类型均值外推，先求各rank轮次均值再取最大；衔接及轮次剩余区间分别计入。'})
    rows.extend({'采样组合': failure['case'], '状态': 'FAILED', '口径与原因': failure['error']}
                for failure in report.get('failures', []))
    return rows



LAYER_MEAN_STATUS = "LAYER_MEAN_ESTIMATE"
LAYER_MEAN_ASSUMPTIONS = [
    "Per-rank/per-round means group by model role and attention kind, pooling hash and normal routing; main layer0 is excluded from reference means by default.",
    "A layer wall is its first HCPre start to final HCPost end, not proof of complete Python-layer ownership. Target depth multiplies these observed wall means.",
    "Retain every observed same-role boundary once; each added layer contributes the same-role reference boundary mean, excluding edges touching excluded main layers.",
    "Cross-role/tail residual equals the measured cycle minus both complete source chain spans and is retained once. Excluded source layer0 is never moved into this residual.",
    "Method1 target Dispatch/Combine=observed duration*remote logical share ratio*role TopK ratio; add only the delta, plus a fixed 14us per Dispatch/Combine target call; no fixed/payload split.",
    "Only named routed GroupedMatmul/Ggmm kernels between Dispatch completion and Combine start scale with TopK at equal efficiency; independent activations are not included.",
    "Packed FP4/MXFP shapes may be opaque. Native operator name plus routed phase is an explicit subset assumption, not proof of a fully decomposed MLP.",
    "H, M, local expert count and quantization remain fixed. Router/expert-width changes are retained in the source layer mean without a separate target delta.",
    "All modeled increments are exposed. Target overlap, routing, padding, graph capacity, context, network contention and stationarity are not measured.",
    "Throughput uses configured new tokens per request per round, including bonus, and explicit DP/physical cards; it is not measured accepted-token throughput.",
]




def _lm_number(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or positive and value <= 0:
        raise ValueError(label + " must be " + ("positive " if positive else "") + "finite")
    return float(value)



def _lm_exact_ns(value):
    return a_exact_ns(value, "HC timestamp")



def _lm_union_ms(intervals):
    return sum(right - left for left, right in a_union(intervals)) / 1e6



def _lm_gmm(task):
    label = re.sub(r"[^a-z0-9]", "", task.label.lower())
    return ("groupedmatmul" in label or "ggmm" in label
            or bool(re.search(r"(?:mxfp[48]|fp[48]).*gmm|gmm.*(?:mxfp[48]|fp[48])", label)))



def _lm_native_layer(tasks, starts, layer, model, ep, require_compute=True):
    left, right = _lm_exact_ns(layer["start_ns"]), _lm_exact_ns(layer["end_ns"])
    inside = tasks[bisect.bisect_left(starts, left):bisect.bisect_left(starts, right)]
    dispatch = [task for task in inside if re.search("MoeDistributeDispatch", task.label, re.I)]
    combine = [task for task in inside if re.search("MoeDistributeCombine", task.label, re.I)]
    if len(dispatch) != 1 or len(combine) != 1:
        raise ValueError("layer-mean requires exactly one native Dispatch and Combine in each reference layer")
    dispatch, combine = dispatch[0], combine[0]
    if not left <= dispatch.start_ns < dispatch.end_ns <= combine.start_ns < combine.end_ns <= right:
        raise ValueError("reference Dispatch/Combine ordering or HC interval bounds conflict")
    named = [task for task in inside if _lm_gmm(task)]
    if any(task.start_ns < dispatch.end_ns or task.end_ns > combine.start_ns for task in named):
        raise ValueError("named routed GMM crosses Dispatch/Combine phase boundaries")
    if not named and require_compute:
        raise ValueError("ROUTED_GMM_UNAVAILABLE: no GroupedMatmul/Ggmm kernel in the Dispatch-to-Combine reference phase")
    local = model["n_routed_experts"] // ep
    logical = 0
    for task in named:
        shapes = [shape for shape in a_dims(task.input_shapes) if shape]
        if len(shapes) >= 2 and len(shapes[0]) == 2 and len(shapes[1]) >= 3 and shapes[1][0] == local:
            logical += 1
    duration = _lm_union_ms((task.start_ns, task.end_ns) for task in named) if named else None
    if duration is not None:
        _lm_number(duration, "routed native GMM union", True)
    evidence = {"role": layer["role"], "layer": layer["layer"], "attention": layer["attention"],
                "names": list(dict.fromkeys(task.name for task in named)),
                "operator_names": list(dict.fromkeys(task.label for task in named)),
                "shapes": list(dict.fromkeys(task.input_shapes for task in named)),
                "input_shapes": list(dict.fromkeys(task.input_shapes for task in named)),
                "matched_calls": len(named), "logical_expert_axis_matches": logical,
                "duration_ms": duration, "routed_compute_ms": duration,
                "row_indices": [getattr(task, "row_index", None) for task in named],
                "phase_basis": "native named GMM subset fully inside Dispatch.end -> Combine.start; independent SwiGlu excluded",
                "shape_basis": ("unidentified_source_compute; unchanged TopK implies zero delta only" if not named else
                                "logical expert axis available" if logical == len(named) else
                                "name_and_routed_phase_assumption_for_opaque_or_packed_shapes")}
    return {"wall_ms": (right - left) / 1e6,
            "dispatch_ms": (dispatch.end_ns - dispatch.start_ns) / 1e6,
            "combine_ms": (combine.end_ns - combine.start_ns) / 1e6,
            "routed_compute_ms": duration}, evidence



_LM_TARGET_STAGE_KEYS = ("attention_prepare", "attention", "attention_output", "attention_post",
                         "moe_prepare", "dispatch", "experts", "combine", "ffn_output", "ffn_post")


_LM_OPERATOR_FIELDS = ("name", "op_type", "input_shapes", "output_shapes", "input_dtypes", "output_dtypes",
                       "stream_id", "task_type")
_LM_OPERATOR_NOTE = (
    "源子项为阶段内整数ns裁剪后的互斥记录覆盖，含并发集合和无设备记录间隙；跨层/跨阶段裁剪仅计覆盖，不证明业务归属。"
    "间隙不等同CPU耗时，记录覆盖不证明关键路径或可节省时间。"
    "成员instances保留同指纹并发实例数；同步/控制记录仍计入覆盖，不等同计算忙时。"
    "source_ms、kernel_sum_ms与calls_mean先按每轮全部参考层平均，再按本rank全部保留轮次平均；未出现指纹计确定0，缺证据不补0。"
    "kernel_sum_ms是每参考层轮次的累计裁剪记录时间均值，可因并发超过阶段跨度，不可相加为周期；"
    "duration_mean_ms/p90是所有实际观察到的阶段内裁剪调用的合并分布，不含未出现调用。"
    "目标仅保留源覆盖并另列父估算的具名模型增减项，不向单个kernel分摊；负增量保留，非实测优化收益。"
)

_LM_PMU_NOTE = (
    "PMU仅为源采样实际kernel整次调用的指标，按整次Duration加权；不乘目标层数、通信或TopK倍率。"
    "阶段裁剪只改变耗时账本，不能测出该子区间的局部PMU；跨阶段引用同次调用时不可相加。"
    "覆盖率分母为本项实际相交的唯一原始调用及其整次Duration，逐指标另列已知/总样本；"
    "未采集、非法或冲突值保持未知，观测到的0保留。流水可重叠，比例不能相加；"
    "bound为保守的流水倾向，非因果证明；MTE高不等于HBM饱和，cube_utilization不是MFU。"
)
_LM_PMU_SOURCES = [
    "https://github.com/mindstudio-docs/master/blob/master/msprof/docs/zh/user_guide/profile_data_file_references.md",
    "https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0009.html",
]


def _lm_operator_pmu_observation(task, start, end, scope):
    raw = getattr(task, "pmu", None)
    raw = raw if isinstance(raw, dict) else {}
    # Identity excludes the stage and clipping bounds: one native call may
    # intersect multiple reference intervals, but must not bias its PMU mean.
    identity = tuple(scope.get(field) for field in ("rank",)) + tuple(getattr(task, field, None)
        for field in ("row_index", "start_ns", "end_ns", "stream_id", "task_id"))
    if isinstance(raw, _ANativePmu):
        metrics, invalid = raw["metrics"], raw["invalid_fields"]
    else:
        metrics = {}
        raw_metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
        for field, value in raw_metrics.items():
            if field not in _A_PMU_FIELDS or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(value) and value >= 0 and (_A_PMU_FIELDS[field] != "ratio" or value <= 1):
                metrics[field] = value
        invalid = [field for field in raw.get("invalid_fields", ()) or () if field in _A_PMU_FIELDS]
        for field in invalid:
            metrics.pop(field, None)
    return {"identity": identity, "duration_ns": task.end_ns - task.start_ns,
            "clipped": start != task.start_ns or end != task.end_ns,
            "metrics": metrics, "invalid_fields": invalid,
            "core_type": str(raw.get("core_type") or getattr(task, "task_type", None) or "")}


def _lm_pmu_core_kind(core_type, metrics):
    text = re.sub(r"[^A-Z0-9]", "", core_type.upper())
    if "AICPU" in text:
        return "aicpu"
    if "MIX" in text:
        return "mix"
    if "VECTOR" in text or text in ("AIV", "AIVECTORCORE"):
        return "aiv"
    if text in ("AICORE", "AIC", "CUBE", "AICUBECORE"):
        return "aic"
    # Names alone do not prove the active accelerator. Positive exported
    # counters can supply a conservative fallback; all-zero counters cannot.
    aic = any(metrics.get("aic_" + pipe + "_ratio", 0) > 0 for pipe in ("mac", "scalar", "mte1", "mte2", "mte3", "fixpipe"))
    aiv = any(metrics.get("aiv_" + pipe + "_ratio", 0) > 0 for pipe in ("vec", "scalar", "mte2", "mte3"))
    return "mix" if aic and aiv else "aic" if aic else "aiv" if aiv else "unknown"


def _lm_pmu_pipeline_direction(prefix, metrics):
    """Conservative heuristics; these thresholds are not hardware limits."""
    compute = prefix + ("_mac_ratio" if prefix == "aic" else "_vec_ratio")
    moves = [prefix + "_" + pipe + "_ratio" for pipe in ("mte1", "mte2", "mte3", "fixpipe")]
    scalar = prefix + "_scalar_ratio"
    values = {field: metrics[field]["value"] for field in (compute, *moves, scalar) if field in metrics}
    if not values or max(values.values()) == 0:
        return "unknown", "未观察到有效流水活动", []
    def shown(field):
        return field + "=" + format(values[field] * 100, ".1f") + "%"
    evidence = [shown(field) for field in sorted(values, key=values.get, reverse=True)[:3]]
    comp = values.get(compute)
    mover = max((field for field in moves if field in values), key=values.get, default=None)
    transfer = values[mover] if mover else None
    compute_name = "矩阵计算" if prefix == "aic" else "向量计算"
    if comp is not None and transfer is not None and min(comp, transfer) >= .85 and abs(comp - transfer) < .10:
        return "mixed", compute_name + "/搬运同时高占用", evidence
    if (comp is not None and comp >= .90 and transfer is not None and comp - transfer >= .10
            and comp >= values.get(scalar, 0) + .10):
        return "compute", "偏" + compute_name + "流水限制", evidence
    if (transfer is not None and transfer >= .75 and comp is not None and transfer - comp >= .15
            and transfer >= values.get(scalar, 0) + .10):
        name = ("读入搬运" if "mte2" in mover else "写出搬运" if "mte3" in mover else "片内搬运")
        return "transfer", "偏" + name + "流水限制", evidence
    if values.get(scalar, 0) >= .60 and values[scalar] - max(comp or 0, transfer or 0) >= .10:
        return "scalar", "标量/控制流水突出", evidence
    if comp is not None and comp >= .75:
        return "busy_compute", compute_name + "流水偏忙，bound尚不确定", evidence
    if transfer is not None and transfer >= .75:
        return "busy_transfer", "搬运流水突出，bound尚不确定", evidence
    return "unknown", "无单一流水显著主导，需核对依赖/分核/启动开销", evidence


def _lm_operator_pmu_bound(member, metrics, coverage, samples):
    label = " ".join(str(member.get(field) or "") for field in ("name", "op_type", "task_type")).upper()
    result = {"code": "unknown", "label": "PMU证据不足，bound未知", "confidence": "低", "evidence": []}
    evidence = result["evidence"]
    if re.search(r"MOEDISTRIBUTE|HCCL|ALLTOALL|ALLGATHER|ALLREDUCE|REDUCESCATTER|SENDRECV", label):
        result.update(code="communication_mixed", label="通信/路由/同步复合任务，PMU不能拆出通信瓶颈")
        evidence.append("需结合跨rank时间线和通信明细；不能由MAC低推断HBM或CPU瓶颈")
        return result
    if member.get("record_kind") == "synchronization_control":
        result.update(code="synchronization_control", label="同步/控制任务，计算搬运bound不适用")
        return result
    kinds = {_lm_pmu_core_kind(sample["core_type"], sample["metrics"]) for sample in samples}
    if kinds == {"aicpu"}:
        result.update(code="aicpu", label="AICPU任务，AIC/AIV流水不能判断服务器CPU瓶颈")
        return result
    cores = set()
    if "mix" in kinds or "aic" in kinds:
        cores.add("aic")
    if "mix" in kinds or "aiv" in kinds:
        cores.add("aiv")
    directions = [(core, *_lm_pmu_pipeline_direction(core, metrics)) for core in sorted(cores)]
    for field, title in (("aicore_time_us", "AIC核内时间"), ("aiv_time_us", "AIV核内时间")):
        if field in metrics:
            evidence.append(title + "=" + format(metrics[field]["value"], ".3f") + "μs（整调用Duration加权；非互斥墙钟时间）")
    for core, _, message, values in directions:
        evidence.append(core.upper() + "：" + message + ("；" + "，".join(values) if values else ""))
    usable = [(core, code, message) for core, code, message, _ in directions if code != "unknown"]
    if len(usable) == 1:
        core, code, message = usable[0]
        result.update(code=core + "_" + code, label=core.upper() + " " + message)
    elif usable:
        result.update(code="mixed_core", label="AIC/AIV联合任务：" + "；".join(core.upper() + " " + message for core, _, message in usable))
        evidence.append("AIC与AIV的比例分母不同，不能直接相加或凭比例比较谁决定总时长")
    elif directions and any(value["value"] > 0 for field, value in metrics.items() if field.endswith("_ratio")):
        result["label"] = "流水无明确主导，需查依赖/分核/调度，bound待确认"
    if usable:
        result["confidence"] = "中"
    relevant = [field for field in metrics if any(field.startswith(core + "_") for core in cores)
                and field.endswith("_ratio")]
    for core in sorted(cores):
        required = [core + ("_mac_ratio" if core == "aic" else "_vec_ratio"), core + "_mte2_ratio", core + "_mte3_ratio"]
        if any(field not in metrics for field in required):
            evidence.append(core.upper() + " 计算/读入/写出指标未采齐，方向判断受限")
            result["confidence"] = "低"
    if coverage["duration_coverage"] < .8 or any(metrics[field]["duration_coverage"] < .8 for field in relevant):
        evidence.append("PMU已知样本Duration覆盖不足80%；缺失指标未补0")
        result["confidence"] = "低"
    total = sum(sample["duration_ns"] for sample in samples)
    short = sum(sample["duration_ns"] for sample in samples if sample["duration_ns"] <= 20000)
    if total and short / total >= .5:
        evidence.append("至少半数耗时来自≤20μs短任务，启动/调度/响应开销需单独核对")
        result["confidence"] = "低"
    gap = 0
    for sample in samples:
        duration_us = sample["duration_ns"] / 1000
        kind = _lm_pmu_core_kind(sample["core_type"], sample["metrics"])
        fields = {"aic": ("aicore_time_us",), "aiv": ("aiv_time_us",),
                  "mix": ("aicore_time_us", "aiv_time_us")}.get(kind, ())
        times = [sample["metrics"].get(field) for field in fields]
        # An unrelated accelerator's time says nothing about the active core.
        # A MIX task can spend its whole duration on the unobserved half; both
        # core times are required before contrasting its task/core spans.
        if not times or any(value is None for value in times):
            continue
        active = [value for value in times if value is not None and value > 0]
        if active and max(active) < .65 * duration_us and duration_us - max(active) > 3:
            gap += sample["duration_ns"]
    if total and gap / total >= .5:
        result.update(code="task_core_time_gap", label="任务跨度显著大于核内时间，需查调度/分核不均/同步", confidence="低")
        evidence.append("aicore_time/aiv_time是按分核周期推导的时间，并非任务Duration；差值不能直接归因等待或CPU")
    if any(metrics[field].get("max", 0) - metrics[field].get("min", 0) >= .35 for field in relevant):
        evidence.append("同指纹调用的流水比例差异≥35个百分点，均值可能掩盖不同执行状态")
        result["confidence"] = "低"
    if coverage["clipped_calls"]:
        evidence.append("存在跨父阶段裁剪：本表引用整次调用PMU，并非该阶段局部测量")
    return result


def _lm_operator_pmu_summary(member, observations):
    # The same physical call can be encountered in two adjacent reference
    # layers/stages. Deduplicate before weighting by its original Duration.
    unique = {}
    for item in observations:
        if item["identity"] not in unique:
            unique[item["identity"]] = dict(item)
        else:
            unique[item["identity"]]["clipped"] |= item["clipped"]
    samples = list(unique.values())
    total_ns = sum(sample["duration_ns"] for sample in samples)
    known = [sample for sample in samples if any(field.endswith("_ratio") or field in ("aicore_time_us", "aiv_time_us")
                                              for field in sample["metrics"])]
    known_ns = sum(sample["duration_ns"] for sample in known)
    coverage = {"known_calls": len(known), "total_calls": len(samples),
                "known_duration_ms": known_ns / 1e6, "total_duration_ms": total_ns / 1e6,
                "duration_coverage": known_ns / total_ns if total_ns else 0.,
                "clipped_calls": sum(sample["clipped"] for sample in samples),
                "invalid_calls": sum(bool(sample["invalid_fields"]) for sample in samples)}
    metrics = {}
    for field in _A_PMU_FIELDS:
        available = [sample for sample in samples if field in sample["metrics"]]
        if not available:
            continue
        duration_ns = sum(sample["duration_ns"] for sample in available)
        if not duration_ns:
            continue
        values = [sample["metrics"][field] for sample in available]
        metrics[field] = {"value": math.fsum(sample["metrics"][field] * sample["duration_ns"] / duration_ns for sample in available),
            "unit": _A_PMU_FIELDS[field], "known_calls": len(available), "total_calls": len(samples),
            "known_duration_ms": duration_ns / 1e6, "total_duration_ms": total_ns / 1e6,
            "duration_coverage": duration_ns / total_ns if total_ns else 0., "min": min(values), "max": max(values)}
    return {"metrics": metrics, "coverage": coverage,
            "bound": _lm_operator_pmu_bound(member, metrics, coverage, samples),
            "scope_note": _LM_PMU_NOTE, "sources": _LM_PMU_SOURCES,
            "core_types": sorted({sample["core_type"] for sample in samples if sample["core_type"]}),
            "aggregation": "native_whole_call_duration_weighted",
            "heuristics": "方向阈值为审慎提示规则：计算≥90%且领先搬运≥10个百分点；搬运≥75%且领先计算≥15个百分点；非硬件bound认证"}


def _lm_operator_member(task, time_index):
    values = tuple(getattr(task, field, None if field in ("stream_id", "task_type") else "")
                   for field in _LM_OPERATOR_FIELDS)
    # Missing export metadata is unknown, not grounds to discard a timed task.
    values = tuple(None if value is None else str(value) for value in values)
    cache = getattr(time_index, "_operator_members", None)
    if cache is None:
        cache = time_index._operator_members = {}
    if values not in cache:
        key = "op_" + hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()
        member = dict(zip(_LM_OPERATOR_FIELDS, values))
        label = re.sub(r"[^A-Z0-9]", "", " ".join(str(member.get(field) or "")
                             for field in ("name", "op_type", "task_type")).upper())
        member.update(key=key, record_kind=("synchronization_control" if any(token in label for token in
            ("EVENTWAIT", "EVENTRECORD", "NOTIFYWAIT", "NOTIFYRECORD", "CONDITIONSQE",
             "STREAMWAITEVENT", "SYNCHRONIZE")) else "device_record"))
        cache[values] = member
    return cache[values]


def _lm_operator_example(task, start, end, scope):
    return {**scope, "row_index": getattr(task, "row_index", None),
            "row_indices": list(getattr(task, "row_indices", [getattr(task, "row_index", None)])),
            "start_ns": str(start), "end_ns": str(end), "clip_start_ns": str(start), "clip_end_ns": str(end),
            "duration_ns": end - start, "task_start_ns": str(task.start_ns), "task_end_ns": str(task.end_ns)}


def _lm_operator_neighbor(time_index, timestamp, *, before):
    if before:
        if not hasattr(time_index, "_operator_ends"):
            time_index._operator_end_tasks = sorted(time_index.tasks,
                key=lambda task: (task.end_ns, task.start_ns, task.row_index))
            time_index._operator_ends = [task.end_ns for task in time_index._operator_end_tasks]
        index = bisect.bisect_right(time_index._operator_ends, timestamp) - 1
        task = time_index._operator_end_tasks[index] if index >= 0 else None
    else:
        index = bisect.bisect_left(time_index.starts, timestamp)
        task = time_index.tasks[index] if index < len(time_index.tasks) else None
    if task is None:
        return None
    return {**_lm_operator_member(task, time_index),
            **_lm_operator_example(task, task.start_ns, task.end_ns, {})}


def _lm_operator_stage(stage, time_index, scope):
    """Exact half-open coverage; active multiplicities preserve parallel calls."""
    left, right = _lm_exact_ns(stage["start_ns"]), _lm_exact_ns(stage["end_ns"])
    result = {"status": "UNAVAILABLE", "reason": None, "rows": [], "kernels": {}}
    try:
        if right < left:
            raise ValueError("operator stage has reversed bounds")
        events, kernels, members, examples = collections.defaultdict(collections.Counter), {}, {}, collections.defaultdict(list)
        events[left]; events[right]
        for task in time_index.overlapping(left, right):
            a, b = max(left, _lm_exact_ns(task.start_ns)), min(right, _lm_exact_ns(task.end_ns))
            if b <= a:
                continue
            member = _lm_operator_member(task, time_index)
            key = member["key"]
            members[key] = member
            events[a][key] += 1
            events[b][key] -= 1
            examples[key].append((a, b, _lm_operator_example(task, a, b, scope)))
            if key not in kernels:
                kernels[key] = {"member": member, "durations_ns": [],
                    "example": _lm_operator_example(task, a, b, scope), "pmu_observations": []}
            kernels[key]["durations_ns"].append(b - a)
            kernels[key]["pmu_observations"].append(_lm_operator_pmu_observation(task, a, b, scope))
        active, rows = collections.Counter(), {}
        points = sorted(events)
        for index, a in enumerate(points):
            active.update(events[a])
            if any(value < 0 for value in active.values()):
                raise ValueError("operator sweep has a negative active count")
            active = collections.Counter({key: value for key, value in active.items() if value})
            if index + 1 == len(points):
                if active:
                    raise ValueError("operator sweep did not finish at the stage boundary")
                break
            b = points[index + 1]
            if b <= a:
                continue
            signature = tuple(sorted(active.items()))
            kind = "gap" if not signature else "operator" if sum(active.values()) == 1 else "parallel"
            key = "gap" if not signature else kind + "_" + hashlib.sha256(
                json.dumps(signature).encode()).hexdigest()
            if key not in rows:
                names = [(members[identity]["name"] or members[identity]["op_type"] or "未导出名称")
                         + (" ×" + str(count) if count > 1 else "") for identity, count in signature]
                label = ("无设备任务记录区间（非CPU归因）" if kind == "gap" else
                         ("并发记录：" if kind == "parallel" else "") + " + ".join(names))
                rows[key] = {"key": key, "label": label[:240], "kind": kind, "duration_ns": 0,
                    "members": [dict(members[identity], instances=count) for identity, count in signature],
                    "example": None}
            if rows[key]["example"] is None or b - a > rows[key]["example"]["duration_ns"]:
                if signature:
                    records = []
                    for identity, count in signature:
                        matching = [value for start, end, value in examples[identity] if start < b and end > a]
                        if len(matching) != count:
                            raise ValueError("operator instance evidence disagrees with active multiplicities")
                        records.extend(dict(value, key=identity, start_ns=str(a), end_ns=str(b),
                                            clip_start_ns=str(a), clip_end_ns=str(b), duration_ns=b - a)
                                       for value in matching)
                    example = dict(records[0], records=records)
                else:
                    example = {**scope, "start_ns": str(a), "end_ns": str(b), "clip_start_ns": str(a), "clip_end_ns": str(b),
                        "duration_ns": b - a, "before": _lm_operator_neighbor(time_index, a, before=True),
                        "after": _lm_operator_neighbor(time_index, b, before=False)}
                rows[key]["example"] = example
            rows[key]["duration_ns"] += b - a
        if sum(row["duration_ns"] for row in rows.values()) != right - left:
            raise ValueError("operator coverage does not close to the exact stage span")
        result.update(status="MATCHED", rows=list(rows.values()), kernels=kernels)
    except (ValueError, KeyError, TypeError, ArithmeticError) as error:
        result["reason"] = str(error)
    return result


def _lm_operator_source_means(source_rounds):
    """Private cached source evidence; no per-round operator payload is exported."""
    output = {}
    keys = {key for cycle in source_rounds for key in cycle.get("class_stage_values", {})}
    for group in keys:
        samples = [cycle.get("class_stage_values", {}).get(group, []) for cycle in source_rounds]
        references = [sample.get("layer") for sample in samples[0]]
        for stage_key in _LM_TARGET_STAGE_KEYS:
            item = {"status": "UNAVAILABLE", "reason": None, "rows": [], "kernels": [],
                    "reference_layers": references, "rounds": len(source_rounds)}
            output[group, stage_key] = item
            try:
                if not references or any([sample.get("layer") for sample in cycle] != references for cycle in samples):
                    raise ValueError("operator references disagree with retained layers/rounds")
                evidence = []
                for cycle in samples:
                    for sample in cycle:
                        stage = sample.get("_operator_stages", {}).get(stage_key)
                        if not stage or stage.get("status") != "MATCHED":
                            raise ValueError((stage or {}).get("reason") or "operator stage evidence unavailable")
                        evidence.append(stage)
                denominator = len(evidence)
                rows, kernels = {}, {}
                for stage in evidence:
                    for row in stage["rows"]:
                        if row["key"] not in rows:
                            rows[row["key"]] = dict(row, duration_ns=0)
                        elif row["example"]["duration_ns"] > rows[row["key"]]["example"]["duration_ns"]:
                            rows[row["key"]]["example"] = row["example"]
                        rows[row["key"]]["duration_ns"] += row["duration_ns"]
                    for key, kernel in stage["kernels"].items():
                        if key not in kernels:
                            kernels[key] = {"member": kernel["member"], "durations_ns": [], "example": kernel["example"],
                                            "pmu_observations": []}
                        kernels[key]["durations_ns"].extend(kernel["durations_ns"])
                        kernels[key]["pmu_observations"].extend(kernel.get("pmu_observations", []))
                item["rows"] = [{key: value for key, value in row.items() if key != "duration_ns"}
                                | {"source_ms": row["duration_ns"] / denominator / 1e6} for row in rows.values()]
                item["kernels"] = [dict(kernel["member"],
                    calls_mean=len(kernel["durations_ns"]) / denominator,
                    kernel_sum_ms=sum(kernel["durations_ns"]) / denominator / 1e6,
                    duration_mean_ms=statistics.fmean(kernel["durations_ns"]) / 1e6,
                    duration_p90_ms=quantile(kernel["durations_ns"], .9) / 1e6,
                    example=kernel["example"], pmu=_lm_operator_pmu_summary(kernel["member"], kernel["pmu_observations"]))
                    for kernel in kernels.values()]
                item["kernels"].sort(key=lambda row: (-row["kernel_sum_ms"], row["key"]))
                item.update(status="MATCHED", layer_round_samples=denominator)
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                item["reason"] = str(error)
    return output


def _lm_operator_target_groups(groups, source_means, cycle_ms):
    """Expand final selected-rank groups without changing any parent number."""
    output = []
    model_labels = {"dispatch": "模型增减：Dispatch通信倍率及固定附加项",
                    "combine": "模型增减：Combine通信倍率及固定附加项",
                    "experts": "模型增减：具名路由GMM按TopK变化（未证明全部暴露）"}
    for group in groups:
        stages = []
        for stage in group["stage_rows"]:
            source = source_means.get(((group["role"], group["attention"]), stage["stage_key"]), {})
            breakdown = {"status": "UNAVAILABLE", "reason": None, "note": _LM_OPERATOR_NOTE,
                         "rows": [], "kernels": []}
            try:
                if (source.get("status") != "MATCHED" or source.get("reference_layers") != group["reference_layers"]
                        or source.get("rounds") != group["rounds"]):
                    raise ValueError(source.get("reason") or "operator sample identities disagree with the parent")
                count = group["target_layers"]
                rows = [dict(row, delta_ms=0., target_ms=row["source_ms"],
                             target_total_ms=count * row["source_ms"],
                             cycle_share=count * row["source_ms"] / cycle_ms) for row in source["rows"]]
                delta = stage["delta_ms"]
                if delta:
                    if stage["stage_key"] not in model_labels:
                        raise ValueError("unexpected parent delta without a named operator model")
                    rows.append({"key": "model_delta_" + stage["stage_key"],
                        "label": model_labels[stage["stage_key"]], "kind": "model_delta", "source_ms": 0.,
                        "delta_ms": delta, "target_ms": delta, "target_total_ms": count * delta,
                        "cycle_share": count * delta / cycle_ms, "members": [], "example": None})
                for field in ("source_ms", "delta_ms", "target_ms", "target_total_ms"):
                    if not math.isclose(math.fsum(row[field] for row in rows), stage[field], rel_tol=1e-12, abs_tol=1e-8):
                        raise ValueError("operator rows do not close to parent " + field)
                rows.sort(key=lambda row: (row["kind"] == "model_delta", -row["source_ms"], row["key"]))
                breakdown.update(status="MATCHED", rows=rows, kernels=source["kernels"],
                    reference_layers=group["reference_layers"], rounds=group["rounds"],
                    layer_round_samples=source["layer_round_samples"])
                if group.get("note"):
                    breakdown["note"] += " " + group["note"]
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                breakdown["reason"] = str(error)
            stages.append(dict(stage, operator_breakdown=breakdown))
        output.append(dict(group, stage_rows=stages))
    return output


def _lm_source_layer_stages(layer, tasks, starts, native, *, time_index=None, round_id=None, rank=None):
    """Optional stage evidence, without changing the parent layer estimator."""
    entry = {"layer": layer["layer"], "status": "UNAVAILABLE", "reason": None,
             "note": None, "stage_rows": []}
    try:
        left, right = _lm_exact_ns(layer["start_ns"]), _lm_exact_ns(layer["end_ns"])
        inside = tasks[bisect.bisect_left(starts, left):bisect.bisect_left(starts, right)]
        stages = layer_stage_plan(layer, inside)
        if (len(stages) != len(_LM_TARGET_STAGE_KEYS)
                or tuple(row["stage_key"] for row in stages) != _LM_TARGET_STAGE_KEYS
                or any(row["status"] != "MATCHED" for row in stages)):
            raise ValueError("; ".join(row.get("reason") or "ten native stage anchors unavailable" for row in stages
                                       if row.get("status") != "MATCHED") or "ten native stages unavailable")
        cursor = left
        for row in stages:
            a, b = _lm_exact_ns(row["start_ns"]), _lm_exact_ns(row["end_ns"])
            if a != cursor or b < a or row["duration_ns"] != b - a:
                raise ValueError("reference stage intervals are not an exclusive continuous partition")
            cursor = b
        if cursor != right or sum(row["duration_ns"] for row in stages) != right - left:
            raise ValueError("reference ten stages do not close to the source layer wall")
        experts = stages[_LM_TARGET_STAGE_KEYS.index("experts")]
        expert_left, expert_right = _lm_exact_ns(experts["start_ns"]), _lm_exact_ns(experts["end_ns"])
        named = [task for task in inside if _lm_gmm(task)]
        if any(task.start_ns < expert_left or task.end_ns > expert_right for task in named):
            raise ValueError("reference routed GMM is outside the experts substage")
        routed = native["routed_compute_ms"]
        if routed is not None:
            observed = _lm_union_ms((task.start_ns, task.end_ns) for task in named)
            if (not math.isclose(observed, routed, rel_tol=1e-12, abs_tol=1e-8)
                    or routed > experts["duration_ms"] + 1e-8):
                raise ValueError("reference routed GMM union conflicts with the experts substage")
        else:
            entry["note"] = ("源 routed_compute_ms 未识别，保持未知；仅在父估算允许的 TopK 不变条件下沿用零计算增量，"
                             "不把未知 GMM 耗时补零。")
        entry.update(status="MATCHED", stage_rows=[{"stage_key": row["stage_key"],
            "stage_order": row["stage_order"], "label": row["label"], "source_ms": row["duration_ms"]} for row in stages])
        time_index = time_index if time_index is not None else ATimeIndex(tasks)
        scope = {"rank": rank, "round_id": round_id, "layer": layer["layer"]}
        entry["_operator_stages"] = {row["stage_key"]: _lm_operator_stage(row, time_index, scope) for row in stages}
    except (ValueError, KeyError, TypeError, ArithmeticError) as error:
        entry["reason"] = str(error)
    return entry


def _lm_target_stage_breakdowns(classes, groups, cycle_ms, *, rounds=1, layer_samples=False):
    """Explain the existing target class means; never recompute cycle totals.

    A single round first averages its reference layers.  Rank results then
    average those round means, using the exact same retained rounds as the
    parent estimator.  Only native D/C/GMM deltas change the ten source spans.
    """
    output = []
    for parent in classes:
        role, attention = parent["role"], parent["attention"]
        references = list(parent["reference_layers"])
        count = parent["target_layers"]
        entry = {"role": role, "attention": attention, "status": "UNAVAILABLE", "reason": None,
                 "note": "细分是父层估算的展开，已包含于目标周期；不得再次加到周期、TPOT 或吞吐。",
                 "reference_layers": references, "target_layers": count, "rounds": rounds,
                 "source_wall_ms": parent["wall_ms"], "target_wall_ms": parent["target_wall_ms"],
                 "target_total_ms": count * parent["target_wall_ms"], "stage_rows": []}
        try:
            samples = groups.get((role, attention), [])
            if layer_samples:
                if [row.get("layer") for row in samples] != references:
                    raise ValueError("native stage references are missing or disagree with parent reference_layers")
            elif len(samples) != rounds or any(row.get("reference_layers") != references for row in samples):
                raise ValueError("stage rounds/reference layers disagree with the retained parent estimate")
            if not samples or any(row.get("status") != "MATCHED" for row in samples):
                reasons = [row.get("reason") or "reference stage evidence unavailable" for row in samples
                           if row.get("status") != "MATCHED"]
                raise ValueError("; ".join(dict.fromkeys(reasons)) or "reference stage evidence unavailable")
            for sample in samples:
                if tuple(row["stage_key"] for row in sample["stage_rows"]) != _LM_TARGET_STAGE_KEYS:
                    raise ValueError("reference stage keys/order do not match the ten-stage definition")
            delta_fields = {"dispatch": "dispatch_delta_ms", "combine": "combine_delta_ms",
                            "experts": "routed_compute_delta_ms"}
            rows = []
            for index, key in enumerate(_LM_TARGET_STAGE_KEYS):
                source_ms = statistics.fmean(_lm_number(sample["stage_rows"][index]["source_ms"], "source stage")
                                             for sample in samples)
                delta = parent[delta_fields[key]] if key in delta_fields else 0.
                target_ms = source_ms + delta
                if source_ms < 0 or not math.isfinite(target_ms) or target_ms < 0:
                    raise ValueError("source or target stage duration is invalid")
                total = count * target_ms
                rows.append({"stage_key": key, "stage_order": index + 1,
                             "label": samples[0]["stage_rows"][index]["label"],
                             "source_ms": source_ms, "delta_ms": delta, "target_ms": target_ms,
                             "target_total_ms": total, "cycle_share": total / cycle_ms})
            for field, expected in (("source_ms", parent["wall_ms"]), ("target_ms", parent["target_wall_ms"]),
                                    ("target_total_ms", entry["target_total_ms"])):
                if not math.isclose(math.fsum(row[field] for row in rows), expected, rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError("ten target stages do not close to parent " + field)
            if any(not 0 <= row["cycle_share"] <= 1 for row in rows):
                raise ValueError("target stage cycle share is outside the parent cycle")
            notes = list(dict.fromkeys(sample["note"].replace(entry["note"], "").strip()
                                      for sample in samples if sample.get("note")))
            notes = [note for note in notes if note]
            entry.update(status="MATCHED", stage_rows=rows)
            if notes:
                entry["note"] += " " + " ".join(notes)
        except (ValueError, KeyError, TypeError, ArithmeticError) as error:
            entry["reason"] = str(error)
            entry["note"] += " 子段不可用时仍保留父层估算总量，不将未知子段补零。"
        output.append(entry)
    return output



def _lm_source_round(cycle, tasks, starts, model, case, source_schedule, excluded, compute_required,
                     *, time_index=None, rank=None):
    start, end = _lm_exact_ns(cycle["start_ns"]), _lm_exact_ns(cycle["end_ns"])
    if end <= start:
        raise ValueError("layer-mean source cycle must have positive exact duration")
    duration = (end - start) / 1e6
    if not math.isclose(_lm_number(cycle["duration_ms"], "source cycle", True), duration, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("source cycle duration disagrees with exact timestamps")
    layers = sorted(cycle["layers"], key=lambda value: _lm_exact_ns(value["start_ns"]))
    expected = [(role, number, attention) for role, number, attention, _ in source_schedule]
    observed = [(layer.get("role"), layer.get("layer"), layer.get("attention")) for layer in layers]
    if observed != expected:
        raise ValueError("layer-mean source schedule is missing, duplicated or inconsistent with model roles/attention")
    previous = start
    for layer in layers:
        left, right = _lm_exact_ns(layer["start_ns"]), _lm_exact_ns(layer["end_ns"])
        if not start <= left < right <= end or left < previous:
            raise ValueError("source HC layers overlap or cross the retained cycle")
        wall = (right - left) / 1e6
        if not math.isclose(_lm_number(layer["wall_ms"], "source HC wall", True), wall, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("source HC wall disagrees with exact timestamps")
        previous = right
    time_index = time_index if time_index is not None else ATimeIndex(tasks)
    roles, class_values, evidence = {}, collections.defaultdict(list), []
    class_stage_values = collections.defaultdict(list)
    for role in ("main", "draft"):
        selected = [layer for layer in layers if layer["role"] == role]
        if not selected:
            raise ValueError("source chain missing for " + role)
        raw_wall = math.fsum((_lm_exact_ns(layer["end_ns"]) - _lm_exact_ns(layer["start_ns"])) / 1e6 for layer in selected)
        gaps, reference_gaps = [], []
        for left, right in zip(selected, selected[1:]):
            gap = (_lm_exact_ns(right["start_ns"]) - _lm_exact_ns(left["end_ns"])) / 1e6
            gaps.append(gap)
            if role != "main" or left["layer"] not in excluded and right["layer"] not in excluded:
                reference_gaps.append(gap)
        chain = (_lm_exact_ns(selected[-1]["end_ns"]) - _lm_exact_ns(selected[0]["start_ns"])) / 1e6
        if not math.isclose(raw_wall + math.fsum(gaps), chain, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("source layer walls plus gaps do not close the " + role + " chain")
        roles[role] = {"source_layer_envelopes_ms": raw_wall, "source_chain_ms": chain,
                       "source_gap_ms": math.fsum(gaps), "source_gap_count": len(gaps),
                       "reference_gap_mean_ms": statistics.fmean(reference_gaps) if reference_gaps else 0.,
                       "reference_gap_count": len(reference_gaps), "source_layers": len(selected)}
        for layer in selected:
            if role == "main" and layer["layer"] in excluded:
                continue
            values, proof = _lm_native_layer(tasks, starts, layer, model, case["base_ep"], compute_required[role])
            class_values[(role, layer["attention"])].append((layer["layer"], values))
            class_stage_values[(role, layer["attention"])].append(_lm_source_layer_stages(
                layer, tasks, starts, values, time_index=time_index, round_id=cycle.get("round_id"), rank=rank))
            evidence.append({"round_id": cycle.get("round_id"), **proof})
    span_sum = math.fsum(value["source_chain_ms"] for value in roles.values())
    residual = duration - span_sum
    if residual < -1e-8:
        raise ValueError("source main/draft chains exceed the retained cycle")
    residual = max(0., residual)
    excluded_wall = math.fsum((_lm_exact_ns(layer["end_ns"]) - _lm_exact_ns(layer["start_ns"])) / 1e6
                             for layer in layers if layer["role"] == "main" and layer["layer"] in excluded)
    partition = {"source_cycle_ms": duration, "source_main_chain_ms": roles["main"]["source_chain_ms"],
                 "source_draft_chain_ms": roles["draft"]["source_chain_ms"],
                 "source_layer_envelopes_ms": math.fsum(value["source_layer_envelopes_ms"] for value in roles.values()),
                 "source_internal_gap_ms": math.fsum(value["source_gap_ms"] for value in roles.values()),
                 "cross_role_tail_residual_ms": residual, "residual_ms": residual,
                 "excluded_reference_layer_wall_ms": excluded_wall,
                 "partition_total_ms": span_sum + residual, "closed": True,
                 "scope": "all original source layer walls and same-role gaps plus cross-role/tail residual, each once; layer0 exclusion changes references only"}
    return {"round_id": cycle.get("round_id"), "source_cycle_ms": duration,
            "roles": roles, "class_values": class_values, "source_partition": partition,
            "compute_evidence": evidence, "class_stage_values": class_stage_values}



def _lm_average_records(rounds, field, key_fields):
    groups = collections.defaultdict(list)
    for cycle in rounds:
        for row in cycle[field]:
            groups[tuple(row[key] for key in key_fields)].append(row)
    output = []
    for _, values in groups.items():
        row = dict(values[0])
        for key, value in row.items():
            if key == "samples":
                row[key] = sum(item[key] for item in values)
            elif type(value) in (int, float) and key not in ("target_layers", "source_layers", "source_gap_count", "reference_gap_count"):
                entries = [item[key] for item in values]
                row[key] = statistics.fmean(entries) if all(type(item) in (int, float) for item in entries) else None
        output.append(row)
    return output



def _lm_partition_mean(rounds):
    first = rounds[0]["source_partition"]
    return {key: statistics.fmean(cycle["source_partition"][key] for cycle in rounds)
            if type(value) in (int, float) else value for key, value in first.items()}



def _lm_source_summary(prepared):
    """Keep measured references available even if a target method is unavailable."""
    output = []
    for rank, source_rounds in prepared.items():
        rounds = []
        for source in source_rounds:
            classes = []
            for (role, attention), values in source["class_values"].items():
                classes.append({"role": role, "attention": attention,
                                "reference_layers": [number for number, _ in values], "samples": len(values),
                                **{key: (statistics.fmean(value[key] for _, value in values)
                                         if all(value[key] is not None for _, value in values) else None)
                                   for key in ("wall_ms", "dispatch_ms", "combine_ms", "routed_compute_ms")}})
            rounds.append({"round_id": source["round_id"], "source_cycle_ms": source["source_cycle_ms"],
                           "cycle_ms": None, "class_means": classes,
                           "role_details": [{"role": role, **values} for role, values in source["roles"].items()],
                           "source_partition": source["source_partition"], "target_stage_breakdowns": []})
        output.append({"rank": rank, "rounds": rounds, "cycle_ms": None,
                       "source_cycle_ms": statistics.fmean(row["source_cycle_ms"] for row in rounds),
                       "class_means": _lm_average_records(rounds, "class_means", ("role", "attention")),
                       "role_details": _lm_average_records(rounds, "role_details", ("role",)),
                       "source_partition": _lm_partition_mean(rounds), "target_stage_breakdowns": []})
    return output



def _lm_estimate_method(result, prepared, shapes, target_schedule, communication, method_id, excluded,
                        operator_cache=None):
    config = result["config"]
    target_counts = collections.Counter((role, attention) for role, _, attention, _ in target_schedule)
    per_rank, all_details, all_evidence = [], [], []
    for rank, source_rounds in prepared.items():
        rounds = []
        for source in source_rounds:
            classes, details = [], []
            for (role, attention), count in target_counts.items():
                values = source["class_values"].get((role, attention))
                if not values:
                    raise ValueError("no retained reference layer for " + role + "/" + attention)
                mean = {key: (statistics.fmean(item[key] for _, item in values)
                              if all(item[key] is not None for _, item in values) else None)
                        for key in ("wall_ms", "dispatch_ms", "combine_ms", "routed_compute_ms")}
                kratio = shapes[role]["target"]["num_experts_per_tok"] / shapes[role]["source"]["num_experts_per_tok"]
                factors = communication["multipliers"][role]
                row = {"role": role, "attention": attention, "reference_layers": [number for number, _ in values],
                       "samples": len(values), "target_layers": count, **mean}
                delta = 0.
                for component, multiplier in (("dispatch", factors["dispatch"]), ("combine", factors["combine"]),
                                              ("routed_compute", kratio)):
                    factor = _lm_number(multiplier, component + " multiplier", True)
                    extra = communication.get("extra_ms_per_call", 0.) if component in ("dispatch", "combine") else 0.
                    source_ms = mean[component + "_ms"]
                    if source_ms is None:
                        if component != "routed_compute" or factor != 1:
                            raise ValueError("missing source component for a nonidentity target delta: " + component)
                        change, target_ms = 0., None
                    else:
                        change, target_ms = source_ms * (factor - 1) + extra, source_ms * factor + extra
                    row[component + "_target_ms"] = target_ms
                    row[component + "_delta_ms"] = change
                    delta += change
                    details.append({"rank": rank, "round_id": source["round_id"], "role": role, "attention": attention,
                                    "class_key": role + "_" + attention, "reference_layers": row["reference_layers"],
                                    "component": component, "target_layers": count, "source_ms": source_ms,
                                    "target_ms": target_ms, "delta_ms": change, "weighted_delta_ms": count * change,
                                    "multiplier": factor, "extra_ms_per_call": extra,
                                    "formula": f"{count} * (source_ms * ({factor}-1) + {extra})",
                                    "status": LAYER_MEAN_STATUS, "basis": "layer_type_mean_and_explicit_scaling_assumption"})
                row["target_wall_ms"] = _lm_number(mean["wall_ms"] + delta, "target layer wall", True)
                classes.append(row)
            roles = []
            for role in ("main", "draft"):
                source_role = source["roles"][role]
                target_layers = sum(count for (name, _), count in target_counts.items() if name == role)
                added = target_layers - source_role["source_layers"]
                if added < 0 or added and not source_role["reference_gap_count"]:
                    raise ValueError("depth reduction or missing same-role reference gaps for " + role)
                class_rows = [row for row in classes if row["role"] == role]
                target_wall = math.fsum(row["target_layers"] * row["target_wall_ms"] for row in class_rows)
                added_gap = added * source_role["reference_gap_mean_ms"]
                target_gap = source_role["source_gap_ms"] + added_gap
                roles.append({"role": role, **source_role, "target_layers": target_layers, "added_layers": added,
                              "target_layer_envelopes_ms": target_wall, "added_gap_ms": added_gap,
                              "target_gap_ms": target_gap, "target_chain_ms": target_wall + target_gap,
                              "dispatch_delta_ms": math.fsum(row["target_layers"] * row["dispatch_delta_ms"] for row in class_rows),
                              "combine_delta_ms": math.fsum(row["target_layers"] * row["combine_delta_ms"] for row in class_rows),
                              "routed_mlp_delta_ms": math.fsum(row["target_layers"] * row["routed_compute_delta_ms"] for row in class_rows)})
            by_role = {row["role"]: row for row in roles}
            residual = source["source_partition"]["residual_ms"]
            breakdown = [{"key": role + suffix, "label": label, "value_ms": by_role[role][field]}
                         for role, suffix, label, field in (
                             ("main", "_layer_envelopes", "主模型按类型均值外推的层区间", "target_layer_envelopes_ms"),
                             ("main", "_internal_gaps", "主模型层间衔接区间", "target_gap_ms"),
                             ("draft", "_layer_envelopes", "草稿按自身SWA均值估算的层区间", "target_layer_envelopes_ms"),
                             ("draft", "_internal_gaps", "草稿层间衔接区间", "target_gap_ms"))]
            breakdown.append({"key": "residual_once", "label": "轮次剩余区间：主草稿切换及尾部（保留一次）", "value_ms": residual})
            cycle = _lm_number(math.fsum(row["target_chain_ms"] for row in roles) + residual, "target layer-mean cycle", True)
            if not math.isclose(math.fsum(item["value_ms"] for item in breakdown), cycle, rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("layer-mean target decomposition does not close")
            rounds.append({"round_id": source["round_id"], "source_cycle_ms": source["source_cycle_ms"],
                           "cycle_ms": cycle, "class_means": classes, "role_details": roles,
                           "source_partition": source["source_partition"], "breakdown": breakdown,
                           "detail_rows": details,
                           "target_stage_breakdowns": _lm_target_stage_breakdowns(classes,
                               source.get("class_stage_values", {}), cycle, layer_samples=True)})
            all_evidence.extend({"rank": rank, **proof} for proof in source["compute_evidence"])
        rank_details = _lm_average_records(rounds, "detail_rows", ("role", "attention", "component"))
        for row in rank_details:
            row.pop("round_id", None)
            row["rounds_averaged"] = len(rounds)
        all_details.extend(rank_details)
        per_rank.append({"rank": rank, "rounds": rounds, "detail_rows": rank_details,
                         "source_cycle_ms": statistics.fmean(cycle["source_cycle_ms"] for cycle in rounds),
                         "cycle_ms": statistics.fmean(cycle["cycle_ms"] for cycle in rounds),
                         "class_means": _lm_average_records(rounds, "class_means", ("role", "attention")),
                         "role_details": _lm_average_records(rounds, "role_details", ("role",)),
                         "source_partition": _lm_partition_mean(rounds),
                         "breakdown": _lm_average_records(rounds, "breakdown", ("key",))})
        stage_groups = collections.defaultdict(list)
        for source_round in rounds:
            for row in source_round["target_stage_breakdowns"]:
                stage_groups[(row["role"], row["attention"])].append(row)
        per_rank[-1]["target_stage_breakdowns"] = _lm_target_stage_breakdowns(
            per_rank[-1]["class_means"], stage_groups, per_rank[-1]["cycle_ms"], rounds=len(rounds))
    if not per_rank:
        raise ValueError("no source ranks/rounds for layer-mean estimation")
    selected = max(per_rank, key=lambda row: row["cycle_ms"])
    operator_cache = operator_cache if operator_cache is not None else {}
    if selected["rank"] not in operator_cache:
        operator_cache[selected["rank"]] = _lm_operator_source_means(prepared[selected["rank"]])
    target_stages = _lm_operator_target_groups(selected["target_stage_breakdowns"],
        operator_cache[selected["rank"]], selected["cycle_ms"])
    cards, dp, bs = (config.get(key) for key in ("target_devices", "target_dp", "bs"))
    chips = config.get("target_chips_per_card", 1)
    if any(type(value) is not int or value <= 0 for value in (cards, dp, bs, chips)):
        raise ValueError("layer-mean target physical cards/chips/DP/BS must be explicit positive integers")
    if max(dp, config["target_ep"]) > cards * chips:
        raise ValueError("target DP/EP exceeds declared physical-card participating chip budget")
    neff = _lm_number(config.get("effective_tokens_per_round"), "configured effective output tokens", True)
    if not 1 <= neff <= 1 + config["draft_tokens"]:
        raise ValueError("configured effective output length is outside [1,S+1]")
    for rank in per_rank:
        for row in rank["rounds"] + [rank]:
            row.update(per_device_tokens_s=1000 * bs * dp * neff / (row["cycle_ms"] * cards),
                       global_tokens_s=1000 * bs * dp * neff / row["cycle_ms"], tpot_ms=row["cycle_ms"] / neff)
    opaque = sum(proof["logical_expert_axis_matches"] < proof["matched_calls"] for proof in all_evidence)
    warnings = ["HC_MARKER_LAYER_WALL_PROXY_NOT_COMPLETE_PYTHON_OWNERSHIP",
                "HASH_NORMAL_POOLED_BY_ATTENTION_KIND",
                "ADDED_LAYER_GAP_MEAN_AND_CROSS_ROLE_TAIL_STATIONARITY_ASSUMED",
                "ROUTER_EXPERT_WIDTH_TARGET_DELTA_NOT_MODELED"]
    if opaque:
        warnings.append("PACKED_OR_OPAQUE_GMM_SHAPES: native names and routed phase used in " + str(opaque) + " reference layers")
    missing_compute = sum(not proof["matched_calls"] for proof in all_evidence)
    if missing_compute:
        warnings.append("ROUTED_GMM_UNMEASURED_UNCHANGED_TOPK: source/target compute metrics remain unknown; zero modeled delta follows from unchanged TopK only")
    return {"id": method_id, "name": "方法1：按层类型均值＋逻辑通信倍率" if method_id == "method1" else "方法2：按层类型均值＋历史通信倍率",
            "status": LAYER_MEAN_STATUS, "reasons": [], "assumptions": list(LAYER_MEAN_ASSUMPTIONS), "warnings": warnings,
            "cycle_ms": selected["cycle_ms"], "per_device_tokens_s": selected["per_device_tokens_s"],
            "global_tokens_s": selected["global_tokens_s"], "tpot_ms": selected["tpot_ms"],
            "source_cycle_ms": selected["source_cycle_ms"], "selected_rank": selected["rank"],
            "aggregation": "max_of_per_rank_mean_of_per_round_layer_type_mean_estimates",
            "per_rank": per_rank, "class_means": selected["class_means"], "role_details": selected["role_details"],
            "target_stage_breakdowns": target_stages,
            "source_partition": selected["source_partition"], "breakdown": selected["breakdown"],
            "detail_rows": all_details, "compute_evidence": all_evidence, "communication": communication, "comm_model": communication,
            "compute_coverage": {"reference_layer_calls": len(all_evidence),
                                 "matched_reference_layers": len(all_evidence) - missing_compute,
                                 "unmatched_reference_layers": missing_compute,
                                 "missing_metric_policy": "None, never zero source cost; zero delta permitted only for unchanged TopK"},
            "reference_policy": {"exclude_main_layers": sorted(excluded), "grouping": "role+attention, hash/normal pooled"},
            "formula": "R=NHCA*mean(Wmain_HCA+delta)+NCSA*mean(Wmain_CSA+delta)+Ndraft*mean(Wdraft_SWA+delta)+source_main_gaps+Nadded_main*mean(reference_main_gaps)+source_draft_gaps+U; U=Rsource-source_main_chain-source_draft_chain; delta=D*(mD-1)+C*(mC-1)+GMM*(Ktarget/Ksource-1)+2*extra_ms_per_call; extra_ms_per_call=" + str(communication.get("extra_ms_per_call", 0.)) + "ms (one fixed addition per Dispatch/Combine target call)"}



def _attach_layer_mean_estimates(report, read_capture, schedule_fn, shapes_fn, comm_fn):
    """Old per-role/attention layer means, with explicit measurement/closure guards.

    Complete-owner flags and component calibration declarations are not inputs
    to this model. Actual missing native tasks, contradictory shapes/topology,
    or unavailable historical communication values are reported as unavailable.
    """
    for result in report["cases"]:
        cached = result.pop('_prepared_layer_means', None)
        methods, prepared, coverage_warnings = [], {}, []
        operator_cache = {}
        source_errors = result.pop('_layer_mean_source_errors', [])
        coverage = {}
        try:
            config, model = result["config"], result["source_model"]
            source_schedule, target_schedule, errors = schedule_fn(model, config, result.get("target_model"))
            shapes, shape_errors = shapes_fn(model, result.get("target_model"), config)
            if errors or shape_errors:
                raise ValueError("; ".join(errors + shape_errors))
            compute_required = {role: shape["target"]["num_experts_per_tok"] != shape["source"]["num_experts_per_tok"]
                                for role, shape in shapes.items()}
            excluded_list = config.get("exclude_target_layers", [0])
            if (not isinstance(excluded_list, list) or any(type(number) is not int or not 0 <= number < model["num_hidden_layers"] for number in excluded_list)
                    or len(set(excluded_list)) != len(excluded_list)):
                raise ValueError("layer-mean excluded main reference layers must be distinct valid integer layer numbers")
            excluded = set(excluded_list)
            cards, chips = config.get("source_devices"), config.get("source_chips_per_card", 1)
            if any(type(value) is not int or value <= 0 for value in (cards, chips)):
                raise ValueError("layer-mean source physical cards/chips must be explicit positive integers")
            captures = {item["rank"]: item for item in result["captures"]}
            ranks = [capture["rank"] for capture in result["measurements"]]
            logical = cards * chips
            if (len(captures) != len(result["captures"]) or len(ranks) != len(set(ranks))
                    or not ranks or set(ranks) != set(captures)):
                raise ValueError("layer-mean requires unique matching source capture/measurement rank identities")
            if len({capture.get("sha256") for capture in captures.values()}) != len(captures):
                raise ValueError("distinct source ranks require distinct native CSV hashes")
            expected_count = config.get("expected_ranks", logical)
            coverage = {"captured_ranks": len(ranks), "expected_ranks": expected_count,
                        "captured_rank_names": ranks, "source_logical_devices": logical,
                        "complete": len(ranks) == expected_count == logical and config.get("_rank_coverage_complete") is not False}
            if not coverage["complete"]:
                coverage_warnings.append("PARTIAL_SOURCE_RANK_COVERAGE: estimates use only captured ranks; the full-system slowest rank is not established")
            if source_errors:
                raise ValueError("; ".join(source_errors))
            for measured in result["measurements"]:
                rank, rounds = measured["rank"], measured["rounds"]
                if not rounds:
                    raise ValueError("source rank has no retained rounds: " + rank)
                if cached is not None:
                    prepared[rank] = cached[rank]
                    continue
                original = captures[rank]
                path = Path(original["path"])
                if not path.is_absolute():
                    path = Path(config.get("_source_point_report_base", ".")) / path
                native = read_capture(path, rank)
                if native["sha256"] != original.get("sha256"):
                    raise ValueError("native source CSV changed after measurement: " + str(path))
                tasks = native["tasks"]
                starts = [task.start_ns for task in tasks]
                time_index = ATimeIndex(tasks)
                prepared[rank] = [_lm_source_round(cycle, tasks, starts, model, config, source_schedule, excluded, compute_required,
                                                  time_index=time_index, rank=rank)
                                  for cycle in rounds]
        except (ValueError, KeyError, OSError, TypeError, ArithmeticError) as error:
            source_errors.append(str(error))
        for method_id in ("method1", "method2"):
            comm = {}
            try:
                if source_errors:
                    raise ValueError("; ".join(source_errors))
                if method_id == "method1":
                    ep0, ep1 = config["base_ep"], config["target_ep"]
                    scope = config.get("traffic_scope", "interhost")
                    if scope == "intranode":
                        if ep1 != ep0 or config.get("ranks_per_node", 0) < ep0:
                            raise ValueError("intranode source cannot establish cross-node EP expansion")
                        ratio = 1.
                    else:
                        local = config.get("ranks_per_node", 8) if scope == "interhost" else 1
                        if scope not in ("interhost", "remote", "all_remote") or not ep1 >= ep0 > local:
                            raise ValueError("logical remote share requires target_ep>=base_ep>local")
                        ratio = (1 - local / ep1) / (1 - local / ep0)
                    comm = {"model": "observed_duration_times_remote_share_times_topk_plus_fixed_overhead", "remote_ratio": ratio,
                            "extra_ms_per_call": METHOD1_EXTRA_US / 1000, "basis": "layer_mean_scaling_assumption",
                            "multipliers": {role: {component: ratio * shape["target"]["num_experts_per_tok"] / shape["source"]["num_experts_per_tok"]
                                                   for component in ("dispatch", "combine")} for role, shape in shapes.items()}}
                else:
                    declaration = {"communication": {"method2": {"basis": "assumed",
                                   "description": "explicit old layer-mean historical ratio transfer assumption",
                                   "fit": "log2_ols_ep16_extrapolation", "reference_bs": config["bs"]}}}
                    comm, errors, assumptions = comm_fn(method_id, model, shapes, config, declaration,
                                                       report.get("communication_reference", {}).get("rows", []))
                    if errors:
                        raise ValueError("; ".join(errors))
                method = _lm_estimate_method(result, prepared, shapes, target_schedule, comm, method_id, excluded,
                                             operator_cache=operator_cache)
                method["warnings"].extend(coverage_warnings)
                method["coverage"] = coverage
                if method_id == "method2":
                    method["assumptions"].extend(assumptions)
            except (ValueError, KeyError, OSError, TypeError, ArithmeticError) as error:
                source_summary = _lm_source_summary(prepared) if prepared else []
                method = {"id": method_id, "name": "方法1：按层类型均值＋逻辑通信倍率" if method_id == "method1" else "方法2：按层类型均值＋历史通信倍率",
                          "status": "UNAVAILABLE", "reasons": [str(error)], "assumptions": list(LAYER_MEAN_ASSUMPTIONS),
                          "warnings": coverage_warnings, "coverage": coverage,
                          "cycle_ms": None, "per_device_tokens_s": None, "global_tokens_s": None,
                          "tpot_ms": None, "selected_rank": None, "per_rank": source_summary,
                          "class_means": source_summary[0]["class_means"] if source_summary else [],
                          "role_details": source_summary[0]["role_details"] if source_summary else [],
                          "source_partition": source_summary[0]["source_partition"] if source_summary else {},
                          "breakdown": [], "detail_rows": [], "compute_evidence": [], "target_stage_breakdowns": [],
                          "communication": comm, "comm_model": comm}
            methods.append(method)
        result["projection"]["layer_mean_methods"] = methods
        result["projection"]["display_method_set"] = "layer_mean_methods"
    report.setdefault("contract", {})["layer_mean"] = "per-rank/per-round attention-kind means; main0 reference excluded by default; original chain boundaries and cross-role/tail residual retained once; explicit uncalibrated estimate"
    return report



def attach_layer_mean_report(report):
    """Project HC layer means; ownership/calibration declarations are not inputs."""
    return _attach_layer_mean_estimates(report, read_capture, b_schedule, b_shapes, b_comm)



def layer_mean_complete(report):
    """Every requested case needs its primary estimate; method2 may lack a reference."""
    return bool(report['cases']) and all(any(method.get('id') == 'method1'
           and method.get('status') == 'LAYER_MEAN_ESTIMATE' and not method.get('reasons')
           for method in displayed_methods(case)) for case in report['cases'])



def verify_result_rows(report, rows):
    """Check both available and unavailable result cells against core values."""
    index = 0
    fields = {'cycle_ms': '周期(ms)', 'per_device_tokens_s': '单卡吞吐(token/s/卡)',
              'tpot_ms': 'TPOT(ms)'}
    for case in report['cases']:
        projection = case.get('projection') or {}
        if projection.get('display_method_set') != 'layer_mean_methods':
            raise ValueError('结果一致性检查失败：未选择当前层均值结果 ' + case['name'])
        for method in projection.get('layer_mean_methods', []):
            if index >= len(rows):
                raise ValueError('结果一致性检查失败：最终结果表缺少 ' + case['name'] + '/' + method['id'])
            row = rows[index]
            index += 1
            label = case['name'] + '/' + method['id']
            available = method.get('status') == LAYER_MEAN_STATUS and not method.get('reasons')
            expected_status = '条件估算' if available else '缺少实际输入'
            if row.get('采样组合') != case['name'] or row.get('方法') != method['name'] or row.get('状态') != expected_status:
                raise ValueError('结果一致性检查失败：' + label + ' 被显示为另一套结果或待估算')
            for key, column in fields.items():
                actual = row.get(column)
                if not available:
                    if actual is not None:
                        raise ValueError(f'结果一致性检查失败：{label} 缺少输入却显示{column}={actual!r}')
                    continue
                expected = method.get(key)
                if (type(expected) not in (int, float) or not math.isfinite(expected) or expected <= 0
                        or type(actual) not in (int, float) or actual != expected):
                    raise ValueError(f'结果一致性检查失败：{label} {column}；核心={expected!r}，显示={actual!r}')
    if len(rows) != index + len(report.get('failures', [])):
        raise ValueError('结果一致性检查失败：最终结果表含额外记录或缺少失败记录')



def iter_report_rows(report, kind):
    """Export flat records; large per-round evidence is never nested in summary cells."""
    for case in report['cases']:
        name, source = case['name'], case['source']
        methods = displayed_methods(case)
        evidence_method = next((method for method in methods if method.get('compute_evidence')), None)
        if kind == 'summary':
            yield {'case': name, 'kind': 'source', **source['aggregate'], 'status': source['status']}
        elif kind == 'per_rank':
            yield from ({'case': name, **item} for item in source['per_rank'])
        elif kind == 'components':
            yield from ({'case': name, **item} for item in source.get('class_stats', []))
        elif kind in ('layer_stage_summary', 'swa_stage_summary'):
            for capture in case.get('measurements', []):
                diagnostic = capture.get('layer_diagnostics' if kind == 'layer_stage_summary' else 'swa_diagnostics') or {}
                yield from ({'case': name, 'rank': capture['rank'], **item}
                            for item in diagnostic.get('stage_summary', []))
        elif kind in ('step_residual_focus_steps', 'step_residual_focus_summary'):
            for capture in case.get('measurements', []):
                focus = (capture.get('step_residuals') or {}).get('focus_diagnostics') or {}
                for stage in focus.get('stages', []):
                    identity = {'case': name, 'rank': capture['rank'], 'step_kind': 'retained_hc_cycle',
                                'parent_stage_key': stage.get('stage_key'), 'parent_label': stage.get('label')}
                    if kind == 'step_residual_focus_steps':
                        # Exact segment evidence lives in the per-capture CSV;
                        # keep this table directly useful for step comparisons.
                        yield from ({**identity, **{key: value for key, value in item.items()
                                     if key not in ('components', 'component_durations_ms')}}
                                    for item in stage.get('per_step', []))
                    else:
                        yield from ({**identity, **item} for item in stage.get('summary', []))
        elif kind in ('step_residual_windows', 'step_residual_summary', 'step_residual_sources',
                      'step_residual_stages', 'step_residual_substages'):
            for capture in case.get('measurements', []):
                diagnostic = capture.get('step_residuals')
                if diagnostic is None:
                    continue
                identity = {'case': name, 'rank': capture['rank'], 'step_kind': 'hc_cycle'}
                if kind == 'step_residual_sources':
                    trace = diagnostic.get('trace') or {}
                    yield {**identity, 'diagnostic_status': diagnostic.get('status'),
                           'trace_status': trace.get('status'),
                           **{key: value for key, value in trace.items() if key != 'status'},
                           **{key: diagnostic.get(key) for key in ('device_tasks_csv', 'host_events_csv', 'stages_csv', 'stage_basis',
                                                                 'source_links_csv', 'source_status', 'source_scope', 'source_task_status_counts')}}
                else:
                    field = {'step_residual_windows': 'windows', 'step_residual_summary': 'summary',
                             'step_residual_stages': 'stage_summary', 'step_residual_substages': 'substage_summary'}[kind]
                    yield from ({**identity, **item} for item in diagnostic.get(field, []))
        elif kind in ('rounds', 'layers', 'time_ledger'):
            for capture in case.get('measurements', []):
                for cycle in capture['rounds']:
                    identity = {'case': name, 'rank': capture['rank'], 'round_id': cycle['round_id']}
                    if kind == 'rounds':
                        yield {**identity, **{key: value for key, value in cycle.items() if key not in ('layers', 'ledger')}}
                    elif kind == 'layers':
                        yield from ({**identity, **item} for item in cycle.get('layers', []))
                    else:
                        yield from ({**identity, **item} for item in cycle.get('ledger', {}).get('rows', []))
        for method in methods:
            identity = {'case': name, 'method': method['id']}
            if kind == 'summary':
                yield {**identity, **{key: method.get(key) for key in
                        ('name', 'status', 'selected_rank', 'cycle_ms', 'per_device_tokens_s', 'tpot_ms')},
                       'reasons': ' | '.join(method.get('reasons', []))}
            elif kind == 'projection_details':
                yield from ({**identity, **item} for item in method.get('detail_rows', []))
            elif kind == 'compute_evidence' and method is evidence_method:
                yield from ({'case': name, **item} for item in method.get('compute_evidence', []))
            for rank in method.get('per_rank', []):
                rank_id = {**identity, 'rank': rank['rank']}
                if kind == 'layer_means':
                    yield from ({**rank_id, **item} for item in rank.get('class_means', []))
                elif kind == 'rank_estimates':
                    yield {**rank_id, **{key: rank.get(key) for key in
                           ('source_cycle_ms', 'cycle_ms', 'per_device_tokens_s', 'tpot_ms')}}
                elif kind == 'round_estimates':
                    for cycle in rank.get('rounds', []):
                        yield {**rank_id, 'round_id': cycle['round_id'],
                               **{key: cycle.get(key) for key in ('source_cycle_ms', 'cycle_ms', 'per_device_tokens_s', 'tpot_ms')},
                               **{role['role'] + '_target_chain_ms': role.get('target_chain_ms') for role in cycle.get('role_details', [])},
                               'residual_ms': cycle.get('source_partition', {}).get('residual_ms')}
                elif kind == 'boundary_accounting':
                    yield from ({**rank_id, **{key: value for key, value in item.items()
                                if key == 'role' or 'gap' in key or 'chain' in key}} for item in rank.get('role_details', []))
                    yield {**rank_id, 'role': 'cross_role_tail', **rank.get('source_partition', {})}
            if kind == 'review':
                for field in ('reasons', 'assumptions', 'warnings'):
                    yield from ({**identity, 'kind': field, 'reason': value} for value in method.get(field, []))
        if kind == 'review':
            for field in ('reasons', 'findings'):
                yield from ({'case': name, 'kind': field, 'reason': value} for value in case.get('review', {}).get(field, []))


def report_json_payload(report):
    """Keep renderable means in JSON; write full round estimates and GMM evidence once to CSV."""
    cases = []
    for case in report['cases']:
        methods = []
        for method in displayed_methods(case):
            compact = {key: value for key, value in method.items() if key not in ('compute_evidence', 'communication')}
            compact['per_rank'] = [{key: value for key, value in rank.items() if key != 'rounds'}
                                   for rank in method.get('per_rank', [])]
            methods.append(compact)
        cases.append({**{key: value for key, value in case.items() if not key.startswith('_')},
                      'projection': {'display_method_set': 'layer_mean_methods', 'layer_mean_methods': methods}})
    return {**report, 'cases': cases, 'detail_tables': {'round_estimates': 'round_estimates.csv',
                                                     'compute_evidence': 'compute_evidence.csv'}}



def export_report(report, out, reference_rows, progress=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, action):
        path = out / name
        if progress is not None:
            progress('报告导出：写入 ' + name)
        started = time.monotonic()
        try:
            action(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f'报告导出失败：{path}；{error}') from error
        if progress is not None:
            progress(f'报告导出完成：{name}（{path.stat().st_size / 1024 / 1024:.2f} MiB，'
                     f'{time.monotonic() - started:.1f} 秒）')

    # Publish the small result table before potentially large evidence exports.
    final_rows = result_rows(report)
    verify_result_rows(report, final_rows)
    write('最终结果.csv', lambda path: write_csv(path, final_rows, progress=progress))
    write('report.html', lambda path: render_html(path, report))
    write('model_architecture_report.json', lambda path: atomic_json(path, report_json_payload(report), progress=progress))
    kinds = ['summary', 'per_rank', 'rounds', 'layers', 'components', 'time_ledger', 'projection_details', 'review']
    kinds += ['layer_means', 'rank_estimates', 'round_estimates', 'boundary_accounting', 'compute_evidence']
    if any('step_residuals' in capture for case in report['cases'] for capture in case.get('measurements', [])):
        kinds += ['step_residual_windows', 'step_residual_summary', 'step_residual_sources',
                  'step_residual_stages', 'step_residual_substages']
    if any((capture.get('step_residuals') or {}).get('focus_diagnostics')
           for case in report['cases'] for capture in case.get('measurements', [])):
        kinds += ['step_residual_focus_steps', 'step_residual_focus_summary']
    if any('swa_diagnostics' in capture for case in report['cases'] for capture in case.get('measurements', [])):
        kinds += ['swa_stage_summary']
    if any('layer_diagnostics' in capture for case in report['cases'] for capture in case.get('measurements', [])):
        kinds += ['layer_stage_summary']
    for kind in kinds:
        write(kind + '.csv', lambda path, kind=kind:
              write_csv(path, lambda: iter_report_rows(report, kind), progress=progress))
    # Exact original communication values are evidence, with unknown units kept.
    write('communication_reference.csv', lambda path: write_csv(path, reference_rows, progress=progress))
    write('problems.txt', lambda path: atomic_text(path, '\n'.join(f['case'] + ': ' + f['error'] for f in report['failures']) + ('\n' if report['failures'] else '')))
    architecture = ['# HC层均值估算', '',
                    '源8主＋3草稿；主0只排除参考，目标层数按配置。',
                    '方法一每次Dispatch/Combine加14μs；方法二历史比值不叠加。',
                    '层间衔接与轮次剩余区间分别计入；单卡吞吐按物理卡数均摊。', '']
    for case in report['cases']:
        architecture += ['## ' + case['name'], '']
        for method in displayed_methods(case):
            architecture += [method['name'] + ': ' + method['status']]
            architecture += ['- ' + reason for reason in method.get('reasons', [])]
    write('model_architecture_report.md', lambda path: atomic_text(path, '\n'.join(architecture) + '\n'))



def render_html(path, report):
    esc = lambda value: html.escape('待估算' if value is None else str(value), quote=True)
    rows = result_rows(report)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>10T模型推理估算</title><style>body{font:15px system-ui;margin:32px;background:#f4f7fb;color:#182c45}h1{font-size:26px}p{line-height:1.8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #d7e0eb;text-align:left}th{background:#163551;color:white}</style>',
             '<h1>10T模型推理估算</h1>',
             '<p>主0只排除参考；HCA/CSA按各rank各轮同类型均值外推，草稿独立。通信及路由GMM只加变化量。方法一每次Dispatch/Combine目标耗时加14μs；方法二不再叠加。</p>',
             '<p>源层间衔接及轮次剩余区间分别保留一次。先求各rank周期均值，再选最大rank；吞吐按目标物理卡数均摊。</p>',
             '<table><thead><tr>' + ''.join('<th>' + esc(key) + '</th>' for key in fields) + '</tr></thead><tbody>']
    parts.extend('<tr>' + ''.join('<td>' + esc(row.get(key)) + '</td>' for key in fields) + '</tr>' for row in rows)
    parts += ['</tbody></table><p>Q_card＝1000×BS×目标DP×Neff/(目标周期ms×目标物理卡数)。</p>',
              '<p><a href="最终结果.csv">目标结果</a> · <a href="per_rank.csv">源基线</a> · <a href="layers.csv">源层原值</a> · <a href="round_estimates.csv">逐轮估算</a> · <a href="review.csv">复核记录</a></p>',
              '<small>生成时间：' + esc(report.get('generated_at')) + '</small></html>']
    atomic_text(path, '\n'.join(parts))



def build_cli_excel(report, out, expected_method_set):
    """Require a renderer that honors the exact result set requested by the CLI."""
    import build_ep128_leadership_excel as renderer
    runtime = renderer.runtime_diagnostics()
    print('Excel生成器：' + runtime['renderer_path'] + '（' + runtime['renderer_version'] + '）', flush=True)
    print('Excel写入器：' + runtime['writer_path'] + '（' + runtime['writer_version'] + '）', flush=True)
    destination = Path(out) / ESTIMATE_XLSX
    print('Excel目标文件：' + str(destination.resolve()), flush=True)
    try:
        renderer.build_workbook(destination, report=Path(out) / 'model_architecture_report.json',
                                report_data=report, expected_method_set=expected_method_set,
                                progress=lambda stage: print('Excel阶段：' + stage, flush=True))
    except TypeError as error:
        raise ValueError('Excel生成器接口不匹配或生成失败；请同步更新profile_ep128.py和build_ep128_leadership_excel.py。'
                         + '实际生成器：' + runtime['renderer_path'] + '（' + runtime['renderer_version'] + '）；'
                         + '本次预测已保存在 ' + str((Path(out) / '最终结果.csv').resolve()) + '；原始错误：' + str(error)) from error



class _CaseProgressStream:
    """Label existing rank progress without changing the analysis API."""
    def __init__(self, stream, name):
        self.stream = stream
        self.prefix = '[case ' + str(name).replace('\n', ' ').replace('\r', ' ') + '] '
        self.at_line_start = True

    def write(self, message):
        for part in message.splitlines(keepends=True):
            if self.at_line_start and part:
                self.stream.write(self.prefix)
            self.stream.write(part)
            self.at_line_start = part.endswith(('\n', '\r'))
        return len(message)

    def flush(self):
        self.stream.flush()


def _cli_case_cgroups(controller):
    """Locate the process cgroup and visible ancestors, including remounted hierarchies."""
    root_path = Path('/sys/fs/cgroup')
    groups = [(root_path, root_path, True), (root_path / controller, root_path / controller, False)]
    try:
        memberships = [line.split(':', 2) for line in Path('/proc/self/cgroup').read_text().splitlines()]
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, right = line.split(' - ', 1)
            fields, mounted = left.split(), right.split()
            unified = mounted[0] == 'cgroup2'
            if not unified and not (mounted[0] == 'cgroup' and controller in mounted[2].split(',')):
                continue
            decode = lambda value: re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)
            root, mount = Path(decode(fields[3])), Path(decode(fields[4]))
            for hierarchy, controllers, member in memberships:
                if (unified and hierarchy == '0' and not controllers) or (not unified and controller in controllers.split(',')):
                    try:
                        relative = Path(member).relative_to(root)
                    except ValueError:
                        continue
                    if '..' not in relative.parts:
                        groups.append((mount / relative, mount, unified))
    except (OSError, ValueError, IndexError):
        pass
    visited, output = set(), []
    for directory, mount, unified in groups:
        while (directory, unified) not in visited:
            visited.add((directory, unified))
            output.append((directory, unified))
            if directory == mount or mount not in directory.parents:
                break
            directory = directory.parent
    return output


def _cli_case_memory_available():
    """Best-effort host and current/ancestor cgroup memory headroom."""
    import os
    available = []
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    if not available:
        try:
            available.append(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        except (AttributeError, OSError, ValueError):
            pass
    # On Windows use the standard-library ctypes interface when available.
    if not available and sys.platform == 'win32':
        try:
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys', 'total_page',
                    'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                available.append(status.avail_phys)
        except (AttributeError, OSError, ValueError):
            pass
    for directory, unified in _cli_case_cgroups('memory'):
        try:
            limit_name, usage_name = ('memory.max', 'memory.current') if unified else ('memory.limit_in_bytes', 'memory.usage_in_bytes')
            limit = int((directory / limit_name).read_text().strip())
            usage = int((directory / usage_name).read_text().strip())
            if 0 < limit < 1 << 60 and usage >= 0:
                available.append(max(0, limit - usage))
        except (OSError, ValueError):
            pass
    return min(available) if available else None


def _cli_case_cpu_budget():
    """Honor affinity and v1/v2 CPU quota, not only the host's reported CPU count."""
    import os
    cpu = max(1, os.cpu_count() or 1)
    try:
        cpu = min(cpu, max(1, len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        pass
    for directory, unified in _cli_case_cgroups('cpu'):
        try:
            if unified:
                quota, period = (directory / 'cpu.max').read_text().split()
            else:
                quota = (directory / 'cpu.cfs_quota_us').read_text().strip()
                period = (directory / 'cpu.cfs_period_us').read_text().strip()
            quota, period = int(quota), int(period)
            if quota > 0 and period > 0:
                cpu = min(cpu, max(1, math.ceil(quota / period)))
        except (OSError, ValueError):
            pass
    return cpu


def _cli_case_workers(requested, case_count):
    """Conservative automatic budget; explicit requests only receive resource warnings."""
    cpu = _cli_case_cpu_budget()
    memory = _cli_case_memory_available()
    low_memory = memory is not None and memory < 4 * 1024 ** 3
    automatic = min(2, cpu, max(1, case_count))
    if low_memory:
        automatic = 1
    workers = min(requested if requested is not None else automatic, max(1, case_count))
    memory_label = '未知' if memory is None else f'{memory / 1024 ** 3:.2f} GiB'
    print(f'执行粒度：case；进程数={workers}；CPU预算={cpu}；可用内存/容器余量={memory_label}；'
          + ('自动选择' if requested is None else '显式指定'), flush=True)
    if requested is not None and requested > automatic:
        print(f'并行提示：当前自动建议上限为{automatic}；按显式请求执行，更多进程会增加内存、磁盘争用。', flush=True)
    return workers


def _cli_case_complete(index, case, model, target, base, reference_rows):
    """Finalize one case before releasing its potentially large prepared evidence."""
    from contextlib import redirect_stdout
    start = time.perf_counter()
    with redirect_stdout(_CaseProgressStream(sys.stdout, case['name'])):
        print('开始读取与统计；本case内各rank顺序执行', flush=True)
        result = analyze_case(case, model, target, base, reference_rows)
        print(f'测量完成（{time.perf_counter() - start:.2f}s）；开始方法1/方法2估算', flush=True)
        partial = {'cases': [result], 'communication_reference': {'rows': reference_rows}}
        attach_layer_mean_report(partial)
        print(f'统计与估算完成（共{time.perf_counter() - start:.2f}s）', flush=True)
    return {'index': index, 'result': result, 'contract': partial['contract']}


def _cli_case_worker(job, spool_directory):
    """Only file identity crosses IPC; large finalized results stay on local disk."""
    import pickle
    index, case, model, target, base, reference_rows = job
    try:
        payload = _cli_case_complete(index, case, model, target, base, reference_rows)
    except (ValueError, KeyError, OSError, ArithmeticError, TypeError, re.error) as error:
        payload = {'index': index, 'failure': {'case': str(case['name']), 'error': str(error)}}
        print('[case ' + str(case['name']) + '] FAILED: ' + str(error), flush=True)
    destination = Path(spool_directory) / ('case-' + str(index) + '.pickle')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=spool_directory, prefix='.case-', delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return index, str(destination)


def _cli_run_case_jobs(jobs, workers):
    """Run one parallel level, restore input order, isolate failures, and clean spools."""
    import multiprocessing
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed
    output = {}
    def failed(job, failure):
        index, case = job[:2]
        known = isinstance(failure, (ValueError, KeyError, OSError, ArithmeticError, TypeError, re.error))
        message = str(failure) if known else 'CASE_WORKER_ERROR: ' + type(failure).__name__ + ': ' + str(failure)
        output[index] = {'index': index, 'failure': {'case': str(case['name']), 'error': message}}
        print('[case ' + str(case['name']) + '] FAILED: ' + message + '；不自动重跑', flush=True)
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            try:
                payload = _cli_case_complete(*job)
            except Exception as error:
                failed(job, error)
            else:
                output[job[0]] = payload
        return [output[index] for index in sorted(output)]
    with tempfile.TemporaryDirectory(prefix='ep128-case-results-') as spool_directory:
        def collect(job, outcome=None, error=None):
            index, case = job[:2]
            try:
                if error is not None:
                    raise error
                expected = Path(spool_directory) / ('case-' + str(index) + '.pickle')
                # Do not load paths supplied by input configuration or arbitrary worker output.
                if outcome != (index, str(expected)):
                    raise ValueError('case worker returned an unexpected result file identity')
                with expected.open('rb') as stream:
                    payload = pickle.load(stream)
                expected.unlink()
                if payload.get('index') != index:
                    raise ValueError('case result index does not match its submitted job')
                output[index] = payload
            except Exception as failure:
                failed(job, failure)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            pending = {}
            for job in jobs:
                try:
                    pending[executor.submit(_cli_case_worker, job, spool_directory)] = job
                except Exception as error:
                    collect(job, error=error)
            for future in as_completed(pending):
                job = pending[future]
                try:
                    outcome = future.result()
                except Exception as error:
                    collect(job, error=error)
                else:
                    collect(job, outcome=outcome)
    return [output[index] for index in sorted(output)]


def _cli_positive_workers(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('--workers需要正整数') from error
    if number <= 0:
        raise argparse.ArgumentTypeError('--workers需要正整数')
    return number


def main():
    parser = argparse.ArgumentParser(description='按HC层均值外推目标周期与单卡吞吐；标准库离线统计')
    parser.add_argument('cases', nargs='?', type=Path, help='原采样cases配置')
    parser.add_argument('--version', action='version', version='EP128 profile ' + PROFILE_VERSION)
    parser.add_argument('--out', type=Path, default=Path('ep128_report'))
    parser.add_argument('--excel', action='store_true', help='同时导出10T模型推理估算Excel')
    parser.add_argument('--comm-reference', type=Path, help='原通信参考表')
    parser.add_argument('--list-inputs', action='store_true', help='只核对输入文件，不读取CSV')
    parser.add_argument('--inventory', type=Path, help='查看少量CSV的字段和设备任务数量')
    parser.add_argument('--workers', type=_cli_positive_workers, default=None,
                        help='按case并行的进程数；默认最多2，低内存自动用1；--workers 1串行')
    args = parser.parse_args()
    print('统计脚本：' + str(Path(__file__).resolve()) + '（' + PROFILE_VERSION + '）', flush=True)
    if args.inventory:
        if args.cases or args.list_inputs or args.excel:
            parser.error('--inventory不能与cases/list-inputs/excel组合')
        for index, path in enumerate(find_files(str(args.inventory), Path.cwd())[:2]):
            capture = read_capture(path, 'trace' + str(index))
            print(json.dumps({key: value for key, value in capture.items() if key != 'tasks'}, ensure_ascii=False))
            print('device task records:', len(capture['tasks']))
        return 0
    if args.cases is None:
        parser.error('provide cases JSON or --inventory')
    if args.list_inputs and args.excel:
        parser.error('--list-inputs不能与--excel组合')
    if args.list_inputs:
        print('输入清单模式：只查找文件，不读取CSV内容、不生成报告。', flush=True)
        print('脚本：' + str(Path(__file__).absolute()), flush=True)
        print('配置：' + str(args.cases.absolute()), flush=True)
    config = json.loads(args.cases.read_text(encoding='utf-8-sig'))
    if type(config.get('schema_version')) is not int or config['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Use the V2 example manifest (schema_version=2)')
    if not isinstance(config.get('cases'), list) or not config['cases']:
        raise ValueError('cases must be a nonempty list')
    defaults = config.get('defaults', {})
    if not isinstance(defaults, dict):
        raise ValueError('defaults must be an object')
    base = args.cases.resolve().parent
    if args.list_inputs:
        return list_inputs(config, base)
    model = referenced_object(config['sample_model'], base, 'sample_model')
    target = referenced_object(config['target_model'], base, 'target_model') if config.get('target_model') is not None else None
    validate_models(model, target)
    reference_path = args.comm_reference
    if reference_path is None:
        configured_reference = config.get('communication_reference')
        candidate = Path(__file__).resolve().parent / 'ep_comm_reference_20261028/raw.csv'
        reference_path = base / configured_reference if configured_reference else candidate if candidate.is_file() else None
    reference_error = None
    try:
        reference_rows = read_reference(reference_path)
    except (OSError, ValueError, csv.Error) as error:
        reference_rows, reference_error = [], str(error)
        print('通信参考不可用，方法1继续：' + reference_error, flush=True)
    report = {'schema_version': SCHEMA_VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
              'source_kind': 'offline_kernel_csv', 'profile_version': PROFILE_VERSION, 'cases': [], 'failures': [],
              'communication_reference': {'path': str(reference_path) if reference_path is not None else None,
                                          'unit': 'unknown; ratios cancel common units', 'rows': reference_rows,
                                          'error': reference_error},
              'contract': {'source_depth': '8 main + 3 draft', 'timestamps': 'integer ns', 'timings': 'ms',
                           'throughput': 'new output token/s/physical card',
                           'projection': 'exclude main0 reference; per-rank per-round attention means, rank means then maximum'}}
    seen, jobs, failed = set(), [], {}
    for index, item in enumerate(config['cases']):
        name = item.get('name', 'unnamed') if isinstance(item, dict) else 'invalid'
        print('统计 ' + str(name), flush=True)
        try:
            if name in seen:
                raise ValueError('Duplicate case name: ' + name)
            seen.add(name)
            case = normalize_case(defaults, item, base)
            slug = re.sub(r'[^\w.-]+', '_', case['name'])[:48] or 'case'
            suffix = hashlib.sha256(case['name'].encode()).hexdigest()[:8]
            case['_step_residual_dir'] = str((args.out / 'step_residual_details' / (slug + '-' + suffix)).resolve())
            case_target = referenced_object(case['target_model'], base, 'case.target_model') if case.get('target_model') is not None else target
            validate_models(model, case_target)
            jobs.append((index, case, model, case_target, base, reference_rows))
        except (ValueError, KeyError, OSError, ArithmeticError, TypeError, re.error) as error:
            failed[index] = {'case': str(name), 'error': str(error)}
            print('  FAILED: ' + str(error), flush=True)
    workers = _cli_case_workers(args.workers, len(jobs))
    for payload in _cli_run_case_jobs(jobs, workers):
        if 'failure' in payload:
            failed[payload['index']] = payload['failure']
        else:
            report['cases'].append(payload['result'])
            report['contract'].update(payload['contract'])
    report['failures'] = [failed[index] for index in sorted(failed)]
    if not report['cases']:
        # Keep the contract identical even when no case reached finalization.
        attach_layer_mean_report(report)
    for result in report['cases']:
        print_case_result(result)
    export_report(report, args.out, reference_rows, progress=lambda message: print(message, flush=True))
    print('结果: ' + str((args.out / 'report.html').resolve()), flush=True)
    if args.excel:
        build_cli_excel(report, args.out, 'layer_mean_methods')
        print('Excel: ' + str((args.out / ESTIMATE_XLSX).resolve()), flush=True)
    return int(bool(report['failures']) or not layer_mean_complete(report))



if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
