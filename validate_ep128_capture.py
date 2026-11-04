#!/usr/bin/env python3
"""Independent, read-only checks of captured 8-main/3-draft device CSVs.

Explicit measurement-only probes accept a configured main depth plus 3 draft.

Uses no estimator imports, Torch, NPU, service calls, or trace JSON. Optionally
compares the small rounds.csv/layers.csv/per_rank.csv exports. Checks device
accounting and input contracts, not target hardware prediction accuracy.
"""
import argparse
import bisect
import collections
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics


PRE = re.compile(r"(?:^|_)HcPre(?:V\d+)?(?:_|$)", re.I)
POST = re.compile(r"(?:^|_)HcPost(?:V\d+)?(?:_|$)", re.I)
DISPATCH = re.compile(r"MoeDistributeDispatch", re.I)
COMBINE = re.compile(r"MoeDistributeCombine", re.I)
INDEXER = re.compile(r"LightningIndexer|IndexerTopK|SparseIndexer", re.I)
NATIVE = re.compile(r"(?:aclnn)?MoeDistributeDispatch(?:V([123]))?", re.I)
NS_PER_MS = 1_000_000


def configured_effective_output(case):
    """Parse the conditional numerator independently of the estimator.

    synthetic_acceptance_length is already mean new output including bonus;
    it is not an accepted-draft count to which one should be added. None of
    these configuration fields independently measures accepted output tokens.
    """
    s = case.get('draft_tokens')
    if type(s) is not int or s < 0:
        raise ValueError('draft_tokens must be an explicit nonnegative integer')

    def bounded(value, name, maximum):
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 < value <= maximum):
            raise ValueError('Invalid configured effective output length: ' + name)
        return float(value)

    explicit = bounded(case['effective_tokens_per_round'], 'effective_tokens_per_round', 1 + s) \
        if 'effective_tokens_per_round' in case else None
    synthetic = bounded(case['synthetic_acceptance_length'], 'synthetic_acceptance_length', 1 + s) \
        if 'synthetic_acceptance_length' in case else None
    if explicit is not None and synthetic is not None and not math.isclose(explicit, synthetic, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('effective_tokens_per_round conflicts with synthetic_acceptance_length')
    if explicit is not None:
        value, field = explicit, 'effective_tokens_per_round'
    elif synthetic is not None:
        value, field = synthetic, 'synthetic_acceptance_length_including_bonus'
    elif case.get('acceptance_semantics') == 'mean_accepted_draft_fraction':
        fraction = case.get('acceptance')
        if (isinstance(fraction, bool) or not isinstance(fraction, (float, int))
                or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError('Invalid configured mean_accepted_draft_fraction')
        value, field = 1 + s * fraction, '1+S*explicit_mean_accepted_draft_fraction'
    else:
        raise ValueError('Invalid configured effective output length: set Neff, synthetic length, or explicit acceptance semantics')
    return value, {'effective_tokens_per_round': value, 'source_field': field,
                   'declared_token_basis': case.get('token_basis'),
                   'basis': 'CONFIGURED_NUMERATOR_ARITHMETIC_ONLY_NOT_MEASURED_ACCEPTED_OUTPUT',
                   'bonus': 'Synthetic length already includes bonus; no additional +1 is applied'}


@dataclass(frozen=True)
class Task:
    row: int
    start: int
    end: int
    name: str
    kind: str
    shapes: str
    dtypes: str
    device: str

    @property
    def label(self):
        return self.name + " " + self.kind


def exact_ns(value):
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError("Invalid microsecond timestamp: " + repr(value)) from error
    if not number.is_finite():
        raise ValueError("Nonfinite timestamp: " + repr(value))
    return int((number * 1000).to_integral_value())


def field(row, *keys):
    entries = [str(row[key]).strip() for key in keys
               if row.get(key) is not None and str(row[key]).strip() not in ("", "N/A", "None")]
    if len(set(entries)) > 1:
        raise ValueError("Conflicting aliases: " + repr(keys))
    return entries[0] if entries else ""


def read_tasks(path):
    before = path.stat()
    tasks, devices, seen, duplicate_rows, dropped = [], set(), set(), 0, 0
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        headers = [str(key).strip() for key in rows.fieldnames or []]
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("Empty or duplicate CSV header")
        rows.fieldnames = headers
        for group in (("Name", "Op Name"), ("Start Time(us)", "Start Time (us)"),
                      ("Duration(us)", "Duration (us)")):
            if not set(group).intersection(headers):
                raise ValueError("Missing column: " + group[0])
        for number, row in enumerate(rows, 1):
            if None in row:
                raise ValueError("CSV record contains extra fields: " + str(number))
            stamp = field(row, "Start Time(us)", "Start Time (us)")
            if not stamp:
                dropped += 1
                continue
            start = exact_ns(stamp)
            duration = exact_ns(field(row, "Duration(us)", "Duration (us)"))
            if duration < 0:
                raise ValueError("Negative duration at data record " + str(number))
            name = field(row, "Name", "Op Name")
            if not name:
                raise ValueError("Missing task name at data record " + str(number))
            device = field(row, "Device_id", "Device ID", "Device Id", "device_id")
            if device:
                devices.add(device)
            metadata = tuple(field(row, *aliases) for aliases in (
                ("Stream ID", "Stream Id", "stream_id"), ("Task ID", "Task Id", "task_id"),
                ("Model ID", "Model Id", "model_id"), ("Context ID", "Context Id", "context_id"),
                ("Batch ID", "Batch Id", "batch_id"), ("Subtask ID", "Subtask Id", "SubTask ID", "subtask_id")))
            shape, dtype, kind = (field(row, key) for key in ("Input Shapes", "Input Data Types", "Type"))
            identity = (start, duration, name, kind, device, metadata, shape, dtype,
                        field(row, "Output Shapes"), field(row, "Output Data Types"),
                        field(row, "Wait Time(us)", "Wait Time (us)"),
                        str(row.get("Task Type") or row.get("Accelerator Core") or "").strip(),
                        field(row, "OP State", "Op State"))
            if identity in seen:
                duplicate_rows += 1
                continue
            seen.add(identity)
            tasks.append(Task(number, start, start + duration, name, kind, shape, dtype, device))
    if len(devices) > 1:
        raise ValueError("One CSV contains multiple device IDs: " + repr(sorted(devices)))
    if not any(task.end > task.start for task in tasks):
        raise ValueError("No nonempty device tasks")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("CSV changed during independent validation")
    return sorted(tasks, key=lambda task: (task.start, task.end, task.row)), {
        "headers": headers, "device_id": next(iter(devices), None),
        "device_tasks": len(tasks), "duplicate_rows": duplicate_rows,
        "untimed_rows": dropped, "csv_sha256": digest.hexdigest()}


def dispatch_shape(task):
    """Check the two explicit supported schemas, never search for width K."""
    versions = [match.group(1) or "1" for label in (task.name, task.kind)
                if (match := NATIVE.fullmatch(label))]
    unknown = {"layout": "unknown", "x": None, "ids": None}
    if not versions or len(set(versions)) != 1:
        return unknown
    shapes = [[int(value) for value in re.findall(r"-?\d+", slot)]
              for slot in task.shapes.split(";")]
    types = [re.sub(r"[^A-Z0-9]", "", value.upper()) for value in task.dtypes.split(";")] if task.dtypes else []
    if types and len(types) != len(shapes):
        return unknown
    is_ids = lambda value: value in ("INT32", "DTINT32", "TORCHINT32")
    is_activation = lambda value: value in (
        "FLOAT", "FLOAT32", "DTFLOAT", "DTFLOAT32", "FP32", "FLOAT16", "DTFLOAT16",
        "FP16", "HALF", "TORCHFLOAT16", "BF16", "DTBF16", "BFLOAT16",
        "DTBFLOAT16", "TORCHBFLOAT16", "INT8", "DTINT8", "TORCHINT8",
        "FLOAT8E4M3FN", "DTFLOAT8E4M3FN", "FLOAT8E5M2", "DTFLOAT8E5M2")
    matrix = lambda shape: len(shape) == 2 and all(value > 0 for value in shape)
    if len(shapes) >= 2 and matrix(shapes[0]) and matrix(shapes[1]):
        if not types or is_activation(types[0]) and is_ids(types[1]):
            return {"layout": "standard" if types else "standard_without_dtype",
                    "x": shapes[0], "ids": shapes[1]}
    if (versions[0] == "3" and len(shapes) >= 3 and types
            and len(shapes[0]) == 1 and shapes[0][0] > 0 and is_ids(types[0])
            and matrix(shapes[1]) and is_activation(types[1])
            and matrix(shapes[2]) and is_ids(types[2])):
        return {"layout": "leading_int32_context", "x": shapes[1], "ids": shapes[2]}
    return unknown


def coverage(tasks, start, end):
    """An independent event-count sweep of clipped intervals across all streams."""
    changes = collections.defaultdict(int)
    kernel_sum, calls = 0, 0
    changes[start] += 0
    changes[end] += 0
    for task in tasks:
        left, right = max(task.start, start), min(task.end, end)
        if right <= left:
            continue
        calls += 1
        kernel_sum += right - left
        changes[left] += 1
        changes[right] -= 1
    count, busy, previous = 0, 0, start
    for point in sorted(changes):
        if count:
            busy += point - previous
        count += changes[point]
        if count < 0:
            raise ValueError("Independent interval sweep became negative")
        previous = point
    if count or not 0 <= busy <= end - start:
        raise ValueError("Independent interval sweep did not close")
    return {"duration_ms": (end - start) / NS_PER_MS, "busy_ms": busy / NS_PER_MS,
            "no_recorded_ms": (end - start - busy) / NS_PER_MS,
            "kernel_sum_ms": kernel_sum / NS_PER_MS, "calls": calls}


class CoverageIndex:
    """Global event integrals; queries include tasks crossing either boundary."""
    def __init__(self, tasks):
        changes = collections.defaultdict(int)
        self.starts, self.ends = [], []
        for task in tasks:
            if task.end <= task.start:
                continue
            changes[task.start] += 1
            changes[task.end] -= 1
            self.starts.append(task.start)
            self.ends.append(task.end)
        self.starts.sort()
        self.ends.sort()
        self.points = sorted(changes)
        self.busy, self.summed, self.counts = [], [], []
        active = busy = summed = 0
        previous = self.points[0]
        for point in self.points:
            busy += (point - previous) if active else 0
            summed += (point - previous) * active
            active += changes[point]
            if active < 0:
                raise ValueError("Global event sweep became negative")
            self.busy.append(busy)
            self.summed.append(summed)
            self.counts.append(active)
            previous = point
        if active:
            raise ValueError("Global event sweep did not close")

    def integral(self, point, summed=False):
        index = bisect.bisect_right(self.points, point) - 1
        if index < 0:
            return 0
        prefix = self.summed[index] if summed else self.busy[index]
        count = self.counts[index] if summed else int(self.counts[index] > 0)
        return prefix + (point - self.points[index]) * count

    def query(self, start, end):
        busy = self.integral(end) - self.integral(start)
        summed = self.integral(end, True) - self.integral(start, True)
        if not 0 <= busy <= end - start:
            raise ValueError("Global event coverage does not close")
        return {"duration_ms": (end - start) / NS_PER_MS, "busy_ms": busy / NS_PER_MS,
                "no_recorded_ms": (end - start - busy) / NS_PER_MS,
                "kernel_sum_ms": summed / NS_PER_MS,
                "calls": bisect.bisect_left(self.starts, end) - bisect.bisect_right(self.ends, start)}


def source_schedule(model, *, measurement_only=False):
    ratios = model.get("compress_ratios", [])
    main = model.get("num_hidden_layers")
    if (type(main) is not int or main < 1 or (not measurement_only and main != 8)
            or not isinstance(ratios, list) or len(ratios) < main+3):
        raise ValueError("Independent validator expects configured 8 main + 3 draft layers")
    if measurement_only and len(ratios) != main + 3:
        raise ValueError("measurement-only requires exact main+3 source compress_ratios")
    hashes = model.get("num_hash_layers")
    if type(hashes) is not int or not 0 <= hashes <= main:
        raise ValueError("Invalid source num_hash_layers")
    schedule = []
    for index, ratio in enumerate(ratios[:main+3]):
        if type(ratio) is not int or ratio not in (0, 4, 128):
            raise ValueError("Unsupported source attention ratio")
        schedule.append(("main" if index < main else "draft", index if index < main else index - main,
                         {0: "SWA", 4: "CSA", 128: "HCA"}[ratio],
                         "hash" if index < hashes else "normal"))
    return schedule


def infer_cycles(tasks, schedule, case):
    """Dispatch-driven HC matching, separate from the estimator's PRE triples."""
    pres = [task for task in tasks if PRE.search(task.name) or PRE.search(task.kind)]
    pre_times = [task.start for task in pres]
    posts = [task for task in tasks if POST.search(task.name) or POST.search(task.kind)]
    dispatches = [task for task in tasks if DISPATCH.search(task.label)]
    combines = [task for task in tasks if COMBINE.search(task.label)]
    indexer = re.compile(case.get("csa_indexer_regex"), re.I) if case.get("csa_indexer_regex") else INDEXER
    indexers = [task for task in tasks if indexer.search(task.label) and "metadata" not in task.label.lower()]
    starts = {kind: [task.start for task in values] for kind, values in (
        ("dispatch", dispatches), ("combine", combines), ("post", posts), ("indexer", indexers))}
    def inside(values, kind, start, end):
        return values[bisect.bisect_left(starts[kind], start):bisect.bisect_left(starts[kind], end)]
    blocks = []
    for dispatch in dispatches:
        ffn = bisect.bisect_right(pre_times, dispatch.start) - 1
        if ffn < 1 or ffn + 1 >= len(pres):
            continue  # Partial capture head/tail cannot form a complete marker block.
        attention, middle, following = pres[ffn - 1], pres[ffn], pres[ffn + 1]
        dcalls = inside(dispatches, "dispatch", middle.start, following.start)
        ccalls = inside(combines, "combine", attention.start, following.start)
        ap = inside(posts, "post", attention.start, middle.start)
        fp = inside(posts, "post", middle.start, following.start)
        if len(dcalls) != 1 or len(ccalls) != 1 or len(ap) != 1 or len(fp) != 1:
            raise ValueError("Independent Dispatch-based HC matching found ambiguous marker counts")
        chain = [attention, ap[0], middle, dispatch, ccalls[0], fp[0], following]
        if any(left.end > right.start for left, right in zip(chain, chain[1:])):
            raise ValueError("Independent HC/Dispatch/Combine dependency ordering failed")
        blocks.append({"start": attention.start, "end": fp[0].end, "next": following.start,
                       "dispatch": dispatch, "combine": ccalls[0],
                       "csa": bool(inside(indexers, "indexer", attention.start, middle.start))})
    period = len(schedule)
    signature = [item[2] == "CSA" for item in schedule]
    override = case.get("phase_offset")
    candidates = []
    for phase in range(period):
        full = (len(blocks) - phase) // period
        if full >= 2 and all(blocks[phase + index]["csa"] == signature[index % period]
                             for index in range(full * period)):
            candidates.append(phase)
    if override is not None:
        if type(override) is not int or not 0 <= override < period:
            raise ValueError("Invalid explicit phase_offset")
        if any(block["csa"] for block in blocks) and override not in candidates:
            raise ValueError("Explicit phase conflicts with independent CSA sequence")
        phase = override
    elif len(candidates) == 1:
        phase = candidates[0]
    else:
        raise ValueError("Independent CSA period has no unique phase: " + repr(candidates))
    windows = []
    for offset in range(phase, len(blocks) - period, period):
        selected = blocks[offset:offset + period]
        if any(left["next"] != right["start"] for left, right in zip(selected, selected[1:])):
            raise ValueError("Independent HC blocks are not contiguous")
        if selected[-1]["next"] != blocks[offset + period]["start"]:
            raise ValueError("Independent HC cycle endpoint is not contiguous")
        start, end = selected[0]["start"], blocks[offset + period]["start"]
        marker_counts = {
            "pre": bisect.bisect_left(pre_times, end) - bisect.bisect_left(pre_times, start),
            "post": len(inside(posts, "post", start, end)),
            "dispatch": len(inside(dispatches, "dispatch", start, end)),
            "combine": len(inside(combines, "combine", start, end))}
        if marker_counts != {"pre": 2*period, "post": 2*period, "dispatch": period, "combine": period}:
            raise ValueError(f"Independent cycle is not {2*period}PRE/{2*period}POST/{period}Dispatch/{period}Combine: " + repr(marker_counts))
        windows.append({"round_id": "hc-" + str(len(windows) + 1), "start": start, "end": end,
                        "blocks": selected, "marker_counts": marker_counts})
    trim = case.get("trim_rounds", 2)
    if type(trim) is not int or trim < 0 or len(windows) <= 2 * trim:
        raise ValueError("Not enough independently complete cycles for configured trim")
    return windows[trim:len(windows) - trim] if trim else windows, {
        "phase_offset": phase, "candidate_phases": candidates, "complete_cycles_before_trim": len(windows)}


def rank_from_path(path):
    for component in reversed(path.parts[:-1]):
        match = re.search(r"(?:^|[_-])dp[_-]?(\d+)(?=$|[_-])", component, re.I)
        if match:
            return "dp" + str(int(match[1]))
        match = re.search(r"(?:^|[_-])rank[_-]?(\d+)(?=$|[_-])", component, re.I)
        if match:
            return "dp" + str(int(match[1]))
    raise ValueError("Cannot independently identify a global rank from " + str(path))


def discover(case, base):
    def files(value):
        path = Path(value).expanduser()
        path = path if path.is_absolute() else base / path
        if path.is_file():
            return [path.resolve()]
        if not path.is_dir():
            raise ValueError("Capture path missing: " + str(path))
        return sorted(Path(directory, name).resolve()
                      for directory, _, names in os.walk(path)
                      for name in names if name == "kernel_details.csv")
    if case.get("inputs") is not None:
        selected = []
        for item in case["inputs"]:
            found = files(item["path"])
            if len(found) != 1:
                raise ValueError("Explicit rank input must select exactly one CSV")
            selected.append((found[0], item["rank"]))
    else:
        found = sorted({path for source in case.get("paths", [case["path"]] if case.get("path") else [])
                        for path in files(source)})
        selected = [(path, rank_from_path(path)) for path in found]
    if not selected:
        raise ValueError("No input CSVs")
    names = [rank for _, rank in selected]
    if len(set(names)) != len(names) or len(set(path for path, _ in selected)) != len(selected):
        raise ValueError("Duplicate global rank or repeated CSV path")
    return selected


def read_exports(directory):
    outputs = {}
    for name in ("rounds", "layers", "per_rank"):
        with (directory / (name + ".csv")).open(encoding="utf-8-sig", newline="") as source:
            outputs[name] = list(csv.DictReader(source))
    for name, fields in (("rounds", ("case", "rank", "round_id")),
                         ("layers", ("case", "rank", "round_id", "role", "layer")),
                         ("per_rank", ("case", "rank"))):
        index = collections.defaultdict(list)
        for row in outputs[name]:
            index[tuple(row.get(field) for field in fields)].append(row)
        outputs[name + "_index"] = index
    return outputs


def near(value, expected, label, tolerance=0.0000011):
    try:
        actual = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(label + ": missing/non-numeric exported value") from error
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=tolerance):
        raise ValueError(label + f": export={actual!r}, independent={expected!r}")


def inspect_step_summary(directory):
    """Inspect native step-summary metadata without treating aggregate Free as idle proof."""
    path = directory / "step_trace_time.csv"
    if not path.is_file():
        return {"status": "NOT_AVAILABLE", "scope": "No independent step-boundary evidence"}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        rows = list(reader)
    labels = [str(row.get("Step", "")).strip() for row in rows if str(row.get("Step", "")).strip()]
    return {"path": str(path), "rows": len(rows), "nonempty_step_labels": labels[:10],
            "nonempty_step_label_count": len(labels), "headers": headers,
            "status": "BOUNDARIES_NOT_ESTABLISHED",
            "scope": "Native aggregate Step/Stage/Free columns do not establish HC-cycle/request-iteration alignment"}


def verify_rank(path, rank, model, case, exported=None):
    tasks, metadata = read_tasks(path)
    layouts, shape_samples = collections.Counter(), []
    for task in tasks:
        if not DISPATCH.search(task.label):
            continue
        shape = dispatch_shape(task)
        layouts[shape["layout"]] += 1
        if len(shape_samples) < 3:
            shape_samples.append({"data_record": task.row, "name": task.name, "type": task.kind,
                                  "input_shapes": task.shapes, "input_dtypes": task.dtypes, **shape})
        if shape["x"] is not None:
            if shape["x"][1] != model["hidden_size"] or shape["ids"][1] != model["num_experts_per_tok"]:
                raise ValueError("Independent Dispatch H/K contradicts source model at data record " + str(task.row))
            if shape["x"][0] != shape["ids"][0]:
                raise ValueError("Independent Dispatch activation/IDs row mismatch")
    schedule = source_schedule(model, measurement_only=case.get("measurement_only") is True)
    windows, alignment = infer_cycles(tasks, schedule, case)
    coverage_index = CoverageIndex(tasks)
    statistics_rows, exported_outputs, numerator_bases = [], [], collections.Counter()
    for window in windows:
        stats = coverage_index.query(window["start"], window["end"])
        statistics_rows.append(stats)
        if exported is None:
            continue
        rounds = exported["rounds_index"].get((case["name"], rank, window["round_id"]), [])
        if len(rounds) != 1:
            raise ValueError("Export does not contain one matching independent cycle: " + rank + "/" + window["round_id"])
        row = rounds[0]
        if int(row["start_ns"]) != window["start"] or int(row["end_ns"]) != window["end"]:
            raise ValueError("Exported source cycle boundary differs from independent markers")
        near(row.get("duration_ms"), stats["duration_ms"], "cycle duration")
        # No output field in a generated report independently proves actual
        # accepted tokens. Still reject impossible values and audit its basis.
        basis = str(row.get("token_basis", ""))
        output = float(row.get("output_tokens", "nan"))
        if not math.isfinite(output) or not 0 <= output <= case["bs"] * (1 + case["draft_tokens"]):
            raise ValueError("Exported output_tokens is outside the physical BS*(1+S) bound")
        if basis == "CONDITIONAL_configured_BS_times_Neff_not_measured_accepted_tokens":
            near(output, case["bs"] * case["effective_tokens_per_round"], "configured numerator")
            numerator_bases["CONFIGURED_NUMERATOR_ARITHMETIC_ONLY"] += 1
        elif basis == "measured_accepted_tokens":
            if output != int(output):
                raise ValueError("Declared measured_accepted_tokens must be an integer")
            numerator_bases["EXPORTED_MEASURED_COUNT_NOT_INDEPENDENTLY_ATTESTED"] += 1
        else:
            numerator_bases["UNVERIFIED_UNKNOWN_TOKEN_BASIS"] += 1
        exported_outputs.append(output)
        for block, (role, number, attention, router) in zip(window["blocks"], schedule):
            layers = exported["layers_index"].get((case["name"], rank, window["round_id"], role, str(number)), [])
            if len(layers) != 1:
                raise ValueError("Exported source layer identity missing/duplicate")
            layer = layers[0]
            if layer.get("attention") != attention or layer.get("router") != router:
                raise ValueError("Exported source layer classification differs from schedule")
            if int(layer["start_ns"]) != block["start"] or int(layer["end_ns"]) != block["end"]:
                raise ValueError("Exported HC marker layer boundaries differ")
            near(layer.get("wall_ms"), (block["end"] - block["start"]) / NS_PER_MS, "HC wall")
            near(layer.get("dispatch_ms"), (block["dispatch"].end - block["dispatch"].start) / NS_PER_MS, "Dispatch")
            near(layer.get("combine_ms"), (block["combine"].end - block["combine"].start) / NS_PER_MS, "Combine")
    mean_cycle = statistics.fmean(row["duration_ms"] for row in statistics_rows)
    mean_busy = statistics.fmean(row["busy_ms"] for row in statistics_rows)
    mean_uncovered = statistics.fmean(row["no_recorded_ms"] for row in statistics_rows)
    conditional_group_rate = 1000 * case["bs"] * case["effective_tokens_per_round"] / mean_cycle
    dp, devices = case.get("source_dp"), case.get("source_devices")
    conditional_card_rate = conditional_group_rate * dp / devices if type(dp) is int and type(devices) is int and devices > 0 else None
    if exported is not None:
        rows = exported["per_rank_index"].get((case["name"], rank), [])
        if len(rows) != 1:
            raise ValueError("Exported per_rank identity missing/duplicate")
        row = rows[0]
        if int(row.get("rounds", -1)) != len(windows):
            raise ValueError("Exported retained cycle count differs")
        near(row.get("cycle_ms_mean"), mean_cycle, "rank cycle mean")
        near(row.get("recorded_busy_ms_mean"), mean_busy, "all-stream busy mean")
        near(row.get("no_recorded_ms_mean"), mean_uncovered, "no-recorded mean")
        exported_rate = 1000 * sum(exported_outputs) / sum(stats["duration_ms"] for stats in statistics_rows)
        near(row.get("conditional_per_rank_tokens_s"), exported_rate, "exported numerator DP-group rate")
        if conditional_card_rate is not None:
            near(row.get("conditional_per_device_tokens_s"), exported_rate * dp / devices,
                 "exported numerator physical-card rate")
    numerator_status = ("NOT_COMPARED_NO_CORE_EXPORT" if exported is None else
                        "UNVERIFIED_UNKNOWN_TOKEN_BASIS" if numerator_bases.get("UNVERIFIED_UNKNOWN_TOKEN_BASIS") else
                        "EXPORTED_MEASURED_COUNT_NOT_INDEPENDENTLY_ATTESTED"
                        if numerator_bases.get("EXPORTED_MEASURED_COUNT_NOT_INDEPENDENTLY_ATTESTED") else
                        "CONFIGURED_NUMERATOR_ARITHMETIC_ONLY")
    return {"rank": rank, "path": str(path), **metadata, **alignment, "retained_cycles": len(windows),
            "dispatch_layout_counts": dict(layouts), "dispatch_samples": shape_samples,
            "source_cycle_mean_ms": mean_cycle, "all_stream_busy_mean_ms": mean_busy,
            "no_recorded_mean_ms": mean_uncovered,
            "accounting_closed": math.isclose(mean_cycle, mean_busy + mean_uncovered, abs_tol=1e-12),
            "conditional_source_card_rate": conditional_card_rate,
            "numerator_scope": "configured_BS_times_Neff_not_independent_measurement_of_accepted_output",
            "numerator_comparison_status": numerator_status, "numerator_basis_counts": dict(numerator_bases),
            "step_marker_crosscheck": inspect_step_summary(path.parent),
            "FULL_graph_identity": "NOT_ESTABLISHED_BY_KERNEL_NAMES_OR_MODEL_ID"}


def validate(manifest, stats_dir=None, *, measurement_only=False):
    manifest = Path(manifest).resolve()
    config = json.loads(manifest.read_text(encoding="utf-8-sig"))
    model = config["sample_model"]
    if isinstance(model, str):
        model = json.loads((manifest.parent / model).read_text(encoding="utf-8-sig"))
    exported = read_exports(Path(stats_dir)) if stats_dir is not None else None
    result = {"validator": "independent_dispatch_driven_csv_accounting_v1", "cases": [],
              "scope": "software_contract_and_device_timestamp_accounting_not_EP128_hardware_accuracy",
              "step_or_graph_evidence": "Separate trace/client evidence is required; no synthetic ownership is created",
              "capture_acceptance_status": "NOT_ESTABLISHED",
              "capture_acceptance_scope": "Device accounting PASS is not same-run/load/graph/accepted-output acceptance"}
    for item in config["cases"]:
        case = {**config.get("defaults", {}), **item}
        if type(case.get("measurement_only", False)) is not bool:
            raise ValueError("measurement_only must be a boolean")
        case["measurement_only"] = measurement_only or case.get("measurement_only", False)
        record = {"case": case.get("name"), "ranks": [], "errors": [],
                  "measurement_only": case["measurement_only"],
                  "capture_cohort_status": "NOT_ESTABLISHED",
                  "capture_cohort_scope": "Global rank names, file timestamps and complete counts do not prove one controller run"}
        try:
            if case.get("draft_layers", 3) != 3 or case.get("draft_passes", 1) != 1:
                raise ValueError("Expected one pass of three draft layers")
            case.setdefault("source_chips_per_card", 1)
            for name in ("bs", "expected_ranks", "source_dp", "source_devices", "source_chips_per_card"):
                if type(case.get(name)) is not int or case[name] < 1:
                    raise ValueError(name + " must be an explicit positive integer")
            logical = case["source_devices"] * case["source_chips_per_card"]
            if case["expected_ranks"] != logical or case["source_dp"] > logical:
                raise ValueError("Expected-rank/DP topology contradicts physical_cards * chips_per_card logical NPU budget")
            if "base_ep" in case and (type(case["base_ep"]) is not int or not 0 < case["base_ep"] <= logical):
                raise ValueError("base_ep must fit the explicit logical NPU budget")
            record["source_topology"] = {"physical_cards": case["source_devices"],
                "chips_per_card": case["source_chips_per_card"], "logical_devices": logical,
                "dp_request_groups": case["source_dp"], "expert_parallel": case.get("base_ep")}
            if type(case.get("draft_tokens")) is not int or case["draft_tokens"] < 0:
                raise ValueError("draft_tokens must be an explicit nonnegative integer")
            neff, numerator_config = configured_effective_output(case)
            case['effective_tokens_per_round'] = neff
            record['conditional_numerator_config'] = numerator_config
            selected = discover(case, manifest.parent)
            expected = case.get("expected_rank_names", ["dp" + str(i) for i in range(case["expected_ranks"])])
            if (not isinstance(expected, list) or len(expected) != case["expected_ranks"]
                    or any(not isinstance(name, str) or not name for name in expected)
                    or len(set(expected)) != len(expected)):
                raise ValueError("expected_rank_names must list every expected global rank exactly once")
            observed = [rank for _, rank in selected]
            record.update(expected_ranks=expected, declared_or_path_inferred_ranks=observed,
                          missing_ranks=[rank for rank in expected if rank not in observed],
                          unexpected_ranks=[rank for rank in observed if rank not in expected])
            if set(expected) != set(observed):
                record["errors"].append("Global rank coverage incomplete")
            for path, rank in selected:
                try:
                    print(f"独立核对 {case['name']} {rank}: {path}", flush=True)
                    record["ranks"].append(verify_rank(path, rank, model, case, exported))
                except (ValueError, OSError, KeyError, ArithmeticError) as error:
                    record["errors"].append(rank + ": " + str(error))
            hashes = [rank["csv_sha256"] for rank in record["ranks"]]
            if len(set(hashes)) != len(hashes):
                record["errors"].append("Identical CSV contents appear under multiple global rank labels")
        except (ValueError, OSError, KeyError, ArithmeticError) as error:
            record["errors"].append(str(error))
        record["status"] = "PASS_DEVICE_ACCOUNTING" if not record["errors"] else "FAILED_OR_INCONCLUSIVE"
        result["cases"].append(record)
    result["all_device_accounting_passed"] = bool(result["cases"]) and all(not case["errors"] for case in result["cases"])
    result["core_exports_compared"] = exported is not None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--stats-dir", type=Path, help="Existing rounds.csv/layers.csv/per_rank.csv directory")
    parser.add_argument("--measurement-only", action="store_true", help="Explicit configured-main-depth + 3-draft measurement probe; no target projection validation")
    parser.add_argument("--out", type=Path, default=Path("independent_capture_validation.json"))
    args = parser.parse_args()
    report = validate(args.cases, args.stats_dir, measurement_only=args.measurement_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in report["cases"]:
        print(case["case"] + ": " + case["status"] + "; " + " | ".join(case["errors"][:5]), flush=True)
    print("NOT_CAPTURE_ACCEPTANCE：同次采样、真实decode/图身份及实际有效输出计数仍须独立证据。", flush=True)
    print("独立核对报告：" + str(args.out.resolve()), flush=True)
    return 0 if report["all_device_accounting_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
