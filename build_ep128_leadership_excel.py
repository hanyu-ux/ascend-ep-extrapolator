#!/usr/bin/env python3
"""Render the EP128 layer-mean report as XLSX and offline HTML.

Standard library + ep128_xlsx_writer.py only. All estimates, means and rank
selection come from profile_ep128.py; this module only displays core values.
"""
import argparse
import collections
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import sys
import traceback
import unicodedata

import ep128_xlsx_writer as xlsx_writer
from ep128_xlsx_writer import MAX_ROW, Workbook, cell_reference

RENDERER_VERSION = "20261110.2.bs_s_sweeps"
PENDING = "待估算"
PREVIEW = "版式预览"
METHOD_SET = "layer_mean_methods"
SHEETS = ("估算总览", "源采样基线", "层与组件实测", "时间账本", "方法1估算",
          "方法2估算", "通信参考原表", "公式与假设", "复核记录")
WIDTHS = [29, 12, 19, 19, 19, 19, 19, 19, 22, 22, 47, 24]
DEFAULT_DETAIL_ROWS = 1000
# Excel cells allow 32,767 UTF-16 code units; leave room for continuation labels.
TEXT_CHUNK_UNITS = 32000
TIME_FIELDS = ("cycle_ms", "per_device_tokens_s", "global_tokens_s", "tpot_ms")
LAYER_MEAN_SCOPE = "采样均值条件估算；主0不作参考，主/草稿分别取样，轮次剩余区间只计一次"
LAYER_GAP_NOTE = "层间衔接区间（HCPost结束→下层HCPre开始，含该段记录任务）"
LAYER_RESIDUAL_NOTE = "轮次剩余区间（主草稿切换及轮次尾部，保留一次）"
NA = "—"
LABELS = {"rank": "采样rank", "role": "主/草稿", "attention": "注意力类型",
          "source_cycle_ms": "源周期(ms)", "source_main_chain_ms": "源主模型层链(ms)",
          "source_draft_chain_ms": "源草稿层链(ms)", "cross_role_tail_residual_ms": "轮次剩余区间(ms)",
          "partition_total_ms": "源分区合计(ms)", "closed": "核心闭合", "scope": "范围"}


def runtime_diagnostics():
    """Describe the code actually loaded, independently of the current directory."""
    writer_path = getattr(xlsx_writer, "__file__", None)
    position_code = getattr(getattr(xlsx_writer, "_position", None), "__code__", None)
    return {
        "renderer_version": RENDERER_VERSION,
        "renderer_path": str(Path(__file__).resolve()),
        "writer_version": getattr(xlsx_writer, "WRITER_VERSION", "unversioned"),
        "writer_path": str(Path(writer_path).resolve()) if writer_path else "unknown",
        "position_code_path": getattr(position_code, "co_filename", "unknown"),
        "position_code_line": getattr(position_code, "co_firstlineno", None),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value):
    if value is None:
        return PENDING
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _text_parts(value):
    """Preserve long evidence verbatim across valid-sized worksheet cells."""
    encoded = value.encode("utf-16-le")
    parts, start = [], 0
    while start < len(encoded):
        end = min(start + 2 * TEXT_CHUNK_UNITS, len(encoded))
        if end < len(encoded) and 0xD800 <= int.from_bytes(encoded[end - 2:end], "little") <= 0xDBFF:
            end -= 2  # Do not split a surrogate pair.
        parts.append(encoded[start:end].decode("utf-16-le"))
        start = end
    return parts or [""]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("schema 2 JSON contains a nonfinite numeric value")


def _validate_report(report):
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 2:
        raise ValueError("仅支持schema_version=2；旧版JSON须用新版统计脚本重算")
    if not isinstance(report.get("cases"), list):
        raise ValueError("schema 2 report must contain a cases list")
    _finite(report)
    for case in report["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise ValueError("每个case必须含name")
        projection = case.get("projection") or {}
        if not isinstance(projection, dict):
            raise ValueError("projection must be an object")
        selected = projection.get("display_method_set")
        if selected != METHOD_SET:
            raise ValueError(f"case={case['name']!r}；仅支持{METHOD_SET}，当前display_method_set={selected!r}；请重跑统计脚本")
        methods = projection.get(METHOD_SET, [])
        if not isinstance(methods, list) or any(not isinstance(item, dict) for item in methods):
            raise ValueError("projection." + METHOD_SET + " must be a list of method objects")
        identities = [item.get("id") for item in methods]
        if any(item not in ("method1", "method2") for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("方法id须为唯一的method1/method2")
        for method in methods:
            if method.get("status") not in ("LAYER_MEAN_ESTIMATE", "PENDING", "UNAVAILABLE"):
                raise ValueError("layer_mean_methods requires LAYER_MEAN_ESTIMATE, PENDING or UNAVAILABLE")
            if method.get("status") != "LAYER_MEAN_ESTIMATE":
                continue
            for field in TIME_FIELDS:
                if not _number(method.get(field)) or method[field] <= 0:
                    raise ValueError("LAYER_MEAN_ESTIMATE requires positive finite core " + field)
                if method.get(field + "_range") is not None:
                    raise ValueError("LAYER_MEAN_ESTIMATE must not contain result ranges")
            selected_rank = method.get("selected_rank")
            ranks = method.get("per_rank") or []
            if (not isinstance(ranks, list) or selected_rank is None or
                    sum(isinstance(item, dict) and item.get("rank") == selected_rank for item in ranks) != 1):
                raise ValueError("LAYER_MEAN_ESTIMATE requires one explicit selected_rank in per_rank")
    return report


def _object(value, base):
    return _read_json(base / value) if isinstance(value, str) else value


def _preview_report(path):
    path = Path(path)
    config = _read_json(path)
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 2:
        raise ValueError("预览仅支持schema_version=2配置")
    if not isinstance(config.get("cases"), list):
        raise ValueError("schema 2配置必须含cases列表")
    _finite(config)
    model = _object(config.get("sample_model"), path.parent) or {}
    target = _object(config.get("target_model"), path.parent) or {}
    cases = []
    for item in config["cases"]:
        normalized = {**config.get("defaults", {}), **item}
        case_target = (_object(normalized["target_model"], path.parent)
                       if normalized.get("target_model") is not None else target)
        cases.append(dict(name=normalized.get("name", "未命名组合"), config=normalized,
            source_model=model, target_model=case_target, captures=[],
            source=dict(status="PREVIEW", per_rank=[], aggregate=dict(scope="PREVIEW")),
            projection=dict(display_method_set=METHOD_SET, layer_mean_methods=[
                dict(id=identity, status="PENDING", reasons=["仅配置版式预览；未读取kernel CSV，未产生性能数值"])
                for identity in ("method1", "method2")])))
    return dict(schema_version=2, generated_at=datetime.now(timezone.utc).isoformat(),
                source_kind="configuration_preview", cases=cases, failures=[])


def _method(case, identity):
    return next((item for item in (case.get("projection") or {}).get(METHOD_SET, [])
                 if item.get("id") == identity),
                dict(id=identity, status="PENDING", reasons=["核心报告未输出本方法：" + identity]))


def _layer_mean_publishable(method):
    return method.get("status") == "LAYER_MEAN_ESTIMATE" and not method.get("reasons")


def _target(method, field):
    value = method.get(field)
    return value if _layer_mean_publishable(method) and _number(value) else PENDING


def _explanation(method):
    if method.get("reasons"):
        return method["reasons"]
    return {"层均值口径": LAYER_MEAN_SCOPE, "假设": method.get("assumptions"),
            "限制": method.get("warnings"), "核心选中rank": method.get("selected_rank")}


def _source(case):
    return (case.get("source") or {}).get("aggregate") or {}


def _sheet(book, name, heading, subtitle):
    sheet = book.add_sheet(name, widths=WIDTHS, paper_size=8)
    sheet.detail_rows = book.detail_rows
    sheet.report_origin = book.report_origin
    sheet.merge(1, 1, 2, 12, heading, "title").row_height(1, 29).row_height(2, 14)
    sheet.merge(3, 1, 3, 12, subtitle, "subtitle").row_height(3, 44)
    sheet.freeze(5, 2)
    return sheet


def _note(sheet, row, text, warning=False, columns=12):
    parts = _text_parts(_text(text))
    for index, part in enumerate(parts):
        prefix = f"（续文 {index + 1}/{len(parts)}）" if index else ""
        sheet.merge(row, 1, row, columns, prefix + part,
                    "warning" if warning else "note").row_height(row, 48)
        row += 1
    return row


def _table(sheet, row, headers, rows, *, record_rows=None, spans=None, blank_columns=()):
    def put(current, col, value, style):
        first, last = spans[col - 1] if spans else (col, col)
        if first == last:
            sheet.write(current, first, value, style)
        else:
            sheet.merge(current, first, current, last, value, style)
    for col, label in enumerate(headers, 1):
        put(row, col, label, "header")
    sheet.row_height(row, 37)
    current, continued = row + 1, False
    for record in rows:
        if record_rows is not None:
            record_rows.append(current)
        cells = []
        for col, value in enumerate(record, 1):
            item = PENDING if value is None and col not in blank_columns else value
            if isinstance(item, (dict, list, tuple, bool)):
                item = _text(item)
            cells.append(_text_parts(item) if isinstance(item, str) else [item])
        physical_rows = max(map(len, cells), default=1)
        if current + physical_rows - 1 > MAX_ROW:
            raise ValueError(f"工作表“{sheet.name}”表格超出Excel行上限："
                             f"起始行={current}，所需续行={physical_rows}；完整数据见{sheet.report_origin}")
        continued = continued or physical_rows > 1
        for offset in range(physical_rows):
            for col, parts in enumerate(cells, 1):
                item = parts[offset] if offset < len(parts) else None
                if col == 1 and offset and item is None:
                    item = f"（上行续文 {offset + 1}/{physical_rows}）"
                put(current, col, item, "number" if _number(item) else
                    "warning" if item == PENDING else "body")
            sheet.row_height(current, 35)
            current += 1
    if continued:
        current = _note(sheet, current, "长文本已按Excel单元格长度限制分成续行；原始记录顺序和数值保留，续行不是新增统计记录。")
    return current + 1


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = prefix + "." + str(key) if prefix else str(key)
            if isinstance(item, dict):
                yield from _flatten(item, path)
            else:
                yield path, item
    else:
        yield prefix, value


def _records(sheet, row, records, preferred=(), *, detail=False, location=""):
    if detail:
        retained, total = [], 0
        for item in records:
            total += 1
            if len(retained) < sheet.detail_rows:
                retained.append(item)
        records = retained
        if total > len(retained):
            row = _note(sheet, row,
                f"明细预览：显示前 {len(retained)} / {total} 条（原始顺序，不作代表性抽样）；"
                "汇总值来自全量核心统计，不用预览重算。完整明细保存在核心JSON："
                + str(sheet.report_origin) + "；位置：" + location, warning=True)
    if not records:
        return _note(sheet, row, "核心未提供本项明细；不以0替代缺失证据。")
    fields = list(dict.fromkeys([key for key in preferred if any(key in item for item in records)] +
                              [key for item in records for key in item]))
    # Keep nested evidence intact; do not average, fit or derive new quantities.
    return _table(sheet, row, [LABELS.get(key, key) for key in fields],
                  [[str(item[key]) if key.endswith("_ns") and item.get(key) is not None else item.get(key)
                    for key in fields] for item in records])


def _breakdown_for_chart(method):
    rows = method.get("breakdown") or []
    total = method.get("cycle_ms")
    if not _layer_mean_publishable(method) or not _number(total) or total <= 0 or not rows:
        return None
    if any(not isinstance(item, dict) or not _number(item.get("value_ms")) for item in rows):
        return None
    if not math.isclose(sum(item["value_ms"] for item in rows), total, rel_tol=1e-9, abs_tol=1e-8):
        return None
    return rows


BS_METRICS = (("per_device_tokens_s", "单卡吞吐", "token/s/物理卡"),
              ("cycle_ms", "完整周期（ITL代理）", "ms/轮次"), ("tpot_ms", "TPOT", "ms/token"))
BS_PARTS = (("main_layer_envelopes", "主模型层区间", "168B8D"),
            ("main_internal_gaps", "主模型层间衔接", "DDA13A"),
            ("draft_layer_envelopes", "草稿层区间", "3975B8"),
            ("draft_internal_gaps", "草稿层间衔接", "9371B5"),
            ("residual_once", "轮次剩余区间U", "D2675B"))
BS_OVERVIEW_HEADERS = ["配置BS\n（每DP请求数）", "采样组合", "方法一rank",
    "方法一单卡吞吐\n(token/s/卡)", "方法一完整周期\n(ms)", "方法一TPOT\n(ms/token)",
    "方法二rank", "方法二单卡吞吐\n(token/s/卡)", "方法二完整周期\n(ms)",
    "方法二TPOT\n(ms/token)", "方法状态"]
BS_OVERVIEW_SPANS = ((1, 1), (2, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                     (8, 8), (9, 9), (10, 10), (11, 11), (12, 12))
BS_METHOD_HEADERS = ["配置BS\n（每DP请求数）", "采样组合", "所选rank",
    *[label + "\n(ms)" for _, label, _ in BS_PARTS], "完整周期\n(ms)", "状态"]
BS_METHOD_SPANS = ((1, 1), (2, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                   (8, 8), (9, 9), (10, 10), (11, 12))
BS_INDEX_TITLE = "本次报告全部采样组合（含单独分组的BS）"
BS_INDEX_HEADERS = ["配置BS\n（每DP请求数）", "BS对比组", "采样组合", "展示状态"]
BS_INDEX_SPANS = ((1, 1), (2, 3), (4, 6), (7, 12))
BS_COMPARISON_SCOPE = ("按报告已声明的非BS条件及观测rank集合分组；横轴取config.bs，按数值升序、等距展示档位。"
    "每个点沿用该BS、该方法核心选中的rank，rank可随BS改变；缺少结果留空并断线，不补0或插值。")
SWEEP_METRICS = (BS_METRICS[0], BS_METRICS[2], BS_METRICS[1])
ITL_PROXY_SCOPE = ("完整周期作为每轮输出间隔的ITL代理值（ms/轮），不是实测逐token发送间隔；"
    "TPOT为周期/配置Neff（ms/token），单卡吞吐按物理卡均摊。")
S_COMPARISON_SCOPE = ("固定BS，横轴取声明的草稿候选token数S，按数值升序、等距展示档位。"
    "每点保留各自配置Neff和各方法原选rank；不从case名称猜S，不重选rank；缺失值留空断线。")
S_OVERVIEW_HEADERS = ["投机步长S", "采样组合", "SAL / Neff\n（含bonus）", "方法一rank / 状态",
    "方法一单卡吞吐\n(token/s/卡)", "方法一TPOT\n(ms/token)", "方法一ITL代理\n(ms/轮)",
    "方法二rank / 状态", "方法二单卡吞吐\n(token/s/卡)", "方法二TPOT\n(ms/token)", "方法二ITL代理\n(ms/轮)"]
S_OVERVIEW_SPANS = ((1, 1), (2, 3), *[(col, col) for col in range(4, 13)])
SWEEP_PART_SPANS = S_OVERVIEW_SPANS
SOURCE_METRIC_SPANS = ((1, 1), (2, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 12))
CASE_PARAMETERS_TITLE = "采样配置对照"
CASE_PARAMETERS_HEADERS = ["采样组合", "配置BS", "草稿S\n（num_speculative_tokens）", "SAL / Neff\n（含bonus）",
                           "声明的平均草稿接受比例\n（配置假设，非实测）"]
CASE_PARAMETERS_SPANS = ((1, 4), (5, 5), (6, 6), (7, 8), (9, 12))
CASE_PARAMETERS_SCOPE = ("S为每轮草稿预测token数；SAL（synthetic_acceptance_length，兼容拼写synthetic_acception_length）"
    "按每请求每轮有效新token数Neff展示，含bonus。接受比例仅在明确声明且与S、Neff一致时展示，"
    "属于配置假设；不从长度反推实测接受率。")


def _parameter_text(value):
    """Use compact parameter labels without rounding any performance result."""
    if not _number(value):
        return "未声明"
    if type(value) is int:
        return str(value)
    if value == 0:
        return "0"
    return format(value, ".6f").rstrip("0").rstrip(".")


def _case_parameter_values(case):
    """Display declared inputs; the consistency check never supplies a rate."""
    config = case.get("config") or {}
    bs, draft, effective = config.get("bs"), _s_value(case), config.get("effective_tokens_per_round")
    rate = config.get("acceptance")
    declared_rate = "未声明"
    if (config.get("acceptance_semantics") == "mean_accepted_draft_fraction"
            and type(draft) is int and draft >= 0 and _number(effective) and 1 <= effective <= 1 + draft
            and _number(rate) and 0 <= rate <= 1
            and math.isclose(effective, 1 + draft * rate, rel_tol=1e-9, abs_tol=1e-9)):
        declared_rate = _parameter_text(100 * rate) + "%"
    return [_parameter_text(bs) if type(bs) is int and bs > 0 else "未声明",
            _parameter_text(draft) if type(draft) is int and draft >= 0 else "未声明",
            _parameter_text(effective) if _number(effective) and effective >= 1 else "未声明", declared_rate]


def _case_parameters(case):
    bs, draft, effective, rate = _case_parameter_values(case)
    label = "BS=" + bs + "；草稿S=" + draft + "；Neff=" + effective + "（含bonus）"
    return label if rate == "未声明" else label + "；声明的平均草稿接受比例=" + rate + "（配置假设，非实测）"


def _case_parameter_rows(report):
    return [[case["name"], *_case_parameter_values(case)] for case in report["cases"]]


def _case_parameter_table(report):
    rows = _case_parameter_rows(report)
    if any(row[-1] != "未声明" for row in rows):
        return CASE_PARAMETERS_HEADERS, rows, CASE_PARAMETERS_SPANS
    return CASE_PARAMETERS_HEADERS[:-1], [row[:-1] for row in rows], (*CASE_PARAMETERS_SPANS[:3], (7, 12))


def _bs_comparison_groups(report):
    """Group declared conditions, retaining all cases and every duplicate BS."""
    groups = {}
    excluded = {"bs", "name", "path", "paths", "inputs", "target_model", "sample_model", "source_model"}
    for case in report["cases"]:
        config = case.get("config") or {}
        ranks = sorted({str(item["rank"]) for item in case.get("captures") or case.get("measurements") or []
                        if item.get("rank") is not None})
        coverage = []
        for identity in ("method1", "method2"):
            original = _method(case, identity).get("coverage") or {}
            fields = {key: original[key] for key in ("captured_ranks", "expected_ranks", "source_logical_devices", "complete")
                      if key in original}
            if "captured_rank_names" in original:
                fields["captured_rank_names"] = sorted(original["captured_rank_names"])
            coverage.append(fields)
        signature = json.dumps({"source_model": case.get("source_model"), "target_model": case.get("target_model"),
            "conditions": {key: value for key, value in config.items() if key not in excluded and not key.startswith("_")},
            "observed_ranks": ranks, "coverage": coverage}, ensure_ascii=False, sort_keys=True, allow_nan=False)
        groups.setdefault(signature, {"cases": []})["cases"].append(case)
    result = []
    for number, group in enumerate(groups.values(), 1):
        group["cases"] = sorted(group["cases"], key=lambda case:
            ((case.get("config") or {}).get("bs") if type((case.get("config") or {}).get("bs")) is int
             and case["config"]["bs"] > 0 else math.inf, case["name"]))
        counts = collections.Counter(case["config"]["bs"] for case in group["cases"]
            if type((case.get("config") or {}).get("bs")) is int and case["config"]["bs"] > 0)
        group["duplicate_bs"] = sorted(bs for bs, count in counts.items() if count > 1)
        group["invalid_bs"] = [case["name"] for case in group["cases"]
            if type((case.get("config") or {}).get("bs")) is not int or case["config"]["bs"] <= 0]
        config = group["cases"][0].get("config") or {}
        group["label"] = ("BS对比组" + str(number) + " · " + _case_scope(group["cases"][0])
            + "；BS=" + _text([(case.get("config") or {}).get("bs") for case in group["cases"]])
            + "；S=" + _parameter_text(config.get("draft_tokens")) + "，Neff=" + _parameter_text(config.get("effective_tokens_per_round"))
            + "；目标DP=" + _text(config.get("target_dp")) + "，目标物理卡=" + _text(config.get("target_devices")))
        result.append(group)
    return result


def _bs_chart_reason(group):
    if group["invalid_bs"]:
        return "存在缺失或无效的配置BS，保留原表，不从采样组合名称推断BS，不绘制本组趋势。"
    if group["duplicate_bs"]:
        return "相同BS存在多条采样组合，未合并趋势；保留全部原值，不取首条、末条或平均。重复BS=" + _text(group["duplicate_bs"])
    if len(group["cases"]) < 2:
        return "本组仅有一个BS，保留单点表格；至少两个不同BS才能展示变化。"
    return None


def _s_value(case):
    """Read only declared S, rejecting conflicting aliases and bool values."""
    config = case.get("config") or {}
    primary = config.get("draft_tokens")
    alias = config.get("num_speculative_tokens")
    if "draft_tokens" in config and "num_speculative_tokens" in config:
        if type(primary) is not int or type(alias) is not int or primary != alias:
            return None
    value = primary if "draft_tokens" in config else alias
    return value if type(value) is int and value >= 0 else None


def _s_comparison_groups(report):
    """Group fixed-BS S sweeps without silently treating Neff as measured acceptance."""
    groups = {}
    excluded = {"name", "path", "paths", "inputs", "target_model", "sample_model", "source_model",
                "draft_tokens", "num_speculative_tokens", "effective_tokens_per_round",
                "synthetic_acceptance_length", "synthetic_acception_length"}
    for case in report["cases"]:
        config = case.get("config") or {}
        bs = config.get("bs")
        valid_bs = type(bs) is int and bs > 0
        rate = config.get("acceptance")
        explicit_rate = (config.get("acceptance_semantics") == "mean_accepted_draft_fraction"
                         and _number(rate) and 0 <= rate <= 1)
        policy = "fixed_acceptance_fraction" if explicit_rate else "per_case_configured_neff"
        ranks = sorted({str(item["rank"]) for item in case.get("captures") or case.get("measurements") or []
                        if item.get("rank") is not None})
        coverage = []
        for identity in ("method1", "method2"):
            original = _method(case, identity).get("coverage") or {}
            fields = {key: original[key] for key in ("captured_ranks", "expected_ranks", "source_logical_devices", "complete")
                      if key in original}
            if "captured_rank_names" in original:
                fields["captured_rank_names"] = sorted(original["captured_rank_names"])
            coverage.append(fields)
        signature = json.dumps({"source_model": case.get("source_model"), "target_model": case.get("target_model"),
            "conditions": {key: value for key, value in config.items() if key not in excluded and not key.startswith("_")},
            "output_policy": policy, "observed_ranks": ranks, "coverage": coverage,
            # Missing BS never establishes a shared fixed-BS experiment.
            "invalid_fixed_bs_case": None if valid_bs else case["name"]},
            ensure_ascii=False, sort_keys=True, allow_nan=False)
        groups.setdefault(signature, {"cases": [], "output_policy": policy})["cases"].append(case)
    result = []
    for number, group in enumerate(groups.values(), 1):
        group["cases"] = sorted(group["cases"], key=lambda case:
            (_s_value(case) if _s_value(case) is not None else math.inf, case["name"]))
        counts = collections.Counter(_s_value(case) for case in group["cases"] if _s_value(case) is not None)
        group["duplicate_s"] = sorted(value for value, count in counts.items() if count > 1)
        group["invalid_s"] = [case["name"] for case in group["cases"] if _s_value(case) is None]
        group["invalid_bs"] = [case["name"] for case in group["cases"]
            if type((case.get("config") or {}).get("bs")) is not int or case["config"]["bs"] <= 0]
        config = group["cases"][0].get("config") or {}
        group["label"] = ("S对比组" + str(number) + " · " + _case_scope(group["cases"][0])
            + "；固定BS=" + (_parameter_text(config.get("bs")) if not group["invalid_bs"] else "无效或未声明")
            + "；S=" + _text([_s_value(case) for case in group["cases"]]))
        if group["output_policy"] == "fixed_acceptance_fraction":
            group["note"] = ("固定BS；声明的平均草稿接受比例r=" + _parameter_text(config.get("acceptance"))
                + "（配置假设，非实测）；各S沿用case配置Neff，含bonus，不从Neff反推实测接受率。")
        else:
            group["note"] = ("固定BS，各S沿用case配置Neff；非固定接受率的单变量对照，"
                "不能把变化仅归因于S；Neff含bonus，不从长度反推实测接受率。")
        if group["invalid_bs"]:
            group["note"] = "固定BS缺失或无效，不能建立固定BS的S趋势；保留全部配置原值。"
        result.append(group)
    return result


def _s_chart_reason(group):
    if group["invalid_bs"]:
        return "固定BS缺失或无效，保留原表；不从采样组合名称推断BS，不绘制本组S趋势。"
    if group["invalid_s"]:
        return "存在缺失、无效或别名冲突的配置S，保留原表；不从采样组合名称推断S，不绘制本组趋势。"
    if group["duplicate_s"]:
        return ("相同S存在多条采样组合，未合并趋势；保留全部原值，不取首条、末条或平均。重复S="
                + _text(group["duplicate_s"]))
    if len(group["cases"]) < 2:
        return "本组仅有一个S，保留单点表格；至少两个不同S才能展示变化。"
    return None


def _bs_index_rows(groups):
    indexed = [(case, group) for group in groups for case in group["cases"]]
    indexed.sort(key=lambda item: ((item[0].get("config") or {}).get("bs")
        if type((item[0].get("config") or {}).get("bs")) is int and item[0]["config"]["bs"] > 0
        else math.inf, item[0]["name"]))
    rows = []
    for case, group in indexed:
        state = "；".join("方法" + str(number) + "=" + _layer_mean_status(_method(case, identity))
                          for number, identity in enumerate(("method1", "method2"), 1))
        state += "。" + (_bs_chart_reason(group) or "本组原值表及趋势图见下方。")
        rows.append([(case.get("config") or {}).get("bs"), group["label"].split(" · ")[0], case["name"], state])
    return rows


def _input_report_diagnostics(report, origin, reused=False):
    """Describe the loaded payload, without discovering or rereading other files."""
    bs_values = sorted({(case.get("config") or {}).get("bs") for case in report["cases"]
        if type((case.get("config") or {}).get("bs")) is int and case["config"]["bs"] > 0})
    kind = ("复用内存报告（来源路径仅为标签）" if reused else
            "配置预览" if report.get("source_kind") == "configuration_preview" else "磁盘JSON")
    messages = ["输入报告：" + str(Path(origin).resolve()) + "；" + kind,
        "输入清单：case=" + str(len(report["cases"])) + "；配置BS=" + str(bs_values)
        + "；失败case=" + str(len(report.get("failures") or [])),
        "报告生成时间：" + _text(report.get("generated_at"))]
    for case in report["cases"]:
        messages.append("输入组合：" + case["name"] + "；config.bs=" + repr((case.get("config") or {}).get("bs"))
            + "；" + "；".join("方法" + str(number) + "=" + _layer_mean_status(_method(case, identity))
                for number, identity in enumerate(("method1", "method2"), 1)))
    messages.extend("BS分组：" + group["label"] for group in _bs_comparison_groups(report))
    messages.extend("失败组合：" + _text(item.get("case")) + "；" + _text(item.get("error"))
                    for item in report.get("failures") or [])
    return messages


def _bs_value(method, field):
    value = _target(method, field)
    return value if _number(value) and value > 0 else None


def _bs_overview_rows(group):
    rows = []
    for case in group["cases"]:
        row = [(case.get("config") or {}).get("bs"), case["name"]]
        states = []
        for number, identity in enumerate(("method1", "method2"), 1):
            method = _method(case, identity)
            row.extend([method.get("selected_rank") if _layer_mean_publishable(method) else NA,
                        *[_bs_value(method, field) for field, _, _ in BS_METRICS]])
            states.append("方法" + str(number) + "：" + _layer_mean_status(method))
        rows.append([*row, "\n".join(states)])
    return rows


def _bs_breakdown_values(method):
    breakdown = _breakdown_for_chart(method)
    if breakdown is None or len(breakdown) != len(BS_PARTS):
        return None
    keys = [item.get("key") for item in breakdown]
    if len(set(keys)) != len(BS_PARTS) or set(keys) != {key for key, _, _ in BS_PARTS}:
        return None
    values = {item["key"]: item["value_ms"] for item in breakdown}
    return [values[key] for key, _, _ in BS_PARTS] if all(value >= 0 for value in values.values()) else None


def _bs_method_rows(group, identity):
    rows = []
    for case in group["cases"]:
        method = _method(case, identity)
        parts = _bs_breakdown_values(method)
        status = _layer_mean_status(method)
        if _layer_mean_publishable(method) and parts is None:
            status = "五项分解缺失或不闭合"
        rows.append([(case.get("config") or {}).get("bs"), case["name"],
            method.get("selected_rank") if _layer_mean_publishable(method) else NA,
            *(parts if parts is not None else [None] * len(BS_PARTS)), _bs_value(method, "cycle_ms"), status])
    return rows


def _bs_table(sheet, row, headers, rows, spans, blank_columns):
    written, header = [], row
    row = _table(sheet, row, headers, rows, spans=spans, record_rows=written, blank_columns=blank_columns)
    sheet.row_height(header, 58)
    for current in written:
        sheet.row_height(current, 48)
    contiguous = bool(written) and written == list(range(written[0], written[0] + len(rows)))
    return row, written if contiguous else None


def _bs_overview_section(sheet, row, report):
    groups = _bs_comparison_groups(report)
    row = _note(sheet, row, BS_INDEX_TITLE)
    row = _note(sheet, row, "本次报告来源：" + str(Path(sheet.report_origin).resolve())
        + "；报告生成时间：" + _text(report.get("generated_at"))
        + "。以下列出报告内全部组合；对比组不同的BS仍保留原值，分组表和图在索引下方。")
    row = _table(sheet, row, BS_INDEX_HEADERS, _bs_index_rows(groups), spans=BS_INDEX_SPANS, blank_columns=(1,))
    if report.get("failures"):
        row = _table(sheet, row, ["统计失败、未纳入BS对比的采样组合", "失败原因"],
            [[item.get("case"), item.get("error")] for item in report["failures"]], spans=((1, 4), (5, 12)))
    row = _note(sheet, row, "不同BS的单卡吞吐、完整周期与TPOT变化")
    row = _note(sheet, row, BS_COMPARISON_SCOPE + ITL_PROXY_SCOPE)
    for group in groups:
        row = _note(sheet, row, group["label"])
        records = _bs_overview_rows(group)
        row, written = _bs_table(sheet, row, BS_OVERVIEW_HEADERS, records, BS_OVERVIEW_SPANS,
                                 (1, 4, 5, 6, 8, 9, 10))
        reason = _bs_chart_reason(group)
        if reason or written is None:
            row = _note(sheet, row, reason or "表格含长文本续行，保留原表，不绘制本组趋势。", warning=bool(reason and len(group["cases"]) > 1))
            continue
        reference = "'" + sheet.name.replace("'", "''") + "'!"
        for metric_index, (field, label, unit) in enumerate(BS_METRICS):
            series = []
            for identity, color, column in (("method1", "168B8D", 5 + metric_index), ("method2", "D4793A", 9 + metric_index)):
                series.append(dict(name=_layer_mean_name({"id": identity}), color=color,
                    values=[_bs_value(_method(case, identity), field) for case in group["cases"]],
                    values_ref=reference + "$" + cell_reference(1, column)[:-1] + "$" + str(written[0])
                        + ":$" + cell_reference(1, column)[:-1] + "$" + str(written[-1])))
            sheet.add_line_chart(label + "随BS变化 · " + group["label"].split(" · ")[0],
                categories=[str(record[0]) for record in records], series=series, anchor=(row, 1, row + 17, 12),
                categories_ref=reference + "$A$" + str(written[0]) + ":$A$" + str(written[-1]),
                x_axis_title="BS（每DP请求数）", y_axis_title=unit)
            row += 19
    return row + 1


def _bs_method_section(sheet, row, report, identity):
    row = _note(sheet, row, "不同BS的整轮五项耗时构成变化 · " + _layer_mean_name({"id": identity}))
    row = _note(sheet, row, BS_COMPARISON_SCOPE + "五项之和须闭合完整周期；D/C和路由计算变化已计入层区间，U保留一次。")
    for group in _bs_comparison_groups(report):
        row = _note(sheet, row, group["label"])
        records = _bs_method_rows(group, identity)
        row, written = _bs_table(sheet, row, BS_METHOD_HEADERS, records, BS_METHOD_SPANS, (1, 4, 5, 6, 7, 8, 9))
        reason = _bs_chart_reason(group)
        if reason or written is None:
            row = _note(sheet, row, reason or "表格含长文本续行，保留原表，不绘制本组趋势。", warning=bool(reason and len(group["cases"]) > 1))
            continue
        reference = "'" + sheet.name.replace("'", "''") + "'!"
        series = []
        for index, (_, label, color) in enumerate(BS_PARTS):
            column = cell_reference(1, 5 + index)[:-1]
            series.append(dict(name=label, color=color, values=[record[3 + index] for record in records],
                values_ref=reference + "$" + column + "$" + str(written[0]) + ":$" + column + "$" + str(written[-1])))
        sheet.add_line_chart("整轮五项耗时构成随BS变化 · " + group["label"].split(" · ")[0],
            categories=[str(record[0]) for record in records], series=series, anchor=(row, 1, row + 21, 12),
            categories_ref=reference + "$A$" + str(written[0]) + ":$A$" + str(written[-1]),
            x_axis_title="BS（每DP请求数）", y_axis_title="ms/完整周期")
        row += 23
    return row + 1


def _s_groups(report):
    # Single points remain in the configuration/BS tables; no repeated S tables.
    return [group for group in _s_comparison_groups(report) if len(group["cases"]) >= 2]


def _source_sweep(case):
    from profile_ep128 import source_sweep_summary
    return source_sweep_summary(case)


def _source_sweep_notes(case, summary):
    notes = [summary[key] for key in ("reason", "breakdown_reason") if summary[key]]
    aggregate = _source(case)
    captured, expected = aggregate.get("captured_ranks"), aggregate.get("expected_ranks")
    if type(captured) is int and type(expected) is int and 0 <= captured < expected:
        notes.append("源rank覆盖=" + str(captured) + "/" + str(expected)
            + "；所选为已采rank中的周期均值最大值，未证明是全系统最慢rank。")
    elif aggregate.get("rank_coverage_complete") is False:
        notes.append("源rank覆盖不完整；所选最大值仅限已采rank。")
    return notes


def _s_overview_rows(group):
    rows = []
    for case in group["cases"]:
        row = [_s_value(case), case["name"], (case.get("config") or {}).get("effective_tokens_per_round")]
        for identity in ("method1", "method2"):
            method = _method(case, identity)
            row.extend([_rank_note(method) + "\n" + _layer_mean_status(method),
                        *[_bs_value(method, field) for field, _, _ in SWEEP_METRICS]])
        rows.append(row)
    return rows


def _s_method_rows(group, identity):
    rows = []
    for case in group["cases"]:
        method = _method(case, identity)
        parts = _bs_breakdown_values(method)
        status = _layer_mean_status(method)
        if _layer_mean_publishable(method) and parts is None:
            status = "五项分解缺失或不闭合"
        rows.append([_s_value(case), case["name"], (case.get("config") or {}).get("effective_tokens_per_round"),
            method.get("selected_rank") if _layer_mean_publishable(method) else NA,
            *(parts if parts is not None else [None] * 5), _bs_value(method, "cycle_ms"), status])
    return rows


def _sweep_part_headers(axis):
    return ["投机步长S" if axis == "S" else "配置BS\n（每DP请求数）", "采样组合", "SAL / Neff\n（含bonus）",
        "所选rank", *[label + "\n(ms)" for _, label, _ in BS_PARTS], "完整周期\n（ITL代理ms/轮）", "状态"]


def _source_metric_rows(group, axis):
    rows = []
    for case in group["cases"]:
        config, summary = case.get("config") or {}, _source_sweep(case)
        rows.append([_s_value(case) if axis == "S" else config.get("bs"), case["name"],
            config.get("bs") if axis == "S" else _s_value(case), config.get("effective_tokens_per_round"),
            summary["selected_rank"], *[summary[field] for field, _, _ in SWEEP_METRICS], summary["status"]])
    return rows


def _source_parts_rows(group, axis):
    rows = []
    for case in group["cases"]:
        summary, config = _source_sweep(case), case.get("config") or {}
        parts = {item["key"]: item["value_ms"] for item in summary["breakdown"]}
        rows.append([_s_value(case) if axis == "S" else config.get("bs"), case["name"],
            config.get("effective_tokens_per_round"), summary["selected_rank"],
            *[parts[key] for key, _, _ in BS_PARTS], summary["cycle_ms"],
            "分解可用" if summary["breakdown_reason"] is None else "分解缺少证据；见组说明"])
    return rows


def _source_metric_headers(axis):
    return ["投机步长S" if axis == "S" else "配置BS\n（每DP请求数）", "采样组合",
        "固定BS" if axis == "S" else "固定S", "SAL / Neff\n（含bonus）", "源所选rank",
        "源单卡吞吐\n(token/s/卡)", "源TPOT\n(ms/token)", "源ITL代理\n(ms/轮)", "状态"]


def _sweep_axis(axis):
    return "投机步长S（草稿候选token数）" if axis == "S" else "BS（每DP请求数）"


def _sweep_excel_chart(sheet, row, title, records, written, specs, unit, axis):
    """Use the exact table coordinates and preserve missing points as blanks."""
    if written is None:
        return _note(sheet, row, "表格含长文本续行，保留原表，不绘制本组趋势。")
    if not any(_number(value) for _, values, _, _ in specs for value in values):
        return _note(sheet, row, "本项没有可用数值，保留空值表格，不绘制趋势。")
    reference = "'" + sheet.name.replace("'", "''") + "'!"
    series = []
    for label, values, color, column in specs:
        letter = cell_reference(1, column)[:-1]
        series.append(dict(name=label, values=values, color=color,
            values_ref=reference + "$" + letter + "$" + str(written[0]) + ":$" + letter + "$" + str(written[-1])))
    sheet.add_line_chart(title, categories=[str(record[0]) for record in records], series=series,
        anchor=(row, 1, row + 19, 12),
        categories_ref=reference + "$A$" + str(written[0]) + ":$A$" + str(written[-1]),
        x_axis_title=_sweep_axis(axis), y_axis_title=unit)
    return row + 21


def _s_overview_section(sheet, row, report):
    row = _note(sheet, row, "固定BS，不同投机步长S的单卡吞吐、TPOT与ITL代理变化")
    row = _note(sheet, row, S_COMPARISON_SCOPE + ITL_PROXY_SCOPE)
    groups = _s_groups(report)
    if not groups:
        return _note(sheet, row, "当前报告没有同一固定BS下的多条S采样；不生成跨条件曲线。")
    for group in groups:
        row = _note(sheet, row, group["label"])
        row = _note(sheet, row, group["note"])
        records = _s_overview_rows(group)
        row, written = _bs_table(sheet, row, S_OVERVIEW_HEADERS, records, S_OVERVIEW_SPANS, (1, 3, 5, 6, 7, 9, 10, 11))
        reason = _s_chart_reason(group)
        if reason:
            row = _note(sheet, row, reason, warning=True)
            continue
        for index, (_, label, unit) in enumerate(SWEEP_METRICS):
            specs = [(_layer_mean_name({"id": identity}), [record[4 + index + shift] for record in records], color, 6 + index + shift)
                for identity, shift, color in (("method1", 0, "168B8D"), ("method2", 4, "D4793A"))]
            row = _sweep_excel_chart(sheet, row, label + "随S变化 · " + group["label"].split(" · ")[0],
                                     records, written, specs, unit, "S")
    return row + 1


def _s_method_section(sheet, row, report, identity):
    row = _note(sheet, row, "固定BS，不同投机步长S的整轮五项耗时构成 · " + _layer_mean_name({"id": identity}))
    row = _note(sheet, row, S_COMPARISON_SCOPE + "五项之和须闭合完整周期；不再次追加通信或路由计算。")
    for group in _s_groups(report):
        row = _note(sheet, row, group["label"] + "。" + group["note"])
        records = _s_method_rows(group, identity)
        row, written = _bs_table(sheet, row, _sweep_part_headers("S"), records, SWEEP_PART_SPANS, (1, 3, 5, 6, 7, 8, 9, 10))
        reason = _s_chart_reason(group)
        if reason:
            row = _note(sheet, row, reason, warning=True)
            continue
        specs = [(label, [record[4 + index] for record in records], color, 6 + index)
                 for index, (_, label, color) in enumerate(BS_PARTS)]
        row = _sweep_excel_chart(sheet, row, "整轮五项耗时构成随S变化 · " + group["label"].split(" · ")[0],
                                 records, written, specs, "ms/完整周期", "S")
    return row + 1


def _source_sweeps_section(sheet, row, report):
    row = _note(sheet, row, "源采样基线：固定S看BS、固定BS看S")
    row = _note(sheet, row, ITL_PROXY_SCOPE + "吞吐和TPOT依赖配置Neff，属于条件换算。源采样沿用源aggregate选中的已采rank周期均值最大值，"
        "每点rank可变化；五项使用该rank全部保留轮次的实际层边界（包含主模型第0层），不使用目标外推分解。")
    for axis, groups in (("BS", _bs_comparison_groups(report)), ("S", _s_groups(report))):
        for group in groups:
            row = _note(sheet, row, group["label"])
            if axis == "S":
                row = _note(sheet, row, group["note"])
            for case in group["cases"]:
                summary = _source_sweep(case)
                reasons = _source_sweep_notes(case, summary)
                if reasons:
                    row = _note(sheet, row, case["name"] + "：" + "；".join(reasons), warning=True)
            reason = (_s_chart_reason if axis == "S" else _bs_chart_reason)(group)
            records = _source_metric_rows(group, axis)
            row, written = _bs_table(sheet, row, _source_metric_headers(axis), records, SOURCE_METRIC_SPANS,
                                     (1, 3, 4, 6, 7, 8))
            if reason:
                row = _note(sheet, row, reason)
            else:
                for index, (_, label, unit) in enumerate(SWEEP_METRICS):
                    specs = [("源采样基线", [record[5 + index] for record in records], "687F94", 7 + index)]
                    row = _sweep_excel_chart(sheet, row, "源采样基线 · " + label + "随" + axis + "变化 · " + group["label"].split(" · ")[0],
                                             records, written, specs, unit, axis)
            records = _source_parts_rows(group, axis)
            row, written = _bs_table(sheet, row, _sweep_part_headers(axis), records, SWEEP_PART_SPANS,
                                     (1, 3, 5, 6, 7, 8, 9, 10))
            if not reason:
                specs = [(label, [record[4 + index] for record in records], color, 6 + index)
                         for index, (_, label, color) in enumerate(BS_PARTS)]
                row = _sweep_excel_chart(sheet, row, "源采样基线 · 整轮五项耗时构成随" + axis + "变化 · " + group["label"].split(" · ")[0],
                                         records, written, specs, "ms/完整周期", axis)
    return row + 1


def _case_scope(case):
    """Describe only explicit core configuration; never invent an EP128 target."""
    config = case.get("config") or {}
    source = case.get("source_model") or {}
    target = case.get("target_model")

    def count(value, label):
        return str(value) + label if type(value) is int and value > 0 else label + "数未配置"

    def ep(value):
        return "EP" + str(value) if type(value) is int and value > 0 else "EP未配置"

    source_scope = ("源" + count(source.get("num_hidden_layers"), "主层") + "＋"
                    + count(config.get("draft_layers"), "草稿层") + "、" + ep(config.get("base_ep")))
    if not isinstance(target, dict) or not target:
        target_scope = "目标模型未配置"
    else:
        main = target.get("main") or {}
        draft = target.get("draft") or {}
        target_scope = ("目标" + count(main.get("num_hidden_layers"), "主层") + "＋"
                        + count(draft.get("num_hidden_layers"), "草稿层") + "、" + ep(config.get("target_ep")))
    return target_scope + "；" + source_scope


def _report_scope(report):
    scopes = list(dict.fromkeys(_case_scope(case) for case in report["cases"]))
    return scopes[0] if len(scopes) == 1 else "源与目标配置按采样组合分别列示；见配置范围表"


def _layer_mean_cards(sheet, row, method, case_index, identity):
    cells = {}
    for field, label, first, last in (
            ("cycle_ms", "完整周期 · ms", 1, 4), ("tpot_ms", "TPOT · ms/token", 5, 8),
            ("per_device_tokens_s", "预测单卡吞吐 · token/s/card", 9, 12)):
        sheet.merge(row, first, row, last, label, "hero_label").row_height(row, 26)
        sheet.merge(row + 1, first, row + 2, last, _target(method, field), "hero")
        sheet.row_height(row + 1, 26).row_height(row + 2, 20)
        cells[field] = (row + 1, first)
    sheet.result_cells[case_index, identity] = cells
    return row + 4


def _layer_mean_value(method, value, target=False):
    return None if target and not _layer_mean_publishable(method) else value


def _layer_mean_status(method):
    return "条件估算" if _layer_mean_publishable(method) else "缺少实际输入"


def _layer_mean_name(method):
    return {"method1": "方法一（逻辑通信倍率）", "method2": "方法二（历史通信比值）"}.get(method.get("id"), "条件估算")


def _rank_note(method):
    selected = method.get("selected_rank")
    if selected is not None:
        return "所选rank=" + _text(selected)
    source = method.get("source_rank")
    return "目标rank未选择" + ("；源参考rank=" + _text(source) if source is not None else "")


def _layer_mean_matrix_rows(method):
    """Copy the core-selected rank's class means; never average or choose ranks."""
    rows = []
    classes = method.get("class_means") or []
    for attention in ("CSA", "HCA", "SWA"):
        roles = {}
        for item in classes:
            if item.get("attention") == attention:
                role = item.get("role")
                if role in roles:
                    raise ValueError("层均值表含重复role/attention：" + str(role) + "/" + attention)
                roles[role] = item
        if not roles:
            continue
        def values(field, target=False):
            return [_layer_mean_value(method, roles[role].get(field), target) if role in roles else NA
                    for role in ("main", "draft")]
        refs = [roles[role].get("reference_layers") if role in roles else NA for role in ("main", "draft")]
        for label, source, target, delta in (
                ("单层", "wall_ms", "target_wall_ms", None),
                ("Dispatch", "dispatch_ms", "dispatch_target_ms", "dispatch_delta_ms"),
                ("Combine", "combine_ms", "combine_target_ms", "combine_delta_ms"),
                ("路由计算", "routed_compute_ms", "routed_compute_target_ms", "routed_compute_delta_ms")):
            rows.append([attention + " " + label, "ms/层", *values(source), *values(target, True),
                         *(values(delta, True) if delta else [NA, NA]), *refs,
                         "所选rank的参考层调用均值；源层已含通信与计算，只追加Δ。", _layer_mean_status(method)])
        rows.append([attention + " 目标层数", "层", NA, NA, *values("target_layers"), NA, NA, *refs,
                     "目标同类型层数取配置；不使用源类型比例推算。", _layer_mean_status(method)])
        rows.append([attention + " 参考样本数", "次", *values("samples"), NA, NA, NA, NA, *refs,
                     "仅该选中rank的参考层调用数；主0不作参考，主/草稿不混合。", _layer_mean_status(method)])
    roles = {item.get("role"): item for item in method.get("role_details") or []}
    def role_values(field, target=False):
        return [_layer_mean_value(method, roles.get(role, {}).get(field), target) for role in ("main", "draft")]
    for label, source, target, delta, note in (
            ("层区间之和", "source_layer_envelopes_ms", "target_layer_envelopes_ms", None, "不含层间衔接区间；目标已含通信和路由计算Δ。"),
            ("层间衔接区间", "source_gap_ms", "target_gap_ms", "added_gap_ms", LAYER_GAP_NOTE),
            ("主/草稿层链", "source_chain_ms", "target_chain_ms", None, "包含本role层区间与内部衔接；整轮使用同一个选中rank。"),
            ("常规衔接参考均值", "reference_gap_mean_ms", None, None, "参考均值由核心计算；不在Excel重新求均值或选边界。"),
            ("Dispatch变化合计", None, None, "dispatch_delta_ms", "已计入目标层区间，不再次加入完整周期。"),
            ("Combine变化合计", None, None, "combine_delta_ms", "已计入目标层区间，不再次加入完整周期。"),
            ("路由计算变化合计", None, None, "routed_mlp_delta_ms", "已计入目标层区间；算子范围以核心说明为准。")):
        rows.append([label, "ms", *(role_values(source) if source else [NA, NA]),
                     *(role_values(target, True) if target else [NA, NA]),
                     *(role_values(delta, True) if delta else [NA, NA]), NA, NA, note, _layer_mean_status(method)])
    return rows


LAYER_MEAN_MATRIX_HEADERS = ["项目", "单位", "源实测主模型", "源实测草稿", "目标主模型", "目标草稿",
                             "主模型Δ", "草稿Δ", "主参考层号", "草稿参考层号", "核心口径/条件", "状态"]


def _layer_mean_closure_rows(method):
    roles = {item.get("role"): item for item in method.get("role_details") or []}
    partition = method.get("source_partition") or {}
    return [
        ["目标主模型层链", _layer_mean_value(method, roles.get("main", {}).get("target_chain_ms"), True), "层区间＋主内部衔接；核心原值"],
        ["目标草稿层链", _layer_mean_value(method, roles.get("draft", {}).get("target_chain_ms"), True), "层区间＋草稿内部衔接；核心原值"],
        ["轮次剩余区间", partition.get("cross_role_tail_residual_ms", partition.get("residual_ms")), LAYER_RESIDUAL_NOTE],
        ["完整周期", _target(method, "cycle_ms"), "由核心闭合；不再次全量追加D/C或路由计算"],
        ["预测单卡吞吐", _target(method, "per_device_tokens_s"), "核心全局产出按目标物理卡数均摊；Excel不重新计算"],
        ["TPOT", _target(method, "tpot_ms"), "核心周期/配置有效输出长度；非实测请求ITL"]]


def _layer_mean_chart(sheet, row, method):
    breakdown = _breakdown_for_chart(method)
    if breakdown is None:
        return _note(sheet, row, "核心未提供可发布且闭合的整轮分解，不绘制数值条；不补造或重算组成。", warning=True, columns=12)
    start = row + 1
    row = _table(sheet, row, ["整轮耗时构成", "ms", "公式/口径"],
                 [[item.get("label", item.get("key")), item["value_ms"], item.get("formula") or item.get("basis") or "已计入完整周期，仅计一次"]
                  for item in breakdown])
    sheet.add_bar_chart("完整周期分解 · " + _layer_mean_name(method),
        categories=[str(item.get("label", item.get("key"))) for item in breakdown],
        categories_ref="'" + sheet.name + "'!$A$" + str(start) + ":$A$" + str(start + len(breakdown) - 1),
        series=[dict(name="整轮耗时构成(ms)", values=[item["value_ms"] for item in breakdown], color="168B8D",
                     values_ref="'" + sheet.name + "'!$B$" + str(start) + ":$B$" + str(start + len(breakdown) - 1))],
        anchor=(row, 1, row + 16, 12), stacked=False, x_axis_title="ms/完整周期；不等于互斥kernel占用")
    return row + 18


def _overview(book, report):
    scope = LAYER_MEAN_SCOPE + "；" + _report_scope(report)
    if report.get("source_kind") == "configuration_preview":
        scope = PREVIEW + "：只读取配置，所有性能数值待估算。" + scope
    sheet = _sheet(book, SHEETS[0], "10T模型推理估算", scope)
    sheet.result_rows, sheet.result_cells = {}, {}
    row = _note(sheet, 5, CASE_PARAMETERS_TITLE)
    row = _note(sheet, row, CASE_PARAMETERS_SCOPE)
    headers, records, spans = _case_parameter_table(report)
    row = _table(sheet, row, headers, records, spans=spans)
    row = _bs_overview_section(sheet, row, report)
    row = _s_overview_section(sheet, row, report)
    for case_index, case in enumerate(report["cases"]):
        row = _note(sheet, row, case["name"] + " · " + _case_scope(case), columns=12)
        row = _note(sheet, row, _case_parameters(case), columns=12)
        for identity in ("method1", "method2"):
            method = _method(case, identity)
            row = _note(sheet, row, _layer_mean_name(method) + " / " + _layer_mean_status(method) +
                        " / " + _rank_note(method) + " / 采样均值", columns=12)
            row = _layer_mean_cards(sheet, row, method, case_index, identity)
            row = _table(sheet, row, ["整轮组成/指标", "核心原值", "说明"], _layer_mean_closure_rows(method))
            row = _note(sheet, row, _explanation(method), warning=True, columns=12)
    return sheet


def _layer_mean_method_section(sheet, row, case, case_index, identity, method):
    row = _note(sheet, row, case["name"] + " · " + _layer_mean_name(method) +
                " / " + _rank_note(method) + " / " + _layer_mean_status(method), columns=12)
    row = _note(sheet, row, _case_parameters(case), columns=12)
    row = _layer_mean_cards(sheet, row, method, case_index, identity)
    rank_scope = ("以下为核心所选rank的均值；不独立取主/草稿或某组件的最大值。"
                  if _layer_mean_publishable(method) else "源参考均值保留；目标输入不足时不选择目标rank。")
    row = _note(sheet, row, LAYER_MEAN_SCOPE + "。" + rank_scope +
                "源层范围为HC标记区间。", warning=True, columns=12)
    row = _table(sheet, row, LAYER_MEAN_MATRIX_HEADERS, _layer_mean_matrix_rows(method))
    row = _table(sheet, row, ["整轮闭合/单卡指标", "核心原值", "说明"], _layer_mean_closure_rows(method))
    if method.get("reasons"):
        row = _note(sheet, row, "；".join(method["reasons"]), warning=True, columns=12)
    row = _layer_mean_chart(sheet, row, method)
    row = _method_residual_section(sheet, row, case, method)
    row = _method_focus_section(sheet, row, case, method)
    row = _method_stage_expansions(sheet, row, case, method)
    row = _table(sheet, row, ["采样rank", "周期均值(ms)", "主模型目标链(ms)", "草稿目标链(ms)", "轮次剩余区间(ms)", "核心选中"],
        [[item.get("rank"), _layer_mean_value(method, item.get("cycle_ms"), True),
          *[_layer_mean_value(method, next((role.get("target_chain_ms") for role in item.get("role_details") or []
                                           if role.get("role") == identity_role), None), True)
            for identity_role in ("main", "draft")],
          (item.get("source_partition") or {}).get("cross_role_tail_residual_ms", (item.get("source_partition") or {}).get("residual_ms")),
          item.get("rank") == method.get("selected_rank")] for item in method.get("per_rank") or []])
    return row + 2


def _source_sheet(book, report):
    sheet = _sheet(book, SHEETS[1], "源采样基线 · 外推输入与核对",
                   "源采样逐rank诊断；配置BS/Neff换算不等于实测有效吞吐。")
    row = _source_sweeps_section(sheet, 5, report)
    for case in report["cases"]:
        row = _note(sheet, row, case["name"] + " / " + _text(_source(case).get("scope")))
        row = _table(sheet, row, ["rank", "保留轮次", "单卡均摊条件值(token/s/card)", "本DP请求组条件输出率(token/s)",
            "周期均值(ms)", "p50(ms)", "p95(ms)", "设备任务覆盖均值(ms)", "无任务记录均值(ms)", "token依据", "范围"],
            [[item.get(key) for key in ("rank", "rounds", "conditional_per_device_tokens_s", "conditional_per_rank_tokens_s",
              "cycle_ms_mean", "cycle_ms_p50", "cycle_ms_p95", "recorded_busy_ms_mean", "no_recorded_ms_mean", "token_basis", "scope")]
             for item in (case.get("source") or {}).get("per_rank") or []])
        row = _table(sheet, row, ["配置/核心字段", "原值"], list(_flatten(case.get("config") or {})))
    return sheet


RESIDUAL_WINDOWS = (("main_to_draft", "主→草稿"), ("draft_to_nextmain", "草稿→下一主"))
METHOD_RESIDUAL_HEADERS = ["过渡区间", "阶段", "耗时均值(ms)", "匹配轮次 / 状态"]
RESIDUAL_SUMMARY_HEADERS = ["剩余窗口", "样本数", "跨度均值(ms)", "设备任务并集(ms)",
                            "无设备任务记录(ms)", "同时段Host覆盖(ms)", "其中与无设备记录重叠(ms)"]
RESIDUAL_STAGE_HEADERS = ["设备锚点阶段", "连续跨度均值(ms)", "设备并集(ms)",
                          "无设备记录(ms)", "同时段Host覆盖(ms)", "匹配轮次 / 状态"]
RESIDUAL_STAGE_SPANS = ((1, 6), (7, 8), (9, 9), (10, 10), (11, 11), (12, 12))
RESIDUAL_SOURCE_HEADERS = ["设备阶段对应的Host来源", "CPU已关联 / 全部任务", "已关联CPU区间并集(ms)",
                           "队列等待并集(ms)", "提交→执行延迟均值(ms)", "来源覆盖状态"]
RESIDUAL_SOURCE_SPANS = ((1, 4), (5, 6), (7, 8), (9, 9), (10, 11), (12, 12))
RESIDUAL_DMA_HEADERS = ["设备锚点阶段", "trace DMA并集(ms)", "其中覆盖CSV未记录(ms)",
                        "扣除DMA仍未覆盖(ms)"]
RESIDUAL_DMA_SPANS = ((1, 6), (7, 8), (9, 10), (11, 12))
RESIDUAL_DMA_SCOPE = ("仅同次trace与CSV时钟已对齐且确认单设备时，补充该阶段的DMA区间覆盖；"
                      "DMA可与CSV记录设备任务重叠，不能把整列追加到阶段跨度或busy。"
                      "扣除DMA后仍未覆盖，也不代表已证实的NPU硬件空闲；无trace或条件不足显示—。")
RESIDUAL_SOURCE_SCOPE = ("按设备阶段反查同次trace的CPU、队列与CANN提交链，来源可以早于U窗口。"
                         "任务数为全部保留轮次合计；时间为每轮已关联子集的均值，PARTIAL仍展示已测子集。"
                         "CPU区间可能含等待，不是CPU忙时；队列等待与其他区间可能重叠。"
                         "提交→执行为Node.start→device.start的延迟，包含设备排队与依赖，不是Host成本。"
                         "共享图缺独立来源，不按kernel数均摊；本表各项不加到U，子阶段已含在父阶段中。")
RESIDUAL_STAGE_SCOPE = ("同次采样内按设备锚点顺序划分连续跨度，包含锚点之间的间隔；"
                        "阶段名对应源码顺序与设备观测范围，不代表完整Python模块耗时。"
                        "Host列仅为同时段活动，可能属于其他轮次，不是本阶段自身的Host调用耗时；"
                        "相关CPU提交可能早于设备窗口，关联来源见下方Host来源表。Host覆盖不额外加到跨度或U。"
                        "仅全部轮次匹配时展示语义阶段均值；整窗未识别时仍展示原窗口实测总耗时，保留状态和依据。")


def _residual_summary_rows(diagnostic):
    names = dict(RESIDUAL_WINDOWS)
    return [[names.get(item.get("window_key"), item.get("window_key")), item.get("rounds"),
             item.get("duration_ms_mean"), item.get("recorded_busy_ms_mean"),
             item.get("no_recorded_ms_mean"), item.get("host_span_union_ms_mean"),
             item.get("host_in_device_no_recorded_ms_mean")]
            for item in diagnostic.get("summary") or []]


def _residual_stages(diagnostic, window_key, field="stage_summary"):
    return sorted((item for item in diagnostic.get(field) or [] if item.get("window_key") == window_key),
                  key=lambda item: item.get("stage_order", 0))


def _whole_unresolved_window(stages, summary):
    """Accept an unknown whole window only when every retained sample is covered."""
    if len(stages) != 1:
        return False
    item = stages[0]
    rounds, value, total = summary.get("rounds"), summary.get("duration_ms_mean"), item.get("duration_ns_total")
    return (item.get("stage_key") == "unresolved" and item.get("status") == "UNRESOLVED"
            and item.get("level", 0) == 0 and not item.get("parent_stage_key")
            and type(rounds) is int and rounds > 0 and item.get("rounds") == rounds
            and type(item.get("matched_windows")) is int and item["matched_windows"] == 0
            and _number(value) and value >= 0 and type(total) is int and total >= 0
            and math.isclose(total, value * rounds * 1e6, rel_tol=1e-9, abs_tol=1e-5))


def _method_residual_data(case, method):
    """Copy only this method's selected rank; check closure before plotting."""
    selected = method.get("selected_rank")
    if selected is None:
        return [], None, "本方法未选中rank，无法展示对应的剩余区间组成。"
    captures = [capture for capture in case.get("measurements") or [] if capture.get("rank") == selected]
    if len(captures) != 1 or not isinstance(captures[0].get("step_residuals"), dict):
        return [], None, "本方法所选rank=" + _text(selected) + "缺少唯一对应的阶段统计；请重跑基础统计。"
    diagnostic = captures[0]["step_residuals"]
    rows, items, problems = [], [], []
    unresolved_windows = False
    whole_unknown_count, expected_items = 0, 0
    summary = {item.get("window_key"): item for item in diagnostic.get("summary") or []}
    for key, label in RESIDUAL_WINDOWS:
        stages = [item for item in _residual_stages(diagnostic, key)
                  if not item.get("parent_stage_key") and item.get("level", 0) == 0]
        expected = 8 if key == "main_to_draft" else 7
        summaries = [item for item in diagnostic.get("summary") or [] if item.get("window_key") == key]
        if len(summaries) != 1:
            problems.append(label + "缺少唯一原窗口统计")
        if len(summaries) == 1 and _whole_unresolved_window(stages, summaries[0]):
            unknown, window = stages[0], summaries[0]
            state = "0/" + str(window["rounds"]) + " · UNRESOLVED"
            if unknown.get("reason"):
                state += "；" + _text(unknown["reason"])
            rows.append([label, label + " · 阶段未识别（原窗口实测总耗时）", window["duration_ms_mean"], state])
            items.append(dict(label=rows[-1][1], value_ms=window["duration_ms_mean"],
                              window_key=key, unresolved_window=True))
            whole_unknown_count += 1
            expected_items += 1
            unresolved_windows = True
            continue
        expected_items += expected
        ordered = not (len(stages) != expected or [item.get("stage_order") for item in stages] != list(range(1, expected + 1))
                or len({item.get("stage_key") for item in stages}) != expected
                or any(not isinstance(item.get("stage_key"), str) or not item["stage_key"] for item in stages))
        if not ordered:
            problems.append(label + "未提供完整有序的" + str(expected) + "个父阶段")
        samples = {item.get("rounds") for item in stages}
        window_rounds = (summary.get(key) or {}).get("rounds")
        same_samples = not (len(samples) > 1 or window_rounds is not None and samples != {window_rounds})
        if not same_samples:
            problems.append(label + "阶段样本数与窗口统计不一致")
        complete = ordered and same_samples and all(
            item.get("status") == "MATCHED" and _number(item.get("duration_ms_mean"))
            and item["duration_ms_mean"] >= 0 and _number(item.get("rounds")) and item["rounds"] > 0
            and item.get("matched_windows") == item["rounds"] for item in stages)
        if not complete:
            unresolved_windows = True
            problems.append(label + "阶段识别未完成")
            states = []
            for item in stages:
                state = str(item.get("matched_windows", NA)) + "/" + str(item.get("rounds", NA)) + " · " + str(item.get("status", "UNRESOLVED"))
                if item.get("reason"):
                    state += "；" + _text(item["reason"])
                states.append(state)
            value = (summary.get(key) or {}).get("duration_ms_mean")
            rows.append([label, "阶段未识别（原窗口实测总耗时）", value if _number(value) and value >= 0 else NA,
                         "；".join(dict.fromkeys(states)) if states else "UNRESOLVED；核心未提供阶段锚点统计"])
            continue
        values = []
        for item in stages:
            value = item.get("duration_ms_mean")
            stage_label = item.get("label") or item.get("stage_key") or "未命名阶段"
            state = str(item.get("matched_windows", NA)) + "/" + str(item.get("rounds", NA)) + " · " + str(item.get("status", "UNRESOLVED"))
            if item.get("reason"):
                state += "；" + _text(item["reason"])
            rows.append([label, stage_label, value, state])
            values.append(value)
            items.append({"label": stage_label, "value_ms": value})
        window_total = (summary.get(key) or {}).get("duration_ms_mean")
        if not _number(window_total) or len(values) != expected or not math.isclose(sum(values), window_total, rel_tol=1e-9, abs_tol=1e-8):
            problems.append(label + "阶段跨度未与源窗口闭合")
    partition = method.get("source_partition") or {}
    residual = partition.get("cross_role_tail_residual_ms", partition.get("residual_ms"))
    total = sum(item["value_ms"] for item in items)
    if whole_unknown_count:
        counts = [summary.get(key, {}).get("rounds") for key, _ in RESIDUAL_WINDOWS]
        if not all(type(count) is int and count > 0 for count in counts) or len(set(counts)) != 1:
            problems.append("未识别整窗与已匹配窗口未使用相同保留轮次")
    if (not _number(residual) or residual <= 0 or not _number(diagnostic.get("residual_ms_mean"))
            or len(items) != expected_items or not math.isclose(total, residual, rel_tol=1e-9, abs_tol=1e-8)
            or not math.isclose(total, diagnostic["residual_ms_mean"], rel_tol=1e-9, abs_tol=1e-8)):
        problems.append(("窗口/阶段" if whole_unknown_count else "15阶段") + "未与本方法源分区中的U闭合")
    if not _layer_mean_publishable(method):
        problems.append("本方法目标结果尚不可发布")
    note = ("仅本方法所选rank=" + _text(selected) + "；以下为已含在完整周期中的轮次剩余区间U组成，"
            "不额外加到完整周期。设备锚点之间的连续跨度包含间隔，不代表完整Python模块耗时；各父阶段的细分子项不另加。")
    if whole_unknown_count and not problems:
        note += ("识别不完整：本图按" + str(len(items)) + "项展示，含" + str(whole_unknown_count)
                 + "个未识别整窗和" + str(len(items) - whole_unknown_count)
                 + "个已匹配阶段，并非15阶段全匹配。未识别项保留原窗口实测总量，不补零、不拆造语义阶段；状态及缺锚原因见表格。")
    if problems:
        note += "未绘图：" + "；".join(dict.fromkeys(problems)) + "。"
        note += ("阶段识别未完成，原窗口耗时仍保留，按具体原因核对原始设备明细；原匹配详情见时间账本。"
                 if unresolved_windows else "请按具体原因核对时间账本和原始设备明细。")
    return rows, None if problems else items, note


def _method_residual_section(sheet, row, case, method):
    rows, items, note = _method_residual_data(case, method)
    row = _note(sheet, row, "轮次剩余区间耗时构成 · " + _rank_note(method))
    row = _note(sheet, row, note, warning=items is None or any(item.get("unresolved_window") for item in items))
    if not rows:
        return row + 1
    written = []
    row = _table(sheet, row, METHOD_RESIDUAL_HEADERS, rows, record_rows=written,
                 spans=((1, 2), (3, 8), (9, 10), (11, 12)))
    if items is None:
        return row
    if written != list(range(written[0], written[0] + len(items))):
        return _note(sheet, row, "阶段表存在长文本续行，未绘图；原始阶段数值保留。", warning=True)
    height = max(18, 2 * len(items) + 6)
    reference = "'" + sheet.name.replace("'", "''") + "'!"
    partial = any(item.get("unresolved_window") for item in items)
    sheet.add_bar_chart("轮次剩余区间耗时构成 · " + _layer_mean_name(method) + ("（含未识别整窗）" if partial else ""),
        categories=[item["label"] for item in items],
        categories_ref=reference + "$C$" + str(written[0]) + ":$C$" + str(written[-1]),
        series=[dict(name="窗口/阶段耗时(ms)" if partial else "阶段耗时(ms)", values=[item["value_ms"] for item in items], color="168B8D",
                     values_ref=reference + "$I$" + str(written[0]) + ":$I$" + str(written[-1]))],
        anchor=(row, 1, row + height, 12), stacked=False, x_axis_title="ms/HC周期；已计入完整周期，不额外相加")
    return row + height + 2


METHOD_STAGE_TITLES = ("SWA层内阶段展开", "下一主模型KV与输入准备 · 阶段展开",
                       "下一主metadata及输入映射准备 · 阶段展开")
MAIN_STAGE_TITLES = ("CSA层内阶段展开", "HCA层内阶段展开")
METHOD_STAGE_SCOPE = ("CSA/HCA/SWA展示核心提供的目标估算；两项U子阶段保持源实测、完整周期只保留一次。"
                      "全部子项已含在完整周期中，不额外计入完整周期或TPOT。")
TARGET_STAGE_HEADERS = ["细分阶段（已含）", "源单层均值(ms)", "目标单层均值(ms)",
                        "目标同类全部层合计(ms/完整周期)", "占完整周期", "状态 / 说明"]
TARGET_STAGE_SPANS = ((1, 3), (4, 4), (5, 5), (6, 7), (8, 8), (9, 12))
SWA_STAGE_KEYS = ("attention_prepare", "attention", "attention_output", "attention_post", "moe_prepare",
                  "dispatch", "experts", "combine", "ffn_output", "ffn_post")


def _matched_stage(item, rounds):
    return (type(rounds) is int and rounds > 0 and item.get("rounds") == rounds
            and item.get("matched_windows") == rounds and item.get("status") == "MATCHED"
            and _number(item.get("duration_ms_mean")) and item["duration_ms_mean"] >= 0)


def _stage_match_text(item):
    text = str(item.get("matched_windows", NA)) + "/" + str(item.get("rounds", NA)) + " · " + str(item.get("status", "UNAVAILABLE"))
    return text + ("；" + _text(item["reason"]) if item.get("reason") else "")


def _empty_expansion(title, reason):
    return dict(title=title, headers=["细分阶段（已含）", "跨度均值(ms)", "匹配轮次 / 状态"],
                spans=((1, 6), (7, 9), (10, 12)), rows=[["核心未提供可核对的细分", NA, "UNAVAILABLE"]],
                series=None, note=reason)


def _target_layer_expansion_data(method, role, attention, title):
    """Validate and display core target values; never estimate from diagnostics."""
    result = dict(title=title + " · 目标估算", headers=TARGET_STAGE_HEADERS, spans=TARGET_STAGE_SPANS,
                  rows=[["核心未提供可核对的目标细分", NA, NA, NA, NA, "UNAVAILABLE"]],
                  series=None, percent_columns=(4,),
                  axis_title="目标同类全部层耗时(ms/完整周期)；已含在完整周期中",
                  scope="本方法所选rank的目标估算；图示同类全部目标层的每周期成本。占比以含其他类型、衔接和U的完整周期为分母，单类占比不要求合计100%。",
                  operator_group=(role, attention))
    groups = method.get("target_stage_breakdowns")
    if not isinstance(groups, list):
        result["note"] = "核心JSON缺少target_stage_breakdowns；请用新版核心重算。源层诊断仅在层与组件实测展示，不能代替目标估算绘图。"
        return result
    groups = [item for item in groups if isinstance(item, dict)
              and item.get("role") == role and item.get("attention") == attention]
    if len(groups) != 1:
        result["note"] = attention + "缺少唯一目标阶段分组，未绘图；请用新版核心重算。"
        return result
    group, problems = groups[0], []
    if group.get("status") != "MATCHED":
        problems.append("核心目标阶段状态=" + _text(group.get("status")))
    if group.get("reason"):
        problems.append(_text(group["reason"]))
    if not _layer_mean_publishable(method) or method.get("selected_rank") is None:
        problems.append("本方法目标结果或选中rank不可发布")
    references, count, rounds = group.get("reference_layers"), group.get("target_layers"), group.get("rounds")
    valid_references = (isinstance(references, list) and bool(references)
                        and all(type(layer) is int and layer >= 0 for layer in references)
                        and len(set(references)) == len(references))
    if not valid_references or type(count) is not int or count <= 0 or type(rounds) is not int or rounds <= 0:
        problems.append("参考层、目标层数或保留轮次无效")
    classes = [item for item in method.get("class_means") or []
               if item.get("role") == role and item.get("attention") == attention]
    matching = classes[0] if len(classes) == 1 else {}
    if (len(classes) != 1 or matching.get("reference_layers") != references
            or matching.get("target_layers") != count):
        problems.append("目标阶段分组与本方法选中rank的同类参考不一致")
    stages = sorted((item for item in group.get("stage_rows") or [] if isinstance(item, dict)),
                    key=lambda item: item.get("stage_order") if type(item.get("stage_order")) is int else -1)
    ordered = ([item.get("stage_order") for item in stages] == list(range(1, 11))
               and tuple(item.get("stage_key") for item in stages) == SWA_STAGE_KEYS)
    if not ordered:
        problems.append("目标10阶段标识或顺序不完整/重复")
    cycle = method.get("cycle_ms")
    valid_cycle = _number(cycle) and cycle > 0
    fields = ("source_ms", "delta_ms", "target_ms", "target_total_ms", "cycle_share")
    complete = ordered and all(all(_number(item.get(field)) for field in fields)
                               and min(item["source_ms"], item["target_ms"], item["target_total_ms"], item["cycle_share"]) >= 0
                               and item["cycle_share"] <= 1 for item in stages)
    if not complete or not valid_cycle:
        problems.append("目标阶段数值或完整周期不可用")
    close = lambda a, b: _number(a) and _number(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)
    if complete and valid_cycle:
        if (not close(math.fsum(item["source_ms"] for item in stages), group.get("source_wall_ms"))
                or not close(math.fsum(item["target_ms"] for item in stages), group.get("target_wall_ms"))
                or not close(math.fsum(item["target_total_ms"] for item in stages), group.get("target_total_ms"))):
            problems.append("目标10阶段未与源单层、目标单层或同类全部层父项闭合")
        if not close(group.get("source_wall_ms"), matching.get("wall_ms")) or not close(group.get("target_wall_ms"), matching.get("target_wall_ms")):
            problems.append("目标阶段父项与本方法同类单层均值不一致")
        parent_total = group.get("target_total_ms")
        if (type(count) is int and count > 0 and not all(close(item["target_total_ms"], item["target_ms"] * count) for item in stages)
                or not all(close(item["target_ms"], item["source_ms"] + item["delta_ms"]) for item in stages)
                or not all(close(item["cycle_share"], item["target_total_ms"] / cycle) for item in stages)
                or not _number(parent_total) or not close(math.fsum(item["cycle_share"] for item in stages), parent_total / cycle)
                or parent_total > cycle + 1e-8):
            problems.append("目标阶段合计或完整周期占比与父项不一致")
    state = "MATCHED" if not problems else "UNAVAILABLE；" + "；".join(dict.fromkeys(problems))
    if stages:
        result["rows"] = [[item.get("label") or item.get("stage_key") or "未命名阶段",
                           *[item.get(field) if _number(item.get(field)) else NA
                             for field in ("source_ms", "target_ms", "target_total_ms", "cycle_share")],
                           state + "；增量=" + _text(item.get("delta_ms")) + " ms/单层"] for item in stages]
    result["note"] = ("参考层=" + _text(references) + "；目标层数=" + _text(count) + "；保留轮次=" + _text(rounds)
                      + "；源单层跨度=" + _text(group.get("source_wall_ms")) + " ms；目标单层跨度="
                      + _text(group.get("target_wall_ms")) + " ms；目标同类全部层=" + _text(group.get("target_total_ms")) + " ms/完整周期。"
                      + ("主模型参考层由核心给定，实际层号见清单；逐源层诊断保留在层与组件实测。" if role == "main" else "草稿同类目标层数由核心提供。")
                      + "Dispatch/Combine增量分别落在对应阶段；专家只追加路由GMM增量，激活与衔接不整体倍增；其他阶段保持核心给定值。"
                      + ("核心说明：" + _text(group["note"]) if group.get("note") else ""))
    if problems:
        result["note"] += "未绘图：" + "；".join(dict.fromkeys(problems)) + "。"
    else:
        result["series"] = [(3, [item["target_total_ms"] for item in stages])]
    return result


def _u_expansion_data(diagnostic, parent_key, title):
    if not isinstance(diagnostic, dict):
        return _empty_expansion(title, "核心缺少U子阶段诊断；请用新版统计脚本重算原始采样。")
    window = "draft_to_nextmain"
    parents = [item for item in diagnostic.get("stage_summary") or []
               if item.get("window_key") == window and item.get("stage_key") == parent_key
               and item.get("level", 0) == 0 and not item.get("parent_stage_key")]
    summaries = [item for item in diagnostic.get("summary") or [] if item.get("window_key") == window]
    children = [item for item in diagnostic.get("substage_summary") or []
                if item.get("window_key") == window and item.get("parent_stage_key") == parent_key
                and item.get("level") == 1]
    if len(parents) != 1 or len(summaries) != 1 or not children:
        return _empty_expansion(title, "缺少唯一父阶段、窗口统计或细分子阶段；原父区间总耗时仍保留在上方U表。")
    parent, summary = parents[0], summaries[0]
    children.sort(key=lambda item: item.get("stage_order") if type(item.get("stage_order")) is int else -1)
    rounds = parent.get("rounds")
    complete = (summary.get("rounds") == rounds and _matched_stage(parent, rounds)
                and all(_matched_stage(item, rounds) for item in children))
    keys = [item.get("stage_key") for item in children]
    orders = [item.get("stage_order") for item in children]
    ordered = (all(isinstance(key, str) and key for key in keys) and len(set(keys)) == len(keys)
               and all(type(order) is int and order >= 0 for order in orders) and len(set(orders)) == len(orders))
    closed = complete and math.isclose(math.fsum(item["duration_ms_mean"] for item in children),
                                      parent["duration_ms_mean"], rel_tol=1e-9, abs_tol=1e-8)
    problems = []
    if not complete:
        problems.append("子阶段未全部匹配或保留轮次与父阶段/窗口不一致")
    if not ordered:
        problems.append("子阶段标识或顺序重复/无效")
    if not closed:
        problems.append("子阶段未与父阶段实测跨度闭合")
    records = [[item.get("label") or item.get("stage_key") or "未命名阶段",
                item["duration_ms_mean"] if _matched_stage(item, rounds) else NA, _stage_match_text(item)]
               for item in children]
    note = "源父阶段跨度=" + _text(parent.get("duration_ms_mean")) + " ms/HC周期；源实测U仅保留一次，不按目标层数放大，不使用通信或专家倍率。"
    if problems:
        note += "未绘图：" + "；".join(problems) + "。"
    return dict(title=title, headers=["细分阶段（已含）", "跨度均值(ms/HC周期)", "匹配轮次 / 状态"],
                spans=((1, 6), (7, 9), (10, 12)), rows=records,
                series=None if problems else [(1, [row[1] for row in records])], note=note,
                scope="仅本方法所选rank的源实测U细分，完整周期只保留一次，不外推。",
                axis_title="源实测U阶段(ms/完整周期)；U仅保留一次")


def _method_expansion_data(case, method):
    """Read the selected capture only; never re-estimate or choose a rank."""
    target_groups = [_target_layer_expansion_data(method, "main", "CSA", MAIN_STAGE_TITLES[0]),
                     _target_layer_expansion_data(method, "main", "HCA", MAIN_STAGE_TITLES[1]),
                     _target_layer_expansion_data(method, "draft", "SWA", METHOD_STAGE_TITLES[0])]
    selected = method.get("selected_rank")
    captures = [capture for capture in case.get("measurements") or [] if capture.get("rank") == selected]
    if selected is None or len(captures) != 1:
        reason = "本方法未选中rank。" if selected is None else "所选rank缺少唯一对应的采样诊断；请重跑基础统计。"
        return target_groups + [_empty_expansion(title, reason) for title in METHOD_STAGE_TITLES[1:]]
    capture = captures[0]
    residual = capture.get("step_residuals")
    if isinstance(residual, dict) and residual.get("rank", selected) != selected:
        residual = None
    return [*target_groups, _u_expansion_data(residual, "main_inputs", METHOD_STAGE_TITLES[1]),
            _u_expansion_data(residual, "main_metadata", METHOD_STAGE_TITLES[2])]


OPERATOR_LEDGER_HEADERS = ["细项 / 父阶段（已含）", "源覆盖×目标层数\n(ms/完整周期)", "模型增减×目标层数\n(ms/完整周期)",
                           "合计(ms)", "占完整周期", "类型 / 算子与来源"]
OPERATOR_LEDGER_SPANS = ((1, 4), (5, 6), (7, 8), (9, 9), (10, 10), (11, 12))
OPERATOR_KERNEL_HEADERS = ["源实测算子 / 父阶段", "相交调用\n/单层均值", "段内累计\n(ms/单层均值)",
                          "段内单次均值\n(ms)", "段内单次P90\n(ms)", "native / 指纹与来源"]
OPERATOR_KERNEL_SPANS = ((1, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 12))
OPERATOR_SCOPE = ("关键路径及优化收益尚未验证。目标互斥区间和有符号模型增减仅展开已有父项，不再计入周期；"
                  "具名并行组合不分摊给成员，无记录区间不等于CPU瓶颈。源kernel累计可重叠，不能当目标关键耗时或优化收益；"
                  "调用按段内相交记录统计，跨阶段不可机械累加调用数。完整指纹、成员及样例见核心JSON。")
OPERATOR_KINDS = {"operator": "单算子覆盖", "parallel": "具名并行组合", "gap": "无设备记录区间", "model_delta": "模型增减（非原生执行）"}
PMU_SCOPE = ("以下顺序与上方源实测Top8一一对应；PMU按原生整次调用Duration加权，保持源实测，不乘目标层数或外推倍率。"
             "各流水可以同时工作，活动比不可相加为100%，也不是MFU或理论峰值利用率。"
             "bound为流水活动证据支持的方向判断，不直接证明HBM带宽跑满、关键路径或优化收益。"
             "若调用跨父阶段，PMU仍属于整次调用；同一调用在不同阶段引用不能相加。")
PMU_METRICS = (("aic_mac_ratio", "AIC MAC\n矩阵计算"),
               ("aic_mte2_ratio", "AIC MTE2\n读入搬运"), ("aic_mte3_ratio", "AIC MTE3\n写出搬运"),
               ("aiv_vec_ratio", "AIV VEC\n向量计算"),
               ("aiv_mte2_ratio", "AIV MTE2\n读入搬运"), ("aiv_mte3_ratio", "AIV MTE3\n写出搬运"))
PMU_HEADERS = ["源热点编号 / 算子", *[label for _, label in PMU_METRICS], "Scalar\nAIC / AIV", "PMU证据覆盖", "bound方向 / 置信度"]
PMU_SPANS = ((1, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12))
PMU_DETAIL_HEADERS = ["源热点编号 / 算子", "完整指纹与高级指标", "bound依据 / 限制"]
PMU_DETAIL_SPANS = ((1, 3), (4, 8), (9, 12))


def _pmu_metric_value(pmu, key):
    item = (pmu.get("metrics") or {}).get(key)
    return item["value"] if isinstance(item, dict) and _number(item.get("value")) else None


def _pmu_metric_text(item):
    if not isinstance(item, dict) or not _number(item.get("value")):
        return "未采集"
    value = format(item["value"], ".1%") if item.get("unit") == "ratio" else format(item["value"], ".6g") + " " + _text(item.get("unit") or "")
    coverage = item.get("duration_coverage")
    if _number(coverage) and coverage < 1 - 1e-9:
        value += "\n覆盖" + format(coverage, ".1%")
    return value


def _pmu_label(kernel, number):
    # Keep names complete without repeating a long shape in every PMU column.
    return "K" + str(number) + " · " + _text(kernel.get("op_type") or kernel.get("name")) + "\n父阶段：" + _text(kernel.get("stage_label")) + "\nstream=" + _text(kernel.get("stream_id"))


def _operator_pmu_data(kernels):
    rows, details = [], []
    for number, kernel in enumerate(kernels, 1):
        pmu = kernel.get("pmu") or {}
        metrics, coverage, bound = pmu.get("metrics") or {}, pmu.get("coverage") or {}, pmu.get("bound") or {}
        coverage_text = (format(coverage["duration_coverage"], ".1%") + "原调用时长\n"
            + _text(coverage.get("known_calls")) + "/" + _text(coverage.get("total_calls")) + "调用"
            if _number(coverage.get("duration_coverage")) else "未采集")
        if coverage.get("clipped_calls"):
            coverage_text += "\n跨段引用=" + _text(coverage["clipped_calls"])
        conclusion = _text(bound.get("label") or "证据不足") + "\n置信度：" + _text(bound.get("confidence") or "未提供")
        label = _pmu_label(kernel, number)
        rows.append([label, *[_pmu_metric_text(metrics.get(key)) for key, _ in PMU_METRICS],
            _pmu_metric_text(metrics.get("aic_scalar_ratio")) + " / " + _pmu_metric_text(metrics.get("aiv_scalar_ratio")),
            coverage_text, conclusion])
        metric_details = []
        for key, item in metrics.items():
            if not isinstance(item, dict):
                continue
            metric_details.append(key + "=" + _pmu_metric_text(item).replace("\n", "；")
                + "；调用=" + _text(item.get("known_calls")) + "/" + _text(item.get("total_calls")))
        evidence = bound.get("evidence") or []
        details.append([label, _operator_fingerprint_text(kernel) + "\n\n" + ("\n".join(metric_details) or "PMU未采集，请用新版脚本重新统计；不会以0替代。"),
            conclusion + "\n" + "\n".join(_text(item) for item in evidence)
            + "\n" + _text(pmu.get("scope_note") or "没有可用PMU证据，不推断bound。")])
    return rows, details


def _operator_pmu_section(sheet, row, attention, kernels, method):
    row = _note(sheet, row, attention + "源实测Top8 · 流水活动与bound诊断")
    row = _note(sheet, row, PMU_SCOPE)
    records, details = _operator_pmu_data(kernels)
    if not kernels or not any((kernel.get("pmu") or {}).get("metrics") for kernel in kernels):
        row = _note(sheet, row, "PMU未采集，请重新统计；上方原热点排序与耗时保留，未知指标不补0。", warning=True)
    row, _, _ = _operator_table(sheet, row, PMU_HEADERS, records, PMU_SPANS)
    row, _, _ = _operator_table(sheet, row, PMU_DETAIL_HEADERS, details, PMU_DETAIL_SPANS)
    return row


def _operator_example_text(example):
    if not isinstance(example, dict):
        return "无独立原生执行样例；模型修正不代表已验证收益"
    text = "rank=" + _text(example.get("rank")) + "；轮=" + _text(example.get("round_id")) + "；层=" + _text(example.get("layer"))
    text += "\n原行=" + _text(example.get("row_indices", example.get("row_index")))
    if _number(example.get("duration_ns")):
        text += "；例跨度=" + format(example["duration_ns"] / 1e6, ".6g") + "ms"
    for label, key in (("前", "before"), ("后", "after")):
        task = example.get(key)
        if isinstance(task, dict):
            text += "\n" + label + "=" + _text(task.get("op_type") or task.get("name"))
            if task.get("name") and task.get("name") != task.get("op_type"):
                text += "；native=" + task["name"]
            text += "[s=" + _text(task.get("stream_id")) + ",row=" + _text(task.get("row_index")) + "]"
        elif key in example:
            text += "；" + label + "=边界处无相邻任务"
    return text


def _operator_fingerprint_text(item):
    return ("指纹#" + item["key"] + "\nnative=" + _text(item.get("name") or "未导出名称")
        + "\nop_type=" + _text(item.get("op_type"))
        + "\nshape in=" + _text(item.get("input_shapes")) + " → out=" + _text(item.get("output_shapes"))
        + "\ndtype in=" + _text(item.get("input_dtypes")) + " → out=" + _text(item.get("output_dtypes"))
        + "\nstream=" + _text(item.get("stream_id")) + "；task=" + _text(item.get("task_type"))
        + ("\n" + _operator_example_text(item.get("example")) if "example" in item else ""))


def _operator_label(item, number):
    if item.get("remainder"):
        return item["label"]
    parts = ["#" + str(number) + " " + _text(item.get("stage_label"))]
    kind = item["kind"]
    if kind in ("operator", "parallel"):
        for index, member in enumerate(item["members"], 1):
            parts.append(("∥ 成员" + str(index) + "：" if kind == "parallel" else "")
                         + _text(member.get("op_type") or member.get("name") or "未导出名称"))
            parts.append("输入shape=" + _text(member.get("input_shapes") or "未记录"))
            parts.append("stream=" + _text(member.get("stream_id"))
                         + ("；并发实例=" + str(member.get("instances", 1)) if member.get("instances", 1) > 1 else ""))
    else:
        parts.append(item["label"].removeprefix("模型增减：").removeprefix("模型增减:") if kind == "model_delta" else item["label"])
    return "\n".join(line for part in parts for line in _operator_svg_label_lines(part, width=72))


def _operator_preview(records, limit):
    ordered = sorted(records, key=lambda item: (-abs(item["target_total_ms"]), item["stage_order"], item["key"]))
    retained, rest = ordered[:limit], ordered[limit:]
    for kind, label in (("operator", "单算子覆盖"), ("parallel", "具名并行组合"), ("gap", "无设备记录区间"),
                        ("model_positive", "正模型增量"), ("model_negative", "负模型修正")):
        subset = [item for item in rest if (item["kind"] == kind or item["kind"] == "model_delta"
                  and (kind == "model_positive" and item["target_total_ms"] >= 0
                       or kind == "model_negative" and item["target_total_ms"] < 0))]
        if not subset:
            continue
        def sample_span(item):
            sample = item.get("example") or {}
            if _number(sample.get("duration_ns")):
                return sample["duration_ns"]
            start, end = sample.get("start_ns"), sample.get("end_ns")
            try:
                return int(end) - int(start)
            except (ValueError, TypeError):
                return -1
        representative = max(subset, key=sample_span)
        retained.append(dict(key="remainder_" + kind, kind="model_delta" if kind.startswith("model_") else kind,
            label="其余" + str(len(subset)) + "项" + label + "合计（非单一热点）", remainder=True,
            members=[], example=representative.get("example"), stage_label="多个父阶段", stage_order=99,
            details="\n\n".join(_operator_label(item, index) for index, item in enumerate(subset[:3], 1)),
            **{field: math.fsum(item[field] for item in subset)
               for field in ("source_ms", "delta_ms", "target_ms", "target_total_ms", "cycle_share")}))
    return retained


def _method_operator_data(method, role, attention, limit=12, kernel_limit=8):
    """Only display a complete core operator contract with the same parent means."""
    result = dict(attention=attention, title=attention + "算子下钻 · 目标贡献", available=False,
                  target_rows=[], kernel_rows=[], target_items=[], kernel_items=[], note="")
    parent = _target_layer_expansion_data(method, role, attention, attention + "层内阶段展开")
    if parent["series"] is None:
        result["note"] = "目标父阶段不可发布，算子下钻未绘图。" + parent["note"]
        return result
    group = next(item for item in method["target_stage_breakdowns"] if item.get("role") == role and item.get("attention") == attention)
    count, cycle, selected = group["target_layers"], method["cycle_ms"], method["selected_rank"]
    result["scope"] = ("rank=" + _text(selected) + "；参考层=" + _text(group["reference_layers"])
        + "；保留轮次=" + str(group["rounds"]) + "；目标同类全部层=" + str(count)
        + "；完整周期=" + _text(cycle) + " ms。"
        + ("主0是否作参考以层号清单为准；目标全层数不再扣除参考排除层。" if role == "main" else "草稿完整同类层数由核心提供。"))
    close = lambda a, b: _number(a) and _number(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)
    all_rows, all_kernels, notes = [], [], []

    def evidence(example, needs_row=False):
        if (not isinstance(example, dict) or example.get("rank") != selected
                or example.get("layer") not in group["reference_layers"] or example.get("round_id") is None):
            return False
        if needs_row:
            rows = example.get("row_indices")
            return (type(example.get("row_index")) is int and example["row_index"] >= 0
                    or isinstance(rows, list) and bool(rows) and all(type(row) is int and row >= 0 for row in rows))
        return True

    try:
        for stage in sorted(group["stage_rows"], key=lambda item: item["stage_order"]):
            prefix = "父阶段" + str(stage["stage_order"]) + "（" + _text(stage.get("label", stage["stage_key"])) + "）"
            breakdown = stage.get("operator_breakdown")
            if not isinstance(breakdown, dict):
                raise ValueError(prefix + "缺少operator_breakdown；请用新版核心重算")
            if breakdown.get("status") != "MATCHED" or breakdown.get("reason"):
                raise ValueError(prefix + "算子证据不可用：" + _text(breakdown.get("reason") or breakdown.get("status")))
            if breakdown.get("reference_layers") != group["reference_layers"] or breakdown.get("rounds") != group["rounds"]:
                raise ValueError(prefix + "算子参考层或保留轮次与父项不一致")
            rows, kernels = breakdown.get("rows"), breakdown.get("kernels")
            if (not isinstance(rows, list) or not isinstance(kernels, list)
                    or any(not isinstance(item, dict) for item in rows + kernels)):
                raise ValueError(prefix + "算子明细/热点列表缺失")
            for records, label in ((rows, "互斥项"), (kernels, "kernel指纹")):
                keys = [item.get("key") for item in records]
                if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
                    raise ValueError(prefix + label + "标识重复/无效")
            fingerprints = {item["key"]: item for item in kernels}
            for kernel in kernels:
                if (not isinstance(kernel.get("name"), str)
                        or not all(field in kernel for field in ("op_type", "input_shapes", "output_shapes", "input_dtypes",
                                                                 "output_dtypes", "stream_id", "task_type"))
                        or not all(_number(kernel.get(field)) and kernel[field] >= 0 for field in
                                   ("calls_mean", "kernel_sum_ms", "duration_mean_ms", "duration_p90_ms"))
                        or kernel["calls_mean"] <= 0 or not evidence(kernel.get("example"), True)
                        or not close(kernel["kernel_sum_ms"], kernel["calls_mean"] * kernel["duration_mean_ms"])):
                    raise ValueError(prefix + "kernel热点数值、完整指纹或原行/轮/层证据无效")
                all_kernels.append({**kernel, "stage_label": stage.get("label") or stage["stage_key"], "stage_order": stage["stage_order"]})
            for item in rows:
                kind, members = item.get("kind"), item.get("members")
                if (kind not in OPERATOR_KINDS or not isinstance(item.get("label"), str) or not item["label"].strip()
                        or not isinstance(members, list) or not all(_number(item.get(field)) for field in
                            ("source_ms", "delta_ms", "target_ms", "target_total_ms", "cycle_share"))
                        or item["source_ms"] < 0 or not close(item["target_ms"], item["source_ms"] + item["delta_ms"])
                        or not close(item["target_total_ms"], item["target_ms"] * count)
                        or not close(item["cycle_share"], item["target_total_ms"] / cycle)):
                    raise ValueError(prefix + "互斥项字段/目标值/占比无效")
                if kind == "model_delta":
                    if members or not close(item["source_ms"], 0):
                        raise ValueError(prefix + "模型修正混入了原生执行覆盖")
                else:
                    if not close(item["delta_ms"], 0) or item["target_total_ms"] < 0 or not evidence(item.get("example"), kind != "gap"):
                        raise ValueError(prefix + "原生覆盖缺定位或混入模型修正")
                    if kind == "gap":
                        if members or not all(key in item["example"] for key in ("before", "after")):
                            raise ValueError(prefix + "无记录区间缺少前后任务定位")
                    else:
                        if not members or any(not isinstance(member, dict) for member in members):
                            raise ValueError(prefix + "缺少具名算子成员")
                        active = 0
                        member_keys = []
                        for member in members:
                            fingerprint = fingerprints.get(member.get("key"))
                            if (fingerprint is None or any(member.get(field) != fingerprint.get(field) for field in
                                ("name", "op_type", "input_shapes", "output_shapes", "input_dtypes", "output_dtypes", "stream_id", "task_type"))
                                    or type(member.get("instances")) is not int or member["instances"] <= 0):
                                raise ValueError(prefix + "成员指纹/并发数量与kernel热点不一致")
                            member_keys.append(member["key"])
                            active += member["instances"]
                        if len(set(member_keys)) != len(member_keys) or (kind == "operator" and active != 1) or (kind == "parallel" and active < 2):
                            raise ValueError(prefix + "单算子/并行组合成员数量不一致")
                all_rows.append({**item, "stage_label": stage.get("label") or stage["stage_key"], "stage_order": stage["stage_order"]})
            if not all(close(math.fsum(item[field] for item in rows), stage[field])
                       for field in ("source_ms", "delta_ms", "target_ms", "target_total_ms", "cycle_share")):
                raise ValueError(prefix + "算子子项未与父阶段闭合")
            if breakdown.get("note"):
                notes.append(_text(breakdown["note"]))
        if not close(math.fsum(item["target_total_ms"] for item in all_rows), group["target_total_ms"]):
            raise ValueError("算子子项未与目标同类全部层闭合")
    except (ValueError, KeyError, TypeError, ArithmeticError) as error:
        result["note"] = "算子下钻未绘图：" + str(error) + "。原十阶段表图与父类结果仍保留；未知子项不补0。"
        return result
    preview = _operator_preview(all_rows, limit)
    target_rows = []
    for number, item in enumerate(preview, 1):
        correction = item["kind"] == "model_delta"
        detail = OPERATOR_KINDS[item["kind"]] + "\n指纹#" + item["key"] + "\n" + _operator_example_text(item.get("example"))
        if item.get("details"):
            detail += "\n前三项预览：\n" + item["details"]
        elif item["members"]:
            detail += "\n\n" + "\n\n".join("成员" + str(index) + "\n" + _operator_fingerprint_text(member)
                                                for index, member in enumerate(item["members"], 1))
        target_rows.append([_operator_label(item, number), 0. if correction else item["target_total_ms"],
                            item["target_total_ms"] if correction else 0., item["target_total_ms"], item["cycle_share"], detail])
    kernels = sorted(all_kernels, key=lambda item: (-item["kernel_sum_ms"], item["stage_order"], item["key"]))[:kernel_limit]
    kernel_rows = [[_operator_label(dict(item, kind="operator", members=[dict(item, instances=1)]), "K" + str(number)),
        item["calls_mean"], item["kernel_sum_ms"], item["duration_mean_ms"],
        item["duration_p90_ms"], _operator_fingerprint_text(item)] for number, item in enumerate(kernels, 1)]
    total_row = ["合计（闭合目标父类）", math.fsum(row[1] for row in target_rows), math.fsum(row[2] for row in target_rows),
                 group["target_total_ms"], math.fsum(item["cycle_share"] for item in all_rows), "完整明细已核对全部10个父阶段；模型增减不代表优化收益"]
    result.update(available=True, target_rows=target_rows, kernel_rows=kernel_rows, target_items=preview, kernel_items=kernels,
                  total_row=total_row, basis_notes=list(dict.fromkeys(notes)),
                  note="目标Top" + str(limit) + "及分类剩余合计：预览" + str(len(preview))
                  + "行 / 完整" + str(len(all_rows)) + "项，合计闭合父类。gap独立列示，定位取最大间隙示例。源热点Top"
                  + str(kernel_limit) + " / 完整" + str(len(all_kernels)) + "个阶段内指纹；单次均值/P90取实测段内裁剪调用，不含未出现调用。")
    return result


def _operator_table(sheet, row, headers, records, spans, percent=False, *, percent_columns=()):
    header, written, expanded = row, [], []
    widths = [max(8, int(sum(WIDTHS[first - 1:last]) - 4)) for first, last in spans]
    for number, record in enumerate(records, 1):
        cells = [_operator_svg_label_lines(value, width=widths[index]) if isinstance(value, str) else [value]
                 for index, value in enumerate(record)]
        pieces = max((math.ceil(len(lines) / 27) for lines in cells), default=1)
        for piece in range(pieces):
            values = []
            for index, lines in enumerate(cells):
                if isinstance(record[index], str):
                    value = "\n".join(lines[piece * 27:(piece + 1) * 27])
                    if index == 0 and piece:
                        value = "（续 #" + str(number) + " · " + str(piece + 1) + "/" + str(pieces) + "）\n" + value
                else:
                    value = record[index] if piece == 0 else None
                values.append(value)
            expanded.append(values)
    row = _table(sheet, row, headers, expanded, spans=spans, record_rows=written,
                 blank_columns=tuple(range(1, len(headers) + 1)))
    sheet.row_height(header, 56)
    for actual, record in zip(written, expanded):
        lines = max((len(_operator_svg_label_lines(value, width=widths[index]))
                     for index, value in enumerate(record) if isinstance(value, str)), default=1)
        sheet.row_height(actual, min(409, max(42, 14 * lines + 12)))
        for column in ((4,) if percent else percent_columns):
            if _number(record[column]):
                sheet.write(actual, spans[column][0], record[column], "percent")
    return row, header, written if len(expanded) == len(records) else []


def _operator_bar(sheet, row, title, headers, records, spans, header, written, columns, axis):
    if not records:
        return _note(sheet, row, "本类没有可绘制算子记录；实测零值及核心JSON保留。")
    if not written or written != list(range(written[0], written[0] + len(records))):
        row = _note(sheet, row, "图表数据（与上表同一组数值，不另加）；正文长文本已在续行完整展示。")
        headers = [headers[0], *[headers[column] for column in columns]]
        records = [[record[0], *[record[column] for column in columns]] for record in records]
        spans = ((1, 8), (9, 12)) if len(columns) == 1 else ((1, 8), (9, 10), (11, 12))
        columns = tuple(range(1, len(columns) + 1))
        header, written = row, []
        row = _table(sheet, row, headers, records, spans=spans, record_rows=written)
        if written != list(range(written[0], written[0] + len(records))):
            return _note(sheet, row, "单条图分类超过Excel单元格长度上限，完整正文已分续行展示；本图不截断分类绘制。", warning=True)
        for actual, record in zip(written, records):
            sheet.row_height(actual, min(409, max(42, 14 * len(record[0].splitlines()) + 12)))
    reference = "'" + sheet.name.replace("'", "''") + "'!"
    series = []
    for index, column in enumerate(columns):
        letter = spans[column][0]
        series.append(dict(name=headers[column], values=[record[column] for record in records], color=("168B8D", "D4793A")[index],
            name_ref=reference + cell_reference(header, letter, True),
            values_ref=reference + cell_reference(written[0], letter, True) + ":" + cell_reference(written[-1], letter, True)))
    height = max(18, math.ceil(sum(max(3, len(record[0].splitlines()) + 1) for record in records) * .85) + 6)
    sheet.add_bar_chart(title, categories=[record[0] for record in records], series=series,
        categories_ref=reference + cell_reference(written[0], 1, True) + ":" + cell_reference(written[-1], 1, True),
        anchor=(row, 1, row + height, 12), stacked=False, x_axis_title=axis)
    return row + height + 2


def _method_operator_section(sheet, row, method, role, attention):
    data = _method_operator_data(method, role, attention)
    sheet.merge(row, 1, row, 12, data["title"], "section").row_height(row, 30)
    row += 1
    row = _note(sheet, row, data.get("scope", "") + data["note"], warning=not data["available"])
    if not data["available"]:
        return row
    row = _note(sheet, row, OPERATOR_SCOPE + "来源JSON=" + str(sheet.report_origin))
    row, header, written = _operator_table(sheet, row, OPERATOR_LEDGER_HEADERS,
        [*data["target_rows"], data["total_row"]], OPERATOR_LEDGER_SPANS, True)
    row = _operator_bar(sheet, row, attention + "算子互斥覆盖与模型增减 · " + _layer_mean_name(method),
        OPERATOR_LEDGER_HEADERS, data["target_rows"], OPERATOR_LEDGER_SPANS, header, written[:-1], (1, 2),
        "ms/目标同类全部层；含有符号模型修正，非优化收益")
    row = _note(sheet, row, attention + "源实测算子热点；相交调用/单层均值，段内单次均值/P90；不加到互斥账本")
    row, header, written = _operator_table(sheet, row, OPERATOR_KERNEL_HEADERS, data["kernel_rows"], OPERATOR_KERNEL_SPANS)
    row = _operator_bar(sheet, row, attention + "源实测算子热点 · " + _layer_mean_name(method),
        OPERATOR_KERNEL_HEADERS, data["kernel_rows"], OPERATOR_KERNEL_SPANS, header, written, (2,),
        "源段内kernel累计(ms/参考单层均值)；可重叠，不是目标贡献")
    return _operator_pmu_section(sheet, row, attention, data["kernel_items"], method)


def _write_method_expansion(sheet, row, group, method):
    row = _note(sheet, row, group["title"] + " · " + _rank_note(method))
    row = _note(sheet, row, group.get("scope", METHOD_STAGE_SCOPE) + group["note"], warning=group["series"] is None)
    header, written = row, []
    row = _table(sheet, row, group["headers"], group["rows"], spans=group["spans"], record_rows=written)
    for actual, record in zip(written, group["rows"]):
        sheet.row_height(actual, max(35, 16 * str(record[-1]).count("\n") + 35))
        for column in group.get("percent_columns", ()):
            if _number(record[column]):
                sheet.write(actual, group["spans"][column][0], record[column], "percent")
    if not group["series"]:
        return row
    if written != list(range(written[0], written[0] + len(group["rows"]))):
        return _note(sheet, row, "阶段表存在长文本续行，未绘图；原始数值保留。", warning=True)
    reference = "'" + sheet.name.replace("'", "''") + "'!"
    series = []
    for index, (column_index, values) in enumerate(group["series"]):
        col = group["spans"][column_index][0]
        series.append(dict(name=group["headers"][column_index], values=values,
            name_ref=reference + cell_reference(header, col, True),
            values_ref=reference + cell_reference(written[0], col, True) + ":" + cell_reference(written[-1], col, True),
            color=("168B8D", "4F90C6", "D59440", "7668AA", "5BA28A")[index % 5]))
    height = max(18, (len(series) + 1) * len(group["rows"]) + 6)
    sheet.add_bar_chart(group["title"] + " · " + _layer_mean_name(method),
        categories=[record[0] for record in group["rows"]], series=series,
        categories_ref=reference + cell_reference(written[0], 1, True) + ":" + cell_reference(written[-1], 1, True),
        anchor=(row, 1, row + height, 12), stacked=False,
        x_axis_title=group.get("axis_title", "源实测阶段耗时(ms)；已含在原层或U中"))
    return row + height + 2


def _method_stage_expansions(sheet, row, case, method):
    index_row, positions = row, []
    row = _note(sheet, row, "CSA/HCA/SWA算子下钻位置：正在列示")
    for group in _method_expansion_data(case, method):
        row = _write_method_expansion(sheet, row, group, method)
        if group.get("operator_group"):
            role, attention = group["operator_group"]
            positions.append(attention + "：第" + str(row) + "行")
            row = _method_operator_section(sheet, row, method, role, attention)
    sheet.write(index_row, 1, "CSA/HCA/SWA算子下钻位置：" + "；".join(positions), "note")
    return row


FOCUS_KEYS = (("main_inputs", "下一主模型KV与输入准备"), ("main_metadata", "下一主metadata及输入映射准备"))
FOCUS_SCOPE = ("保持一个固定rank，逐项使用原保留HC轮次，不逐轮挑最大rank、不拼接rank。"
               "横轴是原HC轮次ID，不是已验证的业务请求step或ProfilerStep；缺失/未识别轮次留空断线。"
               "设备细项按任务起止切成互斥区间，并行任务保留组合、无设备记录区间独立列出；这些细项已含在父段和完整周期内，不额外相加。"
               "源码函数入口是静态对照，不是CPU函数实测时间；没有Host调用关联时不能把间隙归因给某个Python函数。")
FOCUS_PARENT_HEADERS = ["父阶段（已含于U）", "均值(ms)", "P50(ms)", "P90(ms)", "最大(ms)", "观测 / 保留轮次", "状态"]
FOCUS_PARENT_SPANS = ((1, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 10), (11, 12))
FOCUS_STEP_HEADERS = ["保留序号", "原HC轮次ID", "KV与输入(ms)", "metadata(ms)", "KV设备记录(ms)", "KV无记录(ms)",
                      "metadata设备记录(ms)", "metadata无记录(ms)", "状态 / 原始边界"]
FOCUS_STEP_SPANS = ((1, 1), (2, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 12))
FOCUS_COMPONENT_HEADERS = ["细项（已含）", "均值(ms)", "P50(ms)", "P90(ms)", "最大(ms)", "占父段", "出现轮次", "类型 / 映射状态"]
FOCUS_COMPONENT_SPANS = ((1, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 12))
FOCUS_FUNCTION_HEADERS = ["细项（与上表对应）", "设备算子 / 完整形状 / 原行", "对应源码入口（静态对照）", "映射证据 / 范围"]
FOCUS_FUNCTION_SPANS = ((1, 3), (4, 6), (7, 10), (11, 12))
FOCUS_KINDS = {"single_operator": "单算子覆盖", "concurrent_operators": "并行算子组合", "no_device_record": "无设备记录区间"}


def _focus_function_text(mapping):
    if not isinstance(mapping, dict):
        return "未能对应源码入口"
    path = _text(mapping.get("path") or "未提供路径")
    if mapping.get("line") is not None:
        path += ":" + _text(mapping["line"])
    text = _text(mapping.get("function") or "未知函数") + "\n" + path
    if mapping.get("version_basis"):
        text += "\n源码基准：" + _text(mapping["version_basis"])
    if mapping.get("call_chain"):
        text += "\n调用链：" + " → ".join(_text(value) for value in mapping["call_chain"])
    return text


def _focus_component_label(item, index):
    return "C" + str(index).zfill(2) + " · " + _text(item.get("label") or item.get("component_key"))


def _method_focus_data(case, method):
    """Display a single core-selected rank; never turn missing observations into zeros."""
    selected = method.get("focus_selected_rank", method.get("selected_rank"))
    result = dict(rank=selected, available=False, note="", stages=[], parent_rows=[], step_rows=[], series=[], categories=[])
    captures = [item for item in case.get("measurements") or [] if item.get("rank") == selected]
    if selected is None or len(captures) != 1:
        result["note"] = "所选rank缺少唯一采样；逐step与额外下钻未采集，请重新统计。"
        return result
    residual = captures[0].get("step_residuals") or {}
    focus = residual.get("focus_diagnostics")
    if not isinstance(focus, dict):
        result["note"] = "逐step与额外下钻未采集，请用新版脚本重新统计；原两父阶段耗时与原展开保留，未知值不补0。"
        return result
    if focus.get("rank") != selected or residual.get("rank", selected) != selected:
        result["note"] = "focus_diagnostics的rank与所选rank不一致，停止绘制本区，避免跨rank拼接。"
        return result
    groups = focus.get("stages")
    if not isinstance(groups, list):
        result["note"] = "focus_diagnostics未提供阶段列表，请重新统计。"
        return result
    operators, step_order, per_stage_steps = focus.get("operators") or {}, {}, {}
    for key, fallback in FOCUS_KEYS:
        found = [item for item in groups if isinstance(item, dict) and item.get("stage_key") == key]
        if len(found) != 1:
            result["note"] = "focus_diagnostics缺少唯一的" + fallback + "；停止绘制本区。"
            return result
        source, components, functions = found[0], [], []
        summaries = source.get("summary") or []
        component_values = [item.get("duration_ms_mean") for item in summaries]
        valid = (_number(source.get("duration_ms_mean")) and source["duration_ms_mean"] >= 0
            and all(_number(value) and value >= 0 for value in component_values)
            and (bool(summaries) or source["duration_ms_mean"] == 0)
            and math.isclose(math.fsum(component_values), source["duration_ms_mean"], rel_tol=1e-9, abs_tol=1e-8))
        for index, item in enumerate(summaries, 1):
            label = _focus_component_label(item, index)
            kind = FOCUS_KINDS.get(item.get("kind"), _text(item.get("kind") or "未分类"))
            components.append([label, *[item.get(field) if _number(item.get(field)) else None for field in
                ("duration_ms_mean", "duration_ms_p50", "duration_ms_p90", "duration_ms_max")],
                item.get("share_of_parent") if _number(item.get("share_of_parent")) else None,
                item.get("observed_steps"), kind + "\n" + _text(item.get("mapping_status") or "unknown")])
            details, mappings, basis = [], [], []
            for mapping in item.get("source_function_mapping") or []:
                mappings.append(_focus_function_text(mapping))
                if isinstance(mapping, dict) and mapping.get("evidence"):
                    basis.append(_text(mapping["evidence"]))
            for operator_id in item.get("operator_ids") or []:
                operator = operators.get(operator_id) or {}
                details.append(_text(operator.get("op_type") or operator.get("name") or operator_id)
                    + "\nnative=" + _text(operator.get("name")) + "；stream=" + _text(operator.get("stream_id"))
                    + "\nin=" + _text(operator.get("input_shapes")) + "\nout=" + _text(operator.get("output_shapes"))
                    + "\n原行示例=" + _text(operator.get("example_row_indices")))
                for mapping in operator.get("source_function_mapping") or []:
                    mappings.append(_focus_function_text(mapping))
                    if isinstance(mapping, dict) and mapping.get("evidence"):
                        basis.append(_text(mapping["evidence"]))
            if item.get("kind") == "no_device_record":
                for field, boundary in (("previous_operator_ids", "前界任务"), ("following_operator_ids", "后界任务")):
                    for operator_id in item.get(field) or []:
                        operator = operators.get(operator_id) or {}
                        details.append(boundary + "（仅用于定位间隙）=" + _text(operator.get("op_type") or operator.get("name") or operator_id)
                            + "\nstream=" + _text(operator.get("stream_id")) + "；in=" + _text(operator.get("input_shapes"))
                            + "\n原行示例=" + _text(operator.get("example_row_indices")))
            functions.append([label, "\n\n".join(details) or "没有同时段设备记录；不能据此判断CPU在执行哪个函数。",
                "\n\n".join(dict.fromkeys(mappings)) or "未知；没有可确认的函数入口，不补造归因。",
                "静态对照\n" + _text(item.get("mapping_status") or "unknown") + "\n" + "\n".join(dict.fromkeys(basis))])
        label = source.get("label") or fallback
        result["parent_rows"].append([label, *[source.get(field) if _number(source.get(field)) else None for field in
            ("duration_ms_mean", "duration_ms_p50", "duration_ms_p90", "duration_ms_max")],
            _text(source.get("observed_steps")) + "/" + _text(source.get("rounds")), source.get("status") or "未提供"])
        result["stages"].append(dict(key=key, label=label, rows=components, functions=functions,
            closed=valid, note="细项均值之和闭合父段；出现轮次不足时，未出现的该细项在已识别父段中计0。" if valid else
            "细项未与父段均值闭合或数值缺失，保留原表并停止该细项柱状图；不补0。"))
        steps = {}
        for step in source.get("per_step") or []:
            identity = (step.get("step_index"), _text(step.get("round_id")))
            if identity in steps:
                result["note"] = label + "出现重复的原轮次标识，停止绘制逐step，避免重复计时。"
                return result
            steps[identity] = step
            step_order.setdefault(identity, len(step_order))
        per_stage_steps[key] = steps
    identities = sorted(step_order, key=lambda item: (item[0] if type(item[0]) is int else step_order[item], step_order[item]))
    for identity in identities:
        pair = [per_stage_steps[key].get(identity) or {} for key, _ in FOCUS_KEYS]
        def value(step, field):
            item = step.get(field)
            return item if step.get("status") == "MATCHED" and _number(item) and item >= 0 else None
        states = []
        for (key, label), step in zip(FOCUS_KEYS, pair):
            states.append(label + "：" + _text(step.get("status") or "缺失")
                + ("\n原因：" + _text(step["reason"]) if step.get("reason") else "")
                + ("\n边界ns=" + _text(step.get("start_ns")) + " → " + _text(step.get("end_ns")) if step else ""))
        result["step_rows"].append([identity[0], identity[1], *[value(step, "duration_ms") for step in pair],
            value(pair[0], "recorded_busy_ms"), value(pair[0], "no_recorded_ms"),
            value(pair[1], "recorded_busy_ms"), value(pair[1], "no_recorded_ms"), "\n".join(states)])
    result.update(available=True, categories=[row[1] for row in result["step_rows"]],
        series=[(label, [row[2 + index] for row in result["step_rows"]], color)
            for index, ((_, label), color) in enumerate(zip(FOCUS_KEYS, ("168B8D", "D4793A")))],
        note="rank=" + _text(selected) + "；" + _text(focus.get("scope") or "本方法selected_rank（最大完整周期均值rank）")
            + "；完整细分区间=" + _text(focus.get("segments_csv")) + "；逐轮CSV=" + _text(focus.get("steps_csv")))
    return result


def _method_focus_section(sheet, row, case, method):
    data = _method_focus_data(case, method)
    row = _note(sheet, row, "两项U重点诊断 · 固定rank逐轮变化与函数入口对照")
    row = _note(sheet, row, data["note"], warning=not data["available"])
    if not data["available"]:
        return row
    row = _note(sheet, row, FOCUS_SCOPE)
    row, header, written = _operator_table(sheet, row, FOCUS_PARENT_HEADERS, data["parent_rows"], FOCUS_PARENT_SPANS)
    if all(_number(record[1]) and record[1] >= 0 for record in data["parent_rows"]):
        row = _operator_bar(sheet, row, "两项U父阶段均值 · rank=" + _text(data["rank"]),
            FOCUS_PARENT_HEADERS, data["parent_rows"], FOCUS_PARENT_SPANS, header, written, (1,),
            "源实测ms/HC轮次；已含于U，不额外相加")
    row = _note(sheet, row, "同一rank不同原HC轮次的阶段耗时；P50/P90/最大为核心原值，缺失轮次不画0")
    # Reserve the overview before the potentially long per-round table. Range
    # references are bound after the table has its final physical row numbers.
    line_row = row
    if data["step_rows"]:
        row += 24
    row, _, written = _operator_table(sheet, row, FOCUS_STEP_HEADERS, data["step_rows"], FOCUS_STEP_SPANS)
    if written and len(written) == len(data["step_rows"]):
        reference = "'" + sheet.name.replace("'", "''") + "'!"
        series = [dict(name=label, values=values, color=color,
            values_ref=reference + cell_reference(written[0], 4 + index, True) + ":" + cell_reference(written[-1], 4 + index, True))
            for index, (label, values, color) in enumerate(data["series"])]
        sheet.add_line_chart("两项U逐轮耗时变化 · 固定rank=" + _text(data["rank"]),
            categories=data["categories"], series=series, anchor=(line_row, 1, line_row + 22, 12),
            categories_ref=reference + cell_reference(written[0], 2, True) + ":" + cell_reference(written[-1], 2, True),
            x_axis_title="原HC轮次ID（等距；非ProfilerStep）", y_axis_title="源实测阶段跨度(ms)")
    elif data["step_rows"]:
        row = _note(sheet, row, "逐轮表出现长文本续行，保留完整明细，折线未绘制。", warning=True)
    else:
        row = _note(sheet, row, "逐step未采集，请重新统计；均值细分不代替逐轮数据。", warning=True)
    for stage in data["stages"]:
        row = _note(sheet, row, stage["label"] + " · 设备互斥细项（源实测）")
        row = _note(sheet, row, stage["note"], warning=not stage["closed"])
        row, header, written = _operator_table(sheet, row, FOCUS_COMPONENT_HEADERS, stage["rows"], FOCUS_COMPONENT_SPANS,
            percent_columns=(5,))
        if stage["closed"] and stage["rows"]:
            row = _operator_bar(sheet, row, stage["label"] + " · 互斥细项均值 · rank=" + _text(data["rank"]),
                FOCUS_COMPONENT_HEADERS, stage["rows"], FOCUS_COMPONENT_SPANS, header, written, (1,), "源实测ms/HC轮次；细项之和等于父段")
        row = _note(sheet, row, stage["label"] + " · 算子定位与源码函数入口（静态对照，不是函数实测耗时）")
        row, _, _ = _operator_table(sheet, row, FOCUS_FUNCTION_HEADERS, stage["functions"], FOCUS_FUNCTION_SPANS)
    return row + 1


def _residual_stage_rows(stages, window_summary=None):
    fields = ("duration_ms_mean", "recorded_busy_ms_mean", "no_recorded_ms_mean", "host_span_union_ms_mean")
    rows = []
    summary = window_summary or {}
    for item in stages:
        # An unresolved subset is not a whole-window mean. Only use the window
        # summary when this one row accounts for every retained window in full.
        whole_unknown = _whole_unresolved_window(stages, summary)
        source = summary if whole_unknown else item if item.get("status") == "MATCHED" else {}
        rows.append(["阶段未识别（原窗口实测总耗时）" if whole_unknown else item.get("label", item.get("stage_key")),
                     *[source.get(field) if source.get(field) is not None else NA for field in fields],
                     str(item.get("matched_windows", NA)) + "/" + str(item.get("rounds", NA)) + " · "
                     + str(item.get("status", "UNRESOLVED"))])
    return rows


def _residual_stage_basis_rows(stages):
    return [[item.get("label", item.get("stage_key")), (item.get("boundary_basis") or
             "核心未提供边界依据。") + (("；缺锚原因：" + _text(item["reason"])) if item.get("reason") else "")]
            for item in stages]


def _residual_source_rows(stages):
    fields = ("source_cpu_union_ms_mean", "source_queue_wait_union_ms_mean", "source_submit_to_device_ms_mean")
    rows = []
    for item in stages:
        status = item.get("source_status") or "UNAVAILABLE"
        for field, label in (("source_graph_tasks_total", "图共享"), ("source_ambiguous_tasks_total", "关联歧义")):
            if item.get(field):
                status += "\n" + label + "=" + _text(item[field])
        rows.append([item.get("label", item.get("stage_key")),
                     "/".join(_text(item[field]) if item.get(field) is not None else NA
                              for field in ("source_cpu_tasks_total", "source_device_tasks_total")),
                     *[item[field] if item.get(field) is not None else NA for field in fields], status])
    return rows


def _residual_dma_rows(stages):
    fields = ("trace_dma_union_ms_mean", "trace_dma_in_csv_uncovered_ms_mean",
              "uncovered_after_trace_dma_ms_mean")
    return [[item.get("label", item.get("stage_key")),
             *[_diagnostic_value(item, field) for field in fields]] for item in stages]


def _has_dma_fields(stages):
    return any(any(key.startswith("trace_dma_") or key.startswith("uncovered_after_trace_dma_")
                   for key in item) for item in stages)


def _residual_substage_groups(diagnostic, window_key):
    """Keep children with their own parent; never merge independent U trees."""
    parents = {item.get("stage_key"): item.get("label") or item.get("stage_key")
               for item in diagnostic.get("stage_summary") or []
               if item.get("window_key") == window_key}
    groups = collections.defaultdict(list)
    for item in _residual_stages(diagnostic, window_key, "substage_summary"):
        groups[item.get("parent_stage_key")].append(item)
    return [(key, parents.get(key) or key or "未提供父阶段", children)
            for key, children in groups.items()]


def _has_source_fields(stages):
    return any(any(key.startswith("source_") for key in item) for item in stages)


def _residual_stage_tables(sheet, row, diagnostic):
    row = _note(sheet, row, RESIDUAL_STAGE_SCOPE)
    row = _note(sheet, row, RESIDUAL_SOURCE_SCOPE)
    summary = {item.get("window_key"): item for item in diagnostic.get("summary") or []}
    for key, label in RESIDUAL_WINDOWS:
        row = _note(sheet, row, label + " · 有序阶段（全部保留轮次均值）")
        stages = _residual_stages(diagnostic, key)
        if stages:
            row = _table(sheet, row, RESIDUAL_STAGE_HEADERS, _residual_stage_rows(stages, summary.get(key)),
                         spans=RESIDUAL_STAGE_SPANS)
            if _has_dma_fields(stages):
                row = _note(sheet, row, label + " · DMA对CSV未记录区间的补充覆盖；" + RESIDUAL_DMA_SCOPE)
                row = _table(sheet, row, RESIDUAL_DMA_HEADERS, _residual_dma_rows(stages),
                             spans=RESIDUAL_DMA_SPANS)
            row = _note(sheet, row, label + " · 关联Host提交来源（独立于同时段Host覆盖）")
            source_rows, written_rows = _residual_source_rows(stages), []
            row = _table(sheet, row, RESIDUAL_SOURCE_HEADERS, source_rows,
                         spans=RESIDUAL_SOURCE_SPANS, record_rows=written_rows)
            for written, values in zip(written_rows, source_rows):
                sheet.row_height(written, max(35, 16 * (values[-1].count("\n") + 1) + 10))
            row = _table(sheet, row, ["阶段", "边界依据 / 缺锚原因"], _residual_stage_basis_rows(stages),
                         spans=((1, 6), (7, 12)))
        else:
            row = _note(sheet, row, "UNRESOLVED：核心未提供本窗口的阶段锚点统计。", warning=True)
        for parent_key, parent_label, substages in _residual_substage_groups(diagnostic, key):
            row = _note(sheet, row, label + " · 父阶段细分：" + _text(parent_label) +
                        "（" + _text(parent_key) + "）；已包含在父阶段中，以下行不能再次加到窗口或U。")
            row = _table(sheet, row, RESIDUAL_STAGE_HEADERS, _residual_stage_rows(substages),
                         spans=RESIDUAL_STAGE_SPANS)
            if _has_dma_fields(substages):
                row = _note(sheet, row, "本组子阶段的DMA覆盖；" + RESIDUAL_DMA_SCOPE)
                row = _table(sheet, row, RESIDUAL_DMA_HEADERS, _residual_dma_rows(substages),
                             spans=RESIDUAL_DMA_SPANS)
            if _has_source_fields(substages):
                row = _note(sheet, row, "本组子阶段的关联Host提交来源；与父阶段重复覆盖，不额外相加。")
                row = _table(sheet, row, RESIDUAL_SOURCE_HEADERS, _residual_source_rows(substages),
                             spans=RESIDUAL_SOURCE_SPANS)
            row = _table(sheet, row, ["子阶段", "父阶段（已含） / 边界依据"],
                         [[item.get("label", item.get("stage_key")),
                           "父阶段=" + _text(item.get("parent_stage_key")) + "；" + _text(item.get("boundary_basis"))]
                          for item in substages], spans=((1, 6), (7, 12)))
    return row


SWA_LAYER_HEADERS = ["主/草稿", "层号", "attention", "参考可用", "轮次", "层跨度均值(ms)",
                     "层p50(ms)", "层p90(ms)", "设备任务并集(ms)", "无设备记录(ms)",
                     "kernel累加(ms)", "duration+wait累加(ms)"]
SWA_STAGE_HEADERS = ["草稿层号", "阶段标识", "SWA内部阶段（已含）", "匹配轮次 / 状态",
                     "跨度均值(ms)", "p50(ms)", "p90(ms)", "设备任务并集(ms)",
                     "无设备记录(ms)", "kernel累加(ms)", "duration+wait累加(ms)"]
SWA_DIAGNOSTIC_SCOPE = ("同rank、同保留HC周期的逐层对照；层跨度为首HCPre开始→末HCPost结束。"
                        "主0保留展示，是否纳入参考由核心给定；SWA内部阶段只展示draft/SWA。"
                        "所有细分已含在原层跨度中，不加到层链、U、完整周期或TPOT。"
                        "kernel累加可因多流重叠大于跨度；duration+wait为诊断累加，不能当互斥占用或额外时延。"
                        "无设备记录只表示CSV任务未覆盖；未知值显示—，不补0。")


def _diagnostic_value(item, field):
    value = item.get(field)
    return value if _number(value) else NA


def _swa_layer_rows(diagnostic):
    fields = ("wall_ms_mean", "wall_ms_p50", "wall_ms_p90", "recorded_busy_ms_mean",
              "no_recorded_ms_mean", "kernel_sum_ms_mean", "total_cost_ms_mean")
    return [[item.get("role"), item.get("layer"), item.get("attention"),
             item.get("reference_eligible") if item.get("reference_eligible") is not None else NA,
             item.get("rounds"), *[_diagnostic_value(item, field) for field in fields]]
            for item in diagnostic.get("layer_summary") or []]


def _swa_stage_rows(diagnostic):
    fields = ("duration_ms_mean", "duration_ms_p50", "duration_ms_p90", "recorded_busy_ms_mean",
              "no_recorded_ms_mean", "kernel_sum_ms_mean", "total_cost_ms_mean")
    rows = []
    for item in diagnostic.get("stage_summary") or []:
        if item.get("role") != "draft" or item.get("attention") != "SWA":
            continue
        complete = (item.get("status") == "MATCHED" and _number(item.get("rounds"))
                    and item["rounds"] > 0 and item.get("matched_windows") == item["rounds"])
        source = item if complete else {}
        rows.append([item.get("layer"), item.get("stage_key"), item.get("label", item.get("stage_key")),
                     str(item.get("matched_windows", NA)) + "/" + str(item.get("rounds", NA)) +
                     " · " + str(item.get("status", "UNRESOLVED")),
                     *[_diagnostic_value(source, field) for field in fields]])
    return rows


def _swa_tables(sheet, row, diagnostic, rank):
    row = _note(sheet, row, "源rank=" + _text(rank) + " · 逐层对照与SWA内部细分；状态=" +
                _text(diagnostic.get("status")))
    row = _note(sheet, row, SWA_DIAGNOSTIC_SCOPE + "核心scope=" + _text(diagnostic.get("scope")))
    if diagnostic.get("reason"):
        row = _note(sheet, row, "SWA诊断未完成原因：" + _text(diagnostic["reason"]), warning=True)
    row = _table(sheet, row, SWA_LAYER_HEADERS, _swa_layer_rows(diagnostic))
    row = _table(sheet, row, SWA_STAGE_HEADERS, _swa_stage_rows(diagnostic))
    return _note(sheet, row, "逐轮SWA阶段锚点和形状证据：" + _text(diagnostic.get("stages_csv")))


def _ledger_sheet(book, report):
    sheet = _sheet(book, SHEETS[3], "源周期 · 层链与轮次剩余区间",
                   "仅展示核心逐rank均值分区；层链含内部衔接，主草稿切换与轮次尾部保留一次。")
    row = 5
    fields = ("source_cycle_ms", "source_main_chain_ms", "source_draft_chain_ms",
              "cross_role_tail_residual_ms", "partition_total_ms", "closed", "scope")
    for case in report["cases"]:
        row = _note(sheet, row, case["name"] + "；" + LAYER_RESIDUAL_NOTE)
        for identity in ("method1", "method2"):
            method = _method(case, identity)
            row = _note(sheet, row, _layer_mean_name(method))
            records = (dict(rank=rank.get("rank"), **(rank.get("source_partition") or {}))
                       for rank in method.get("per_rank") or [])
            row = _records(sheet, row, records, ("rank", *fields), detail=True,
                           location=f"case={case['name']}; projection.{METHOD_SET}[id={identity}].per_rank[].source_partition")
        for capture in case.get("measurements") or []:
            diagnostic = capture.get("step_residuals")
            if diagnostic is None:
                continue
            trace = diagnostic.get("trace") or {}
            row = _note(sheet, row, "源rank=" + _text(capture.get("rank")) + " · HC周期剩余区间实测；状态="
                        + _text(diagnostic.get("status")) + "；host trace=" + _text(trace.get("status")))
            row = _table(sheet, row, RESIDUAL_SUMMARY_HEADERS, _residual_summary_rows(diagnostic),
                         spans=((1, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 9), (10, 12)))
            row = _table(sheet, row, ["每HC周期轮次剩余区间U合计(ms)", "核心原值"],
                         [[LAYER_RESIDUAL_NOTE, diagnostic.get("residual_ms_mean")]], spans=((1, 8), (9, 12)))
            row = _residual_stage_tables(sheet, row, diagnostic)
            row = _note(sheet, row, "源HC周期诊断，不等同完整请求step。Host跨度可包含等待，与设备任务重叠，不能额外加到U；"
                        "未记录设备任务不自动归因CPU或空闲。逐轮阶段锚点见：" + _text(diagnostic.get("stages_csv")) +
                        "；原始任务/形状与Host事件见明细CSV。")
            row = _note(sheet, row, "逐任务Host来源链：" + _text(diagnostic.get("source_links_csv") or "核心未提供来源链CSV") +
                        "；CPU提交跨度、CANN API并集及完整关联计数见阶段汇总step_residual_stages.csv。")
    return sheet


def _method_sheet(book, report, identity, sheet_name):
    sheet = _sheet(book, sheet_name, _layer_mean_name({"id": identity}) + " · 采样均值与轮次估算",
                   "源实测与目标估算 · 主/草稿独立列；层内已有成本只追加增量。")
    sheet.result_rows, sheet.result_cells = {}, {}
    row = _bs_method_section(sheet, 5, report, identity)
    row = _s_method_section(sheet, row, report, identity)
    for case_index, case in enumerate(report["cases"]):
        row = _layer_mean_method_section(sheet, row, case, case_index, identity, _method(case, identity))
    return sheet


def _reference_rows(report, reference):
    archived = report.get("communication_reference") or {}
    if isinstance(archived, dict) and isinstance(archived.get("rows"), list):
        return archived["rows"]
    if reference is not None:
        with Path(reference).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    return []


def _reference_sheet(book, report, reference, unit):
    sheet = _sheet(book, SHEETS[6], "通信参考 · 原始表",
                   "原表仅核对来源；优先核心归档数据，外部CSV只补缺。外部参考单位标签：" + unit)
    _records(sheet, 5, _reference_rows(report, reference), detail=True,
             location="communication_reference.rows[]")
    return sheet


def _layers_sheet(book, report):
    sheet = _sheet(book, SHEETS[2], "层与组件 · 排除主0的分类参考均值", "按attention分类；所有数值来自本方法逐rank参考均值，不使用旧Hash分组或构造完整Router时长。")
    row = 5
    for case in report["cases"]:
        row = _note(sheet, row, case["name"] + "：参考层号和样本数由核心给定；主/草稿独立统计。", columns=12)
        for identity in ("method1", "method2"):
            method = _method(case, identity)
            rows = [[case["name"], _layer_mean_name(method), rank.get("rank"), item.get("role"), item.get("attention"),
                     item.get("reference_layers"), item.get("samples"), item.get("wall_ms"), item.get("dispatch_ms"),
                     item.get("combine_ms"), item.get("routed_compute_ms"), rank.get("rank") == method.get("selected_rank")]
                    for rank in method.get("per_rank") or [] for item in rank.get("class_means") or []]
            row = _table(sheet, row, ["采样组合", "方法", "rank", "主/草稿", "attention", "参考层号（主0排除）",
                         "参考样本数", "源单层(ms)", "源Dispatch(ms)", "源Combine(ms)", "源路由GMM(ms)", "核心选中rank"], rows)
        for capture in case.get("measurements") or []:
            diagnostic = capture.get("swa_diagnostics")
            if isinstance(diagnostic, dict):
                row = _swa_tables(sheet, row, diagnostic, capture.get("rank"))
    return sheet


def _formula_sheet(book, report):
    sheet = _sheet(book, SHEETS[7], "采样均值 · 公式与条件", "本次条件估算的公式、采样参数与依据；不重新计算时延、取均值或选择rank。")
    row = _table(sheet, 5, ["项目", "口径"], [
        ["层类别参考", LAYER_MEAN_SCOPE], ["完整周期", "主模型目标链＋草稿目标链＋轮次剩余区间；D/C/GMM增量已在层内，仅保留一次"],
        ["单卡指标", "使用核心per_device_tokens_s；按目标物理卡数均摊，不把全局产出当单卡吞吐"],
        ["层间衔接区间", LAYER_GAP_NOTE], ["轮次剩余区间", LAYER_RESIDUAL_NOTE]])
    for case in report["cases"]:
        row = _note(sheet, row, case["name"] + " · " + _case_scope(case), columns=12)
        config = case.get("config") or {}
        row = _table(sheet, row, ["配置", "核心原值"], [[key, config.get(key)] for key in (
            "bs", "draft_tokens", "effective_tokens_per_round", "base_ep", "target_ep", "source_dp", "target_dp",
            "source_devices", "target_devices", "ranks_per_node") if key in config])
        for identity in ("method1", "method2"):
            method = _method(case, identity)
            row = _table(sheet, row, ["方法", "状态", "核心选中rank", "公式（仅显示）"],
                         [[_layer_mean_name(method), _layer_mean_status(method), method.get("selected_rank"), method.get("formula")]])
            communication = method.get("comm_model") or method.get("communication") or {}
            if "extra_ms_per_call" in communication:
                extra = communication["extra_ms_per_call"]
                note = ("方法一每次Dispatch/Combine附加14μs；已计入核心增量"
                        if identity == "method1" and extra == .014 else
                        "方法二按历史通信倍率换算，无额外14μs"
                        if identity == "method2" and extra == 0 else "核心给定的每次附加通信成本")
                row = _table(sheet, row, ["通信参数", "核心原值(ms/次)", "说明"],
                             [["extra_ms_per_call", extra, note]])
            for field in ("assumptions", "warnings", "reasons"):
                row = _table(sheet, row, ["本次条件/限制", "核心原文"], [[field, value] for value in method.get(field) or []])
    return sheet


def _review_sheet(book, report, origin):
    sheet = _sheet(book, SHEETS[8], "层均值结果 · 来源与复核", "本次估算的状态、缺输入原因与采样来源。")
    row = _table(sheet, 5, ["来源", "原值"], [["核心JSON", str(origin)], ["renderer_version", RENDERER_VERSION],
                    ["profile_version", report.get("profile_version")], ["generated_at", report.get("generated_at")]])
    for case in report["cases"]:
        row = _note(sheet, row, case["name"], columns=12)
        for identity in ("method1", "method2"):
            method = _method(case, identity)
            row = _table(sheet, row, ["方法", "状态", "核心选中rank", "汇总方式"],
                         [[_layer_mean_name(method), _layer_mean_status(method), method.get("selected_rank"), "各rank先取轮次均值，再选最大周期均值"]])
            row = _table(sheet, row, ["本次原因/限制", "核心原文"],
                         [[field, value] for field in ("reasons", "warnings") for value in method.get(field) or []])
        row = _table(sheet, row, ["采样rank", "CSV来源", "SHA256"],
                     [[item.get("rank"), item.get("path"), item.get("sha256")] for item in case.get("captures") or []])
    if report.get("failures"):
        _records(sheet, row, report["failures"])
    return sheet


def _html_table(headers, rows):
    esc = lambda value: html.escape(_text(value), quote=True)
    return '<table><thead><tr>' + ''.join('<th>' + esc(value) + '</th>' for value in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(value) + '</td>' for value in row) + '</tr>' for row in rows) + '</tbody></table>'


def _method_residual_html(case, method):
    esc = lambda value: html.escape(_text(value), quote=True)
    rows, items, note = _method_residual_data(case, method)
    parts = ['<h4>轮次剩余区间耗时构成 · ' + esc(_rank_note(method)) + '</h4><p>' + esc(note) + '</p>']
    if rows:
        parts.append('<div class="method-residual">' + _html_table(METHOD_RESIDUAL_HEADERS, rows) + '</div>')
    if items is not None:
        maximum = max(item["value_ms"] for item in items)
        unknown_count = sum(bool(item.get("unresolved_window")) for item in items)
        chart_label = ("轮次剩余区间" + str(len(items)) + "项耗时构成，含" + str(unknown_count) + "个未识别整窗，单位毫秒"
                       if unknown_count else "轮次剩余区间15阶段耗时构成，单位毫秒")
        parts.append('<div class="u-bars" role="img" aria-label="' + esc(chart_label) + '">')
        for record, item in zip(rows, items):
            parts.append('<div class="u-bar"><span>' + esc(record[0] + ' · ' + item["label"]) +
                         '</span><span class="u-track"><i style="width:' + str(100 * item["value_ms"] / maximum) +
                         '%"></i></span><span>' + format(item["value_ms"], '.6f') + ' ms</span></div>')
        parts.append('</div>')
    return ''.join(parts)


def _method_focus_html(case, method):
    esc = lambda value: html.escape(_text(value), quote=True)
    data = _method_focus_data(case, method)
    parts = ['<section class="u-focus"><h4>两项U重点诊断 · 固定rank逐轮变化与函数入口对照</h4><p>'
             + esc(data["note"]) + '</p>']
    if not data["available"]:
        return ''.join(parts) + '</section>'
    parts.append('<p>' + esc(FOCUS_SCOPE) + '</p>' + _html_table(FOCUS_PARENT_HEADERS, data["parent_rows"]))
    if all(_number(record[1]) and record[1] >= 0 for record in data["parent_rows"]):
        parts.append(_operator_svg('两项U父阶段均值 · rank=' + _text(data["rank"]),
            [record[0] for record in data["parent_rows"]],
            [('源实测均值', [record[1] for record in data["parent_rows"]], '168B8D')], 'ms/HC轮次；已含于U'))
    if data["step_rows"]:
        parts.append(_bs_svg('两项U逐轮耗时变化 · 固定rank=' + _text(data["rank"]), data["categories"], data["series"],
            '源实测阶段跨度(ms)', x_axis_title='原HC轮次ID（等距；非ProfilerStep）', category_name='原HC轮次ID',
            description='一个固定rank的原保留HC轮次；缺少或未识别结果时断线，不以0替代。', max_labels=14))
        parts.append('<details><summary>展开全部原HC轮次、设备记录与无记录区间</summary>'
            + _html_table(FOCUS_STEP_HEADERS, [['' if value is None else value for value in row] for row in data["step_rows"]])
            + '</details>')
    else:
        parts.append('<p>逐step未采集，请重新统计；均值细分不代替逐轮数据。</p>')
    for stage in data["stages"]:
        parts.append('<h5>' + esc(stage["label"] + ' · 设备互斥细项（源实测）') + '</h5><p>' + esc(stage["note"]) + '</p>')
        records = [[format(value, '.3%') if index == 5 and _number(value) else value for index, value in enumerate(row)]
            for row in stage["rows"]]
        parts.append(_html_table(FOCUS_COMPONENT_HEADERS, records))
        if stage["closed"] and stage["rows"]:
            parts.append(_operator_svg(stage["label"] + ' · 互斥细项均值 · rank=' + _text(data["rank"]),
                [record[0] for record in stage["rows"]], [('源实测均值', [record[1] for record in stage["rows"]], '168B8D')],
                'ms/HC轮次；细项之和等于父段，已经计入完整周期'))
        parts.append('<h5>' + esc(stage["label"] + ' · 算子定位与源码函数入口（静态对照）') + '</h5>'
            + _html_table(FOCUS_FUNCTION_HEADERS, stage["functions"]))
    return ''.join(parts) + '</section>'


def _operator_svg_label_lines(value, width=68):
    """Fit mixed CJK/ASCII labels into the fixed SVG label column."""
    lines, line, used = [], "", 0
    for char in _text(value):
        if char == "\n":
            lines.append(line)
            line, used = "", 0
            continue
        cells = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if used + cells > width:
            lines.append(line)
            line, used = "", 0
        line += char
        used += cells
    return [*lines, line] if line or not lines else lines


def _operator_svg(title, labels, series, axis):
    """Draw signed values around a real zero axis, never a negative CSS width."""
    esc = lambda value: html.escape(_text(value), quote=True)
    values = [value for _, entries, _ in series for value in entries]
    low, high = min([0., *values]), max([0., *values])
    if low == high:
        high = low + 1
    left, right, top = 490, 1000, 46
    layout, bottom = [], top
    for label in labels:
        lines = _operator_svg_label_lines(label)
        content_height = max(len(lines) * 14, len(series) * 15)
        layout.append((bottom, content_height, lines))
        bottom += content_height + 14
    legends = [(name, _operator_svg_label_lines(name, width=166), color) for name, _, color in series]
    legend_height = sum(len(lines) * 14 + 8 for _, lines, _ in legends)
    axis_lines = _operator_svg_label_lines(axis, width=175)
    chart_height = bottom + 40 + legend_height + 14 * len(axis_lines) + 14
    x = lambda value: left + (value - low) / (high - low) * (right - left)
    parts = ['<div class="operator-chart"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 '
        + str(chart_height) + '" role="img"><title>' + esc(title) + '</title>',
        '<text x="12" y="23" font-weight="bold">' + esc(title) + '</text>',
        f'<line x1="{x(0):.3f}" y1="{top - 10}" x2="{x(0):.3f}" y2="{bottom}" stroke="#60778c"/>']
    for index, label in enumerate(labels):
        position, content_height, lines = layout[index]
        parts.append('<text x="12" y="' + str(position + 12) + '"><title>' + esc(label) + '</title>'
            + ''.join('<tspan x="12" dy="' + str(0 if index == 0 else 14) + '">' + esc(line) + '</tspan>'
                      for index, line in enumerate(lines)) + '</text>')
        for series_index, (name, entries, color) in enumerate(series):
            value, height = entries[index], 12
            y = position + (content_height - len(series) * 15) / 2 + series_index * 15
            parts.append(f'<rect x="{min(x(0), x(value)):.3f}" y="{y}" width="{abs(x(value) - x(0)):.3f}" '
                + f'height="{height}" fill="#{color}"><title>' + esc(label + ' · ' + name + ' = ' + format(value, '.6f') + ' ms') + '</title></rect>')
            if value:
                anchor = "end" if value < 0 else "start"
                offset = -5 if value < 0 else 5
                parts.append(f'<text x="{x(value) + offset:.3f}" y="{y + 11}" text-anchor="{anchor}">'
                             + esc(format(value, '.6g')) + '</text>')
    parts.append(f'<text x="{left}" y="{bottom + 20}">{low:.6g}</text>'
                 + f'<text x="{right}" y="{bottom + 20}" text-anchor="end">{high:.6g} ms</text>')
    legend_y = bottom + 38
    for _, lines, color in legends:
        parts.append('<rect x="12" y="' + str(legend_y - 10) + '" width="12" height="12" fill="#' + color + '"/>')
        parts.append('<text x="34" y="' + str(legend_y) + '">'
            + ''.join('<tspan x="34" dy="' + str(0 if index == 0 else 14) + '">' + esc(line) + '</tspan>'
                      for index, line in enumerate(lines)) + '</text>')
        legend_y += len(lines) * 14 + 8
    parts.append('<text x="12" y="' + str(legend_y + 6) + '">'
        + ''.join('<tspan x="12" dy="' + str(0 if index == 0 else 14) + '">' + esc(line) + '</tspan>'
                  for index, line in enumerate(axis_lines)) + '</text></svg></div>')
    return ''.join(parts)


def _method_operator_html(method, role, attention):
    esc = lambda value: html.escape(_text(value), quote=True)
    data = _method_operator_data(method, role, attention)
    parts = ['<section class="operator-drilldown"><h4>' + esc(data["title"]) + '</h4><p>'
             + esc(data.get("scope", "") + data["note"]) + '</p>']
    if not data["available"]:
        return ''.join(parts) + '</section>'
    parts.append('<p>' + esc(OPERATOR_SCOPE) + '</p>')
    displayed = [[format(value, '.3%') if index == 4 and _number(value) else value
                  for index, value in enumerate(row)] for row in [*data["target_rows"], data["total_row"]]]
    parts.append('<div class="operator-ledger">' + _html_table(OPERATOR_LEDGER_HEADERS, displayed) + '</div>')
    parts.append(_operator_svg(attention + '算子互斥覆盖与模型增减', [row[0] for row in data["target_rows"]],
        [(OPERATOR_LEDGER_HEADERS[column], [row[column] for row in data["target_rows"]], color)
         for column, color in ((1, "168B8D"), (2, "D4793A"))], '目标同类全部层，ms；模型修正不是已验证收益'))
    parts.append('<h5>' + esc(attention + '源实测算子热点；段内相交调用/单层均值，单次均值/P90') + '</h5>')
    parts.append('<div class="operator-hotspots">' + _html_table(OPERATOR_KERNEL_HEADERS, data["kernel_rows"]) + '</div>')
    if data["kernel_rows"]:
        parts.append(_operator_svg(attention + '源实测算子热点', [row[0] for row in data["kernel_rows"]],
            [(OPERATOR_KERNEL_HEADERS[2], [row[2] for row in data["kernel_rows"]], "168B8D")],
            '源段内kernel累计，可重叠；不是目标贡献或critical time'))
    pmu_rows, pmu_details = _operator_pmu_data(data["kernel_items"])
    parts.append('<h5>' + esc(attention + '源实测Top8 · 流水活动与bound诊断') + '</h5><p>' + esc(PMU_SCOPE) + '</p>')
    if not any((kernel.get("pmu") or {}).get("metrics") for kernel in data["kernel_items"]):
        parts.append('<p>PMU未采集，请重新统计；原热点排序与耗时保留，未知指标不补0。</p>')
    parts.append('<div class="pmu-metrics">' + _html_table(PMU_HEADERS, pmu_rows) + '</div>')
    parts.append('<details><summary>展开完整指纹、高级PMU指标及bound依据</summary>'
                 + _html_table(PMU_DETAIL_HEADERS, pmu_details) + '</details>')
    parts.append('<details><summary>本页源热点的完整指纹与原行/轮/层示例（全部明细见核心JSON）</summary><pre>'
        + esc(json.dumps({"core_basis_notes": data.get("basis_notes"), "target_preview": data["target_items"], "kernel_hotspots": data["kernel_items"]},
                         ensure_ascii=False, indent=2)) + '</pre></details></section>')
    return ''.join(parts)


def _method_stage_expansions_html(case, method):
    esc = lambda value: html.escape(_text(value), quote=True)
    parts = []
    for group in _method_expansion_data(case, method):
        display_rows = [[format(value, '.3%') if column in group.get("percent_columns", ()) and _number(value) else value
                         for column, value in enumerate(record)] for record in group["rows"]]
        parts.append('<section class="method-stage-expansion"><h4>' + esc(group["title"]) +
                     ' · ' + esc(_rank_note(method)) + '</h4><p>' + esc(group.get("scope", METHOD_STAGE_SCOPE) + group["note"]) +
                     '</p>' + _html_table(group["headers"], display_rows))
        if group["series"]:
            maximum = max(value for _, values in group["series"] for value in values) or 1
            parts.append('<div class="u-bars" role="img" aria-label="' + esc(group["title"] + '；' + group.get("axis_title", '源实测阶段耗时，单位毫秒')) + '">')
            for index, (column, values) in enumerate(group["series"]):
                color = ("168B8D", "4F90C6", "D59440", "7668AA", "5BA28A")[index % 5]
                for record, value in zip(group["rows"], values):
                    parts.append('<div class="u-bar"><span>' + esc(record[0] + ' · ' + group["headers"][column]) +
                        '</span><span class="u-track"><i style="background:#' + color + ';width:' +
                        str(100 * value / maximum) + '%"></i></span><span>' + format(value, '.6f') + ' ms'
                        + (' · ' + format(record[4], '.3%') if group.get("percent_columns") and _number(record[4]) else '')
                        + '</span></div>')
            parts.append('</div>')
        parts.append('</section>')
        if group.get("operator_group"):
            parts.append(_method_operator_html(method, *group["operator_group"]))
    return ''.join(parts)


def _residual_html(diagnostic, rank):
    esc = lambda value: html.escape(_text(value), quote=True)
    trace = diagnostic.get("trace") or {}
    summary = {item.get("window_key"): item for item in diagnostic.get("summary") or []}
    parts = ['<section><h3>HC周期剩余区间 · rank=' + esc(rank) + '</h3><p>状态：' +
             esc(diagnostic.get("status")) + '；Host trace：' + esc(trace.get("status")) + '</p>',
             _html_table(RESIDUAL_SUMMARY_HEADERS, _residual_summary_rows(diagnostic)),
             '<p>每HC周期U合计（ms，核心原值）：' + esc(diagnostic.get("residual_ms_mean")) + '</p>',
             '<p>' + esc(RESIDUAL_STAGE_SCOPE) + '</p><p>' + esc(RESIDUAL_SOURCE_SCOPE) + '</p>']
    for key, label in RESIDUAL_WINDOWS:
        parts.append('<h4>' + esc(label) + ' · 有序阶段（全部保留轮次均值）</h4>')
        stages = _residual_stages(diagnostic, key)
        if stages:
            parts.append('<div class="stages">' + _html_table(RESIDUAL_STAGE_HEADERS, _residual_stage_rows(stages, summary.get(key))) + '</div>')
            if _has_dma_fields(stages):
                parts.append('<p>' + esc(label) + ' · DMA对CSV未记录区间的补充覆盖；' + esc(RESIDUAL_DMA_SCOPE) + '</p>')
                parts.append(_html_table(RESIDUAL_DMA_HEADERS, _residual_dma_rows(stages)))
            parts.append('<p>' + esc(label) + ' · 关联Host提交来源（独立于同时段Host覆盖）</p>')
            parts.append('<div class="stages sources">' + _html_table(RESIDUAL_SOURCE_HEADERS, _residual_source_rows(stages)) + '</div>')
            parts.append(_html_table(["阶段", "边界依据 / 缺锚原因"], _residual_stage_basis_rows(stages)))
        else:
            parts.append('<p>UNRESOLVED：核心未提供本窗口的阶段锚点统计。</p>')
        for parent_key, parent_label, substages in _residual_substage_groups(diagnostic, key):
            parts.append('<h4>' + esc(label) + ' · 父阶段细分：' + esc(parent_label) +
                         '（' + esc(parent_key) + '）</h4><p>已包含在父阶段中，以下行不能再次加到窗口或U。</p>')
            parts.append('<div class="stages">' + _html_table(RESIDUAL_STAGE_HEADERS, _residual_stage_rows(substages)) + '</div>')
            if _has_dma_fields(substages):
                parts.append('<p>本组子阶段的DMA覆盖；' + esc(RESIDUAL_DMA_SCOPE) + '</p>')
                parts.append(_html_table(RESIDUAL_DMA_HEADERS, _residual_dma_rows(substages)))
            if _has_source_fields(substages):
                parts.append('<p>本组子阶段的关联Host提交来源；与父阶段重复覆盖，不额外相加。</p>')
                parts.append('<div class="stages sources">' +
                             _html_table(RESIDUAL_SOURCE_HEADERS, _residual_source_rows(substages)) + '</div>')
            parts.append(_html_table(["子阶段", "父阶段（已含） / 边界依据"],
                         [[item.get("label", item.get("stage_key")),
                           "父阶段=" + _text(item.get("parent_stage_key")) + "；" + _text(item.get("boundary_basis"))]
                          for item in substages]))
    parts.append('<p>源HC周期诊断，不等同完整请求step。未记录设备任务不自动归因CPU或空闲。逐轮阶段锚点：<code>' +
                 esc(diagnostic.get("stages_csv")) + '</code>；原始任务/形状与Host事件见明细CSV。</p>')
    parts.append('<p>逐任务Host来源链：<code>' + esc(diagnostic.get("source_links_csv") or "核心未提供来源链CSV") +
                 '</code>；CPU提交跨度、CANN API并集及完整关联计数见阶段汇总step_residual_stages.csv。</p></section>')
    return ''.join(parts)


def _swa_html(diagnostic, rank):
    esc = lambda value: html.escape(_text(value), quote=True)
    return ('<section><h3>逐层对照与SWA内部细分 · rank=' + esc(rank) + '</h3><p>状态：' +
            esc(diagnostic.get("status")) + '；scope=' + esc(diagnostic.get("scope")) + '</p><p>' +
            esc(SWA_DIAGNOSTIC_SCOPE) + '</p>' +
            ('<p>SWA诊断未完成原因：' + esc(diagnostic["reason"]) + '</p>' if diagnostic.get("reason") else '') +
            _html_table(SWA_LAYER_HEADERS, _swa_layer_rows(diagnostic)) +
            _html_table(SWA_STAGE_HEADERS, _swa_stage_rows(diagnostic)) +
            '<p>逐轮SWA阶段锚点和形状证据：<code>' + esc(diagnostic.get("stages_csv")) + '</code></p></section>')


def _bs_svg(title, categories, series, unit, *, x_axis_title="BS（每DP请求数）", category_name="BS",
            description="横轴为按数值升序的等距BS档位；缺少结果时断线。", max_labels=None):
    """Offline line preview; missing observations split the visible line."""
    esc = lambda value: html.escape(str(value), quote=True)
    observed = [value for _, values, _ in series for value in values if value is not None]
    maximum = max(observed, default=0.) or 1.
    maximum *= 1.1
    left, right, top, bottom = 100, 1060, 48, 324
    width = right - left
    x = lambda index: left + width * index / max(1, len(categories) - 1)
    y = lambda value: bottom - (bottom - top) * value / maximum
    parts = ['<figure class="bs-trend"><figcaption>' + esc(title) + '</figcaption>',
        '<svg viewBox="0 0 1120 420" role="img" aria-label="' + esc(title + '；' + unit) + '">',
        '<title>' + esc(title) + '</title><desc>' + esc(description) + '</desc>',
        '<text x="16" y="24">' + esc(unit) + '</text>']
    for tick in range(5):
        value = maximum * tick / 4
        position = y(value)
        parts.append(f'<line x1="{left}" y1="{position:.3f}" x2="{right}" y2="{position:.3f}" stroke="#dce4ed"/>'
            f'<text x="{left - 12}" y="{position + 5:.3f}" text-anchor="end">{value:.4g}</text>')
    parts.append(f'<path d="M {left},{top} V {bottom} H {right}" fill="none" stroke="#687f94"/>')
    for index, category in enumerate(categories):
        if max_labels and index % max(1, math.ceil(len(categories) / max_labels)) and index != len(categories) - 1:
            continue
        parts.append(f'<text x="{x(index):.3f}" y="{bottom + 26}" text-anchor="middle">{esc(category)}</text>')
    parts.append('<text x="580" y="382" text-anchor="middle">' + esc(x_axis_title) + '</text>')
    for series_index, (label, values, color) in enumerate(series):
        segments, segment = [], []
        for index, value in enumerate(values):
            if value is None:
                if segment:
                    segments.append(segment)
                    segment = []
            else:
                segment.append((index, value))
        if segment:
            segments.append(segment)
        for segment in segments:
            points = ' '.join(f'{x(index):.3f},{y(value):.3f}' for index, value in segment)
            parts.append(f'<polyline points="{points}" fill="none" stroke="#{color}" stroke-width="2.5"/>')
            for index, value in segment:
                parts.append(f'<circle cx="{x(index):.3f}" cy="{y(value):.3f}" r="4" fill="#{color}">'
                    + '<title>' + esc(f'{label}；{category_name}={categories[index]}；{value} {unit}') + '</title></circle>')
        legend_x = 22 + series_index * (1080 / max(1, len(series)))
        parts.append(f'<line x1="{legend_x:.3f}" y1="410" x2="{legend_x + 23:.3f}" y2="410" stroke="#{color}" stroke-width="3"/>'
            f'<text x="{legend_x + 30:.3f}" y="415">{esc(label)}</text>')
    return ''.join(parts) + '</svg></figure>'


def _bs_comparison_html(report):
    esc = lambda value: html.escape(_text(value), quote=True)
    groups = _bs_comparison_groups(report)
    parts = ['<section class="bs-comparison"><h2>' + esc(BS_INDEX_TITLE) + '</h2>',
        _html_table(BS_INDEX_HEADERS, [['' if value is None else value for value in row]
                                      for row in _bs_index_rows(groups)])]
    if report.get('failures'):
        parts.append(_html_table(['统计失败、未纳入BS对比的采样组合', '失败原因'],
            [[item.get('case'), item.get('error')] for item in report['failures']]))
    parts.append('<h2>不同BS的变化对比</h2><p>' + esc(BS_COMPARISON_SCOPE + ITL_PROXY_SCOPE) + '</p>')
    for group in groups:
        parts.append('<h3>' + esc(group['label']) + '</h3><h4>单卡吞吐、完整周期与TPOT</h4>')
        overview = _bs_overview_rows(group)
        parts.append('<div class="bs-table">' + _html_table(BS_OVERVIEW_HEADERS,
            [['' if value is None else value for value in row] for row in overview]) + '</div>')
        reason = _bs_chart_reason(group)
        if reason:
            parts.append('<p>' + esc(reason) + '</p>')
        else:
            categories = [row[0] for row in overview]
            for field, label, unit in BS_METRICS:
                series = [(_layer_mean_name({'id': identity}), [_bs_value(_method(case, identity), field)
                    for case in group['cases']], color) for identity, color in
                    (('method1', '168B8D'), ('method2', 'D4793A'))]
                parts.append(_bs_svg(label + '随BS变化', categories, series, unit))
        for identity in ('method1', 'method2'):
            records = _bs_method_rows(group, identity)
            name = _layer_mean_name({'id': identity})
            parts.append('<h4>' + esc(name + ' · 整轮五项耗时构成') + '</h4>')
            parts.append('<div class="bs-table">' + _html_table(BS_METHOD_HEADERS,
                [['' if value is None else value for value in row] for row in records]) + '</div>')
            if not reason:
                series = [(label, [row[3 + index] for row in records], color)
                    for index, (_, label, color) in enumerate(BS_PARTS)]
                parts.append(_bs_svg(name + ' · 整轮五项耗时构成随BS变化', categories, series, 'ms/完整周期'))
    parts.append('<p>五项之和闭合完整周期；D/C和路由计算变化已计入层区间，U保留一次。</p>')
    return ''.join(parts) + '</section>'


def _sweep_html_table(headers, records):
    return '<div class="bs-table">' + _html_table(headers,
        [['' if value is None else value for value in row] for row in records]) + '</div>'


def _sweep_html_chart(title, records, series, unit, axis):
    if not any(_number(value) for _, values, _ in series for value in values):
        return '<p>本项没有可用数值，保留空值表格，不绘制趋势。</p>'
    return _bs_svg(title, [record[0] for record in records], series, unit, x_axis_title=_sweep_axis(axis),
        category_name=axis, description="横轴为声明的" + axis + "数值升序等距档位；缺少结果时断线，不补0或插值。")


def _s_comparison_html(report):
    esc = lambda value: html.escape(_text(value), quote=True)
    parts = ['<section class="s-comparison"><h2>固定BS，不同投机步长S的变化对比</h2><p>'
        + esc(S_COMPARISON_SCOPE + ITL_PROXY_SCOPE) + '</p>']
    groups = _s_groups(report)
    if not groups:
        parts.append('<p>当前报告没有同一固定BS下的多条S采样；不生成跨条件曲线。</p>')
    for group in groups:
        parts.append('<h3>' + esc(group['label']) + '</h3><p>' + esc(group['note']) + '</p>')
        records = _s_overview_rows(group)
        parts.append(_sweep_html_table(S_OVERVIEW_HEADERS, records))
        reason = _s_chart_reason(group)
        if reason:
            parts.append('<p>' + esc(reason) + '</p>')
        else:
            for index, (_, label, unit) in enumerate(SWEEP_METRICS):
                series = [(_layer_mean_name({'id': identity}), [record[4 + index + shift] for record in records], color)
                    for identity, shift, color in (('method1', 0, '168B8D'), ('method2', 4, 'D4793A'))]
                parts.append(_sweep_html_chart(label + '随S变化', records, series, unit, 'S'))
        for identity in ('method1', 'method2'):
            records = _s_method_rows(group, identity)
            name = _layer_mean_name({'id': identity})
            parts.append('<h4>' + esc(name + ' · 整轮五项耗时构成') + '</h4>')
            parts.append(_sweep_html_table(_sweep_part_headers('S'), records))
            if not reason:
                series = [(label, [record[4 + index] for record in records], color)
                          for index, (_, label, color) in enumerate(BS_PARTS)]
                parts.append(_sweep_html_chart(name + ' · 整轮五项耗时构成随S变化', records, series, 'ms/完整周期', 'S'))
    return ''.join(parts) + '</section>'


def _source_sweeps_html(report):
    esc = lambda value: html.escape(_text(value), quote=True)
    parts = ['<section class="source-sweeps"><h2>源采样基线：固定S看BS、固定BS看S</h2><p>'
        + esc(ITL_PROXY_SCOPE) + '吞吐和TPOT依赖配置Neff，属于条件换算。沿用源aggregate在已采rank中所选的最大均值rank，包含全部主层（含main0），不使用目标分解。</p>']
    for axis, groups in (('BS', _bs_comparison_groups(report)), ('S', _s_groups(report))):
        for group in groups:
            parts.append('<h3>' + esc(group['label']) + '</h3>')
            if axis == 'S':
                parts.append('<p>' + esc(group['note']) + '</p>')
            for case in group['cases']:
                summary = _source_sweep(case)
                reasons = _source_sweep_notes(case, summary)
                if reasons:
                    parts.append('<p>' + esc(case['name'] + '：' + '；'.join(reasons)) + '</p>')
            reason = (_s_chart_reason if axis == 'S' else _bs_chart_reason)(group)
            records = _source_metric_rows(group, axis)
            parts.append(_sweep_html_table(_source_metric_headers(axis), records))
            if reason:
                parts.append('<p>' + esc(reason) + '</p>')
            else:
                for index, (_, label, unit) in enumerate(SWEEP_METRICS):
                    series = [('源采样基线', [record[5 + index] for record in records], '687F94')]
                    parts.append(_sweep_html_chart('源采样基线 · ' + label + '随' + axis + '变化', records, series, unit, axis))
            records = _source_parts_rows(group, axis)
            parts.append(_sweep_html_table(_sweep_part_headers(axis), records))
            if not reason:
                series = [(label, [record[4 + index] for record in records], color)
                          for index, (_, label, color) in enumerate(BS_PARTS)]
                parts.append(_sweep_html_chart('源采样基线 · 整轮五项耗时构成随' + axis + '变化', records, series, 'ms/完整周期', axis))
    return ''.join(parts) + '</section>'


def _html_preview(report, out):
    esc = lambda value: html.escape(_text(value), quote=True)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>10T模型推理估算</title><style>body{margin:24px;font:16px system-ui;color:#173451}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #d2dde8;padding:10px}th{background:#edf2f7}p{line-height:1.7}section{margin:32px 0}code{white-space:pre-wrap;overflow-wrap:anywhere}.stages table{table-layout:fixed}.stages th:first-child{width:42%}.stages th:last-child{width:14%}.stages th,.stages td{overflow-wrap:anywhere}.sources th:first-child{width:31%}.sources td:last-child{white-space:pre-line}.method-residual th:nth-child(2){width:45%}.u-bars{margin:24px 0}.u-bar{display:grid;grid-template-columns:minmax(16em,2fr) 3fr 8em;gap:12px;align-items:center;margin:10px 0}.u-track{background:#edf2f7;height:18px}.u-track i{display:block;background:#168b8d;height:100%}.u-bar>span:last-child{text-align:right;font-variant-numeric:tabular-nums}</style>',
             '<h1>10T模型推理估算</h1><p>' + esc(LAYER_MEAN_SCOPE) + '</p>',
             '<p><a href="' + esc(Path(out).name) + '">打开Excel（9个工作表）</a>；来源：' + esc(report.get('source_kind')) + '</p>']
    parts.append('<style>.bs-table{overflow-x:auto}.bs-table table{min-width:1200px}.bs-table th,.bs-table td{white-space:pre-line}.bs-trend{margin:28px 0;max-width:1120px}.bs-trend svg{display:block;width:100%;font:14px system-ui}.bs-trend figcaption{font-weight:600}</style>')
    parts.append('<style>.operator-chart svg{width:100%;font:12px system-ui;display:block;margin:24px 0}.operator-drilldown table{table-layout:fixed;font-size:13px}.operator-drilldown th:first-child{width:34%}.operator-drilldown th:last-child{width:29%}.operator-drilldown td{overflow-wrap:anywhere;white-space:pre-line}.operator-drilldown pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>')
    parts.append('<style>.u-focus table{table-layout:fixed;font-size:13px}.u-focus td,.u-focus th{white-space:pre-line;overflow-wrap:anywhere}.u-focus th:first-child{width:25%}.u-focus th:last-child{width:25%}.pmu-metrics{overflow-x:auto}.pmu-metrics table{min-width:1500px}.pmu-metrics th:first-child{width:20%}.pmu-metrics th:last-child{width:14%}details>summary{cursor:pointer;font-weight:600;padding:12px;background:#edf2f7}</style>')
    parameter_headers, parameter_rows, _ = _case_parameter_table(report)
    parts.append('<section class="case-parameters"><h2>' + esc(CASE_PARAMETERS_TITLE) + '</h2><p>'
                 + esc(CASE_PARAMETERS_SCOPE) + '</p>'
                 + _html_table(parameter_headers, parameter_rows) + '</section>')
    parts.append(_bs_comparison_html(report))
    parts.append(_s_comparison_html(report))
    parts.append(_source_sweeps_html(report))
    for case in report["cases"]:
        parts.append('<h2>' + esc(case["name"]) + '</h2><p>' + esc(_case_scope(case)) + '</p>')
        parts.append('<p>' + esc(_case_parameters(case)) + '</p>')
        for identity in ("method1", "method2"):
            method = _method(case, identity)
            parts.append('<section><h3>' + esc(_layer_mean_name(method)) + '</h3><p>' +
                         esc(_rank_note(method)) + '；采样均值</p>')
            parts.append('<p>' + esc(_case_parameters(case)) + '</p>')
            parts.append(_html_table(["完整周期(ms)", "TPOT(ms/token)", "预测单卡吞吐(token/s/card)", "状态"],
                         [[_target(method, "cycle_ms"), _target(method, "tpot_ms"), _target(method, "per_device_tokens_s"), _layer_mean_status(method)]]))
            parts.append(_html_table(LAYER_MEAN_MATRIX_HEADERS, _layer_mean_matrix_rows(method)))
            parts.append(_html_table(["整轮闭合/单卡指标", "核心原值", "说明"], _layer_mean_closure_rows(method)))
            parts.append(_method_residual_html(case, method))
            parts.append(_method_focus_html(case, method))
            parts.append(_method_stage_expansions_html(case, method))
            parts.append('<p><code>' + esc(method.get("formula")) + '</code></p><p>' + esc(_explanation(method)) + '</p></section>')
        for capture in case.get("measurements") or []:
            swa = capture.get("swa_diagnostics")
            if isinstance(swa, dict):
                parts.append(_swa_html(swa, capture.get("rank")))
            diagnostic = capture.get("step_residuals") or {}
            if diagnostic:
                parts.append(_residual_html(diagnostic, capture.get("rank")))
    parts.append('</html>')
    path = Path(out).with_suffix('.html')
    path.write_text(''.join(parts), encoding='utf-8')
    return path


def _check_expected_method_set(report, expected, origin):
    if expected is None:
        return
    for case in report["cases"]:
        actual = (case.get("projection") or {}).get("display_method_set")
        if actual != expected:
            raise ValueError(f"导出方法集不一致；来源={origin}；case={case['name']!r}；"
                             f"expected_method_set={expected!r}；display_method_set={actual!r}")


def _check_result_cells(book, report):
    """Compare written headline cards directly with core values, without display helpers."""
    sheets = {sheet.name: sheet for sheet in book.sheets}
    for case_index, case in enumerate(report["cases"]):
        canonical = {method.get("id"): method for method in (case.get("projection") or {}).get(METHOD_SET, [])}
        for identity, method_sheet in (("method1", SHEETS[4]), ("method2", SHEETS[5])):
            method = canonical.get(identity) or {}
            publishable = method.get("status") == "LAYER_MEAN_ESTIMATE" and not method.get("reasons")
            for sheet_name in (SHEETS[0], method_sheet):
                sheet = sheets[sheet_name]
                cards = sheet.result_cells.get((case_index, identity), {})
                for field in ("per_device_tokens_s", "cycle_ms", "tpot_ms"):
                    row, column = cards.get(field, (None, None))
                    expected = method[field] if publishable else PENDING
                    cell = sheet.cells.get((row, column))
                    actual = cell[1] if cell is not None else None
                    valid = (cell is not None and cell[3] is None and
                             ((cell[0] == "n" and actual == str(expected)) if publishable
                              else cell[0] == "str" and actual == PENDING))
                    if not valid:
                        address = cell_reference(row, column) if row is not None and column is not None else "未登记结果单元格"
                        raise ValueError(f"Excel关键结果不一致；来源={book.report_origin}；case={case['name']!r}；"
                                         f"method={identity}；工作表={sheet_name!r}；单元格={address}；"
                                         f"字段={field}；核心值={expected!r}；显示值={actual!r}。已停止保存。")


def build_workbook(out, *, report=None, report_data=None, cases=None, reference=None,
                   reference_unit='unknown', preview=False, detail_rows=DEFAULT_DETAIL_ROWS, progress=None,
                   expected_method_set=None):
    """Render a core report or an honestly empty configuration preview.

    ``report_data`` reuses an already loaded report without parsing its JSON
    again. ``report`` remains required as its source label. The supplied object
    undergoes the usual report validation and is not copied or modified.
    ``expected_method_set`` prevents rendering a different method set from the
    one requested by the caller.
    """
    if reference_unit not in ('unknown','us','ms'):
        raise ValueError('reference_unit must be unknown, us or ms')
    if type(detail_rows) is not int or not 1 <= detail_rows <= 10000:
        raise ValueError('detail_rows必须是1到10000之间的整数；完整明细始终保存在核心JSON')
    if expected_method_set is not None and expected_method_set != METHOD_SET:
        raise ValueError("expected_method_set must be " + METHOD_SET)
    if (report is None) == (cases is None):
        raise ValueError('provide exactly one of report or cases')
    if report_data is not None:
        if report is None or cases is not None or preview:
            raise ValueError('report_data必须同时提供report来源路径，不能配合cases或preview')
        if not isinstance(report_data, dict):
            raise ValueError('report_data必须是已加载的核心报告dict')
    if cases is not None and not preview:
        raise ValueError('配置只能配合--preview；不能从配置伪造性能结果')
    if report is not None and preview:
        raise ValueError('--preview只用于--cases；真实报告状态由核心JSON决定')
    runtime = runtime_diagnostics()
    if not callable(getattr(xlsx_writer.Worksheet, "add_line_chart", None)):
        raise ValueError("Excel写入器缺少BS趋势图接口；请同步更新build_ep128_leadership_excel.py和ep128_xlsx_writer.py。"
                         + "实际写入器：" + runtime["writer_path"] + "（" + runtime["writer_version"] + "）")

    def stage(name, action):
        if progress is not None:
            progress(name)
        try:
            return action()
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Excel导出失败；阶段={name}；"
                f"renderer={runtime['renderer_path']}（{runtime['renderer_version']}）；"
                f"writer={runtime['writer_path']}（{runtime['writer_version']}）；"
                f"原始错误：{error}") from error

    if report_data is not None:
        payload = stage("复用已加载核心报告", lambda: report_data)
    else:
        payload = stage("读取配置预览" if cases is not None else "读取核心JSON",
                        lambda: _preview_report(cases) if cases is not None else _read_json(report))
    stage("校验核心报告", lambda: _validate_report(payload))
    if expected_method_set is not None:
        stage("核对导出方法集", lambda: _check_expected_method_set(payload, expected_method_set, report or cases))
    if progress is not None:
        for message in _input_report_diagnostics(payload, report or cases, reused=report_data is not None):
            progress(message)
    book = Workbook()
    book.detail_rows = detail_rows
    book.report_origin = str(report or cases)
    stage(SHEETS[0], lambda: _overview(book,payload))
    stage(SHEETS[1], lambda: _source_sheet(book,payload))
    stage(SHEETS[2], lambda: _layers_sheet(book,payload))
    stage(SHEETS[3], lambda: _ledger_sheet(book,payload))
    stage(SHEETS[4], lambda: _method_sheet(book,payload,'method1',SHEETS[4]))
    stage(SHEETS[5], lambda: _method_sheet(book,payload,'method2',SHEETS[5]))
    stage(SHEETS[6], lambda: _reference_sheet(book,payload,reference,reference_unit))
    stage(SHEETS[7], lambda: _formula_sheet(book,payload))
    stage(SHEETS[8], lambda: _review_sheet(book,payload,report or cases))
    stage("核对Excel关键结果", lambda: _check_result_cells(book, payload))
    path = stage("写入XLSX", lambda: book.save(out))
    stage("写入HTML预览", lambda: _html_preview(payload,path))
    return Path(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',action='version',version='EP128 Excel '+RENDERER_VERSION)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--report',type=Path,help='schema 2 model_architecture_report.json')
    group.add_argument('--cases',type=Path,help='cases配置，仅可搭配--preview')
    parser.add_argument('--preview',action='store_true',help='只看配置版式，性能全部待估算')
    parser.add_argument('--reference',type=Path,help='原通信CSV仅展示；不参与renderer估算')
    parser.add_argument('--reference-unit',choices=('unknown','us','ms'),default='unknown')
    parser.add_argument('--out',type=Path,default=Path('10T模型推理估算.xlsx'))
    parser.add_argument('--detail-rows',type=int,default=DEFAULT_DETAIL_ROWS,
                        help='每case每类明细在Excel中最多预览的条数，默认1000，范围1..10000；核心统计及完整JSON不变')
    args=parser.parse_args()
    runtime = runtime_diagnostics()
    print(f"Excel导出版本：{runtime['renderer_version']}；写入器版本：{runtime['writer_version']}", file=sys.stderr, flush=True)
    print('生成器：'+runtime['renderer_path'], file=sys.stderr, flush=True)
    print('实际导入写入器：'+runtime['writer_path'], file=sys.stderr, flush=True)
    print('坐标校验代码：'+runtime['position_code_path']+':'+str(runtime['position_code_line']), file=sys.stderr, flush=True)
    print('Python：'+runtime['python_executable']+' ('+runtime['python_version']+')', file=sys.stderr, flush=True)
    try:
        path=build_workbook(args.out,report=args.report,cases=args.cases,reference=args.reference,
                            reference_unit=args.reference_unit,preview=args.preview,detail_rows=args.detail_rows,
                            progress=lambda name: print('Excel阶段：'+name, file=sys.stderr, flush=True))
    except (OSError,ValueError,KeyError,TypeError) as error:
        traceback.print_exc(file=sys.stderr)
        return 1
    print('Excel：'+str(path.resolve()))
    print('HTML预览：'+str(path.with_suffix('.html').resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
