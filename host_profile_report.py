"""Dependency-free exports for measured Host profiling, without extrapolation.

JSON and CSV retain all aggregate rows. HTML and optional XLSX contain bounded
previews, so a long capture does not create an unmanageably large workbook.
"""
import csv
import html
import json
import math
from pathlib import Path


PREVIEW_ROWS = 50
UNAVAILABLE = "未采集 / 不可用"

# Stable CSV field names also make these exports usable outside the workbook.
TABLES = {
    "threads": ("threads.csv", "Host 线程轨道", [
        ("pid", "PID"), ("tid", "TID"), ("process", "进程"),
        ("thread", "线程"), ("calls", "调用数"),
        ("wall_union_ms", "可见区间 union（ms）"),
        ("inclusive_sum_ms", "inclusive 合计（ms）"),
        ("self_sum_ms", "self 合计（ms）"),
        ("self_unavailable_calls", "self 不可用调用数")]),
    "hotspots": ("host_hotspots.csv", "Host 热点", [
        ("pid", "PID"), ("tid", "TID"), ("process", "进程"),
        ("thread", "线程"), ("kind", "类型"), ("name", "事件"),
        ("category", "trace 类别"), ("calls", "调用数"),
        ("inclusive_sum_ms", "inclusive 合计（ms）"),
        ("self_sum_ms", "self 合计（ms）"),
        ("self_unavailable_calls", "self 不可用调用数"),
        ("mean_ms", "平均 inclusive（ms）"), ("p50_ms", "P50（ms）"),
        ("p95_ms", "P95（ms）"), ("max_ms", "最大（ms）"),
        ("status", "self 状态")]),
    "apis": ("host_api.csv", "API、同步与队列", [
        ("pid", "PID"), ("tid", "TID"), ("process", "进程"),
        ("thread", "线程"), ("kind", "类型"), ("name", "事件"),
        ("category", "trace 类别"), ("calls", "调用数"),
        ("inclusive_sum_ms", "inclusive 合计（ms）"),
        ("self_sum_ms", "self 合计（ms）"),
        ("self_unavailable_calls", "self 不可用调用数"),
        ("mean_ms", "平均 inclusive（ms）"), ("p50_ms", "P50（ms）"),
        ("p95_ms", "P95（ms）"), ("max_ms", "最大（ms）"),
        ("status", "self 状态")]),
    "steps": ("steps.csv", "步骤 / 标注区间", [
        ("event_id", "事件 ID"), ("name", "原始标注"),
        ("pid", "PID"), ("tid", "TID"),
        ("start_ns", "起点（ns）"), ("end_ns", "终点（ns）"),
        ("wall_ms", "区间 wall（ms）"),
        ("host_wall_union_ms", "可见 Host union（ms）"),
        ("device_recorded_union_ms", "设备已记录 union（ms）"),
        ("device_gap_ms", "设备无记录区间（ms）")]),
    "gaps": ("device_gaps.csv", "设备无记录区间", [
        ("device_pid", "设备 PID"), ("start_ns", "起点（ns）"),
        ("end_ns", "终点（ns）"), ("duration_ms", "区间（ms）"),
        ("kind", "区间类型"), ("previous_name", "前一任务"),
        ("previous_tid", "前一 TID"), ("next_name", "后一任务"),
        ("next_tid", "后一 TID"),
        ("host_overlap_ms", "Host 重叠（ms）"),
        ("sync_overlap_ms", "同步重叠（ms）"),
        ("copy_overlap_ms", "拷贝重叠（ms）"),
        ("api_overlap_ms", "API 重叠（ms）"), ("evidence", "证据边界")]),
    "links": ("source_links.csv", "设备任务来源", [
        ("device_event_id", "设备事件 ID"), ("device_pid", "设备 PID"),
        ("device_tid", "设备 TID"), ("device_name", "设备任务"),
        ("start_ns", "设备起点（ns）"), ("api_event_id", "API 事件 ID"),
        ("api_pid", "API PID"), ("api_tid", "API TID"),
        ("api_name", "API"), ("cpu_event_id", "CPU 事件 ID"),
        ("cpu_name", "可见 CPU 调用"),
        ("enqueue_event_id", "Enqueue 事件 ID"),
        ("dequeue_event_id", "Dequeue 事件 ID"),
        ("queue_wait_ms", "队列等待（ms）"),
        ("submit_to_device_ms", "提交至设备起点（ms）"),
        ("status", "关联状态"), ("reason", "依据 / 不可关联原因")]),
    "devices": ("device_summary.csv", "设备记录覆盖", [
        ("pid", "设备 PID"), ("process", "进程"), ("calls", "任务数"),
        ("recorded_union_ms", "已记录 union（ms）"),
        ("recorded_sum_ms", "任务时长合计（ms）"),
        ("no_recorded_ms", "无记录区间（ms）"),
        ("aicpu_union_ms", "AICPU union（ms）"),
        ("dma_union_ms", "DMA union（ms）"),
        ("host_overlap_in_gaps_ms", "无记录区间内 Host 重叠（ms）")]),
}


def _display(value):
    if value is None:
        return UNAVAILABLE
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return format(value, ".6f").rstrip("0").rstrip(".") if math.isfinite(value) else UNAVAILABLE
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _escape(value):
    return html.escape(_display(value), quote=True)


def _csv_value(value):
    """Keep missing data empty and prevent spreadsheet formula interpretation."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, str):
        # Leading whitespace can hide a formula prefix from spreadsheet readers.
        stripped = value.lstrip()
        if value.startswith(("\t", "\r", "\n")) or stripped.startswith(("=", "+", "-", "@")):
            return "'" + value
    return value


def _excel_value(value):
    if value is None:
        return UNAVAILABLE
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    # Long numeric identifiers/timestamps would otherwise lose precision in Excel.
    if type(value) is int and abs(value) >= 10 ** 15:
        value = str(value)
    if isinstance(value, str):
        value = "".join(char if (char in "\t\n\r" or 0x20 <= ord(char) <= 0xD7FF
                                or 0xE000 <= ord(char) <= 0xFFFD
                                or 0x10000 <= ord(char) <= 0x10FFFF) else "\ufffd" for char in value)
        return value[:32767]
    return value


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _preview(rows, metric=None):
    if metric:
        rows = sorted(rows, key=lambda row: row.get(metric) if _number(row.get(metric)) else -1,
                      reverse=True)
    return rows[:PREVIEW_ROWS]


def _table_html(key, rows, *, metric=None, detail_limits=None):
    filename, title, columns = TABLES[key]
    shown = _preview(rows, metric)
    sampled = bool((detail_limits or {}).get(key + "_truncated"))
    parts = ["<section><h2>" + _escape(title) + "</h2>",
             '<p class="muted">显示 ' + str(len(shown)) + " / " + str(len(rows))
             + (' 行；已导出的明细样例：' if sampled else ' 行；完整导出：')
             + '<a href="' + filename + '">' + filename + "</a>。</p>"]
    if sampled:
        parts.append('<p class="unavailable">' + _escape(_detail_note(key, detail_limits)) + "</p>")
    if shown:
        parts.append('<div class="table-scroll"><table><thead><tr>')
        parts.extend("<th>" + _escape(label) + "</th>" for _, label in columns)
        parts.append("</tr></thead><tbody>")
        for row in shown:
            parts.append("<tr>" + "".join("<td>" + _escape(row.get(field)) + "</td>"
                                            for field, _ in columns) + "</tr>")
        parts.append("</tbody></table></div>")
    else:
        parts.append('<p class="unavailable">没有可展示的记录；请结合覆盖状态判断是否采集。</p>')
    parts.append("</section>")
    return "".join(parts)


def _hotspot_chart(rows, topic="线程内热点"):
    """Self comparisons use only complete groups; an inclusive fallback is labelled."""
    valid = [row for row in rows if _number(row.get("self_sum_ms"))
             and not row.get("self_unavailable_calls")]
    if valid:
        metric, label = "self_sum_ms", topic + " self wall 合计（ms）"
        note = "仅展示 self 可完整计算的分组。self wall 排除了同线程子调用，仍可能包含等待、阻塞或被抢占。"
    else:
        valid = [row for row in rows if _number(row.get("inclusive_sum_ms"))]
        metric, label = "inclusive_sum_ms", topic + " inclusive wall 合计（ms）"
        note = "self 无可用分组，改用 inclusive wall；嵌套调用会重复计时，柱值不可相加为 CPU 忙时。"
    valid = sorted(valid, key=lambda row: row[metric], reverse=True)[:10]
    return valid, metric, label, note


def _chart_html(rows, topic="线程内热点"):
    selected, metric, label, note = _hotspot_chart(rows, topic)
    parts = ["<section><h2>" + _escape(label) + "</h2><p>" + _escape(note) + "</p>"]
    if not selected:
        parts.append('<p class="unavailable">没有可用热点数据。</p>')
    else:
        maximum = max(row[metric] for row in selected)
        for row in selected:
            width = 0 if maximum <= 0 else max(0, row[metric] / maximum * 100)
            name = str(row.get("pid", "")) + "/" + str(row.get("tid", "")) + " · " + str(row.get("name", ""))
            parts.append('<div class="bar-row"><span class="bar-label">' + _escape(name)
                         + '</span><div class="bar-track"><div class="bar" style="width:'
                         + format(width, ".3f") + '%"></div></div><span>'
                         + _escape(row[metric]) + " ms</span></div>")
    parts.append("</section>")
    return "".join(parts)


def _summary_rows(report):
    summary = report.get("summary", {})
    scope = report.get("scope", {})
    context = report.get("context", {})
    coverage = report.get("coverage", {})
    return [
        ("工具版本", report.get("tool_version")), ("生成时间", report.get("generated_at")),
        ("模型（用户声明，未验证）", context.get("model")),
        ("硬件（用户声明，未验证）", context.get("hardware")),
        ("仅主模型（用户声明）", context.get("main_only")),
        ("架构适配状态", context.get("architecture_status")),
        ("分析范围", scope.get("mode")), ("范围起点（ns）", scope.get("start_ns")),
        ("范围终点（ns）", scope.get("end_ns")), ("范围 wall（ms）", scope.get("wall_ms")),
        ("Host 覆盖状态", coverage.get("host_status")),
        ("设备覆盖状态", coverage.get("device_status")),
        ("来源关联状态", coverage.get("source_status")),
        ("可见 Host 调用数", summary.get("host_events")),
        ("可见 Host 线程轨道数（PID / TID）", summary.get("host_threads")),
        ("Host 跨线程 wall union（ms）", summary.get("host_wall_union_ms")),
        ("各线程轨道 wall union 合计（ms）", summary.get("host_thread_union_sum_ms")),
        ("Host inclusive 合计（ms）", summary.get("host_inclusive_sum_ms")),
        ("Host self 合计（ms）", summary.get("host_self_sum_ms")),
        ("self 不可用调用数", summary.get("self_unavailable_events")),
        ("设备任务数", summary.get("device_events")),
        ("OS CPU running（ms）", summary.get("cpu_running_ms")),
        ("OS CPU 利用率（%）", summary.get("cpu_utilization_percent")),
        ("设备区间明细范围", _detail_note("gaps", report.get("detail_limits", {}))),
        ("来源关联明细范围", _detail_note("links", report.get("detail_limits", {}))),
    ]


def _detail_note(key, limits):
    singular = {"gaps": "gap", "links": "link"}[key]
    if limits.get(key + "_truncated"):
        order = "时长最大的区间" if key == "gaps" else "任务关联样例"
        return ("JSON / CSV 明细仅保留 " + str(limits.get(key, "")) + " 条" + order
                + "，原始共 " + str(limits.get(singular + "_rows_total", ""))
                + " 条；覆盖统计使用全部记录。CSV 的 detail_* 列保留此范围。")
    return "JSON / CSV 包含全部已统计明细；HTML / XLSX 最多预览 50 行。"


BASE_SEMANTICS = [
    "本报告为采样实测分析，不计算外推耗时、TPOT、接受长度或加速比。模型和硬件名称仅来自输入声明。",
    "Host wall 包含等待、阻塞和被抢占；self wall 仅排除可验证的同线程子调用，不等于 OS CPU 实际运行时间。",
    "inclusive 合计、各线程 union 合计和跨线程 union 的口径不同；API、同步及来源延迟也不能与这些值相加。",
    "线程轨道以 trace 的 PID / TID 区分；CANN 与 Python 的合成 PID 可能对应同一物理线程，因此轨道数不是 OS 线程数，各轨道 union 合计也可能重复。",
    "设备无记录区间只表示本次 trace 未记录该设备任务，不证明设备空闲；与 Host 重叠不证明由 Host 导致。",
    "同步 API 可能正在等待设备；AICPU 属于设备侧，不能合并到服务器 Host CPU 计算中。",
    "未采集或不可关联项保留为空 / 不可用。没有步骤标注时，整个采样范围不代表一个请求或一轮解码。",
]


def _semantics(report):
    return list(dict.fromkeys(BASE_SEMANTICS + list(report.get("semantics", []))))


def _render_html(report):
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Host 采样实测分析</title><style>',
             'body{font:15px/1.6 system-ui,sans-serif;color:#24364b;background:#edf2f7;margin:0}'
             'main{max-width:1500px;margin:auto;padding:28px}h1{font-size:28px}'
             'section{background:white;padding:20px;margin:20px 0;border-radius:10px}'
             'h2{font-size:20px;margin-top:0}.muted,.unavailable{color:#617286}'
             '.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%}'
             'th,td{border:1px solid #d2dde8;padding:8px;text-align:left;vertical-align:top;min-width:80px;overflow-wrap:anywhere}'
             'th{background:#173451;color:white}tbody tr:nth-child(even){background:#f4f7fa}'
             '.summary td:first-child{width:38%}.bar-row{display:grid;grid-template-columns:minmax(180px,40%) 1fr 120px;gap:12px;align-items:center;margin:9px 0}'
             '.bar-label{overflow-wrap:anywhere}.bar-track{background:#edf2f7;height:18px}.bar{height:18px;background:#167d8d}'
             'a{color:#126c9b}@media(max-width:700px){main{padding:12px}.bar-row{grid-template-columns:1fr}.bar-track{height:14px}}',
             '</style></head><body><main><h1>Host 采样实测分析</h1>',
             '<p>独立的 Host / 设备记录分析；覆盖范围与缺失证据随报告保留。</p>',
             '<section><h2>总览</h2><table class="summary"><tbody>']
    for label, value in _summary_rows(report):
        parts.append("<tr><td>" + _escape(label) + "</td><td>" + _escape(value) + "</td></tr>")
    parts.append("</tbody></table></section><section><h2>统计口径</h2><ul>")
    parts.extend("<li>" + _escape(text) + "</li>" for text in _semantics(report))
    parts.append("</ul></section>")
    inventory = report.get("inventory", [])
    parts.append("<section><h2>采样进程清单</h2><p>PID 用于选择分析范围；此处为 trace 进程轨道，不等于完整服务器进程清单。显示 "
                 + str(min(PREVIEW_ROWS, len(inventory))) + " / " + str(len(inventory))
                 + " 行；完整清单见 JSON 的 inventory。</p><table><thead><tr><th>PID</th><th>名称</th><th>类型</th></tr></thead><tbody>")
    for item in inventory[:PREVIEW_ROWS]:
        parts.append("<tr>" + "".join("<td>" + _escape(item.get(key)) + "</td>"
                                        for key in ("pid", "name", "kind")) + "</tr>")
    parts.append("</tbody></table></section>")
    parts.append(_chart_html(report.get("hotspots", [])))
    if report.get("apis"):
        parts.append(_chart_html(report["apis"], "API、同步与队列（线程内）"))
    for key, metric in (("threads", "wall_union_ms"), ("hotspots", "inclusive_sum_ms"),
                        ("apis", "inclusive_sum_ms"), ("steps", None),
                        ("devices", "recorded_union_ms"), ("gaps", "duration_ms"), ("links", None)):
        parts.append(_table_html(key, report.get(key, []), metric=metric,
                                 detail_limits=report.get("detail_limits", {})))
    parts.append("<section><h2>来源关联覆盖</h2><table><tbody>")
    for status, count in sorted(report.get("coverage", {}).get("source_status_counts", {}).items()):
        parts.append("<tr><td>" + _escape(status) + "</td><td>" + _escape(count) + "</td></tr>")
    parts.append("</tbody></table></section><section><h2>采样与解析诊断</h2><table><thead><tr><th>代码</th><th>数量</th><th>说明</th></tr></thead><tbody>")
    for diagnostic in report.get("diagnostics", [])[:PREVIEW_ROWS]:
        parts.append("<tr>" + "".join("<td>" + _escape(diagnostic.get(key)) + "</td>"
                                        for key in ("code", "count", "message")) + "</tr>")
    parts.append("</tbody></table><h3>原始输入</h3><table><tbody>")
    for key in ("path", "sha256", "bytes_read", "parser_max_buffer_chars", "time_quantization"):
        parts.append("<tr><td>" + _escape(key) + "</td><td>"
                     + _escape(report.get("source", {}).get(key)) + "</td></tr>")
    parts.append('</tbody></table><p><a href="host_profile_report.json">完整结构化报告 JSON</a></p></section></main></body></html>')
    return "".join(parts)


def _xlsx_table(workbook, sheet_name, key, rows, metric=None, *, detail_limits=None):
    _, title, columns = TABLES[key]
    sheet = workbook.add_sheet(sheet_name, widths=[24] * len(columns))
    sheet.merge(1, 1, 1, len(columns), title, "title").row_height(1, 36)
    shown = _preview(rows, metric)
    sheet.merge(2, 1, 2, len(columns),
                "预览 " + str(len(shown)) + " / " + str(len(rows))
                + " 行；完整聚合 / 有上限的明细见 " + TABLES[key][0] + "。未采集项保持不可用。", "note").row_height(2, 32)
    if (detail_limits or {}).get(key + "_truncated"):
        sheet.merge(3, 1, 3, len(columns), _detail_note(key, detail_limits), "warning").row_height(3, 36)
    for col, (_, label) in enumerate(columns, 1):
        sheet.write(4, col, label, "header")
    sheet.row_height(4, 38).freeze(4, 0)
    for index, item in enumerate(shown, 5):
        for col, (field, _) in enumerate(columns, 1):
            value = _excel_value(item.get(field))
            sheet.write(index, col, value, "number" if _number(value) else "body")
        sheet.row_height(index, 36)
    if not shown:
        sheet.merge(5, 1, 5, len(columns), "没有可展示的记录；请结合覆盖状态判断是否采集。", "note")
    return sheet, max(6, 5 + len(shown))


def _xlsx_chart(sheet, sheet_name, rows, last, topic="线程内热点"):
    from ep128_xlsx_writer import cell_reference

    selected, metric, title, note = _hotspot_chart(rows, topic)
    sheet.merge(last + 1, 1, last + 1, 16, title, "section")
    sheet.merge(last + 2, 1, last + 2, 16, note, "note").row_height(last + 2, 32)
    if not selected:
        sheet.merge(last + 4, 1, last + 4, 16, "没有可用热点数据。", "note")
        return
    start = last + 4
    sheet.write(start, 1, "PID / TID · 事件", "header")
    sheet.write(start, 2, title, "header")
    labels = []
    for index, item in enumerate(selected, start + 1):
        name = _excel_value(str(item.get("pid", "")) + "/" + str(item.get("tid", ""))
                            + " · " + str(item.get("name", "")))
        labels.append(name)
        sheet.write(index, 1, name, "body")
        sheet.write(index, 2, item[metric], "number")
        sheet.row_height(index, 36)
    end = start + len(selected)
    sheet_ref = "'" + sheet_name.replace("'", "''") + "'!"
    sheet.add_bar_chart(title, labels,
                        [{"name": title, "values": [item[metric] for item in selected],
                          "values_ref": sheet_ref + cell_reference(start + 1, 2, True) + ":" + cell_reference(end, 2, True),
                          "name_ref": sheet_ref + cell_reference(start, 2, True)}],
                        (start, 4, start + 22, 12), stacked=False, x_axis_title="wall 合计（ms）",
                        categories_ref=sheet_ref + cell_reference(start + 1, 1, True) + ":" + cell_reference(end, 1, True))


def _write_xlsx(report, path):
    from ep128_xlsx_writer import Workbook

    workbook = Workbook()
    overview = workbook.add_sheet("总览", widths=[46, 34, 24, 24, 24, 24])
    overview.merge(1, 1, 1, 6, "Host 采样实测分析", "title").row_height(1, 36)
    overview.merge(2, 1, 2, 6, "模型和硬件来自用户声明；本报告不计算外推指标。", "note").row_height(2, 28)
    row = 4
    for label, value in _summary_rows(report):
        overview.write(row, 1, label, "label")
        overview.merge(row, 2, row, 6, _excel_value(value), "body")
        overview.row_height(row, 26)
        row += 1
    row += 1
    overview.merge(row, 1, row, 6, "统计口径", "section")
    for text in _semantics(report):
        row += 1
        overview.merge(row, 1, row, 6, _excel_value(text), "note").row_height(row, 42)
    row += 2
    overview.merge(row, 1, row, 6, "采样与解析诊断", "section")
    for item in report.get("diagnostics", [])[:PREVIEW_ROWS]:
        row += 1
        overview.write(row, 1, _excel_value(item.get("code")), "label")
        overview.write(row, 2, _excel_value(item.get("count")), "integer")
        overview.merge(row, 3, row, 6, _excel_value(item.get("message")), "body").row_height(row, 40)
    row += 2
    overview.merge(row, 1, row, 6, "原始输入", "section")
    for key in ("path", "sha256", "bytes_read", "parser_max_buffer_chars", "time_quantization"):
        row += 1
        overview.write(row, 1, key, "label")
        overview.merge(row, 2, row, 6, _excel_value(report.get("source", {}).get(key)), "body").row_height(row, 40)
    inventory = report.get("inventory", [])
    row += 2
    overview.merge(row, 1, row, 6, "采样进程清单（trace 进程轨道）", "section")
    row += 1
    overview.merge(row, 1, row, 6, "PID 用于选择范围；预览最多 50 行，完整 inventory 见 JSON。此处不是服务器全部进程。", "note").row_height(row, 32)
    row += 1
    for col, label in enumerate(("PID", "名称", "类型"), 1):
        overview.write(row, col, label, "header")
    for item in inventory[:PREVIEW_ROWS]:
        row += 1
        for col, key in enumerate(("pid", "name", "kind"), 1):
            overview.write(row, col, _excel_value(item.get(key)), "body")
        overview.row_height(row, 32)

    _xlsx_table(workbook, "Host线程", "threads", report.get("threads", []), "wall_union_ms")
    hotspots, last = _xlsx_table(workbook, "Host热点", "hotspots", report.get("hotspots", []), "inclusive_sum_ms")
    _xlsx_chart(hotspots, "Host热点", report.get("hotspots", []), last)
    api_sheet, api_last = _xlsx_table(workbook, "API与同步", "apis", report.get("apis", []), "inclusive_sum_ms")
    if report.get("apis"):
        _xlsx_chart(api_sheet, "API与同步", report["apis"], api_last, "API、同步与队列（线程内）")
    _xlsx_table(workbook, "步骤区间", "steps", report.get("steps", []))
    gap_sheet, gap_last = _xlsx_table(workbook, "设备空档", "gaps", report.get("gaps", []), "duration_ms",
                                      detail_limits=report.get("detail_limits", {}))
    # Keep device summaries on the device sheet as a distinct, bounded table.
    gap_sheet.merge(gap_last + 1, 1, gap_last + 1, 15, "设备记录覆盖（详见 device_summary.csv）", "section")
    device_cols = TABLES["devices"][2]
    for col, (_, label) in enumerate(device_cols, 1):
        gap_sheet.write(gap_last + 3, col, label, "header")
    gap_sheet.row_height(gap_last + 3, 40)
    for index, item in enumerate(_preview(report.get("devices", []), "recorded_union_ms"), gap_last + 4):
        for col, (field, _) in enumerate(device_cols, 1):
            value = _excel_value(item.get(field))
            gap_sheet.write(index, col, value, "number" if _number(value) else "body")
        gap_sheet.row_height(index, 36)
    source_sheet, _ = _xlsx_table(workbook, "来源覆盖", "links", report.get("links", []),
                                 detail_limits=report.get("detail_limits", {}))
    counts = report.get("coverage", {}).get("source_status_counts", {})
    # The stable row number avoids ever merging through the source preview.
    row = 60
    source_sheet.merge(row, 1, row, 17, "来源关联覆盖（各状态任务数）", "section")
    for status, count in sorted(counts.items()):
        row += 1
        source_sheet.write(row, 1, _excel_value(status), "label")
        source_sheet.write(row, 2, _excel_value(count), "integer")
    workbook.save(path)


def write_reports(report, out, *, excel=False):
    """Write portable reports; return an artifact-name to absolute-path mapping."""
    out = Path(out).resolve()
    filenames = ["host_profile_report.json", "host_profile_report.html"]
    filenames.extend(spec[0] for spec in TABLES.values())
    if excel:
        filenames.append("host_profile_report.xlsx")
    source = report.get("source", {}).get("path")
    if source:
        source_path = Path(source).expanduser().resolve()
        for filename in filenames:
            target = out / filename
            if (target.resolve() == source_path or
                    target.exists() and source_path.exists() and target.samefile(source_path)):
                raise ValueError("Report output would overwrite the source trace: " + str(target))
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    json_path = out / "host_profile_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifacts["json"] = str(json_path)
    for key, (filename, _, columns) in TABLES.items():
        path = out / filename
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            detail = key in ("gaps", "links")
            fields = [field for field, _ in columns]
            writer.writerow(fields + (["detail_sampled", "detail_rows_total", "detail_rows_exported"] if detail else []))
            rows = report.get(key, [])
            limits = report.get("detail_limits", {})
            for row in rows:
                values = [_csv_value(row.get(field)) for field in fields]
                if detail:
                    singular = {"gaps": "gap", "links": "link"}[key]
                    values.extend([bool(limits.get(key + "_truncated")),
                                   limits.get(singular + "_rows_total", len(rows)), len(rows)])
                writer.writerow(values)
        artifacts[Path(filename).stem + "_csv"] = str(path)
    html_path = out / "host_profile_report.html"
    html_path.write_text(_render_html(report), encoding="utf-8")
    artifacts["html"] = str(html_path)
    if excel:
        xlsx_path = out / "host_profile_report.xlsx"
        _write_xlsx(report, xlsx_path)
        artifacts["xlsx"] = str(xlsx_path)
    return artifacts
