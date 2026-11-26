"""Place native Host/HC/U diagrams on the existing extra-chart worksheet."""

import html
import math

from ep128_swimlane_plot import LANE_LABELS, panel_drawing, render_svg


SCOPE = ("固定rank的源实测代表轮；0为该case所选轮主HC起点，所有case共用毫秒轴。"
         "propose来自本轮sample_tokens，其后四条Host线为下一次execute_model的准备。"
         "HC仅为层链包络，不代表完整FULL图或持续设备忙碌。Host条为完整调用wall time，"
         "按函数着色；竖线只说明时间对应，提交因果以来源链为准。缺失留空，不作0或设备idle。"
         "该区不修改主报告的外推均值或结果。")


def _detail_values(item):
    role = item.get("role")
    label = LANE_LABELS.get(role, "NPU U阶段" if role == "device_u" else role)
    name = item.get("name")
    if item.get("record_type") == "u_stage":
        name = "U" + str(item.get("stage_number", "?")) + " · " + str(name)
    relation = item.get("relation")
    if relation is None:
        relation = "设备阶段边界" if item.get("record_type") == "u_stage" else "父调用范围"
    return [label, name, item.get("event_index"), item.get("start_ms"), item.get("end_ms"),
            item.get("duration_ms"), item.get("depth"), relation]


def _ordered(summary):
    def numeric(value):
        return (value is None, float(value) if value is not None else 0)
    return sorted(summary.get("panels", []), key=lambda p: (numeric(p.get("bs")), numeric(p.get("S")), str(p.get("case", ""))))


def _drawing(sheet, row, drawing, first=1, last=25):
    # The containing worksheet uses its established 12-character columns.
    width_px = sum(int(sheet.widths[i] * 7 + 5) if i < len(sheet.widths) else 89
                   for i in range(first - 1, last - 1))
    height_px = drawing["height"] * width_px / drawing["width"]
    rows = max(1, math.ceil(height_px / 20))
    for index in range(row, row + rows + 1):
        sheet.row_height(index, 15)
    sheet.add_vector_drawing(drawing["elements"], view_box=(drawing["width"], drawing["height"]),
                             anchor=(row, first, row + rows, last),
                             description=drawing.get("description", "Host/HC/U真实时间泳道"))
    return row + rows + 2


def append_swimlanes(sheet, row, summary, *, note, table):
    """All diagrams precede the full child tables, with internal navigation."""
    panels = _ordered(summary)
    row = note(sheet, row, "Host / HC / U 同轮泳道 · " + str(summary.get("rank", "")), columns=24)
    row = note(sheet, row, SCOPE, columns=24)
    if summary.get("input_json"):
        row = note(sheet, row, "泳道来源：" + str(summary["input_json"]), columns=24)
    if not panels:
        return note(sheet, row, "没有可绘制case，未构造数据。", warning=True, columns=24)
    axes = {key: summary[key] for key in ("x_min_ms", "x_max_ms")}
    overview_links = []
    row = note(sheet, row, "全部case总览 · 按BS、S排序；点击case进入放大泳道", columns=24)
    for offset in range(0, len(panels), 3):
        top, end = row, row
        for column, panel in enumerate(panels[offset:offset + 3]):
            left, right = 1 + column * 8, 9 + column * 8
            label = str(panel.get("case", "unknown")) + " · 放大图"
            sheet.merge(top, left, top, right - 1, label, "link").row_height(top, 25)
            overview_links.append((top, left, label, offset + column))
            drawing = panel_drawing(panel, width=620, overview=True, **axes)
            end = max(end, _drawing(sheet, top + 1, drawing, left, right))
        row = end + 1
    row = note(sheet, row, "各case放大泳道 · 函数名、一级语义子调用及U边界；下方附全部已保留调用明细", columns=24)
    detail_positions, table_links = {}, []
    for index, panel in enumerate(panels):
        title = str(panel.get("case", "unknown")) + " · rank=" + str(panel.get("rank", summary.get("rank", "")))
        title += " · " + str(panel.get("round_id", panel.get("selected_round_id", "未知轮次")))
        detail_positions[index] = row
        sheet.merge(row, 1, row, 19, title, "header").row_height(row, 30)
        table_links.append((row, 20, "该case全部子调用明细", index))
        sheet.merge(row, 20, row, 24, "该case全部子调用明细", "link")
        row = _drawing(sheet, row + 1, panel_drawing(panel, width=1600, overview=False, **axes))
    for top, left, label, index in overview_links:
        sheet.add_internal_link(top, left, label, sheet.name, "A" + str(detail_positions[index]))
    table_positions = {}
    row = note(sheet, row, "源数据与调用明细 · 精确时间用ms存为数值，原event_index保留便于回查", columns=24)
    for index, panel in enumerate(panels):
        table_positions[index] = row
        sheet.merge(row, 1, row, 19, str(panel.get("case", "unknown")) + " · 源调用明细", "header").row_height(row, 30)
        sheet.merge(row, 20, row, 24, "返回该case放大图", "link")
        sheet.add_internal_link(row, 20, "返回该case放大图", sheet.name, "A" + str(detail_positions[index]))
        row += 1
        reason = panel.get("reason") or panel.get("reasons")
        if reason:
            row = note(sheet, row, str(reason), warning=True, columns=24)
        values = panel.get("detail_rows") or []
        row = table(sheet, row, ["角色/父范围", "函数/子调用完整名称", "event_index", "开始(ms)", "结束(ms)",
                                "wall(ms)", "层级", "归属证据"],
                    [_detail_values(item) for item in values],
                    spans=((1, 3), (4, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 21), (22, 24)),
                    blank_columns=(3, 4, 5, 6, 7))
        if not values:
            row = note(sheet, row, "Host调用不可用；保留错误及设备侧证据，不构造Host时间。", warning=True, columns=24)
        edges = panel.get("edges") or []
        if edges:
            row = note(sheet, row, "函数到设备锚点的全部来源候选 · 图上每条Host线只举一例；虚线不能当作已证实提交", columns=24)
            names = {item.get("event_index"): item.get("name") for item in values if item.get("event_index") is not None}
            row = table(sheet, row, ["Host函数/event", "原CSV任务行", "U阶段/边界", "Host来源(ms)", "设备锚点(ms)",
                                    "设备来源状态", "Host关系", "画线证据"],
                        [[names.get(edge.get("host_event_index"), "event=" + str(edge.get("host_event_index"))),
                          edge.get("row_index"), ("U" + str(edge["stage_number"]) if edge.get("stage_number") else edge.get("anchor_kind")),
                          edge.get("source_time_ms"), edge.get("anchor_time_ms"), edge.get("source_status"), edge.get("host_relation"),
                          "实线：已证实提交" if edge.get("runtime_submission_confirmed") else "虚线：推断/来源不完整"] for edge in edges],
                        spans=((1, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 18), (19, 21), (22, 24)),
                        blank_columns=(2, 4, 5))
    for top, left, label, index in table_links:
        sheet.add_internal_link(top, left, label, sheet.name, "A" + str(table_positions[index]))
    return row + 1


def html_section(summary):
    panels = _ordered(summary)
    esc = lambda value: html.escape(str(value), quote=True)
    axes = {key: summary[key] for key in ("x_min_ms", "x_max_ms")}
    parts = ['<section class="swimlane-section"><h2>Host / HC / U 同轮泳道</h2><p>' + esc(SCOPE) + '</p>',
             '<style>.swimlane-grid{display:grid;grid-template-columns:repeat(3,minmax(520px,1fr));gap:18px;overflow:auto}'
             '.swimlane-detail{overflow:auto;margin:24px 0}.swimlane-detail svg{min-width:1400px;width:100%;height:auto}'
             '.swimlane-grid svg{width:100%;height:auto}.swimlane-detail table{font-size:12px}.swimlane-detail td{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
             '<div class="swimlane-grid">']
    for index, panel in enumerate(panels):
        parts.append('<article><a href="#swimlane-' + str(index) + '">' + esc(panel.get("case")) + ' · 放大图</a>'
                     + render_svg(panel_drawing(panel, width=620, overview=True, **axes)) + '</article>')
    parts.append('</div>')
    for index, panel in enumerate(panels):
        parts.append('<article class="swimlane-detail" id="swimlane-' + str(index) + '"><h3>' + esc(panel.get("case")) + '</h3>'
                     + render_svg(panel_drawing(panel, width=1600, overview=False, **axes)))
        parts.append('<details><summary>全部已保留Host调用与子调用</summary><table><thead><tr>'
                     + ''.join('<th>' + name + '</th>' for name in ('角色', '函数', 'event_index', '开始(ms)', '结束(ms)', 'wall(ms)', '层级', '归属证据'))
                     + '</tr></thead><tbody>')
        for item in panel.get("detail_rows") or []:
            parts.append('<tr>' + ''.join('<td>' + esc(value if value is not None else '—') + '</td>'
                for value in _detail_values(item)) + '</tr>')
        parts.append('</tbody></table></details></article>')
    parts.append('</section>')
    return ''.join(parts)
