"""Place native Host/HC/U diagrams on the existing extra-chart worksheet."""

import html
import math

from ep128_swimlane_plot import LANE_LABELS, _name, panel_drawing, render_svg
from ep128_swimlane_mechanism import mechanism_drawing


SCOPE = ("固定rank的源实测代表轮；0为该case所选轮主HC起点，所有case共用毫秒轴。"
         "propose来自本轮sample_tokens，其后四条Host线为下一次execute_model的准备。"
         "HC仅为层链包络，不代表完整FULL图或持续设备忙碌。Host父条为完整wall，子色带互斥闭合，父子不相加。"
         "原始嵌套范围另表留存，不参与互斥求和；其余wall不是纯CPU计算。"
         "30个U起止边界全部审计，实线只画已记录CPU/API来源；Python归属另行标明。缺失不补线、不作0或设备idle。"
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


def _exclusive_values(panel):
    rows = []
    for group in panel.get("child_plot_rows", []):
        role = group["lane"]
        parent = (panel.get("host_lanes") or {}).get(role, {})
        duration = parent.get("duration_ns", 0)
        for piece in group["events"]:
            rows.append([LANE_LABELS.get(role, role), "#"+str(piece["piece_id"]), piece["label"],
                         piece["start_ms"], piece["end_ms"], piece["duration_ms"],
                         100*piece["duration_ns"]/duration if duration else None,
                         piece.get("selection_reason")])
    return rows


def _anchor_values(panel):
    return [[entry["anchor_id"], entry.get("time_ms"), entry.get("physical_anchor_id"),
             str(entry.get("device_rows", [])), _name(entry.get("source_name")) if entry.get("source_name") else None,
             entry.get("source_start_ms"), entry.get("source_end_ms"), entry.get("grade_label", entry.get("grade")),
             ("已证实：" + "; ".join(_name(owner.get("name")) for owner in entry["verified_python_owners"])
              if entry.get("verified_python_owners") else entry.get("python_owner_grade")),
             "; ".join(str(v) for v in (entry.get("endpoint_note"), entry.get("reason")) if v)]
            for entry in (panel.get("anchor_audit") or {}).get("audit_entries", [])]


def _exclusive_totals(panel):
    totals = {}
    for group in panel.get("child_plot_rows", []):
        for piece in group["events"]:
            key = (group["lane"], piece["label"])
            total = totals.setdefault(key, [LANE_LABELS.get(group["lane"], group["lane"]), piece["label"], 0, 0.])
            total[2] += 1
            total[3] += piece["duration_ms"]
    return list(totals.values())


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


def append_swimlanes(sheet, row, summary, *, note, table, progress=None):
    """All diagrams precede the full child tables, with internal navigation."""
    panels = _ordered(summary)
    row = note(sheet, row, "Host / HC / U 同轮泳道 · " + str(summary.get("rank", "")), columns=24)
    row = note(sheet, row, SCOPE, columns=24)
    if summary.get("input_json"):
        row = note(sheet, row, "泳道来源：" + str(summary["input_json"]), columns=24)
    if not panels:
        return note(sheet, row, "没有可绘制case，未构造数据。", warning=True, columns=24)
    axes = {key: summary[key] for key in ("x_min_ms", "x_max_ms")}
    row = note(sheet, row, "先读业务机制示意，再对照实测时间；下图横向不表示时间，也不补造EngineCore/输出线程实测。", columns=24)
    row = _drawing(sheet, row, mechanism_drawing())
    overview_links = []
    row = note(sheet, row, "全部case总览 · 按BS、S排序；点击case进入放大泳道", columns=24)
    for offset in range(0, len(panels), 3):
        if progress:
            progress("泳道总览 " + str(offset+1) + "–" + str(min(offset+3, len(panels))) + "/" + str(len(panels)))
        top, end = row, row
        for column, panel in enumerate(panels[offset:offset + 3]):
            left, right = 1 + column * 8, 9 + column * 8
            label = str(panel.get("case", "unknown")) + " · 放大图"
            sheet.merge(top, left, top, right - 1, label, "link").row_height(top, 25)
            overview_links.append((top, left, label, offset + column))
            drawing = panel_drawing(panel, width=620, overview=True, **axes)
            end = max(end, _drawing(sheet, top + 1, drawing, left, right))
        row = end + 1
    row = note(sheet, row, "各case放大泳道 · 父范围＋互斥子色带；下方逐项列出U15起止来源，未知不补线", columns=24)
    detail_positions, table_links = {}, []
    for index, panel in enumerate(panels):
        if progress:
            progress("泳道放大图 " + str(index+1) + "/" + str(len(panels)) + "：" + str(panel.get("case")))
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
        if progress:
            progress("泳道调用明细 " + str(index+1) + "/" + str(len(panels)) + "：" + str(panel.get("case"))
                     + "；记录=" + str(len(panel.get("detail_rows") or [])))
        table_positions[index] = row
        sheet.merge(row, 1, row, 19, str(panel.get("case", "unknown")) + " · 源调用明细", "header").row_height(row, 30)
        sheet.merge(row, 20, row, 24, "返回该case放大图", "link")
        sheet.add_internal_link(row, 20, "返回该case放大图", sheet.name, "A" + str(detail_positions[index]))
        row += 1
        reason = panel.get("reason") or panel.get("reasons")
        if reason:
            row = note(sheet, row, str(reason), warning=True, columns=24)
        business = panel.get("worker_business") or {}
        row = note(sheet,row,"Worker主线程互斥业务线 · 外层三个方法及间隙串接；未知间隙不是idle，所跨Host范围不等于设备整轮。",columns=24)
        row = table(sheet,row,["业务步骤", "所属Host轮", "开始(ms)", "结束(ms)", "wall(ms)", "event / 来源", "归属依据"],
                    [[p["label"],p["role"],p["start_ms"],p["end_ms"],p["duration_ms"],str(p.get("source_event_indexes",[])),p.get("selection_reason")]
                     for p in business.get("pieces",[])],
                    spans=((1,7),(8,10),(11,12),(13,14),(15,16),(17,20),(21,24)),blank_columns=(3,4,5))
        for observation in business.get("trace_observations",[]):
            row = note(sheet,row,"本轮时间证据："+str(observation.get("scope"))+"；Host返回早于 "+str(observation.get("device_role"))+" 包络结束 "
                       +str(observation.get("host_return_before_hc_end_ms"))+" ms（负数表示较晚；仅时间关系，不证明某个wait依赖）。",columns=24)
        if business.get("diagnostics"):
            row=note(sheet,row,"Worker业务归属诊断："+str(business["diagnostics"]),warning=True,columns=24)
        row = note(sheet, row, "互斥子阶段分账 · 下列子段可按父范围相加；父条自身不可再次相加。#编号对应该父色带。", columns=24)
        row = table(sheet, row, ["父范围", "子段", "互斥函数/其余wall", "开始(ms)", "结束(ms)", "wall(ms)", "父wall占比(%)", "归属依据"],
                    _exclusive_values(panel),
                    spans=((1, 3), (4, 4), (5, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 24)),
                    blank_columns=(4, 5, 6, 7))
        for role, partition in (panel.get("exclusive_partitions") or {}).items():
            row = note(sheet, row, LANE_LABELS.get(role, role) + "：父wall=" + f"{partition['parent_duration_ns']/1e6:.6f} ms"
                       + "；互斥和=" + f"{partition['total_duration_ns']/1e6:.6f} ms"
                       + "；未细分=" + f"{partition['parent_remainder_ns']/1e6:.6f} ms"
                       + "；归属不明=" + f"{partition['ambiguous_duration_ns']/1e6:.6f} ms"
                       + "；closed=" + str(partition["closed"]), columns=24)
        row = note(sheet, row, "U15全边界来源审计 · 15阶段×起止=30项，不是30次下发；相邻阶段可共用同一task端点。", columns=24)
        row = table(sheet, row, ["目的边界", "设备时刻(ms)", "物理task端点", "CSV行", "CPU/API来源", "源开始(ms)", "源结束(ms)",
                                "来源证据", "Python归属", "端点/缺失说明"], _anchor_values(panel),
                    spans=((1, 2), (3, 4), (5, 7), (8, 9), (10, 13), (14, 15), (16, 17), (18, 20), (21, 22), (23, 24)),
                    blank_columns=(2, 6, 7))
        row = note(sheet, row, "原始父/子嵌套调用 · 仅供回查，不可将wall列相加；图中已改为互斥分账。", columns=24)
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
            row = note(sheet, row, "原始函数归属候选 · 仅时间包含不是下发证据；图上用全边界审计取代每Host一例", columns=24)
            names = {item.get("event_index"): item.get("name") for item in values if item.get("event_index") is not None}
            row = table(sheet, row, ["Host函数/event", "原CSV任务行", "U阶段/边界", "Host来源(ms)", "设备锚点(ms)",
                                    "设备来源状态", "Host关系", "画线证据"],
                        [[names.get(edge.get("host_event_index"), "event=" + str(edge.get("host_event_index"))),
                          edge.get("row_index"), ("U" + str(edge["stage_number"]) if edge.get("stage_number") else edge.get("anchor_kind")),
                          edge.get("source_time_ms"), edge.get("anchor_time_ms"), edge.get("source_status"), edge.get("host_relation"),
                          "已证实提交" if edge.get("runtime_submission_confirmed") else "候选：未证实/来源不完整"] for edge in edges],
                        spans=((1, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 18), (19, 21), (22, 24)),
                        blank_columns=(2, 4, 5))
    for top, left, label, index in table_links:
        sheet.add_internal_link(top, left, label, sheet.name, "A" + str(table_positions[index]))
    return row + 1


def html_section(summary):
    panels = _ordered(summary)
    esc = lambda value: html.escape(str(value), quote=True)
    axes = {key: summary[key] for key in ("x_min_ms", "x_max_ms")}
    def html_table(headers, rows):
        return ('<table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers)
                + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(v if v is not None else '—')+'</td>'
                    for v in row)+'</tr>' for row in rows) + '</tbody></table>')
    parts = ['<section class="swimlane-section"><h2>Host / HC / U 同轮泳道</h2><p>' + esc(SCOPE) + '</p>',
             '<style>.swimlane-grid{display:grid;grid-template-columns:repeat(3,minmax(520px,1fr));gap:18px;overflow:auto}'
             '.swimlane-detail{overflow:auto;margin:24px 0}.swimlane-detail svg{min-width:1400px;width:100%;height:auto}'
             '.swimlane-grid svg{width:100%;height:auto}.swimlane-detail table{font-size:12px}.swimlane-detail td{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
             '<div class="swimlane-grid">']
    parts.insert(-1, '<h3>异步业务关系（非时间比例）</h3>'+render_svg(mechanism_drawing()))
    for index, panel in enumerate(panels):
        parts.append('<article><a href="#swimlane-' + str(index) + '">' + esc(panel.get("case")) + ' · 放大图</a>'
                     + render_svg(panel_drawing(panel, width=620, overview=True, **axes)) + '</article>')
    parts.append('</div>')
    for index, panel in enumerate(panels):
        parts.append('<article class="swimlane-detail" id="swimlane-' + str(index) + '"><h3>' + esc(panel.get("case")) + '</h3>'
                     + render_svg(panel_drawing(panel, width=1600, overview=False, **axes)))
        business=panel.get("worker_business") or {}
        parts.append('<h4>Worker连续业务线（未知间隙不等于idle）</h4>'
                     +html_table(('业务步骤','所属Host轮','开始(ms)','结束(ms)','wall(ms)','来源events','归属依据'),
                         [[p['label'],p['role'],p['start_ms'],p['end_ms'],p['duration_ms'],p.get('source_event_indexes'),p.get('selection_reason')]
                          for p in business.get('pieces',[])]))
        parts.append('<h4>互斥子阶段汇总（同父内可相加，父条不可再次相加）</h4>'
                     + html_table(('父范围', '函数/其余wall', '互斥片段数', 'wall(ms)'), _exclusive_totals(panel)))
        parts.append('<details><summary>互斥子色带逐段精确时间（#编号在各父范围内独立）</summary>'
                     + html_table(('父范围', '子段', '互斥函数', '开始(ms)', '结束(ms)', 'wall(ms)', '父wall占比(%)', '归属依据'),
                                  _exclusive_values(panel))+'</details>')
        parts.append('<h4>U15全边界来源审计：30项，不等于30次下发</h4>'
                     + html_table(('目的边界', '设备时刻(ms)', '物理task端点', 'CSV行', 'CPU/API来源', '源开始(ms)', '源结束(ms)',
                                   '来源证据', 'Python归属', '端点/缺失说明'), _anchor_values(panel)))
        parts.append('<details><summary>原始父/子嵌套调用（不可相加）</summary><table><thead><tr>'
                     + ''.join('<th>' + name + '</th>' for name in ('角色', '函数', 'event_index', '开始(ms)', '结束(ms)', 'wall(ms)', '层级', '归属证据'))
                     + '</tr></thead><tbody>')
        for item in panel.get("detail_rows") or []:
            parts.append('<tr>' + ''.join('<td>' + esc(value if value is not None else '—') + '</td>'
                for value in _detail_values(item)) + '</tr>')
        parts.append('</tbody></table></details></article>')
    parts.append('</section>')
    return ''.join(parts)
