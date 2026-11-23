"""Compact source-U charts from saved report means; never read a capture."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re


WINDOWS = (("main_to_draft", 8, 0), ("draft_to_nextmain", 7, 8))
SCOPE = ("源实测U：仅统计设备锚点之间的15个父阶段，包含任务间隔；不是CPU函数独占耗时。"
         "按耗时降序取前N项，其余合计；所有占比均为阶段均值÷整个U，不对前N项重新归一化。"
         "U只保留一次，不按外推层数放大；未识别、缺失或不闭合的数据不补零。")
RANK_BASIS_LABELS = {"fixed_rank": "命令指定固定rank", "saved_source_rank": "报告保存的源rank",
                     "saved_method_rank": "报告各方法共同的rank", "only_captured_rank": "唯一已采rank"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def _rank(case, requested):
    if requested is not None:
        return requested, "fixed_rank"
    aggregate = (case.get("source") or {}).get("aggregate") or {}
    if aggregate.get("selected_rank") is not None:
        return aggregate["selected_rank"], "saved_source_rank"
    methods = (case.get("projection") or {}).get("layer_mean_methods") or []
    ranks = {method["selected_rank"] for method in methods if method.get("selected_rank") is not None}
    if len(ranks) == 1:
        return next(iter(ranks)), "saved_method_rank"
    if len(ranks) > 1:
        raise ValueError("旧报告各方法所选rank不一致；请显式指定 --u-hotspots-rank")
    captures = case.get("measurements") or []
    if len(captures) == 1 and captures[0].get("rank") is not None:
        return captures[0]["rank"], "only_captured_rank"
    raise ValueError("报告没有唯一已保存的源rank；请显式指定 --u-hotspots-rank")


def _parents(diagnostic):
    """Require each complete window to close before ranking any of its parents."""
    total = diagnostic.get("residual_ms_mean")
    if not _number(total):
        raise ValueError("源U均值缺失或无效")
    parents, counts, window_totals = [], {}, []
    all_stages = diagnostic.get("stage_summary") or []
    if any(row.get("window_key") not in dict((key, count) for key, count, _ in WINDOWS)
           for row in all_stages if row.get("level", 0) == 0 and not row.get("parent_stage_key")):
        raise ValueError("父阶段含未知窗口，不能构造15阶段占比")
    for key, expected, offset in WINDOWS:
        windows = [row for row in diagnostic.get("summary") or [] if row.get("window_key") == key]
        if len(windows) != 1:
            raise ValueError(key + " 缺少唯一窗口统计")
        window = windows[0]
        rounds = window.get("rounds")
        value = window.get("duration_ms_mean")
        if type(rounds) is not int or rounds <= 0 or not _number(value):
            raise ValueError(key + " 窗口均值或样本数不可用")
        stages = [row for row in all_stages if row.get("window_key") == key
                  and row.get("level", 0) == 0 and not row.get("parent_stage_key")]
        if (len(stages) != expected or any(type(row.get("stage_order")) is not int for row in stages)
                or sorted(row["stage_order"] for row in stages) != list(range(1, expected + 1))
                or any(not isinstance(row.get("stage_key"), str) or not row["stage_key"] for row in stages)
                or len({row["stage_key"] for row in stages}) != expected):
            raise ValueError(key + " 未提供完整有序的" + str(expected) + "个父阶段；不猜补缺失阶段")
        if any(row.get("status") != "MATCHED" or row.get("rounds") != rounds
               or row.get("matched_windows") != rounds or not _number(row.get("duration_ms_mean"))
               for row in stages):
            raise ValueError(key + " 父阶段未全部匹配或样本数与窗口不一致")
        if not _close(math.fsum(row["duration_ms_mean"] for row in stages), value):
            raise ValueError(key + " 父阶段合计与窗口耗时不闭合")
        counts[key] = rounds
        window_totals.append(value)
        for row in stages:
            number = offset + row["stage_order"]
            parents.append(dict(stage_number=number, stage_key=row["stage_key"],
                label="U" + str(number) + " · " + str(row.get("label") or row["stage_key"]),
                value_ms=row["duration_ms_mean"], kind="stage", members=[number]))
    if len(set(counts.values())) != 1:
        raise ValueError("两段窗口的保留轮次数不一致；不能拼为同一组15阶段U均值")
    if not _close(math.fsum(window_totals), total):
        raise ValueError("两段窗口与源U均值不闭合；不能计算占比")
    if total == 0:
        raise ValueError("U=0，占比无定义；保留零值但不绘制占比图")
    return total, parents, counts


def summarize_u_hotspots(report, top=5, rank=None, names=None):
    """Read source-only means without extrapolating, reselecting or pooling ranks."""
    if type(top) is not int or not 1 <= top <= 15:
        raise ValueError("--u-top必须为1..15的整数")
    if not isinstance(report, dict) or report.get("schema_version") != 2 or not isinstance(report.get("cases"), list):
        raise ValueError("请提供schema_version=2的model_architecture_report.json，不是cases配置或trace")
    selected = set(names or [])
    present = {case.get("name") for case in report["cases"]}
    if selected - present:
        raise ValueError("报告中找不到case：" + ", ".join(sorted(selected - present)))
    panels = []
    for case in report["cases"]:
        if selected and case.get("name") not in selected:
            continue
        config = case.get("config") or {}
        panel = dict(case=case.get("name", "unnamed"), rank=rank, bs=config.get("bs"),
            S=config.get("draft_tokens", config.get("num_speculative_tokens")),
            status="UNAVAILABLE", reason="", u_ms=None, rows=[], top_share=None,
            rank_basis=None, window_rounds={}, top=top)
        try:
            panel["rank"], panel["rank_basis"] = _rank(case, rank)
            captures = [capture for capture in case.get("measurements") or []
                        if capture.get("rank") == panel["rank"]]
            if len(captures) != 1:
                raise ValueError("此rank缺少唯一采样统计；不借用其他rank")
            diagnostic = captures[0].get("step_residuals")
            if not isinstance(diagnostic, dict):
                raise ValueError("已有报告缺少U阶段统计；重绘不能补出实测阶段")
            if diagnostic.get("rank", panel["rank"]) != panel["rank"]:
                raise ValueError("阶段统计rank与采样rank不一致")
            if _number(diagnostic.get("residual_ms_mean")):
                panel["u_ms"] = diagnostic["residual_ms_mean"]
            total, parents, counts = _parents(diagnostic)
            ordered = sorted(parents, key=lambda row: (-row["value_ms"], row["stage_number"]))
            rows = [dict(row) for row in ordered[:top]]
            if len(ordered) > top:
                omitted = ordered[top:]
                rows.append(dict(stage_number=None, stage_key="other", kind="other",
                    label="其余" + str(len(omitted)) + "阶段合计",
                    value_ms=math.fsum(row["value_ms"] for row in omitted),
                    members=sorted(row["stage_number"] for row in omitted)))
            for row in rows:
                row["share_u"] = row["value_ms"] / total
                row["label_text"] = f'{row["value_ms"]:.3f} ms · {row["share_u"]:.1%}'
            panel.update(status="MATCHED", u_ms=total, rows=rows, window_rounds=counts,
                top_share=math.fsum(row["share_u"] for row in rows if row["kind"] == "stage"))
        except ValueError as error:
            panel["reason"] = str(error)
        panels.append(panel)

    def order(panel):
        return tuple((not _number(panel[key]), panel[key] if _number(panel[key]) else 0)
                     for key in ("S", "bs")) + (str(panel["case"]),)

    return sorted(panels, key=order)


def _esc(value):
    return html.escape(str(value), quote=True)


def panel_title(panel):
    return f'{panel["case"]} / {panel["rank"]} · BS{panel["bs"]} / S{panel["S"]}'


def hotspot_svg(panel, axis_max=None):
    """Editable vector chart; all panels in an export share the same ms axis."""
    rows = panel["rows"]
    width, left, plot, row_h = 1200, 480, 465, 74
    height = 155 + max(len(rows), 1) * row_h
    largest = max((row["value_ms"] for row in rows), default=0)
    maximum = axis_max if _number(axis_max) and axis_max > 0 else max(largest * 1.08, .001)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<title>' + _esc(panel_title(panel) + "；源实测U主要耗时") + '</title>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<g font-family="Microsoft YaHei,Arial,sans-serif" fill="#18334d">',
        '<text x="22" y="33" font-size="24" font-weight="700">' + _esc(panel_title(panel)) + '</text>']
    if not rows:
        pieces.append('<text x="22" y="90" font-size="19" fill="#9b421a">未绘图：' + _esc(panel["reason"]) + '</text>')
    else:
        pieces.append(f'<text x="22" y="63" font-size="18">U = {panel["u_ms"]:.3f} ms；前{panel["top"]}项占U {panel["top_share"]:.1%}；横轴为ms</text>')
        for tick in range(5):
            x = left + plot * tick / 4
            pieces.append(f'<path d="M{x:.2f} 88V{height-53}" stroke="#e1e8ef"/>')
            pieces.append(f'<text x="{x:.2f}" y="{height-27}" text-anchor="middle" font-size="16">{maximum*tick/4:.2f}</text>')
        for index, row in enumerate(rows):
            y = 91 + index * row_h
            label = row["label"]
            for part in range((len(label) + 25) // 26):
                # The tooltip and CSV retain even pathological long source labels.
                if part == 2:
                    pieces.append(f'<text x="22" y="{y+49}" font-size="18">…（完整名称见CSV）</text>')
                    break
                pieces.append(f'<text x="22" y="{y+21+part*23}" font-size="18">{_esc(label[part*26:(part+1)*26])}</text>')
            length = plot * row["value_ms"] / maximum
            color = "#9aa9b8" if row["kind"] == "other" else "#168b8d"
            pieces.append(f'<g class="u-hotspot-bar"><title>{_esc(label)}: {_esc(row["label_text"])}</title>'
                f'<rect x="{left}" y="{y}" width="{length:.3f}" height="38" rx="3" fill="{color}"/>'
                f'<text x="{left+length+10:.3f}" y="{y+25}" font-size="19" font-weight="700">{_esc(row["label_text"])}</text></g>')
    pieces.append('</g></svg>')
    return "".join(pieces)


def _xlsx(panels, destination):
    from ep128_xlsx_writer import Workbook

    book = Workbook()
    sheet = book.add_sheet("U主要耗时", widths=[48, 16, 16, 30, 30, 24, 20, 20, 20, 20, 20, 20])
    sheet.merge(1, 1, 1, 12, "源实测U · 主要耗时阶段", "title")
    sheet.merge(2, 1, 3, 12, SCOPE, "note")
    row = 5
    for panel in panels:
        sheet.merge(row, 1, row, 12, panel_title(panel), "section")
        row += 1
        if panel["status"] != "MATCHED":
            sheet.merge(row, 1, row + 2, 12, "未绘图：" + panel["reason"], "warning")
            row += 5
            continue
        sheet.merge(row, 1, row, 12, f'U={panel["u_ms"]:.6f} ms；前{panel["top"]}项占U {panel["top_share"]:.1%}；rank依据：{RANK_BASIS_LABELS.get(panel["rank_basis"], "不可用")}；各图独立刻度', "note")
        row += 1
        for col, label in enumerate(("阶段", "耗时(ms)", "占整个U", "柱上标签", "包含阶段编号"), 1):
            sheet.write(row, col, label, "header")
        start = row + 1
        for item in panel["rows"]:
            row += 1
            for col, value in enumerate((item["label"], item["value_ms"], item["share_u"],
                                         item["label_text"], ",".join(map(str, item["members"]))), 1):
                sheet.write(row, col, value, "percent" if col == 3 else "number" if col == 2 else "body")
        ref = "'U主要耗时'!"
        chart_row = row + 2
        chart_h = max(18, len(panel["rows"]) * 3 + 5)
        sheet.add_bar_chart("U主要耗时 · " + panel_title(panel),
            categories=[item["label"] for item in panel["rows"]],
            categories_ref=ref + f"$A${start}:$A${row}",
            series=[dict(name="阶段均值(ms)", values=[item["value_ms"] for item in panel["rows"]], color="168B8D",
                values_ref=ref + f"$B${start}:$B${row}", data_labels=[item["label_text"] for item in panel["rows"]],
                data_labels_ref=ref + f"$D${start}:$D${row}")],
            anchor=(chart_row, 1, chart_row + chart_h, 12), stacked=False,
            x_axis_title="阶段耗时均值(ms)；标签占比=阶段/整个U")
        row = chart_row + chart_h + 3
    book.save(destination)


def export_u_hotspots(report, out, top=5, rank=None, names=None):
    """Write a small standalone view, CSV, native Excel charts and SVGs."""
    panels = summarize_u_hotspots(report, top=top, rank=rank, names=names)
    out = Path(out)
    svg_dir = out / "u_hotspots_svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(html=out / "u_hotspots.html", csv=out / "u_hotspots.csv", xlsx=out / "u_hotspots.xlsx", svg=[], panels=panels)
    maximum = max((row["value_ms"] for panel in panels for row in panel["rows"]), default=0) * 1.08
    sections = []
    for panel in panels:
        image = hotspot_svg(panel, maximum)
        identity = str(panel["case"]) + "\0" + str(panel["rank"])
        stem = re.sub(r"[^\w.-]+", "_", str(panel["case"]) + "-" + str(panel["rank"]))[:70]
        path = svg_dir / (stem + "-" + hashlib.sha256(identity.encode()).hexdigest()[:8] + ".svg")
        path.write_text(image, encoding="utf-8")
        paths["svg"].append(path)
        window_labels = {"main_to_draft": "主→草稿", "draft_to_nextmain": "草稿→下一主"}
        sample_note = "；".join(f'{window_labels[key]}: {count}轮' for key, count in panel["window_rounds"].items())
        sections.append('<article>' + image + '<p>rank依据：' + _esc(RANK_BASIS_LABELS.get(panel["rank_basis"], "不可用")) + '；' + _esc(sample_note)
                        + '。数值为已保存的阶段均值；非单个代表轮，未重新配对轮次。</p></article>')
    notice = ("固定rank=" + rank + "，各case不另选rank。" if rank is not None else
              "沿用每个case已保存的源rank；可能因case而异。固定rank横比请加 --u-hotspots-rank dp4。")
    paths["html"].write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>U主要耗时</title>'
        '<style>body{font:16px "Microsoft YaHei",system-ui;margin:28px;color:#18334d;background:#f2f6fa}'
        'p{line-height:1.7}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}'
        'article{background:white;padding:12px;border:1px solid #dbe4ec;border-radius:8px}svg{width:100%;height:auto}'
        'article p{font-size:12px;margin:6px 12px}@media(max-width:1250px){main{grid-template-columns:1fr}}'
        '@media print{main{display:block}article{break-inside:avoid}}</style><h1>U主要耗时 · 前'
        + str(top) + '项＋其余合计</h1><p>' + _esc(SCOPE) + '</p><p>' + _esc(notice)
        + ' 横轴为统一毫秒尺度；灰条为其余阶段合计，不是新增阶段。不同模型等条件仍须核对可比性。</p><main>'
        + ''.join(sections) + '</main></html>', encoding="utf-8")
    fields = ("case", "rank", "bs", "S", "status", "reason", "rank_basis", "u_ms", "kind", "stage_number", "stage_key", "label", "value_ms", "share_u", "label_text", "members")
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for panel in panels:
            context = {key: panel[key] for key in fields[:8]}
            for row in panel["rows"] or [{}]:
                writer.writerow({**context, **row, "members": ",".join(map(str, row.get("members", [])))})
    _xlsx(panels, paths["xlsx"])
    return paths
