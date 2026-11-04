#!/usr/bin/env python3
"""Validate another scenario's communication table and export observed EP ratios.

Example:
  python3 profile_ep128_comm_reference.py ep_comm_reference_20261028/raw.csv \
      --out ep128_empirical_report

Standard library only. No model, Torch, NPU, network, or trace processing is used.
This reports the supplied measurements; it does not predict EP16 or TPOT.
"""

import argparse
import csv
import html
import itertools
import json
import math
from pathlib import Path
import re
import sys


PARTS = ("send", "return", "total")
EP_COLUMN = re.compile(r"ep([1-9][0-9]*)_(send|return|total)\Z")


def read_measurements(path):
    """Return sorted EP sizes and rows, rejecting incomplete or invalid data."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise ValueError("输入 CSV 没有表头")
        if len(fields) != len(set(fields)):
            raise ValueError("CSV 表头包含重复字段")
        if "bs" not in fields:
            raise ValueError("CSV 表头缺少 bs")
        eps = set()
        for field in fields:
            if field == "bs":
                continue
            match = EP_COLUMN.fullmatch(field)
            if not match:
                raise ValueError(f"不支持的表头字段 {field!r}；应为 ep32_send 等 epN_send/return/total")
            eps.add(int(match.group(1)))
        eps = sorted(eps)
        for ep in eps:
            missing = [f"ep{ep}_{part}" for part in PARTS
                       if f"ep{ep}_{part}" not in fields]
            if missing:
                raise ValueError("CSV 表头缺少：" + ", ".join(missing))
        if len(eps) < 2:
            raise ValueError("至少需要两个 EP 规模，每个规模均须有 send、return、total 三列")
        rows = []
        seen_bs = set()
        for source in reader:
            line = reader.line_num
            if None in source:
                raise ValueError(f"第 {line} 行的字段数超过表头")
            bs_text = (source.get("bs") or "").strip()
            if not re.fullmatch(r"[1-9][0-9]*", bs_text):
                raise ValueError(f"第 {line} 行 bs 必须是正整数，实际为 {bs_text!r}")
            bs = int(bs_text)
            if bs in seen_bs:
                raise ValueError(f"第 {line} 行 bs={bs} 重复")
            seen_bs.add(bs)
            row = {"bs": bs}
            for ep in eps:
                for part in PARTS:
                    key = f"ep{ep}_{part}"
                    try:
                        value = float(source[key])
                    except (TypeError, ValueError):
                        raise ValueError(f"第 {line} 行 {key} 必须是有限正数") from None
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"第 {line} 行 {key} 必须是有限正数，实际为 {source[key]!r}")
                    row[key] = value
                send = row[f"ep{ep}_send"]
                returned = row[f"ep{ep}_return"]
                total = row[f"ep{ep}_total"]
                if not math.isclose(send + returned, total, rel_tol=1e-7, abs_tol=1e-6):
                    raise ValueError(f"第 {line} 行 bs={bs}、EP{ep}：send + return != total "
                                     f"({send} + {returned} != {total})")
            rows.append(row)
    if not rows:
        raise ValueError("输入 CSV 没有数据行")
    return eps, sorted(rows, key=lambda row: row["bs"])


def observed_ratios(eps, rows):
    """Compute each measured direction separately, without extrapolating EP16."""
    pairs = list(itertools.combinations(eps, 2))
    fields = ["bs"] + [f"ep{target}_over_ep{base}_{part}"
                       for base, target in pairs for part in PARTS]
    output = []
    for row in rows:
        result = {"bs": row["bs"]}
        for base, target in pairs:
            for part in PARTS:
                ratio = row[f"ep{target}_{part}"] / row[f"ep{base}_{part}"]
                if not math.isfinite(ratio) or ratio <= 0:
                    raise ValueError(f"bs={row['bs']} EP{base}->EP{target} {part} 倍率溢出或下溢")
                result[f"ep{target}_over_ep{base}_{part}"] = ratio
        output.append(result)
    return fields, output


def theory_payload_ratios(eps, ranks_per_node=8):
    """Same-TopK payload ratios, distinct from the measured latency ratios."""
    result = []
    for base, target in itertools.combinations(eps, 2):
        if base <= ranks_per_node:
            continue
        result.append({
            "base_ep": base,
            "target_ep": target,
            "logical_interhost_payload_ratio": ((1 - ranks_per_node / target)
                                                 / (1 - ranks_per_node / base)),
            "assumption": (f"{ranks_per_node} ranks per node; fixed per-rank tokens/TopK/hidden/dtype; "
                           "uniform routing; per-expert assignment payload; complete nodes"),
        })
    return result


def write_csv(path, fields, rows, ratio_fields=()):
    ratio_fields = set(ratio_fields)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{row[key]:.6f}" if key in ratio_fields
                                   else format(row[key], ".15g") if isinstance(row[key], float)
                                   else row[key]) for key in fields})


def html_table(fields, rows, ratio_fields=()):
    ratio_fields = set(ratio_fields)
    heading = "".join("<th>" + html.escape(key) + "</th>" for key in fields)
    body = []
    for row in rows:
        cells = []
        for key in fields:
            value = f"{row[key]:.6f}" if key in ratio_fields else str(row[key])
            cells.append("<td>" + html.escape(value) + "</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<div class='table'><table><thead><tr>" + heading + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def build_report(source, out_dir, unit="unknown", ranks_per_node=8):
    if unit not in ("unknown", "us", "ms"):
        raise ValueError("unit 必须是 unknown、us 或 ms")
    if isinstance(ranks_per_node, bool) or not isinstance(ranks_per_node, int) or ranks_per_node <= 0:
        raise ValueError("ranks_per_node 必须是正整数")
    source = Path(source).expanduser().resolve()
    eps, rows = read_measurements(source)
    ratio_fields, ratios = observed_ratios(eps, rows)
    theory = theory_payload_ratios(eps, ranks_per_node)
    raw_fields = ["bs"] + [f"ep{ep}_{part}" for ep in eps for part in PARTS]
    theory_fields = ["base_ep", "target_ep", "logical_interhost_payload_ratio", "assumption"]
    metadata = {
        "source": str(source),
        "unit_label": unit,
        "unit_conversion_applied": False,
        "ep_sizes": eps,
        "validated_bs_rows": len(rows),
        "validated_send_return_total_groups": len(rows) * len(eps),
        "observed_ratio_columns": len(ratio_fields) - 1,
        "current_target_usable": False,
        "unconfirmed_source_context": ["BS definition", "TopK", "hidden size", "expert intermediate size",
                                       "communication dtype", "hardware", "operator/statistics scope",
                                       "repeat count and dispersion"],
        "current_target_context": {"ep": 128, "topk": 24, "expert_intermediate_size": 4096},
        "target_topk_or_width_scaling_applied": False,
        "extra_latency_us_added": 0,
        "theory_payload_assumptions": {
            "ranks_per_node": ranks_per_node,
            "formula": "(1 - ranks_per_node / target_ep) / (1 - ranks_per_node / base_ep)",
            "same_per_rank_tokens_topk_hidden_dtype": True,
            "uniform_routing": True,
            "complete_nodes": True,
            "is_latency_prediction": False,
        },
    }
    note = (
        f"输入：{source}\n"
        f"时间单位标签：{unit}。仅标注，不转换；默认 unknown 表示未确认。\n"
        f"校验通过：{len(rows)} 个 BS，{len(eps)} 个 EP 规模，"
        f"共 {len(rows) * len(eps)} 组 send + return = total；所有测量值均为有限正数。\n"
        "send/return 保留源表称呼，具体对应的算子、统计范围和 BS 定义尚未确认。\n"
        "当前目标为 EP128、TopK=24、专家中间维度 4096；源表 TopK、hidden、专家宽度、dtype 和硬件未确认。\n"
        "current_target_usable=false：这些倍率仅为其他场景实测参考，暂不能直接用于当前目标时延预测。\n"
        "observed_ratios.csv 仅给输入表中实际存在的 EP 规模之间的倍率；没有推造缺失的 EP16。\n"
        "send 和 return 分别计算倍率；total 倍率不能作为两者共同的统一倍率。\n"
        "若口径确认可迁移，应分别使用 target_send=base_send*Rsend(BS)、"
        "target_return=base_return*Rreturn(BS)，再求和；本脚本不自动迁移到当前目标。\n"
        "实测耗时已含该场景原有固定和同步开销；不额外增加 14us，不擅自乘 TopK=24 的倍率。\n"
        "theory_payload_ratios.csv 是独立的同 TopK 理论跨机逻辑数据量比，不是观测耗时比或时延模型。\n"
        f"理论比假设：每节点 {ranks_per_node} rank、完整节点、均匀路由、每 rank token/TopK/hidden/dtype 不变；"
        "仅对源表中 base_ep > ranks_per_node 的规模对计算。\n"
        "两种指标应按同一起点比较；理论数据量比不能单独验证或否定观测时延。\n"
        "投机解码须按每次 MoE 调用的实际 token 数映射 BS，主模型与草稿分别处理。\n"
        "本脚本不读取 trace，不产生完整 ITL/TPOT 预测；未提供重复采样，不能判断非单调点的显著性。\n"
    )
    report = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>其他场景实测通信参考</title>"
        "<style>body{font:15px/1.6 system-ui,sans-serif;margin:28px;max-width:1500px}"
        ".notice{padding:12px 16px;background:#fff3cd;border:1px solid #e1c574}"
        ".table{overflow-x:auto}table{border-collapse:collapse;margin:16px 0;font-size:13px}"
        "th,td{border:1px solid #ccc;padding:5px 9px;text-align:right;white-space:nowrap}"
        "th{background:#eef1f5}pre{white-space:pre-wrap}</style>"
        "<h1>其他场景实测通信参考</h1>"
        "<p class='notice'><strong>current_target_usable=false</strong>：源表口径未确认，"
        "暂不能直接预测当前 EP128 / TopK=24 / 专家中间维度 4096 的时延。</p>"
        f"<p>输入：{html.escape(str(source))}<br>单位标签：{html.escape(unit)}（未转换）；"
        f"{len(rows)} 个 BS，{len(rows) * len(eps)} 组和式校验通过。</p>"
        "<h2>源表校验结果</h2>" + html_table(raw_fields, rows)
        + "<h2>直接观测到的耗时倍率</h2><p>发送、收回、合计各自取比；未补 EP16、未加 14us、未乘 TopK 倍率。</p>"
        + html_table(ratio_fields, ratios, ratio_fields[1:])
        + "<h2>同 TopK 理论数据量比（独立对照）</h2><p>以下是逻辑数据量比，不能作为实测耗时倍率。</p>"
        + html_table(theory_fields, theory, ["logical_interhost_payload_ratio"])
        + "<h2>口径与限制</h2><pre>" + html.escape(note) + "</pre></html>"
    )
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "raw_checked.csv", raw_fields, rows)
    write_csv(out_dir / "observed_ratios.csv", ratio_fields, ratios, ratio_fields[1:])
    write_csv(out_dir / "theory_payload_ratios.csv", theory_fields, theory,
              ["logical_interhost_payload_ratio"])
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "说明.txt").write_text(note, encoding="utf-8")
    (out_dir / "report.html").write_text(report, encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="其他场景实测通信表：bs、epN_send/return/total")
    parser.add_argument("--out", type=Path, default=Path("ep128_empirical_report"), help="输出目录")
    parser.add_argument("--unit", choices=("unknown", "us", "ms"), default="unknown",
                        help="仅标注时间单位，不转换；默认 unknown")
    parser.add_argument("--ranks-per-node", type=int, default=8,
                        help="独立理论数据量对照采用的每节点 rank 数，默认 8")
    args = parser.parse_args(argv)
    try:
        metadata = build_report(args.csv, args.out, args.unit, args.ranks_per_node)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(f"校验通过：{metadata['validated_bs_rows']} 个 BS，"
          f"{metadata['validated_send_return_total_groups']} 组 send+return=total。")
    print(f"已输出：{args.out.resolve() / 'report.html'}")
    print("current_target_usable=false：源表口径尚未确认，仅作其他场景参考。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
