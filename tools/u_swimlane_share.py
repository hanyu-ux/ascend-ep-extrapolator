#!/usr/bin/env python3
"""Make copy/paste PPT evidence from an existing full swimlane JSON.

Reads only the saved u_swimlane JSON. Never opens kernel CSVs or traces, starts
workers or accesses accelerator software. Keeps one original representative
round per case with complete intervals, children and referenced source proof.
The full local export remains unchanged; this packet is not full run history.
"""

import argparse
import base64
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_ep128 as core
import ep128_swimlane_trace as host_analysis
from ep128_swimlane_share import build_ppt_evidence
import u_swimlane_probe as probe


TEXT_CHUNK = 40000


def _stamp(path):
    info = core._io_path(path).stat()
    return info.st_size, info.st_mtime_ns


def read_source(path):
    """Hash the existing JSON without retaining a second 300-MB byte copy."""
    before = _stamp(path)
    digest = hashlib.sha256()
    with core._io_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    with core._io_path(path).open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if _stamp(path) != before:
        raise ValueError("Source JSON changed while reading")
    if not isinstance(payload, dict) or payload.get("schema") not in ("u_swimlane_v2", "u_swimlane_v3"):
        raise ValueError("Provide the existing full u_swimlane_v2/v3 JSON, not a text part or cases manifest")
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("Source swimlane JSON has no cases")
    return payload, {"sha256": digest.hexdigest(), "bytes": before[0], "schema": payload["schema"]}, before


def write_packet(payload, out, rank):
    """Remove no fields to satisfy a byte limit; split all representative bytes."""
    stem = "u_swimlane_ppt_" + rank
    raw = (json.dumps(probe.wire(payload), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")) + "\n").encode("utf-8")
    source_sha = hashlib.sha256(raw).hexdigest()
    with core.atomic_writer(core._io_path(out / (stem + ".json"))) as handle:
        handle.write(raw.decode("utf-8"))
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    data = base64.b64encode(packed).decode("ascii")
    packed_sha = hashlib.sha256(packed).hexdigest()
    count = max(1, (len(data) + TEXT_CHUNK - 1) // TEXT_CHUNK)
    directory = out / (stem + "_parts_" + source_sha[:12])
    core._io_path(directory).mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(count):
        record = {"schema": "u_swimlane_ppt_gzip_base64_v1", "part": index + 1,
                  "parts": count, "raw_bytes": len(raw), "raw_sha256": source_sha,
                  "gzip_sha256": packed_sha, "encoding": "gzip+base64",
                  "data": data[index * TEXT_CHUNK:(index + 1) * TEXT_CHUNK]}
        blob = (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")
        if len(blob) > 65536:
            raise ValueError("Text evidence part exceeds 64 KiB")
        target = directory / ("part" + str(index + 1).zfill(3) + ".txt")
        core._io_path(target).write_bytes(blob)
        names.append(target.relative_to(out).as_posix())
    package = {"schema": "u_swimlane_ppt_package_v1", "parts": names, "part_count": count,
               "raw_bytes": len(raw), "raw_sha256": source_sha, "gzip_sha256": packed_sha,
               "scope": "One representative round per case, not full history; no selected interval is clipped",
               "truncated": False, "clipboard_required": False,
               "send": "Display each text part in the terminal and copy its complete JSON content using your available text-transfer method"}
    core.atomic_json(core._io_path(out / (stem + "_package.json")), package)
    return package


def write_status(payload, out, rank):
    fields = ("case", "rank", "bs", "S", "status", "representative_status", "source_case_status", "selected_round_id",
              "source_round_count", "omitted_round_count", "trace_status", "error", "share_error")
    path = out / ("u_swimlane_ppt_" + rank + "_status.csv")
    with core._io_path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in payload["cases"]:
            values = {key: row.get(key) for key in fields}
            values["trace_status"] = (row.get("trace") or {}).get("status")
            writer.writerow(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing u_swimlane_dp4.json, not cases JSON or a partNNN.txt")
    parser.add_argument("--rank", default="dp4")
    parser.add_argument("--expected-cases", type=core._cli_positive_workers)
    parser.add_argument("--out", type=Path, default=Path("u_swimlane_ppt_dp4"))
    args = parser.parse_args(argv)
    if not args.rank or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for c in args.rank):
        parser.error("--rank must be a simple rank identifier")
    source, out = core._source_path(args.source), core._source_path(args.out)
    stem = "u_swimlane_ppt_" + args.rank
    try:
        relative = source.relative_to(out)
    except ValueError:
        relative = None
    if relative is not None and (source.parent == out and source.name.casefold().startswith(stem.casefold())
                                or relative.parts and relative.parts[0].casefold().startswith(stem.casefold() + "_parts_")):
        parser.error("Output directory would overwrite the source JSON")
    print("Reading existing swimlane JSON only; no CSV/trace scan or new capture", flush=True)
    try:
        payload, info, before = read_source(source)
        if args.expected_cases is not None and len(payload["cases"]) != args.expected_cases:
            raise ValueError("Expected " + str(args.expected_cases) + " cases, found " + str(len(payload["cases"])))
        result = build_ppt_evidence(payload, core, host_analysis, probe.attach_host, info,
                                    rank=args.rank, progress=lambda s: print(s, flush=True))
        if _stamp(source) != before:
            raise ValueError("Source JSON changed while making the representative packet")
        core._io_path(out).mkdir(parents=True, exist_ok=True)
        package = write_packet(result, out, args.rank)
        write_status(result, out, args.rank)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        parser.error(type(error).__name__ + ": " + str(error))
    print("Cases retained: " + str(len(result["cases"])) + "; text parts: " + str(package["part_count"]), flush=True)
    print("Representative status counts: " + json.dumps(result.get("representative_status_counts", {}),
                                                       ensure_ascii=True, sort_keys=True), flush=True)
    print("Representative evidence bytes: " + str(package["raw_bytes"]), flush=True)
    print("Text part list: " + str(out / (stem + "_package.json")), flush=True)
    print("First text part: " + str(out / package["parts"][0]), flush=True)
    print("Original full JSON is unchanged. Do not send its hundreds of full-history parts.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
