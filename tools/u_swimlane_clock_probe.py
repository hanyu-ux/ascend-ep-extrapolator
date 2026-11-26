#!/usr/bin/env python3
"""Explain saved fixed-rank clock failures using existing CSVs and traces.

This is an offline provenance probe, not a clock repair or a sampling launcher.
The exact core anchor selection, identity rejection and tolerance are unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_ep128 as core
from tools.u_overlap_dp4_s15_probe import diagnose_clock


def _paths(config, manifest, names, rank):
    protected = {manifest}
    for key in ("sample_model", "target_model", "communication_reference"):
        if isinstance(config.get(key), str):
            p = Path(config[key])
            protected.add(core._source_path(p if p.is_absolute() else manifest.parent/p))
    sources = []
    for item in config["cases"]:
        # Protect explicit inputs, including cases that are not selected.
        for source in item.get("inputs", config.get("defaults", {}).get("inputs", [])) or []:
            if not isinstance(source, dict):
                raise ValueError("Explicit inputs must be objects")
            for key in ("path", "trace_path"):
                if isinstance(source.get(key), str):
                    p = Path(source[key])
                    protected.add(core._source_path(p if p.is_absolute() else manifest.parent/p))
        if item["name"] not in names:
            continue
        entry = {"case": item["name"]}
        try:
            case = core.normalize_case(config.get("defaults", {}), item, manifest.parent,
                                       capture_purpose="host_stack_diagnostic")
            kernel = core._source_path(core._fixed_overlap_input(case, rank, manifest.parent))
            trace = core._source_path(core._case_trace_path(case, rank, manifest.parent) or kernel.parent/"trace_view.json")
            protected.update((kernel, trace))
            entry.update(bs=case["bs"], S=case["draft_tokens"], kernel=kernel, trace=trace)
        except (ValueError, TypeError, KeyError, OSError) as error:
            entry.update(status="ERROR", error=type(error).__name__+": "+str(error))
        sources.append(entry)
    return sources, protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="Existing cases manifest; not trace_view.json or a swimlane JSON")
    parser.add_argument("--rank", default="dp4")
    parser.add_argument("--case", dest="names", action="append", help="Default: bs2-s9 and bs16-s9")
    parser.add_argument("--out", type=Path, default=Path("u_clock_dp4_s9.json"))
    args = parser.parse_args(argv)
    manifest = core._source_path(args.cases)
    try:
        config = json.loads(core._io_path(manifest).read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or config.get("schema_version") != core.SCHEMA_VERSION or not isinstance(config.get("cases"), list):
            raise ValueError("Provide the existing schema_version=2 cases JSON")
        names = args.names or ["bs2-s9", "bs16-s9"]
        available = [item.get("name") for item in config["cases"] if isinstance(item, dict)]
        if (len(available) != len(config["cases"]) or any(not isinstance(n, str) or not n for n in available)
                or len(set(available)) != len(available)):
            raise ValueError("Cases require unique names")
        if len(set(names)) != len(names) or set(names)-set(available):
            raise ValueError("Selected --case names must be unique and present in cases JSON")
        if not args.rank or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for c in args.rank):
            raise ValueError("--rank must be a simple identifier")
        sources, protected = _paths(config, manifest, names, args.rank)
        output = core._source_path(args.out)
        if output in protected:
            raise ValueError("Output would overwrite a source input")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    result = {"schema": "u_swimlane_clock_probe_v1", "profile_version": core.PROFILE_VERSION,
              "rank": args.rank, "manifest_sha256": hashlib.sha256(core._io_path(manifest).read_bytes()).hexdigest(),
              "scope": "Existing fixed-rank input clock diagnostics only; no threshold changes, forced alignment or sampling",
              "cases": []}
    for entry in sources:
        row = {k: v for k, v in entry.items() if k not in ("kernel", "trace")}
        print("Clock probe: " + entry["case"] + " / " + args.rank, flush=True)
        if entry.get("status") != "ERROR":
            try:
                capture = core.read_capture(entry["kernel"], args.rank)
                row.update(kernel_sha256=capture["sha256"], kernel_path=str(entry["kernel"]), trace_path=str(entry["trace"]))
                row.update(diagnose_clock(core, capture, entry["trace"]))
            except (OSError, ValueError, KeyError, ArithmeticError, TypeError, RuntimeError) as error:
                row.update(status="ERROR", error=type(error).__name__+": "+str(error))
        result["cases"].append(row)
        print("  " + str(row.get("status")) + "; anchors=" + str(row.get("anchor_count"))
              + "; max residual(ns)=" + str(row.get("max_clock_residual_ns")), flush=True)
    blob = (json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))+"\n").encode("ascii")
    if len(blob) > 65536:
        raise ValueError("Diagnostic exceeds 64 KiB; select fewer cases. No evidence was cropped.")
    core._io_path(output).parent.mkdir(parents=True, exist_ok=True)
    with core.atomic_writer(core._io_path(output)) as handle:
        handle.write(blob.decode("ascii"))
    print(str(output) + " ("+str(len(blob))+" bytes)", flush=True)
    return int(any(row.get("status") != "ALIGNED_X_EVENTS" for row in result["cases"]))


if __name__ == "__main__":
    sys.exit(main())
