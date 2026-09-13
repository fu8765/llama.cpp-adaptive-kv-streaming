#!/usr/bin/env python3
"""Combine the no-spec, MTP and DFlash adaptive-streaming sweeps into one table and graph.

Reads results.jsonl produced by benchmark_upstream_vs_mtp.py (config upstream)
and benchmark_mtp_streaming.py (modes default/keep) and writes a combined CSV
plus separate decode and prefill plots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"

SERIES = [
    ("upstream", "upstream (no spec)", RESULTS / "draft-thresh-nospec" / "results.jsonl", "#1f77b4", "o"),
    ("mtp-eject", "MTP (eject at threshold)", RESULTS / "draft-thresh-mtp" / "results.jsonl", "#d62728", "s"),
    ("mtp-keep", "MTP (keep draft)", RESULTS / "draft-thresh-mtp" / "results.jsonl", "#ff9896", "^"),
    ("dflash-eject", "DFlash2 (eject at threshold)", RESULTS / "draft-thresh-dflash" / "results.jsonl", "#2ca02c", "D"),
    ("dflash-keep", "DFlash2 (keep draft)", RESULTS / "draft-thresh-dflash" / "results.jsonl", "#98df8a", "v"),
]


def load(path: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") != "measurement" or row.get("status") != "ok":
            continue
        if "mode" in row:
            key = (row["mode"], row["context"])
        else:
            key = ("default", row["context"])
        rows[key] = row
    return rows


def lookup(rows: dict[tuple[str, int], dict], series: str, context: int) -> dict | None:
    if series == "upstream":
        return rows.get(("default", context))
    if series.endswith("-keep"):
        return rows.get(("keep", context))
    return rows.get(("default", context))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {name: load(path) for name, _, path, _, _ in SERIES}
    contexts = sorted({
        context for rows in loaded.values() for (_, context) in rows
    })

    with (args.output_dir / "draft-thresholds.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "context",
            "upstream_decode", "upstream_prefill",
            "mtp_eject_decode", "mtp_eject_prefill",
            "mtp_keep_decode", "mtp_keep_prefill",
            "dflash_eject_decode", "dflash_eject_prefill",
            "dflash_keep_decode", "dflash_keep_prefill",
            "mtp_ejected_pages", "dflash_ejected_pages",
        ])
        for context in contexts:
            row = {"context": context}
            for name, _, _, _, _ in SERIES:
                point = lookup(loaded[name], name, context)
                key = name.replace("-", "_")
                row[f"{key}_decode"] = point["decode_tps"] if point else None
                row[f"{key}_prefill"] = point["prefill_tps"] if point else None
            mtp = loaded["mtp-eject"].get(("default", context))
            dflash = loaded["dflash-eject"].get(("default", context))
            row["mtp_ejected_pages"] = mtp.get("ejected_pages") if mtp else None
            row["dflash_ejected_pages"] = dflash.get("ejected_pages") if dflash else None
            writer.writerow([row.get(key) for key in [
                "context",
                "upstream_decode", "upstream_prefill",
                "mtp_eject_decode", "mtp_eject_prefill",
                "mtp_keep_decode", "mtp_keep_prefill",
                "dflash_eject_decode", "dflash_eject_prefill",
                "dflash_keep_decode", "dflash_keep_prefill",
                "mtp_ejected_pages", "dflash_ejected_pages",
            ]])

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "svg.fonttype": "none",
    })

    media = ROOT / "media"
    media.mkdir(exist_ok=True)

    for metric, title, filename, annotate_eject in (
        ("decode_tps", "Decode throughput vs context", "draft-thresholds-decode", True),
        ("prefill_tps", "Prefill throughput vs context", "draft-thresholds-prefill", False),
    ):
        fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
        for name, label, _, color, marker in SERIES:
            points = [(context, lookup(loaded[name], name, context)) for context in contexts]
            points = [(context, point) for context, point in points if point]
            if not points:
                continue
            x = [context / 1000 for context, _ in points]
            ax.plot(x, [point[metric] for _, point in points],
                    color=color, marker=marker, linewidth=2.0, markersize=6, label=label)

        if annotate_eject:
            eject_notes = []
            for name, color in (("mtp-eject", "#d62728"), ("dflash-eject", "#2ca02c")):
                for context in contexts:
                    point = lookup(loaded[name], name, context)
                    if point and point.get("ejected_pages"):
                        ax.axvline(context / 1000, color=color, linewidth=1.4,
                                   linestyle=":", alpha=0.7)
                        eject_notes.append(
                            f"{name.split('-')[0]} (first eject {point['ejected_pages']} pages)")
                        break
            if eject_notes:
                ax.text(0.98, 0.95, "dotted lines mark first decode eject:\n" + "\n".join(eject_notes),
                        transform=ax.transAxes, ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

        ax.set_title(title)
        ax.set_xlabel("context capacity (thousands of tokens)")
        ax.set_ylabel("tokens/s")
        ax.grid(True, alpha=0.3)
        ax.legend()

        for suffix in ("png", "svg"):
            fig.savefig(args.output_dir / f"{filename}.{suffix}", dpi=150)
            fig.savefig(media / f"{filename}.{suffix}", dpi=150)
        plt.close(fig)
        print(f"wrote {args.output_dir / (filename + '.png/svg')}")
        print(f"wrote {media / (filename + '.png/svg')}")

    print(f"wrote {args.output_dir / 'draft-thresholds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
