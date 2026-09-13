#!/usr/bin/env python3
"""Measure how the MTP draft KV type moves the dynamic MTP crossover.

Runs this fork with the preset MTP settings once per MTP draft KV type (F16,
q8_0/q4_0, q4_0/q4_0) across the requested prompt sizes. Each point uses a
fresh server with a fixed --ctx-size, so the automatic pin and the decode
window are sized the same way for every point. Records prefill/decode
throughput and whether MTP ejected, then plots decode t/s versus prompt size.

Reuses the server/HTTP/corpus helpers from benchmark_upstream_vs_mtp.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import benchmark_kv_stream as bks
import benchmark_upstream_vs_mtp as um


KV_CONFIGS = ("f16", "q8q4", "q4q4")
KV_ARGS = {
    "f16": [],
    "q8q4": ["-ctkd", "q8_0", "-ctvd", "q4_0"],
    "q4q4": ["-ctkd", "q4_0", "-ctvd", "q4_0"],
}


def server_argv(config: str, args: argparse.Namespace) -> list[str]:
    return [
        str(args.server),
        "-m", str(args.model),
        "--alias", "bench",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ctx-size", str(args.n_ctx),
        "-fa", "on",
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-ngl", "all",
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "--spec-type", "draft-mtp",
        "-np", "1",
        "--no-mmproj",
        "--no-warmup",
        "--fit", "off",
        "--reasoning-format", "none",
        "-lv", "5",
        "--kv-stream-arena-mib", str(args.arena_mib),
        "--spec-draft-n-max", str(args.spec_draft_n_max),
        "--model-draft", str(args.mtp_model),
        "--spec-draft-ngl", "all",
        "--kv-stream-spec-dynamic",
        "--kv-stream-spec-reenable-pages", "8",
        "--kv-stream-spec-stable-decodes", "4",
    ] + KV_ARGS[config]


def token_server_argv(args: argparse.Namespace) -> list[str]:
    return [
        str(args.server),
        "-m", str(args.model),
        "--alias", "bench",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ctx-size", "4096",
        "-ngl", "all",
        "-b", "256",
        "-ub", "256",
        "-np", "1",
        "--no-mmproj",
        "--no-warmup",
        "--fit", "off",
        "--reasoning-format", "none",
        "--kv-stream-arena-mib", "512",
    ]


def run_point(
    config: str,
    args: argparse.Namespace,
    context: int,
    corpus_tokens: list[int],
    suffix_tokens: list[int],
    log_path: Path,
) -> dict:
    server = um.ServerHandle(server_argv(config, args), args, log_path)
    try:
        need = context - args.decode_tokens
        prefix_count = need - len(suffix_tokens)
        if prefix_count < 0:
            raise RuntimeError("context is smaller than the instruction suffix")
        if prefix_count > len(corpus_tokens):
            raise RuntimeError(
                f"corpus has {len(corpus_tokens)} tokens, need {prefix_count}; "
                "increase --corpus-max-chars"
            )
        prompt = corpus_tokens[:prefix_count] + suffix_tokens
        response = bks.http_json(
            server.url("/completion"),
            {
                "prompt": prompt,
                "n_predict": args.decode_tokens,
                "ignore_eos": True,
                "cache_prompt": False,
                "temperature": 0,
                "seed": 1,
                "reasoning_format": "none",
                "response_fields": ["timings"],
            },
            args.request_timeout,
        )
        timings = response.get("timings") or {}
        if timings.get("predicted_n") != args.decode_tokens:
            raise RuntimeError(
                f"incomplete decode: expected {args.decode_tokens}, "
                f"received {timings.get('predicted_n')}"
            )
        log_text = log_path.read_text(errors="replace")
        cap = um.MTP_CAP_RE.search(log_text)
        return {
            "type": "measurement",
            "status": "ok",
            "config": config,
            "context": context,
            "n_ctx": args.n_ctx,
            "prompt_tokens": len(prompt),
            "decode_tokens": args.decode_tokens,
            "arena_mib": args.arena_mib,
            "spec_type": "draft-mtp",
            "prefill_tps": timings.get("prompt_per_second"),
            "decode_tps": timings.get("predicted_per_second"),
            "prompt_ms": timings.get("prompt_ms"),
            "predicted_ms": timings.get("predicted_ms"),
            "mtp_pin_pages": int(cap.group(1)) if cap else None,
            "mtp_window_pages": int(cap.group(3)) if cap else None,
            "mtp_ejected": bool(um.MTP_EJECT_RE.search(log_text)),
        }
    finally:
        server.stop()


def write_csv(path: Path, results: dict[tuple[str, int], dict]) -> None:
    columns = [
        "config", "context", "n_ctx", "prompt_tokens", "decode_tokens",
        "arena_mib", "prefill_tps", "decode_tps", "prompt_ms", "predicted_ms",
        "mtp_pin_pages", "mtp_window_pages", "mtp_ejected",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(results, key=lambda item: (item[0], item[1])):
            writer.writerow(results[key])


def plot_results(output_dir: Path, results: dict[tuple[str, int], dict], plt) -> None:
    if not results:
        return
    styles = {
        "f16": ("#1f77b4", "o", "F16"),
        "q8q4": ("#2ca02c", "s", "q8_0 K / q4_0 V"),
        "q4q4": ("#d62728", "^", "q4_0 K / q4_0 V"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for config in KV_CONFIGS:
        points = sorted(
            (context, row) for (name, context), row in results.items() if name == config
        )
        if not points:
            continue
        color, marker, label = styles[config]
        xs = [context / 1000 for context, _ in points]
        decode = [row["decode_tps"] for _, row in points]
        prefill = [row["prefill_tps"] for _, row in points]
        for ax, values, title in (
            (axes[0], decode, "Token generation (decode) throughput"),
            (axes[1], prefill, "Prompt processing (prefill) throughput"),
        ):
            ax.plot(xs, values, color=color, marker=marker, linewidth=2.2, label=label)
            ax.set_title(title)
            ax.set_ylabel("tokens/s")
            ax.set_xlabel("prompt size (thousands of tokens)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        for context, row in points:
            if row.get("mtp_ejected"):
                axes[0].annotate(
                    "ejected",
                    (context / 1000, row["decode_tps"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=color,
                )
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"mtp-kv-quant.{suffix}", dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build/bin/llama-server")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mtp-model", type=Path, required=True)
    parser.add_argument("--contexts", type=um.parse_context_list,
                        default=[40000, 50000, 52000, 54000, 80000, 160000])
    parser.add_argument("--n-ctx", type=int, default=160000)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--corpus-max-chars", type=int, default=700000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--spec-draft-n-max", type=int, default=3)
    parser.add_argument("--arena-mib", type=int, default=3264)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--release-timeout", type=int, default=90)
    parser.add_argument("--release-slack-mib", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    results_path = args.output_dir / "results.jsonl"

    blob, digest, used = um.build_corpus(um.ROOT, args.corpus_max_chars)
    print(f"corpus: {len(blob)} chars from {used} files, sha256 {digest[:12]}", flush=True)

    results = um.load_results(results_path)
    baseline = bks.query_gpu_memory(args.nvidia_smi, args.gpu_index).used_mib
    corpus_tokens: list[int] | None = None
    token_server = um.ServerHandle(token_server_argv(args), args, logs / "tokenize.log")
    try:
        corpus_tokens = um.load_or_tokenize(
            token_server, blob, args.output_dir / "corpus-tokens.json",
            args.request_timeout,
        )
        suffix_tokens = um.tokenize(token_server, um.DEFAULT_INSTRUCTION, args.request_timeout)
    finally:
        token_server.stop()
    bks.wait_for_release(args, baseline)

    for config in KV_CONFIGS:
        for context in args.contexts:
            if (config, context) in results:
                continue
            log_path = logs / f"{config}-{context}.log"
            row = run_point(config, args, context, corpus_tokens, suffix_tokens, log_path)
            um.append_jsonl(results_path, row)
            results[(config, context)] = row
            bks.wait_for_release(args, baseline)

    write_csv(args.output_dir / "results.csv", results)
    plt = bks.require_matplotlib()
    plt.switch_backend("Agg")
    plot_results(args.output_dir, results, plt)
    print(json.dumps({"points": len(results), "output": str(args.output_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
