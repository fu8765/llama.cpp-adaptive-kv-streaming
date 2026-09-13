#!/usr/bin/env python3
"""Compare dynamic MTP eject against keeping MTP active while streaming.

Both modes use the automatic MTP KV pin. `default` ejects MTP when the working
set outgrows the MTP-active pool; `keep` stays active and lets the target stream
under the smaller pool. Records decode/prefill throughput and the speculative
draft counters so the crossover can be read off directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import benchmark_kv_stream as bks
import benchmark_upstream_vs_mtp as um

ROOT = Path(__file__).resolve().parents[1]

MODES = ("default", "keep")


def server_argv(mode: str, args: argparse.Namespace) -> list[str]:
    argv = [
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
        "--kv-stream-arena-mib", str(args.arena_mib),
        "--spec-draft-n-max", str(args.spec_draft_n_max),
        "--model-draft", str(args.mtp_model),
        "--spec-draft-ngl", "all",
        "-ctkd", args.draft_cache_type_k,
        "-ctvd", args.draft_cache_type_v,
        "--kv-stream-spec-dynamic",
        "--kv-stream-spec-eject-pages", "0",
        "--kv-stream-spec-reenable-pages", "8",
        "--kv-stream-spec-stable-decodes", "4",
        "-lv", "5",
    ]
    if mode == "keep":
        argv += ["--kv-stream-spec-keep-pages", str(args.keep_pages)]
    return argv


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
    mode: str,
    args: argparse.Namespace,
    context: int,
    corpus_tokens: list[int],
    suffix_tokens: list[int],
    log_path: Path,
) -> dict:
    server = um.ServerHandle(server_argv(mode, args), args, log_path)
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
        predicted_n = timings.get("predicted_n") or 0
        if predicted_n < args.decode_tokens - 4:
            raise RuntimeError(
                f"incomplete decode: expected {args.decode_tokens}, "
                f"received {predicted_n}"
            )
        if predicted_n != args.decode_tokens:
            print(f"warning: {mode} ctx {context}: decoded {predicted_n} of {args.decode_tokens}")
        log_text = log_path.read_text(errors="replace")
        cap = um.MTP_CAP_RE.search(log_text)
        return {
            "type": "measurement",
            "status": "ok",
            "mode": mode,
            "context": context,
            "n_ctx": args.n_ctx,
            "prompt_tokens": len(prompt),
            "decode_tokens": args.decode_tokens,
            "decode_actual": predicted_n,
            "arena_mib": args.arena_mib,
            "spec_type": "draft-mtp",
            "prefill_tps": timings.get("prompt_per_second"),
            "decode_tps": timings.get("predicted_per_second"),
            "prompt_ms": timings.get("prompt_ms"),
            "predicted_ms": timings.get("predicted_ms"),
            "draft_n": timings.get("draft_n"),
            "draft_n_accepted": timings.get("draft_n_accepted"),
            "mtp_pin_pages": int(cap.group(1)) if cap else None,
            "mtp_window_pages": int(cap.group(3)) if cap else None,
            "mtp_ejected": bool(um.MTP_EJECT_RE.search(log_text)),
            "draft_errors": log_text.count("llama_decode(ctx_dft)"),
        }
    finally:
        server.stop()


def write_csv(path: Path, results: dict[tuple[str, int], dict]) -> None:
    columns = [
        "mode", "context", "n_ctx", "prompt_tokens", "decode_tokens",
        "decode_actual", "arena_mib", "prefill_tps", "decode_tps", "prompt_ms", "predicted_ms",
        "draft_n", "draft_n_accepted", "mtp_pin_pages", "mtp_window_pages",
        "mtp_ejected", "draft_errors",
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
        "default": ("#1f77b4", "o", "dynamic eject (default)"),
        "keep": ("#d62728", "s", "keep MTP while streaming"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for mode in MODES:
        points = sorted(
            (context, row) for (name, context), row in results.items() if name == mode
        )
        if not points:
            continue
        color, marker, label = styles[mode]
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
        fig.savefig(output_dir / f"mtp-streaming.{suffix}", dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/llama-server")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mtp-model", type=Path, required=True)
    parser.add_argument("--contexts", type=um.parse_context_list,
                        default=[50000, 80000, 120000, 160000])
    parser.add_argument("--n-ctx", type=int, default=160000)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--corpus-max-chars", type=int, default=700000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18092)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--draft-cache-type-k", default="q8_0")
    parser.add_argument("--draft-cache-type-v", default="q4_0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--spec-draft-n-max", type=int, default=3)
    parser.add_argument("--keep-pages", type=int, default=1000000)
    parser.add_argument("--arena-mib", type=int, default=3072)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--release-timeout", type=int, default=90)
    parser.add_argument("--release-slack-mib", type=int, default=64)
    return parser.parse_args(argv)


def load_results(path: Path) -> dict[tuple[str, int], dict]:
    results = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                results[(row["mode"], row["context"])] = row
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    results_path = args.output_dir / "results.jsonl"

    blob, digest, used = um.build_corpus(um.ROOT, args.corpus_max_chars)
    print(f"corpus: {len(blob)} chars from {used} files, sha256 {digest[:12]}", flush=True)

    results = load_results(results_path)
    baseline = bks.query_gpu_memory(args.nvidia_smi, args.gpu_index).used_mib
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

    for mode in MODES:
        for context in args.contexts:
            if (mode, context) in results:
                continue
            log_path = logs / f"{mode}-{context}.log"
            row = run_point(mode, args, context, corpus_tokens, suffix_tokens, log_path)
            um.append_jsonl(results_path, row)
            results[(mode, context)] = row
            bks.wait_for_release(args, baseline)

    write_csv(args.output_dir / "results.csv", results)
    plt = bks.require_matplotlib()
    plt.switch_backend("Agg")
    plot_results(args.output_dir, results, plt)
    print(json.dumps({"points": len(results), "output": str(args.output_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
