# Adaptive KV streaming benchmark

`benchmark_kv_stream.py` performs the complete context-matched benchmark and
creates its graph. The only required inputs are the GGUF model and the largest
context capacity to test:

```bash
python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K
```

The default server is `build/bin/llama-server`. Use `--server` when the
binary is elsewhere. Matplotlib is the only Python dependency:

```bash
python3 -m pip install matplotlib
```

## What the script does

For every configured context capacity from 8K through `--max-context`, in 8K
steps, the script:

1. Starts a fresh adaptive KV streaming server with a 1536 MiB total arena.
2. If that arena cannot fit the prefill graph plus minimum KV layout, increases it in 256 MiB steps and retries.
3. Measures free VRAM after model initialization and a warm-up request.
4. Adds all measured free VRAM to the successful probe arena, rounded down to 32 MiB.
5. Runs a full prompt and 256-token decode to validate that arena.
6. If the candidate fails, reduces it by 64 MiB and retries.
7. Records prefill speed, decode speed, selected arena size, and VRAM telemetry.
8. Updates the CSV and Matplotlib graph after every successful point.

There is no fixed VRAM safety reserve. Actual server execution is the
validation: allocation failures are handled by automatic arena backoff.

The arena is one physical CUDA allocation shared between resident/ring KV and
the active phase's compute workspace. The configured MiB value therefore
includes both. Prompt processing reserves its measured scheduler workspace;
TG1 decode releases that large slice and gives the reclaimed bytes to KV.

The prompt length at each point is the configured context capacity minus the
256 decode tokens. For example, the 192K point starts the server with
`--ctx-size 196608`, prefills 196352 tokens, and then decodes 256 tokens.
If the maximum is not a multiple of 8K, the exact maximum is appended as the
last point.

The driver uses the configuration currently supported and validated by this
branch:

- Flash Attention enabled
- K cache `q8_0`
- V cache `q4_0`
- all model layers on the GPU
- one server slot
- 256-token batch and micro-batch by default
- ordinary CUDA allocation, without UVM

Use `--batch-size` and `--ubatch-size` to benchmark other logical and physical batch sizes. The micro-batch must not exceed the logical batch. Both values are included in result metadata and the resume signature.

```bash
python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K \
  --batch-size 512 \
  --ubatch-size 512
```

## Results and resuming

By default, a timestamped directory is created under
`benchmarks/results/adaptive-kv-sweep-*`. It contains:

- `results.jsonl`: metadata, arena probes, retries, and measurements
- `results.csv`: one successful measurement per context capacity
- `kv-stream-sweep.png` and `kv-stream-sweep.svg`: decode, prefill, and arena
  size plots
- `logs/`: one server log per probe and benchmark attempt

Use an explicit output directory to resume an interrupted sweep:

```bash
python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K \
  --output-dir benchmarks/results/my-sweep
```

Re-run the same command after an interruption. Completed contexts are skipped.
The script rejects a resume if the model or benchmark settings differ, avoiding
mixed data in one result set.

Run `python3 benchmarks/benchmark_kv_stream.py --help` for optional GPU,
timeout, arena probing/backoff, output, and server arguments. Old `--*-pool-*`
driver spellings remain accepted as compatibility aliases.

Do not run another GPU workload during the sweep. Its allocations would change
the automatically selected arena and invalidate comparisons between points.

## Speculative draft threshold sweeps

`benchmark_mtp_streaming.py` compares the dynamic eject controller ("default")
against keeping the draft for the whole run ("keep") for any pinned draft. It
runs the same prompt-and-decode workload across a list of context capacities and
writes a resumable `results.jsonl`, a CSV, and a plot. Select the draft with
`--spec-type` and `--draft-model` (`--mtp-model` still works):

```bash
python3 benchmarks/benchmark_mtp_streaming.py \
  --model /path/to/target.gguf \
  --draft-model /path/to/draft.gguf \
  --spec-type draft-dflash \
  --arena-mib 3200 \
  --tag dflash \
  --contexts 8192,16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072,139264,147456,155648,160000 \
  --output-dir benchmarks/results/draft-thresh-dflash
```

`--arena-mib` must be the largest arena that validates for that draft on the
GPU; probe it first. `benchmark_upstream_vs_mtp.py --configs upstream` measures
the no-spec baseline. `plot_draft_thresholds.py` combines the three legs into
`benchmarks/results/draft-thresholds.csv` and two separate figures (decode and
prefill), and copies them to `media/draft-thresholds-decode.png` and
`media/draft-thresholds-prefill.png` for the README:

```bash
python3 benchmarks/plot_draft_thresholds.py
```

The measured thresholds and the recommended eject point are in the fork README.
