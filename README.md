# fastslow

`fastslow` is a runnable experiment repo for the fast/slow multi-timescale residual-stream idea described in `FastSlow.docx`.

The repo contains four variants on the same long-range associative-recall task:

- `baseline`: standard pre-LN causal Transformer.
- `fastslow`: main stream plus a narrow slow stream updated every `k` layers.
- `fastslow_every_layer`: same two-stream parameter budget, but the slow stream updates every layer.
- `widened_baseline`: standard Transformer widened until it roughly matches the `fastslow` parameter count.

The experiment is set up to use the latest `voltropy/volta2` path when a real Volta backend is configured. Local JSONL tracking still exists for explicit smoke-test mode, but the training process no longer silently falls back when Volta has been requested.

## Experiment summary

The hypothesis is narrow and testable:

- A persistent narrow slow stream should help preserve longer-horizon information.
- The gain should come from timescale separation, not just extra state or extra parameters.

The task used here is synthetic associative recall:

- The model reads a long sequence of key/value pairs.
- A query token at the end asks for the value associated with an earlier key.
- The queried key is biased toward the earliest quarter of the sequence, which forces long-range retention.

This is intentionally simple. It gives a clean first read on whether a slow stream helps memory over long contexts.

## Repository layout

- `src/fastslow/data.py`: on-the-fly associative-recall batch generation.
- `src/fastslow/models.py`: baseline and fast/slow Transformer implementations.
- `src/fastslow/train.py`: CLI entrypoint, training loop, evaluation, checkpoints, summaries.
- `src/fastslow/tracking.py`: strict Volta tracker bootstrap plus explicit JSONL debug mode.
- `scripts/launch_ddp.sh`: `torchrun` helper for full-node launches.
- `Dockerfile`: container build that starts from `volta-runtime:latest`.
- `tests/`: light unit coverage for data/model plumbing.

## Local development

Use Python 3.10-3.13 if you want the most predictable PyTorch install path. Python 3.14 may work depending on wheel availability in the target environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Quick smoke run without Volta:

```bash
source .venv/bin/activate
python -m fastslow.train \
  --variant fastslow \
  --preset quick \
  --tracking-backend jsonl \
  --run-name local-smoke \
  --artifacts-dir artifacts/local-smoke
```

That uses explicit local JSONL tracking in `artifacts/local-smoke/metrics.jsonl`.

## OpenClaw runbook

These are the DevOps steps OpenClaw should carry out before running the real experiment through Volta.

### 1. Build the container

Use the repo Dockerfile. It now starts from `volta-runtime:latest`.

```bash
docker build -t fastslow:latest .
```

### 2. Provision the runtime

- Create a Python 3.10-3.13 environment.
- Install the repo:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

If not using the base image, install the Volta Python bindings explicitly:

```bash
python -m pip install -e ".[volta]"
```

### 3. Install the latest Volta from `voltropy/volta2`

The latest Volta revision I inspected was commit `385e2210e515cc510116592e57320314acbf485c` from March 14, 2026 UTC.

OpenClaw should install from the live repo rather than pinning to this README forever:

```bash
git clone https://github.com/voltropy/volta2.git /tmp/volta2
cd /tmp/volta2
cargo build --release
python -m pip install maturin
maturin develop --features python
export PATH="/tmp/volta2/target/release:$PATH"
```

### 4. Provision a real Volta backend

The latest Volta Python binding only persists runs when a backend is configured. This repo treats Volta mode as strict: if `volta` import fails, credentials are bad, or `VOLTA_SPANNER_DATABASE` is missing, training should crash instead of quietly switching to JSONL.

For a real tracked run, OpenClaw should provision and export:

- `VOLTA_SPANNER_DATABASE`
- `VOLTA_GCS_BUCKET`
- `GOOGLE_APPLICATION_CREDENTIALS`

Local JSONL mode is only for explicit debug runs via `--tracking-backend jsonl`.

### 5. Execution model for 8x B200 machines

Primary intent: one variant per 8x B200 node, using all 8 GPUs with DDP via `torchrun`.

- This repo auto-enables distributed mode when launched under `torchrun`.
- `batch_size` is per-rank. Global batch size is `batch_size * WORLD_SIZE`.
- Only rank 0 talks to Volta and writes artifacts.
- Checkpoint saves transition through `phase("checkpoint")`.

If throughput matters more than per-run speed, OpenClaw can still pack multiple single-GPU jobs onto a node with `CUDA_VISIBLE_DEVICES`, but the validated path is full-node DDP.

### 6. Create and launch the Volta experiment

Choose a machine label and run name, then:

```bash
export RUN_NAME=fs-k4-seed0
export MACHINE_LABEL=openclaw-b200x8

volta create "fastslow/${RUN_NAME}" \
  --git-repo https://github.com/voltropy/fastslow \
  --machine "${MACHINE_LABEL}" \
  --docker-image fastslow:latest \
  --command "bash scripts/launch_ddp.sh --variant fastslow --preset default --tracking-backend volta --run-name ${RUN_NAME} --artifacts-dir artifacts/${RUN_NAME}" \
  --config '{"variant":"fastslow","preset":"default","seed":0,"tracking_backend":"volta","nproc_per_node":8}' \
  --description "Fast/slow residual-stream associative-recall baseline run on 8x B200"

volta launch "fastslow/${RUN_NAME}" "${MACHINE_LABEL}"
```

Then export:

```bash
export VOLTA_PROJECT=fastslow
export VOLTA_EXPERIMENT="${RUN_NAME}"
```

### 7. Run the full comparison set

Run at least these four jobs:

```bash
bash scripts/launch_ddp.sh --variant baseline --preset default --tracking-backend volta --run-name baseline-seed0 --artifacts-dir artifacts/baseline-seed0
bash scripts/launch_ddp.sh --variant fastslow --preset default --tracking-backend volta --run-name fs-k4-seed0 --artifacts-dir artifacts/fs-k4-seed0
bash scripts/launch_ddp.sh --variant fastslow_every_layer --preset default --tracking-backend volta --run-name fs-k1-seed0 --artifacts-dir artifacts/fs-k1-seed0
bash scripts/launch_ddp.sh --variant widened_baseline --preset default --tracking-backend volta --run-name wide-seed0 --artifacts-dir artifacts/wide-seed0
```

For each run, OpenClaw should create and launch a matching Volta experiment first, then set `VOLTA_PROJECT` and `VOLTA_EXPERIMENT` to the matching name before starting training.

### 8. Compare the results

Target signals:

- `fastslow` should beat `baseline` on the longest evaluation lengths.
- `fastslow` should beat `fastslow_every_layer` if timescale separation is doing real work.
- `fastslow` should remain competitive with `widened_baseline` after parameter matching.

The repo writes:

- `summary.json`: final metrics and parameter counts.
- `history.jsonl`: structured training/eval log.
- `best.pt`: best checkpoint by longest-context eval accuracy.

With Volta enabled, metrics also stream into the configured backend.

## Research Results (2026-03-14, 8 rounds on B200 fleet)

### Summary
**FastSlow provides no architectural advantage** on synthetic associative recall. Curriculum learning is the real discovery.

### Key Findings
- **Curriculum learning (32→96 pairs over 40K steps) + 0.1× slow stream LR** achieves 100% accuracy on 96-pair recall with FastSlow. Baseline stuck at 5.3% without curriculum.
- **BUT**: multi-seed validation (Round 8) killed the advantage. Baseline seed=42 gets 60.3% @160 pairs, beating all FastSlow variants. Widened baseline seed=0 gets 53.6%.
- **FastSlow adds fragility**: seed=123 never learns at all (5.4% @96p). 1/3 failure rate. Baseline never fails to learn.
- **Curriculum learning IS the real discovery** — works for all architectures, not FastSlow-specific.
- **Slow-LR ablation**: 0.1× optimal. 0.01× too aggressive, 0.3× similar near-transfer but worse far-transfer.

### What went wrong
Round 7 showed "440× advantage" — but that compared fastslow seed=0 (48.8% @160p) vs baseline seed=0 (0.11%). Baseline seed=0 was an outlier. **A single seed comparison is not science.**

### Lessons
- ALWAYS validate with 3+ seeds before claiming architectural advantage
- "Best during training" is not the result — final eval is what matters
- Curriculum learning (progressive difficulty) genuinely enables long-range generalization for any architecture at this scale
- Both architectures show extreme seed sensitivity

Code: commit `3c879a3` added `--curriculum-start`, `--curriculum-end-step`, `--slow-lr-scale`.

## Useful commands

Quick sweep:

```bash
bash scripts/launch_ddp.sh --variant fastslow --preset default --tracking-backend volta --seed 0 --run-name fs-k4-seed0
```

Baseline:

```bash
bash scripts/launch_ddp.sh --variant baseline --preset default --tracking-backend volta --seed 0 --run-name baseline-seed0
```

Parameter-matched widened baseline:

```bash
bash scripts/launch_ddp.sh --variant widened_baseline --preset default --tracking-backend volta --seed 0 --run-name wide-seed0
```
