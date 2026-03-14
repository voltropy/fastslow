# fastslow

`fastslow` is a runnable experiment repo for the fast/slow multi-timescale residual-stream idea described in `FastSlow.docx`.

The repo contains four variants on the same long-range associative-recall task:

- `baseline`: standard pre-LN causal Transformer.
- `fastslow`: main stream plus a narrow slow stream updated every `k` layers.
- `fastslow_every_layer`: same two-stream parameter budget, but the slow stream updates every layer.
- `widened_baseline`: standard Transformer widened until it roughly matches the `fastslow` parameter count.

The experiment is set up to use the latest `voltropy/volta2` path when a real Volta backend is configured. For local smoke tests without backend provisioning, it falls back to JSONL tracking so the training code still runs.

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
- `src/fastslow/tracking.py`: Volta-first tracker with JSONL fallback.
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
  --run-name local-smoke \
  --artifacts-dir artifacts/local-smoke
```

That uses local JSONL tracking in `artifacts/local-smoke/metrics.jsonl`.

## OpenClaw runbook

These are the DevOps steps OpenClaw should carry out before running the real experiment through Volta.

### 1. Provision the runtime

- Create a Python 3.10-3.13 environment.
- Install the repo:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Install the latest Volta from `voltropy/volta2`

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

### 3. Provision a real Volta backend

The latest Volta Python binding only persists runs when a backend is configured. For a real tracked run, OpenClaw should provision and export:

- `VOLTA_SPANNER_DATABASE`
- `VOLTA_GCS_BUCKET`
- `GOOGLE_APPLICATION_CREDENTIALS`

If those are absent, this repo falls back to local JSONL tracking for debugging only.

### 4. Create and launch the Volta experiment

Choose a machine label and run name, then:

```bash
export RUN_NAME=fs-k4-seed0
export MACHINE_LABEL=openclaw-gpu

volta create "fastslow/${RUN_NAME}" \
  --git-repo https://github.com/voltropy/fastslow \
  --machine "${MACHINE_LABEL}" \
  --command "python -m fastslow.train --variant fastslow --preset default --run-name ${RUN_NAME} --artifacts-dir artifacts/${RUN_NAME}" \
  --config '{"variant":"fastslow","preset":"default","seed":0}' \
  --description "Fast/slow residual-stream associative-recall baseline run"

volta launch "fastslow/${RUN_NAME}" "${MACHINE_LABEL}"
```

Then export:

```bash
export VOLTA_PROJECT=fastslow
export VOLTA_EXPERIMENT="${RUN_NAME}"
```

### 5. Run the full comparison set

Run at least these four jobs:

```bash
python -m fastslow.train --variant baseline --preset default --run-name baseline-seed0 --artifacts-dir artifacts/baseline-seed0
python -m fastslow.train --variant fastslow --preset default --run-name fs-k4-seed0 --artifacts-dir artifacts/fs-k4-seed0
python -m fastslow.train --variant fastslow_every_layer --preset default --run-name fs-k1-seed0 --artifacts-dir artifacts/fs-k1-seed0
python -m fastslow.train --variant widened_baseline --preset default --run-name wide-seed0 --artifacts-dir artifacts/wide-seed0
```

For each run, OpenClaw should create and launch a matching Volta experiment first, then set `VOLTA_PROJECT` and `VOLTA_EXPERIMENT` to the matching name before starting training.

### 6. Compare the results

Target signals:

- `fastslow` should beat `baseline` on the longest evaluation lengths.
- `fastslow` should beat `fastslow_every_layer` if timescale separation is doing real work.
- `fastslow` should remain competitive with `widened_baseline` after parameter matching.

The repo writes:

- `summary.json`: final metrics and parameter counts.
- `history.jsonl`: structured training/eval log.
- `best.pt`: best checkpoint by longest-context eval accuracy.

With Volta enabled, metrics also stream into the configured backend.

## Useful commands

Quick sweep:

```bash
python -m fastslow.train --variant fastslow --preset default --seed 0 --run-name fs-k4-seed0
```

Baseline:

```bash
python -m fastslow.train --variant baseline --preset default --seed 0 --run-name baseline-seed0
```

Parameter-matched widened baseline:

```bash
python -m fastslow.train --variant widened_baseline --preset default --seed 0 --run-name wide-seed0
```
