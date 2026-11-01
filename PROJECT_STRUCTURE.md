# Project Structure and Training Workflow

## Repository layout

```text
Solar-Filament-Segmentation/
├── configs/
│   └── data.yaml                 # data-preparation settings
├── experiments/
│   ├── condinst/
│   │   └── config.yaml           # CondInst comparison experiment
│   ├── unet/
│   │   └── config.yaml           # sole U-Net baseline
│   └── boundary_aware_unet/
│       └── config.yaml           # complete boundary-aware experiment
├── data/
│   ├── raw/                      # original dataset; never modified
│   ├── splits/                   # train/validation manifests
│   └── processed/                # prepared images and masks
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train.py                  # train one named experiment
│   ├── run_all_experiments.py    # repeat each experiment across seeds
│   └── run_parallel_experiments.py # one model family per TPU core
├── src/
│   ├── data/                     # preparation and annotation utilities
│   ├── models/                   # one module per registered architecture
│   └── training/
│       ├── dataset.py
│       ├── losses.py
│       ├── metrics.py
│       ├── postprocessing.py
│       └── trainer.py            # shared training and validation loop
├── runs/
│   └── <model>/<experiment>/<run-tag>/
│       ├── best_model.pt
│       └── metrics.json
└── tests/
```

The standard U-Net is the project's sole baseline. CondInst, implemented in
`src/models/condinst.py` with an anchor-free center controller and dynamic
per-instance mask heads, and boundary-aware U-Net are comparison experiments.
CondInst uses the existing processed instance-ID masks and the trainer's
PQ-based validation/checkpoint selection.

Add an experiment by making a directory under `experiments/` with a
`config.yaml`. Set its `model.name`, unique `experiment.name`, data/loss
settings (including maximum epochs and early-stopping patience), and validation
evaluation options there. Add new
architectures as separate modules under `src/models/` and register them in
`src/models/__init__.py`. Training and evaluation are shared across models.

## Start one training run

Use Python 3.8 or newer. On systems where `python` points to Python 2, use
`python3` for all commands. From the repository root, run the U-Net baseline:

```bash
python3 scripts/train.py --experiment unet
```

Run the comparison experiments:

```bash
python3 scripts/train.py --experiment condinst --device cuda
python3 scripts/train.py --experiment boundary_aware_unet
```

Override the random seed or choose a label for the output directory:

```bash
python3 scripts/train.py --experiment unet --seed 43 --output-tag seed43
```

Set the maximum epoch count and early-stopping patience from the command line:

```bash
python3 scripts/train.py --experiment unet --max-epochs 100 --patience 15
```

`--max-epochs` overrides the experiment config for this run. `--patience`
overrides early stopping; set `--patience 0` to disable it and run through the
maximum epoch count.

Run on TPU explicitly:

```bash
python3 scripts/train.py --experiment condinst --device xla
```

`--device xla` uses PyTorch/XLA. The equivalent environment setting is
`GLS_DEVICE=xla`; `auto` (the config default) selects TPU, then CUDA, then CPU.
CondInst training, validation, and checkpoint saving support the shared TPU
path. Use the TPU runtime's compatible PyTorch/XLA installation.
Before importing PyTorch/XLA, the trainer explicitly sets `PJRT_DEVICE=TPU`
when the backend is not already selected. It then checks access to
`/dev/accel*`; permission errors are reported without triggering an XLA process
abort. You can also select PJRT before Python starts:

```bash
PJRT_DEVICE=TPU GLS_DEVICE=xla python3 scripts/train.py --experiment condinst
```

Training displays an epoch progress bar, batch progress during training, and
validation-image progress. `training.max_epochs` is the upper bound;
`training.patience` stops training when validation PQ has not improved for the
configured number of consecutive epochs. The best-PQ checkpoint is retained.

Interrupted runs can continue from their latest checkpoint, which stores the
model and optimizer state, completed epoch, and early-stopping history:

```bash
python3 scripts/train.py --experiment condinst --seed 43 --output-tag seed43 \
  --resume-from runs/condinst/condinst-comparison/seed43/last_checkpoint.pt \
  --device xla
```

Use the same experiment, seed, and output tag as the original run. Older
`best_model.pt` files can warm-start training, but do not include optimizer
state and therefore cannot restore an exact continuation.

The previous environment-variable style also works:

```bash
GLS_DEVICE=xla python3 scripts/train.py --experiment condinst
```

## Run every experiment for multiple seeds

The runner discovers `experiments/*/config.yaml` and launches each experiment
once per requested seed. Runs have separate output directories.

```bash
python3 scripts/run_all_experiments.py --n-runs 5
```

For TPU:

```bash
python3 scripts/run_all_experiments.py --n-runs 5 --device xla
```

By default, seeds start at 42. Change that with `--seed-start`. Run output is
written to `runs/<model>/<experiment>/<run-tag>/`; for example,
`runs/unet/unet-baseline/seed42/`.

## Run four model families concurrently on TPU

On a four-core TPU VM, the parallel runner assigns U-Net, CondInst,
boundary-aware U-Net, and the tuned boundary-aware U-Net to separate XLA cores.
It schedules individual model/seed runs dynamically: it starts one run for
each model, prioritizes the remaining CondInst seeds, then queues the other
runs. A core picks up another run as soon as its current run finishes. By
default it runs seeds 42-46 for 100 epochs with early stopping disabled.
Run failures are logged and that worker continues with queued runs; the runner
reports a nonzero exit status after all workers finish if any run failed. It
refuses to overwrite output folders, so use a new `--run-suffix` for each
attempt. The tmux wrapper starts the runner in its own process group and stops
its PJRT child workers when interrupted. Check the device mapping before
training:

```bash
PJRT_DEVICE=TPU python3 scripts/run_parallel_experiments.py --preflight-only
```

Start the runs in a managed tmux session with:

```bash
tmux new-session -d -s solar-parallel-training -c "$PWD" \
  'bash scripts/run_parallel_experiments_tmux.sh corrected-data-dynamic-20261031 runs/parallel-corrected-data-dynamic-20261031.log'
```

Follow progress with `tail -f runs/parallel-corrected-data-dynamic-20261031.log`.
Attach with `tmux attach -t solar-parallel-training` and detach with Ctrl-b,
then d. To stop the run, send Ctrl-C inside tmux; the wrapper terminates the
runner's process group. If a TPU is already in use, the runner exits without
terminating another process; verify ownership before stopping it. Use
`--seeds`, `--max-epochs`, and `--run-suffix` to change the defaults when
launching the Python runner directly. The default
tuned configuration is read from
`runs/boundary_aware_unet/boundary-aware-unet/pyhopper-seed43/best_config.yaml`.
It starts from random initialization; its parameter search predates the
corrected-data regeneration, so this run compares those settings but does not
validate them as tuned on the corrected data.

## Prepare data and test

Use the existing data pipeline independently of model training:

```bash
python3 scripts/prepare_data.py --help
python3 -m pytest -q
```
