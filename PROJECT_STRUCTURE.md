# Project Structure and Training Workflow



## Repository layout

```text
Solar-Filament-Segmentation/
├── configs/
│   └── data.yaml                 # data-preparation settings
├── experiments/
│   ├── unet/
│   │   └── config.yaml           # complete U-Net baseline experiment
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
│   └── run_all_experiments.py    # repeat each experiment across seeds
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

Run the boundary-aware model:

```bash
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
python3 scripts/train.py --experiment unet --device xla
```

`--device xla` uses PyTorch/XLA. The equivalent environment setting is
`GLS_DEVICE=xla`; `auto` (the config default) selects TPU, then CUDA, then CPU.
Use the TPU runtime's compatible PyTorch/XLA installation.
Before importing PyTorch/XLA, the trainer explicitly sets `PJRT_DEVICE=TPU`
when the backend is not already selected. It then checks access to
`/dev/accel*`; permission errors are reported without triggering an XLA process
abort. You can also select PJRT before Python starts:

```bash
PJRT_DEVICE=TPU GLS_DEVICE=xla python3 scripts/train.py --experiment unet
```

Training displays an epoch progress bar, batch progress during training, and
validation-image progress. `training.max_epochs` is the upper bound;
`training.patience` stops training when validation PQ has not improved for the
configured number of consecutive epochs. The best-PQ checkpoint is retained.

The previous environment-variable style also works:

```bash
GLS_DEVICE=xla python3 scripts/train.py --experiment unet
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

## Prepare data and test

Use the existing data pipeline independently of model training:

```bash
python3 scripts/prepare_data.py --help
python3 -m pytest -q
```
