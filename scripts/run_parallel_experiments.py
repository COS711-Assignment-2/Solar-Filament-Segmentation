"""Run four model families concurrently, one independent job per TPU core."""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import queue
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.trainer import load_config, train


DEFAULT_CONFIGS = (
    Path("experiments/unet/config.yaml"),
    Path("experiments/condinst/config.yaml"),
    Path("experiments/boundary_aware_unet/config.yaml"),
    Path(
        "runs/boundary_aware_unet/boundary-aware-unet/"
        "pyhopper-seed43/best_config.yaml"
    ),
)
TPU_CORE_COUNT = 4


def _find_tpu_blocking_pids() -> list[int]:
    """Find any external processes currently holding /dev/accel* handles."""
    import glob

    pids = set()
    current_pid = os.getpid()
    for fd_path in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            target = os.readlink(fd_path)
            if target.startswith("/dev/accel"):
                pid = int(fd_path.split("/")[2])
                if pid != current_pid:
                    pids.add(pid)
        except (OSError, ValueError):
            continue
    return sorted(pids)


class IndependentXlaModel:
    """XLA wrapper for independent TPU core jobs that bypasses mesh collectives.

    Standard torch_xla xm.optimizer_step() executes an all_reduce collective
    across all replicas when world_size > 1. For independent model runs
    concurrently executing across TPU cores, cross-replica reduction crashes
    due to mismatched parameters, topologies, and step counts. This wrapper
    issues local optimizer steps and mark_step() without collective reductions,
    and saves checkpoints without collective rendezvous or master-only filtering.
    """

    def __init__(self, xm: Any) -> None:
        self._xm = xm

    def __getattr__(self, name: str) -> Any:
        return getattr(self._xm, name)

    def optimizer_step(
        self, optimizer: Any, barrier: bool = True, **kwargs: Any
    ) -> Any:
        loss = optimizer.step()
        if barrier:
            self._xm.mark_step()
        return loss

    def save(self, data: Any, file_or_path: Any, **kwargs: Any) -> None:
        import torch

        cpu_data = self._xm._maybe_convert_to_cpu(data, convert=True)
        torch.save(cpu_data, file_or_path)


def _build_tasks(
    jobs: list[dict[str, Any]], seeds: list[int]
) -> list[tuple[int, int]]:
    """Queue one run per model first, then prioritize remaining CondInst runs."""
    first_seed = seeds[0]
    tasks = [(job_index, first_seed) for job_index in range(len(jobs))]
    remaining_seeds = seeds[1:]
    condinst_index = next(
        index for index, job in enumerate(jobs) if job["model_name"] == "condinst"
    )
    tasks.extend((condinst_index, seed) for seed in remaining_seeds)
    tasks.extend(
        (job_index, seed)
        for seed in remaining_seeds
        for job_index in range(len(jobs))
        if job_index != condinst_index
    )
    return tasks


def _load_jobs(config_paths: list[Path]) -> list[dict[str, Any]]:
    if len(config_paths) != TPU_CORE_COUNT:
        raise ValueError(f"Exactly {TPU_CORE_COUNT} model configs are required")
    jobs = []
    for index, config_path in enumerate(config_paths):
        resolved = (ROOT / config_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Experiment config not found: {resolved}")
        config = load_config(resolved)
        if index == 3:
            config["experiment"]["name"] = (
                "boundary-aware-unet-pyhopper-corrected-data"
            )
        jobs.append(
            {
                "config_path": str(resolved),
                "config": config,
                "model_name": config["model"]["name"],
                "experiment_name": config["experiment"]["name"],
            }
        )
    model_names = [job["model_name"] for job in jobs]
    expected_models = [
        "unet",
        "condinst",
        "boundary_aware_unet",
        "boundary_aware_unet",
    ]
    if model_names != expected_models:
        raise ValueError(
            "Configs must be ordered as U-Net, CondInst, boundary-aware U-Net, "
            f"and tuned boundary-aware U-Net; got {model_names}"
        )
    unet_options = {
        key: value
        for key, value in jobs[0]["config"]["model"].items()
        if key != "name"
    }
    boundary_options = {
        key: value
        for key, value in jobs[2]["config"]["model"].items()
        if key != "name"
    }
    if unet_options != boundary_options:
        raise ValueError("The two U-Net comparison configs must use the same architecture")
    return jobs


def _validate_output_paths(
    jobs: list[dict[str, Any]], seeds: list[int], run_suffix: str
) -> None:
    collisions = []
    for job in jobs:
        for seed in seeds:
            output_dir = (
                Path(job["config"]["training"]["output_dir"])
                / job["model_name"]
                / job["experiment_name"]
                / f"{run_suffix}-seed{seed}"
            )
            if output_dir.exists():
                collisions.append(str(output_dir))
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing run output(s): "
            + ", ".join(collisions)
            + ". Choose a new --run-suffix."
        )


def _run_worker_tasks(
    rank: int,
    global_ordinal: int,
    device: Any,
    jobs: list[dict[str, Any]],
    task_queue: Any,
    failure_queue: Any,
    max_epochs: int,
    run_suffix: str,
    xla_model: Any,
    train_fn: Any = train,
) -> None:
    while True:
        try:
            job_index, seed = task_queue.get_nowait()
        except queue.Empty:
            return
        job = jobs[job_index]
        config = copy.deepcopy(job["config"])
        print(
            f"[worker {rank} / TPU core {global_ordinal} ({device})] starting "
            f"{job['experiment_name']} seed={seed}, "
            f"epochs={max_epochs}, early_stopping=disabled",
            flush=True,
        )
        try:
            result = train_fn(
                config,
                seed_override=seed,
                output_tag=f"{run_suffix}-seed{seed}",
                device_override="tpu",
                max_epochs_override=max_epochs,
                patience_override=0,
                initialized_xla_model=xla_model,
            )
        except Exception as error:
            failure = (
                rank,
                global_ordinal,
                job["experiment_name"],
                seed,
                repr(error),
            )
            failure_queue.put(failure)
            print(
                f"[worker {rank} / TPU core {global_ordinal} ({device})] "
                f"FAILED {job['experiment_name']} seed={seed}: {error}",
                flush=True,
            )
            traceback.print_exc()
            continue
        print(
            f"[worker {rank} / TPU core {global_ordinal} ({device})] finished "
            f"{job['experiment_name']} seed={seed}, "
            f"best_val_pq={result['best_val_pq']:.6f}, "
            f"best_epoch={result['best_epoch']}",
            flush=True,
        )


def _worker(
    rank: int,
    jobs: list[dict[str, Any]],
    task_queue: Any,
    failure_queue: Any,
    max_epochs: int,
    run_suffix: str,
    preflight_only: bool,
) -> None:
    import torch_xla.core.xla_model as xm
    import torch_xla.experimental.pjrt as pjrt

    device_count = pjrt.global_device_count()
    if device_count != TPU_CORE_COUNT:
        raise RuntimeError(
            f"Expected exactly {TPU_CORE_COUNT} TPU devices, found "
            f"{device_count} global devices"
        )
    device = xm.xla_device()
    global_ordinal = pjrt.global_ordinal()
    if global_ordinal != rank:
        raise RuntimeError(
            f"Worker {rank} was assigned global ordinal {global_ordinal} "
            f"and device {device}; expected TPU core {rank}"
        )
    print(
        f"[worker {rank} / TPU core {global_ordinal} ({device})] ready for "
        "dynamically scheduled runs",
        flush=True,
    )
    if preflight_only:
        return

    _run_worker_tasks(
        rank,
        global_ordinal,
        device,
        jobs,
        task_queue,
        failure_queue,
        max_epochs,
        run_suffix,
        IndependentXlaModel(xm),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        type=Path,
        nargs=TPU_CORE_COUNT,
        default=list(DEFAULT_CONFIGS),
        metavar="CONFIG",
        help=(
            "Four YAML configs ordered as U-Net, CondInst, boundary-aware "
            "U-Net, tuned boundary-aware U-Net."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds queued as independent model/seed runs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs for each seed/model training run.",
    )
    parser.add_argument(
        "--run-suffix",
        default="corrected-data",
        help="Unique output tag prefix so existing runs are preserved.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify four-worker TPU-core assignment without starting training.",
    )
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error("--max-epochs must be at least one")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be non-empty and contain no duplicates")
    if (
        not args.run_suffix
        or Path(args.run_suffix).name != args.run_suffix
        or args.run_suffix in (".", "..")
    ):
        parser.error("--run-suffix must be a single path-safe name")
    if os.environ.get("PJRT_DEVICE", "TPU").strip().upper() != "TPU":
        parser.error("PJRT_DEVICE must be unset or set to TPU")
    os.environ["PJRT_DEVICE"] = "TPU"

    blocking_pids = _find_tpu_blocking_pids()
    if blocking_pids:
        parser.error(
            f"TPU devices are currently in use by process(es): {blocking_pids}. "
            "Stop the intended process after verifying its owner, or wait for it "
            "to finish before starting this runner."
        )

    jobs = _load_jobs(args.configs)
    if not args.preflight_only:
        try:
            _validate_output_paths(jobs, args.seeds, args.run_suffix)
        except FileExistsError as error:
            parser.error(str(error))
    import torch_xla.distributed.xla_multiprocessing as xmp

    task_manager = None
    task_queue = None
    failure_queue = None
    if not args.preflight_only:
        task_manager = multiprocessing.get_context("spawn").Manager()
        task_queue = task_manager.Queue()
        failure_queue = task_manager.Queue()
        for job_index, seed in _build_tasks(jobs, args.seeds):
            task_queue.put((job_index, seed))
    if args.preflight_only:
        print(
            f"Checking worker assignment across {TPU_CORE_COUNT} XLA devices.",
            flush=True,
        )
    else:
        print(
            f"Launching {len(jobs) * len(args.seeds)} dynamically scheduled "
            f"model/seed runs across {TPU_CORE_COUNT} XLA devices; "
            f"seeds={args.seeds}; epochs={args.max_epochs}; "
            "early stopping disabled.",
            flush=True,
        )
    failures = []
    try:
        xmp.spawn(
            _worker,
            args=(
                jobs,
                task_queue,
                failure_queue,
                args.max_epochs,
                args.run_suffix,
                args.preflight_only,
            ),
            nprocs=None,
            start_method="spawn",
        )
        if failure_queue is not None:
            while True:
                try:
                    failures.append(failure_queue.get_nowait())
                except queue.Empty:
                    break
    finally:
        if task_manager is not None:
            task_manager.shutdown()
    if failures:
        print(
            f"{len(failures)} model/seed run(s) failed; see worker errors "
            "above for details.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
