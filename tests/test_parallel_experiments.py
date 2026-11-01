import multiprocessing
import queue
import threading
import time

from scripts.run_parallel_experiments import (
    IndependentXlaModel,
    _build_tasks,
    _find_tpu_blocking_pids,
    _run_worker_tasks,
    _validate_output_paths,
)


def test_parallel_tasks_prioritize_condinst_after_starting_each_model():
    jobs = [
        {"model_name": "unet"},
        {"model_name": "condinst"},
        {"model_name": "boundary_aware_unet"},
        {"model_name": "boundary_aware_unet"},
    ]

    tasks = _build_tasks(jobs, [42, 43, 44])

    assert tasks[:4] == [(0, 42), (1, 42), (2, 42), (3, 42)]
    assert tasks[4:6] == [(1, 43), (1, 44)]
    assert len(tasks) == len(jobs) * 3
    assert sorted(tasks) == sorted(
        (job_index, seed)
        for job_index in range(len(jobs))
        for seed in (42, 43, 44)
    )


def test_worker_failure_does_not_cancel_other_queued_runs():
    jobs = [
        {"model_name": "unet", "experiment_name": "unet"},
        {"model_name": "condinst", "experiment_name": "condinst"},
    ]
    for job in jobs:
        job["config"] = {
            "model": {"name": job["model_name"]},
            "training": {"output_dir": "/tmp"},
        }
    task_queue = queue.Queue()
    for task in [(0, 42), (1, 42), (1, 43)]:
        task_queue.put(task)
    failures = queue.Queue()
    finished = []
    finished_lock = threading.Lock()

    def fake_train(config, **kwargs):
        seed = kwargs["seed_override"]
        if seed == 42 and config["model"]["name"] == "unet":
            time.sleep(0.02)
            raise RuntimeError("simulated failed run")
        with finished_lock:
            finished.append((config, seed))
        return {"best_val_pq": 0.5, "best_epoch": 1}

    threads = [
        threading.Thread(
            target=_run_worker_tasks,
            args=(
                rank,
                rank,
                f"xla:{rank}",
                jobs,
                task_queue,
                failures,
                1,
                "test",
                None,
                fake_train,
            ),
        )
        for rank in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert len(finished) == 2
    assert failures.get_nowait()[2:] == ("unet", 42, "RuntimeError('simulated failed run')")
    assert task_queue.empty()


def test_worker_finishing_does_not_interrupt_another_active_run():
    jobs = [
        {"model_name": "fast", "experiment_name": "fast"},
        {"model_name": "slow", "experiment_name": "slow"},
    ]
    for job in jobs:
        job["config"] = {
            "model": {"name": job["model_name"]},
            "training": {"output_dir": "/tmp"},
        }
    task_queue = queue.Queue()
    task_queue.put((0, 42))
    task_queue.put((1, 42))
    failures = queue.Queue()
    fast_finished = threading.Event()
    slow_finished = threading.Event()

    def fake_train(config, **kwargs):
        if config["model"]["name"] == "fast":
            fast_finished.set()
        else:
            assert fast_finished.wait(timeout=1)
            slow_finished.set()
        return {"best_val_pq": 0.5, "best_epoch": 1}

    threads = [
        threading.Thread(
            target=_run_worker_tasks,
            args=(
                rank,
                rank,
                f"xla:{rank}",
                jobs,
                task_queue,
                failures,
                1,
                "test",
                None,
                fake_train,
            ),
        )
        for rank in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert fast_finished.is_set()
    assert slow_finished.is_set()
    assert failures.empty()


def test_output_path_validation_refuses_to_overwrite_existing_run(tmp_path):
    output_dir = tmp_path / "runs"
    existing = output_dir / "unet" / "baseline" / "trial-seed42"
    existing.mkdir(parents=True)
    jobs = [
        {
            "model_name": "unet",
            "experiment_name": "baseline",
            "config": {"training": {"output_dir": str(output_dir)}},
        }
    ]

    try:
        _validate_output_paths(jobs, [42], "trial")
    except FileExistsError as error:
        assert str(existing) in str(error)
    else:
        raise AssertionError("Expected existing output to be rejected")


def test_worker_claims_manager_queue_tasks_and_returns_without_hanging():
    job = {
        "model_name": "unet",
        "experiment_name": "unet",
        "config": {
            "model": {"name": "unet"},
            "training": {"output_dir": "/tmp"},
        },
    }
    with multiprocessing.get_context("spawn").Manager() as manager:
        task_queue = manager.Queue()
        failure_queue = manager.Queue()
        task_queue.put((0, 42))
        completed = []

        def fake_train(config, **kwargs):
            completed.append(kwargs["seed_override"])
            return {"best_val_pq": 0.5, "best_epoch": 1}

        _run_worker_tasks(
            0,
            0,
            "xla:0",
            [job],
            task_queue,
            failure_queue,
            1,
            "test",
            None,
            fake_train,
        )

        assert completed == [42]
        assert failure_queue.empty()


def test_independent_xla_model_bypasses_collective_reduction(tmp_path):
    calls = []

    class FakeXm:
        @staticmethod
        def mark_step():
            calls.append("mark_step")

        @staticmethod
        def _maybe_convert_to_cpu(data, convert=True):
            calls.append("convert_cpu")
            return data

    class FakeOptimizer:
        @staticmethod
        def step():
            calls.append("optimizer_step")

    wrapper = IndependentXlaModel(FakeXm)
    wrapper.optimizer_step(FakeOptimizer, barrier=True)
    assert calls == ["optimizer_step", "mark_step"]

    checkpoint_file = tmp_path / "checkpoint.pt"
    wrapper.save({"epoch": 1}, checkpoint_file)
    assert "convert_cpu" in calls
    assert checkpoint_file.is_file()


def test_find_tpu_blocking_pids_returns_list():
    pids = _find_tpu_blocking_pids()
    assert isinstance(pids, list)
