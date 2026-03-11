"""Tests for hotcb.launch — programmatic autopilot launch API."""
import json
import os
import tempfile
import threading
import time

import pytest

from hotcb.launch import launch, LaunchHandle, _resolve_train_fn, _seed_run_dir


def _dummy_train(run_dir, max_steps, step_delay, stop_event):
    """Minimal training function for testing."""
    for step in range(max_steps):
        if stop_event.is_set():
            break
        metrics = {"loss": 1.0 / (step + 1), "lr": 0.001, "val_loss": 1.0 / (step + 1) + 0.05}
        record = {"step": step, "metrics": metrics}
        with open(os.path.join(run_dir, "hotcb.metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        # Check commands
        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        if os.path.exists(cmd_path):
            with open(cmd_path) as f:
                for line in f:
                    pass  # just consume
        time.sleep(step_delay)


class TestResolveTrainFn:
    def test_callable(self):
        fn = _resolve_train_fn(_dummy_train)
        assert fn is _dummy_train

    def test_string_import(self):
        fn = _resolve_train_fn("hotcb.tests.test_launch:_dummy_train")
        assert callable(fn)

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            _resolve_train_fn("not_a_module_path")

    def test_bad_import(self):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            _resolve_train_fn("nonexistent.module:fn")


class TestSeedRunDir:
    def test_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "hotcb.metrics.jsonl"))
            assert os.path.exists(os.path.join(tmpdir, "hotcb.commands.jsonl"))
            assert os.path.exists(os.path.join(tmpdir, "hotcb.applied.jsonl"))
            assert os.path.exists(os.path.join(tmpdir, "hotcb.freeze.json"))

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            # Write some data
            with open(os.path.join(tmpdir, "hotcb.metrics.jsonl"), "w") as f:
                f.write('{"step": 0}\n')
            _seed_run_dir(tmpdir)
            # Data should still be there (doesn't overwrite)
            with open(os.path.join(tmpdir, "hotcb.metrics.jsonl")) as f:
                assert f.read().strip() == '{"step": 0}'


class TestLaunchHandle:
    def test_metrics_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            handle = LaunchHandle(
                run_dir=tmpdir,
                server_url="http://localhost:8421",
                autopilot_mode="off",
                key_metric="val_loss",
                _stop_event=threading.Event(),
            )
            assert handle.metrics() == []
            assert handle.latest_metrics() == {}

    def test_metrics_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            # Write some metrics
            with open(os.path.join(tmpdir, "hotcb.metrics.jsonl"), "w") as f:
                f.write(json.dumps({"step": 0, "metrics": {"loss": 0.5}}) + "\n")
                f.write(json.dumps({"step": 1, "metrics": {"loss": 0.4}}) + "\n")

            handle = LaunchHandle(
                run_dir=tmpdir,
                server_url="http://localhost:8421",
                autopilot_mode="off",
                key_metric="val_loss",
                _stop_event=threading.Event(),
            )
            assert len(handle.metrics(last_n=10)) == 2
            assert handle.latest_metrics()["loss"] == 0.4

    def test_metric_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            with open(os.path.join(tmpdir, "hotcb.metrics.jsonl"), "w") as f:
                for i in range(5):
                    f.write(json.dumps({"step": i, "metrics": {"loss": 1.0 - i * 0.1}}) + "\n")

            handle = LaunchHandle(
                run_dir=tmpdir,
                server_url="",
                autopilot_mode="off",
                key_metric="loss",
                _stop_event=threading.Event(),
            )
            history = handle.metric_history("loss")
            assert len(history) == 5
            assert history[-1] == pytest.approx(0.6)

    def test_send_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            handle = LaunchHandle(
                run_dir=tmpdir,
                server_url="",
                autopilot_mode="off",
                key_metric="val_loss",
                _stop_event=threading.Event(),
            )
            handle.send_command({"module": "opt", "op": "set_params", "params": {"lr": 0.001}})
            with open(os.path.join(tmpdir, "hotcb.commands.jsonl")) as f:
                cmd = json.loads(f.read().strip())
            assert cmd["params"]["lr"] == 0.001

    def test_set_param(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            handle = LaunchHandle(
                run_dir=tmpdir,
                server_url="",
                autopilot_mode="off",
                key_metric="val_loss",
                _stop_event=threading.Event(),
            )
            handle.set_param(lr=0.005, weight_decay=0.01)
            with open(os.path.join(tmpdir, "hotcb.commands.jsonl")) as f:
                cmd = json.loads(f.read().strip())
            assert cmd["module"] == "opt"
            assert cmd["params"]["lr"] == 0.005

    def test_ai_status_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            handle = LaunchHandle(
                run_dir=tmpdir,
                server_url="",
                autopilot_mode="off",
                key_metric="val_loss",
                _stop_event=threading.Event(),
            )
            assert handle.ai_status() == {}

    def test_ai_status_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            state = {"key_metric": "val_acc", "run_number": 2}
            with open(os.path.join(tmpdir, "hotcb.ai.state.json"), "w") as f:
                json.dump(state, f)

            handle = LaunchHandle(
                run_dir=tmpdir,
                server_url="",
                autopilot_mode="ai_suggest",
                key_metric="val_acc",
                _stop_event=threading.Event(),
            )
            assert handle.ai_status()["key_metric"] == "val_acc"

    def test_running_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_run_dir(tmpdir)
            handle = LaunchHandle(
                run_dir=tmpdir,
                server_url="",
                autopilot_mode="off",
                key_metric="val_loss",
                _stop_event=threading.Event(),
            )
            assert not handle.running  # no thread


class TestLaunch:
    def test_launch_with_custom_fn_no_server(self):
        """Launch with custom function, no server, immediate stop."""
        handle = launch(
            train_fn=_dummy_train,
            max_steps=5,
            step_delay=0.01,
            serve=False,
        )
        assert handle.running
        handle.wait(timeout=10)
        assert not handle.running
        # Should have metrics
        m = handle.latest_metrics()
        assert "loss" in m
        assert m["loss"] > 0

    def test_launch_builtin_no_server(self):
        """Launch with built-in config, no server."""
        handle = launch(
            config="simple",
            max_steps=3,
            step_delay=0.01,
            serve=False,
        )
        handle.wait(timeout=30)
        assert not handle.running

    def test_launch_ai_state_written(self):
        """AI mode writes state file before training starts."""
        handle = launch(
            train_fn=_dummy_train,
            max_steps=3,
            step_delay=0.01,
            autopilot="ai_suggest",
            key_metric="val_loss",
            watch_metrics=["grad_norm"],
            serve=False,
        )
        handle.wait(timeout=10)
        state = handle.ai_status()
        assert state["key_metric"] == "val_loss"
        assert "grad_norm" in state["watch_metrics"]

    def test_launch_creates_run_dir(self):
        """Launch creates temp run dir if not specified."""
        handle = launch(
            train_fn=_dummy_train,
            max_steps=3,
            step_delay=0.01,
            serve=False,
        )
        assert os.path.isdir(handle.run_dir)
        assert os.path.exists(os.path.join(handle.run_dir, "hotcb.run.json"))
        handle.wait(timeout=10)

    def test_launch_stop(self):
        """Stop should terminate training early."""
        handle = launch(
            train_fn=_dummy_train,
            max_steps=9999,
            step_delay=0.05,
            serve=False,
        )
        assert handle.running
        time.sleep(0.2)
        handle.stop()
        time.sleep(0.2)
        assert not handle.running

    def test_launch_invalid_config(self):
        with pytest.raises(ValueError, match="Unknown config"):
            launch(config="nonexistent", serve=False)

    def test_launch_max_time(self):
        """max_time should stop training after wall-clock limit."""
        handle = launch(
            train_fn=_dummy_train,
            max_steps=99999,
            step_delay=0.05,
            max_time=0.5,
            serve=False,
        )
        assert handle.running
        handle.wait(timeout=5)
        assert not handle.running
        # Should have run some steps but not all 99999
        m = handle.metrics(last_n=99999)
        assert 0 < len(m) < 99999

    def test_launch_max_time_in_metadata(self):
        """max_time should be recorded in run metadata."""
        handle = launch(
            train_fn=_dummy_train,
            max_steps=3,
            step_delay=0.01,
            max_time=300.0,
            serve=False,
        )
        handle.wait(timeout=10)
        with open(os.path.join(handle.run_dir, "hotcb.run.json")) as f:
            meta = json.load(f)
        assert meta["max_time"] == 300.0

    def test_launch_config_file(self):
        """hotcb.launch.json should provide defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "hotcb.launch.json")
            with open(cfg_path, "w") as f:
                json.dump({
                    "train_fn": "hotcb.tests.test_launch:_dummy_train",
                    "key_metric": "loss",
                    "max_steps": 3,
                    "max_time": 60,
                    "seed": 99,
                }, f)
            handle = launch(
                config_file=cfg_path,
                step_delay=0.01,
                serve=False,
            )
            handle.wait(timeout=10)
            with open(os.path.join(handle.run_dir, "hotcb.run.json")) as f:
                meta = json.load(f)
            assert meta["key_metric"] == "loss"
            assert meta["seed"] == 99
            assert meta["max_steps"] == 3
            assert meta["max_time"] == 60

    def test_launch_config_file_kwargs_override(self):
        """Explicit kwargs should override config file values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "hotcb.launch.json")
            with open(cfg_path, "w") as f:
                json.dump({
                    "train_fn": "hotcb.tests.test_launch:_dummy_train",
                    "key_metric": "loss",
                    "max_steps": 999,
                }, f)
            handle = launch(
                config_file=cfg_path,
                max_steps=3,
                step_delay=0.01,
                serve=False,
            )
            handle.wait(timeout=10)
            with open(os.path.join(handle.run_dir, "hotcb.run.json")) as f:
                meta = json.load(f)
            # Explicit max_steps=3 should win over config's 999
            assert meta["max_steps"] == 3

    def test_launch_with_seed(self):
        handle = launch(
            train_fn=_dummy_train,
            max_steps=3,
            step_delay=0.01,
            seed=42,
            serve=False,
        )
        handle.wait(timeout=10)
        with open(os.path.join(handle.run_dir, "hotcb.run.json")) as f:
            meta = json.load(f)
        assert meta["seed"] == 42
