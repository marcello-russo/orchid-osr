"""Test del watchdog di alimentazione.

Il caso che conta: il caricatore non tiene il passo, quindi con l'AC collegato
la batteria scende lo stesso. Il watchdog deve fermare il training prima che la
macchina si spenga e riprenderlo quando la batteria e' risalita.
"""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "power-watchdog.py"


def _load():
    spec = importlib.util.spec_from_file_location("power_watchdog", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


wd = _load()


class TestDecide:
    def test_battery_high_keeps_running(self):
        assert wd.decide(80, "Charging", 1, paused=False) == "keep"

    def test_battery_low_pauses(self):
        assert wd.decide(30, "Discharging", 1, paused=False) == "pause"

    def test_paused_stays_paused_until_resume_threshold(self):
        assert wd.decide(40, "Charging", 1, paused=True) == "keep"

    def test_resumes_above_threshold_when_charging(self):
        assert wd.decide(61, "Charging", 1, paused=True) == "resume"

    def test_does_not_resume_with_the_cable_out(self):
        assert wd.decide(95, "Discharging", 0, paused=True) == "keep"

    def test_resumes_while_plugged_even_if_slow_draining(self):
        assert wd.decide(65, "Discharging", 1, paused=True) == "resume"

    def test_unplugged_below_ninety_pauses_immediately(self):
        assert wd.decide(70, "Discharging", 0, paused=False) == "pause"

    def test_slow_drain_while_plugged_uses_the_normal_threshold(self):
        """Su questa macchina l'AC non tiene il passo: 80% in scarica da spina
        non e' un cavo staccato, e non deve fermare il training."""
        assert wd.decide(80, "Discharging", 1, paused=False) == "keep"
        assert wd.decide(34, "Discharging", 1, paused=False) == "pause"

    def test_no_battery_does_nothing(self):
        assert wd.decide(None, None, None, paused=False) == "keep"

    def test_custom_thresholds(self):
        assert wd.decide(50, "Charging", 1, paused=False, pause_at=55) == "pause"
        assert wd.decide(50, "Charging", 1, paused=True, resume_at=45) == "resume"


class TestProcessTree:
    def test_collects_children_and_grandchildren(self):
        code = "import subprocess,time; subprocess.Popen(['sleep','30']); time.sleep(30)"
        parent = subprocess.Popen([sys.executable, "-c", code])  # noqa: S603
        try:
            deadline = time.time() + 5
            tree = wd.collect_tree(parent.pid)
            while len(tree) < 2 and time.time() < deadline:
                time.sleep(0.1)
                tree = wd.collect_tree(parent.pid)
            assert parent.pid in tree
            assert len(tree) >= 2, "il figlio non e' stato trovato"
        finally:
            parent.kill()
            parent.wait(timeout=5)


class TestStateFile:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wd, "STATE_DIR", tmp_path)
        monkeypatch.setattr(wd, "STATE_FILE", tmp_path / "state.json")
        wd.save_state(4242, paused=True)
        state = wd.load_state()
        assert state["pid"] == 4242 and state["paused"] is True

    def test_missing_file_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wd, "STATE_FILE", tmp_path / "nope.json")
        assert wd.load_state() == {}


class TestSignalIntegration:
    def _run_once(self, tmp_path, pid, capacity, status, ac):
        source = tmp_path / "battery.json"
        source.write_text(json.dumps({"capacity": capacity, "status": status, "ac_online": ac}), encoding="utf-8")
        return subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(SCRIPT),
                "--pid",
                str(pid),
                "--once",
                "--capacity-source",
                str(source),
                "--state-file",
                str(tmp_path / "state.json"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    @staticmethod
    def _proc_state(pid):
        return Path(f"/proc/{pid}/stat").read_text().split()[2]

    def test_pauses_and_resumes_a_real_process(self, tmp_path):
        child = subprocess.Popen(["sleep", "60"])  # noqa: S603, S607
        try:
            done = self._run_once(tmp_path, child.pid, 20, "Discharging", 1)
            assert done.returncode == 0, done.stderr
            assert self._proc_state(child.pid) == "T"

            done = self._run_once(tmp_path, child.pid, 90, "Charging", 1)
            assert done.returncode == 0, done.stderr
            assert self._proc_state(child.pid) != "T"
        finally:
            try:
                child.send_signal(18)  # SIGCONT, se il test fallisce a meta'
            except ProcessLookupError:
                pass
            child.kill()
            child.wait(timeout=5)

    def test_dry_run_sends_no_signal(self, tmp_path):
        child = subprocess.Popen(["sleep", "30"])  # noqa: S603, S607
        try:
            source = tmp_path / "battery.json"
            source.write_text(json.dumps({"capacity": 5, "status": "Discharging", "ac_online": 0}), encoding="utf-8")
            done = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    str(SCRIPT),
                    "--pid",
                    str(child.pid),
                    "--once",
                    "--dry-run",
                    "--capacity-source",
                    str(source),
                    "--state-file",
                    str(tmp_path / "state.json"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert done.returncode == 0
            assert "[dry-run] PAUSA" in done.stdout
            assert self._proc_state(child.pid) != "T"
        finally:
            child.kill()
            child.wait(timeout=5)

    def test_dead_pid_exits_cleanly(self, tmp_path):
        source = tmp_path / "battery.json"
        source.write_text(json.dumps({"capacity": 50, "status": "Charging", "ac_online": 1}), encoding="utf-8")
        done = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(SCRIPT),
                "--pid",
                "999999",
                "--once",
                "--capacity-source",
                str(source),
                "--state-file",
                str(tmp_path / "state.json"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert done.returncode == 0
        assert "terminato" in done.stdout
