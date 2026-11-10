import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kaggle_bundle as kb  # noqa: E402


class CheckSourcesComplete(unittest.TestCase):
    def run_check(self, statuses):
        entries = [{"name": "n"}]
        with mock.patch.object(kb, "source_refs", return_value=list(statuses)):
            kb.check_sources_complete({"owner": "o"}, None, entries, lambda r: statuses[r])

    def test_all_complete(self):
        self.run_check({"o/a": ("complete", None)})

    def test_parsed_not_complete(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_check({"o/a": ("running", None)})
        msg = str(cm.exception)
        self.assertIn("not COMPLETE", msg)
        self.assertIn("n needs o/a (running)", msg)
        self.assertNotIn("could not read", msg)

    def test_unknown_reported_separately(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_check({"o/a": (None, "rate limited"), "o/b": ("error", None)})
        msg = str(cm.exception)
        self.assertIn("could not read status of source kernel(s): n needs o/a (rate limited / error); retry later", msg)
        self.assertIn("not COMPLETE on Kaggle: n needs o/b (error)", msg)


class StatusRetry(unittest.TestCase):
    def test_rate_limit_retried_then_unknown(self):
        r = mock.Mock(stdout="You have been rate limited. Wait a little while before retrying.",
                      stderr="", returncode=1)
        with mock.patch.object(kb.subprocess, "run", return_value=r) as run:
            res = kb.kernel_status_ex("o/a", sleep=lambda s: None)
        self.assertEqual(res, (None, "rate limited"))
        self.assertEqual(run.call_count, len(kb.STATUS_RETRY_DELAYS) + 1)

    def test_rate_limit_then_success(self):
        limited = mock.Mock(stdout="You have been rate limited.", stderr="", returncode=1)
        ok = mock.Mock(stdout='o/a has status "KernelWorkerStatus.COMPLETE"', stderr="", returncode=0)
        with mock.patch.object(kb.subprocess, "run", side_effect=[limited, ok]):
            self.assertEqual(kb.kernel_status_ex("o/a", sleep=lambda s: None), ("complete", None))


class ActiveSessions(unittest.TestCase):
    def test_unknown_counts_active_and_absent_rechecked(self):
        res = {"a": ("queued", None), "b": (None, "error"), "c": (None, "not accessible")}
        self.assertEqual(kb.active_sessions(["a", "b", "c"], None, lambda r: res[r], lambda s: None), 2)
        res["c"] = ("running", None)  # pushed mid-wait: counted on the next poll
        self.assertEqual(kb.active_sessions(["a", "b", "c"], None, lambda r: res[r], lambda s: None), 3)


class WaitSlot(unittest.TestCase):
    def test_absent_rechecked_each_poll(self):
        # limit 2: a runs; b absent on poll 1 (slot free, returns), later b runs (slot full, times out)
        res = {"a": ("running", None), "b": (None, "not accessible")}
        kb.wait_slot(["a", "b"], 2, 10, "CPU", 25, lambda r: res[r], lambda s: None, lambda: 0)
        res["b"] = ("running", None)
        with self.assertRaises(SystemExit) as cm:
            kb.wait_slot(["a", "b"], 2, 10, "CPU", 0, lambda r: res[r], lambda s: None, lambda: 0)
        self.assertIn("b (running)", str(cm.exception))

    def test_timeout_exits_and_names_kernels(self):
        t = [0]
        def sleep(s):
            t[0] += s
        with self.assertRaises(SystemExit) as cm:
            kb.wait_slot(["a", "b"], 1, 60, "GPU", 150,
                         lambda r: ("running", None) if r == "a" else (None, "error"), sleep, lambda: t[0])
        msg = str(cm.exception)
        self.assertIn("a (running)", msg)
        self.assertIn("b (unreadable: error)", msg)

    def test_returns_when_slot_frees(self):
        res = iter([("running", None), ("complete", None)])
        kb.wait_slot(["a"], 1, 5, "GPU", 100, lambda r: next(res), lambda s: None, lambda: 0)


class Titles(unittest.TestCase):
    def test_build_rejects_long_kernel_title(self):
        entry = {"name": "Long", "kaggle": {"slug": "x" * 46}}
        kb.check_title(kb.kernel_slug(entry, False), "ok")
        with self.assertRaises(SystemExit) as cm:
            kb.check_title(kb.kernel_slug(entry, True), "kernel entry 'Long'")
        self.assertIn("kernel entry 'Long'", str(cm.exception))


class Layout(unittest.TestCase):
    def test_single_data_entry_gets_layout_file(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            repo = Path(t) / "repo"
            (repo / "splits").mkdir(parents=True)
            (repo / "splits" / "a.csv").write_text("x\n")
            (repo / "pkg").mkdir()
            (repo / "pkg" / "m.py").write_text("")
            cfg = {"owner": "o", "code_dataset": "code-ds", "data_dataset": "data-ds",
                   "code_paths": ["pkg"], "data": [{"src": "splits", "dest": "splits"}]}
            out = Path(t) / "out"
            with mock.patch.object(kb, "REPO", repo):
                kb.build_datasets(cfg, out)
            data = out / "datasets" / "data"
            layout = json.loads((data / "LAYOUT.json").read_text())
            self.assertEqual(layout["entries"], ["splits"])
            self.assertTrue((data / "splits").is_dir())
            code = json.loads((out / "datasets" / "code" / "LAYOUT.json").read_text())
            self.assertEqual(code["entries"], ["pkg"])


if __name__ == "__main__":
    unittest.main()
