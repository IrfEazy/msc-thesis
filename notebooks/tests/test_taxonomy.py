import tempfile
import unittest
from pathlib import Path

from notebooks import taxonomy as tx


def watson(*paths):
    return repr({"categories": [{"label": p, "score": 0.5} for p in paths], "usage": {}})


TC = "/technology and computing"


class TestWatsonList(unittest.TestCase):
    def test_all_nodes_depth_first(self):
        raw = watson(f"{TC}/hardware/computer/servers", f"{TC}/computer security/network security")
        self.assertEqual(
            tx.watson_list(raw),
            ["technology and computing", "hardware", "computer", "servers",
             "computer security", "network security"],
        )

    def test_dict_input_and_missing_categories(self):
        self.assertEqual(tx.watson_list({"usage": {}}), [])
        self.assertEqual(tx.watson_list("{'categories': []}"), [])
        self.assertEqual(tx.watson_list({"categories": [{"label": "/a/b"}]}), ["a", "b"])

    def test_rejects_bad_input(self):
        with self.assertRaises(TypeError):
            tx.watson_list(None)


class TestTopLabels(unittest.TestCase):
    def wl(self, *paths):
        return tx.watson_list(watson(*paths))

    def test_two_step_mapped_and_sorted(self):
        wl = self.wl(f"{TC}/software/databases", f"{TC}/computer security/antivirus and malware")
        self.assertEqual(tx.top_labels(wl, True, "two_step"), ["computer security", "software"])

    def test_alias_nodes(self):
        wl = self.wl(f"{TC}/programming languages/java", f"{TC}/internet technology/email",
                     f"{TC}/electronic components")
        self.assertEqual(tx.top_labels(wl, True, "two_step"), ["hardware", "networking", "software"])

    def test_missing_categories_other(self):
        self.assertEqual(tx.top_labels([], True, "two_step"), ["other"])
        wl = self.wl("/sports/football", f"{TC}/tech news")
        self.assertEqual(tx.top_labels(wl, True, "one_step"), ["other"])

    def test_two_step_rejects_irrelevant(self):
        with self.assertRaises(ValueError):
            tx.top_labels(["hardware"], False, "two_step")

    def test_one_step(self):
        wl = self.wl(f"{TC}/hardware/computer")
        self.assertEqual(tx.top_labels(wl, False, "one_step"), ["other"])
        self.assertEqual(tx.top_labels(wl, True, "one_step"), ["hardware"])

    def test_bad_scheme_and_relevance(self):
        with self.assertRaises(ValueError):
            tx.top_labels([], True, "three_step")
        with self.assertRaises(TypeError):
            tx.top_labels([], "True", "one_step")
        with self.assertRaises(TypeError):
            tx.top_labels("hardware", True, "one_step")


class TestSubLabels(unittest.TestCase):
    def test_leaves_outside_hierarchy_keep_parent(self):
        wl = tx.watson_list(watson(f"{TC}/hardware/computer/servers",
                                   f"{TC}/hardware/computer networking/router"))
        self.assertEqual(tx.sub_labels(wl, "hardware"), ["computer", "computer networking"])
        self.assertEqual(tx.top_labels(wl, True, "two_step"), ["hardware"])

    def test_sub_other(self):
        wl = tx.watson_list(watson(f"{TC}/hardware/computer peripherals/printers"))
        self.assertEqual(tx.sub_labels(wl, "hardware"), ["other"])
        self.assertEqual(tx.sub_labels(["software"], "software"), ["other"])

    def test_other_levels(self):
        wl = tx.watson_list(watson(f"{TC}/operating systems/mac os", f"{TC}/operating systems/linux"))
        self.assertEqual(tx.sub_labels(wl, "operating systems"), ["linux", "mac os"])
        self.assertEqual(tx.sub_labels(wl, "computer security"), ["other"])

    def test_no_sub_level(self):
        for parent in ("networking", "other", "top", "servers"):
            with self.assertRaises(ValueError):
                tx.sub_labels([], parent)


class TestClasses(unittest.TestCase):
    def test_class_order(self):
        self.assertEqual(
            tx.classes("top"),
            ["computer security", "hardware", "networking", "operating systems", "other", "software"],
        )
        self.assertEqual(tx.classes("computer security"), ["antivirus and malware", "network security", "other"])
        self.assertEqual(tx.classes("hardware"), ["computer", "computer components", "computer networking", "other"])
        self.assertEqual(tx.classes("operating systems"), ["linux", "mac os", "other", "unix", "windows"])
        self.assertEqual(tx.classes("software"), ["databases", "other"])
        self.assertEqual(tx.parents(), ["computer security", "hardware", "operating systems", "software"])
        with self.assertRaises(ValueError):
            tx.classes("networking")

    def test_not_labels(self):
        all_labels = set(tx.classes("top")).union(*(tx.classes(p) for p in tx.parents()))
        for leaf in ("servers", "portable computer", "desktop computer", "router", "wireless technology"):
            self.assertNotIn(leaf, all_labels)

    def test_sha_ignores_comments_and_validation(self):
        text = tx.TAXONOMY_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.toml"
            p.write_text("# extra comment\n" + text, encoding="utf-8")
            self.assertEqual(tx.load_taxonomy(p).sha256, tx.taxonomy_sha256())
            p.write_text(text.replace('"databases" = "databases"', '"databases" = "nosuch"'), encoding="utf-8")
            with self.assertRaises(ValueError):
                tx.load_taxonomy(p)


if __name__ == "__main__":
    unittest.main()
