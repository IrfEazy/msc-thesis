import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import thesis_figures as tf  # noqa: E402

GOLDS = [["other"], ["other"], ["computer security", "antivirus and malware"],
         ["computer security"], ["computer security", "antivirus and malware"], ["mac os", "operating systems"]]


class Counting(unittest.TestCase):
    def test_label_counts(self):
        c = tf.label_counts(GOLDS)
        self.assertEqual((c["other"], c["computer security"], c["antivirus and malware"], c["linux"]), (2, 3, 2, 0))

    def test_labelset_counts_order_and_top(self):
        sets = tf.labelset_counts(GOLDS, 2)
        self.assertEqual(sets, [(("computer security", "antivirus and malware"), 2), (("other",), 2)])
        self.assertEqual(len(tf.labelset_counts(GOLDS, 10)), 4)

    def test_display_name(self):
        self.assertEqual(tf.display_name("antivirus and malware"), "Antivirus and Malware")
        self.assertEqual(tf.display_name("mac os"), "MacOS")


if __name__ == "__main__":
    unittest.main()
