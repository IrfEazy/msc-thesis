import unittest

from notebooks.textprep import TEXTPREP_VERSION, group_key, preprocess_text


class TestPreprocessText(unittest.TestCase):
    GOLDEN = [
        ("", ""),
        ("   ", ""),
        ("https://t.co/AbC123", ""),
        ("@user", ""),
        ("@user_1 @Other", ""),
        ("#infosec", ""),
        ("New #ransomware hits #NHS hospitals", "New hits hospitals"),
        ("local #privilege#escalation bug", "local bug"),
        ("Contact admin@example.com now", "Contact now"),
        ("mail me: first.last+tag@sub.example.org.", "mail me: ."),
        ("Patch now http://example.com/a?b=1#frag !", "Patch now !"),
        ("Patch now HTTPS://Example.com/x", "Patch now"),
        ("Read www.example.com/page today", "Read today"),
        ("see t.co/xyz and pic.twitter.com/abc", "see and"),
        ("details:https://t.co/abc", "details:"),
        ("RT @news: Breach at   Acme\n\tInc https://t.co/x #breach", "RT : Breach at Acme Inc"),
        ("Keep CASE, emoji \U0001F600 and 555-123-4567", "Keep CASE, emoji \U0001F600 and 555-123-4567"),
        ("C# and F# devs &#39;ok&#39; a@b", "C# and F# devs &#39;ok&#39; a@b"),
        ("running servers", "running servers"),
    ]

    def test_golden(self):
        for raw, expected in self.GOLDEN:
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw), expected)

    def test_idempotent(self):
        for raw, _ in self.GOLDEN:
            once = preprocess_text(raw)
            self.assertEqual(preprocess_text(once), once)

    def test_rejects_non_string(self):
        for bad in (None, float("nan"), 3):
            with self.assertRaises(TypeError):
                preprocess_text(bad)

    def test_version(self):
        self.assertIsInstance(TEXTPREP_VERSION, str)


class TestGroupKey(unittest.TestCase):
    def test_copies_share_key(self):
        a = "@a: Massive DDoS attack!!! https://t.co/1 #ddos"
        b = "massive ddos attack https://t.co/2 @b"
        self.assertEqual(group_key(a), group_key(b))
        self.assertEqual(group_key(a), "massive ddos attack")

    def test_leading_retweet_marker_stripped_only_at_start(self):
        self.assertEqual(group_key("RT @a: Massive DDoS attack"), "massive ddos attack")
        self.assertEqual(group_key("rt: Massive DDoS attack"), "massive ddos attack")
        # Only one leading marker; "rt" inside the text, inside a word or alone is kept.
        self.assertEqual(group_key("RT RT x"), "rt x")
        self.assertEqual(group_key("Please RT this attack"), "please rt this attack")
        self.assertEqual(group_key("RTX driver bug"), "rtx driver bug")
        self.assertEqual(group_key("art attack"), "art attack")
        self.assertEqual(group_key("RT"), "rt")
        # preprocess_text (the model input) keeps the marker.
        self.assertEqual(preprocess_text("RT @a: x"), "RT : x")

    def test_punctuation_removed_and_spaces_collapsed(self):
        self.assertEqual(group_key("  Don't -- panic,  ok?  "), "dont panic ok")

    def test_unicode_letters_kept(self):
        self.assertEqual(group_key("Sécurité \U0001F512 Ünïcode"), "sécurité ünïcode")

    def test_empty_after_preprocessing(self):
        self.assertEqual(group_key("https://t.co/x @u #h"), "")
        self.assertEqual(group_key("!!!"), "")

    def test_different_texts_differ(self):
        self.assertNotEqual(group_key("windows exploit"), group_key("linux exploit"))


if __name__ == "__main__":
    unittest.main()
