"""Thesis text preprocessing (Methodology.tex, "Data Preprocessing").

``preprocess_text`` is the single cleaning rule applied to every post before it is
embedded, in every notebook and in the System: URLs, emails, mentions and hashtags are
removed, whitespace is collapsed. Nothing else changes (no lowercasing, no phone-number,
emoji or punctuation removal, no lemmatisation, no HTML unescaping).

``group_key`` is the normalised identity used by the grouped split (``notebooks.splits``):
two rows with the same key are copies of the same post and must stay on the same side
(``notebooks.splits`` additionally merges truncated copies whose key is a prefix of another).

Both functions are pure and deterministic. Bump ``TEXTPREP_VERSION`` whenever their
output can change; split manifests record it and fail verification on a mismatch.
"""

import re

TEXTPREP_VERSION = "2"

# Order matters: URLs first (they may contain "@" and "#"), then emails (so the domain
# of an address is not taken for a mention), then mentions and hashtags.
_URL = re.compile(
    r"(?i)"
    r"(?:https?://\S+"  # scheme URLs, also glued to a preceding word ("see:https://...")
    r"|(?<![\w.-])www\.\S+"  # www. forms
    r"|(?<![\w./-])(?:t\.co|pic\.twitter\.com)/\S*)"  # bare t.co and pic.twitter.com links
)
_EMAIL = re.compile(r"(?<![\w.%+-])[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# A mention or hashtag is a whole token: "@user", "#topic", chained "#a#b"; "a@b", "C#" and
# "&#39;" stay.
_MENTION = re.compile(r"(?<![\w@])@\w+")
_HASHTAG = re.compile(r"(?<![\w&#])#\w+(?:#\w+)*")
_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")
# Retweet marker, matched on the normalised key only at its start ("RT @user: ..." -> "rt ...").
_LEADING_RT = re.compile(r"^rt\s+")


def preprocess_text(text: str) -> str:
    """Apply the thesis preprocessing rule to one post.

    Removes URLs (``http(s)://...``, ``www....``, bare ``t.co/...`` and
    ``pic.twitter.com/...``), email addresses, ``@mentions`` and ``#hashtags`` (the whole
    token, including chains such as ``#a#b``), then collapses runs of whitespace to one
    space and strips the ends. Case, emojis, punctuation and numbers are kept. A post made
    only of removed components becomes ``""``.

    Raises ``TypeError`` for non-string input (e.g. NaN), so missing texts are not
    silently embedded as ``"nan"``.
    """
    if not isinstance(text, str):
        raise TypeError(f"preprocess_text expects str, got {type(text).__name__}")
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def group_key(text: str) -> str:
    """Duplicate-group key: ``preprocess_text``, lowercased, non-word characters removed
    (``\\w`` is Unicode-aware, so letters of any script, digits and ``_`` stay), whitespace
    collapsed and stripped, then one leading retweet marker (``rt`` followed by whitespace)
    removed, so ``"RT @user: X"`` and ``"X"`` share a key; ``rt`` elsewhere is kept. Posts
    that are empty after preprocessing all share key ``""`` (identical model input, so they
    form one group)."""
    key = _NON_WORD.sub("", preprocess_text(text).lower())
    key = _SPACE.sub(" ", key).strip()
    return _LEADING_RT.sub("", key, count=1)
