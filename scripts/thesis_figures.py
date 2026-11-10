"""Regenerate the CyberPosts label distribution figures of the thesis from the frozen split.

Run from the repo root:
    uv run python scripts/thesis_figures.py [--split-dir runs/splits/data] [--out thesis/Images]

Writes cyberposts_label_distribution.pdf and cyberposts_labelset_distribution.pdf (vector, deterministic)
and prints the counts to stdout.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

TRAIN_COLOR = "#0072B2"  # Okabe-Ito blue
TEST_COLOR = "#E69F00"  # Okabe-Ito orange
DISPLAY = {"mac os": "MacOS"}
SMALL = {"and"}


def display_name(label):
    """Thesis display name of a lowercase public label id."""
    if label in DISPLAY:
        return DISPLAY[label]
    return " ".join(w if w in SMALL else w.capitalize() for w in label.split())


def label_counts(golds):
    """Counter of public labels over a list of gold label lists."""
    return Counter(label for gold in golds for label in gold)


def labelset_counts(golds, top=15):
    """The `top` most frequent label-sets as (tuple, count), by count then set for determinism."""
    counts = Counter(tuple(gold) for gold in golds)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


def load_golds(path):
    import pandas as pd

    sys.path.insert(0, ".")
    from notebooks import system as sy, taxonomy as tx

    df = pd.read_csv(path, usecols=["relevant", "watson"])
    rel = [tx.as_bool(v) if isinstance(v, bool) else str(v) == "True" for v in df["relevant"]]
    return [sy.gold_public(tx.watson_list(w), r) for w, r in zip(df["watson"], rel)]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42})
    return plt


def plot_label_distribution(labels, train, test, path):
    plt = _style()
    import numpy as np

    y = np.arange(len(labels))
    h = 0.4
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    ax.barh(y - h / 2, [train[l] for l in labels], h, color=TRAIN_COLOR, label="Train")
    ax.barh(y + h / 2, [test[l] for l in labels], h, color=TEST_COLOR, label="Test")
    ax.set_yticks(y)
    ax.set_yticklabels([display_name(l) for l in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Number of posts")
    ax.set_ylabel("Public label")
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, format="pdf", bbox_inches="tight", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)


def plot_labelset_distribution(sets, path):
    plt = _style()

    import textwrap

    names = [textwrap.fill(", ".join(display_name(l) for l in s), 34) for s, _ in sets]
    counts = [n for _, n in sets]
    fig, ax = plt.subplots(figsize=(4.4, 7.4))
    bars = ax.barh(range(len(sets)), counts, 0.7, color=TRAIN_COLOR)
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Number of posts")
    ax.set_ylabel("Label-set")
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.bar_label(bars, labels=[f"{n:,}" for n in counts], padding=3, fontsize=9)
    ax.set_xlim(0, max(counts) * 1.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, format="pdf", bbox_inches="tight", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-dir", type=Path, default=Path("runs/splits/data"))
    ap.add_argument("--out", type=Path, default=Path("thesis/Images"))
    args = ap.parse_args(argv)

    sys.path.insert(0, ".")
    from notebooks import system as sy

    labels = sy.public_labels()
    train_golds = load_golds(args.split_dir / "split_train.csv")
    test_golds = load_golds(args.split_dir / "split_test.csv")
    train, test = label_counts(train_golds), label_counts(test_golds)
    print("label | train | test")
    for l in labels:
        print(f"{display_name(l)} | {train[l]} | {test[l]}")
    sets = labelset_counts(train_golds + test_golds, 15)
    print("\nMost frequent label-sets (train+test):")
    for s, n in sets:
        print(f"{n}: ({', '.join(display_name(l) for l in s)})")

    args.out.mkdir(parents=True, exist_ok=True)
    plot_label_distribution(labels, train, test, args.out / "cyberposts_label_distribution.pdf")
    plot_labelset_distribution(sets, args.out / "cyberposts_labelset_distribution.pdf")


if __name__ == "__main__":
    main()
