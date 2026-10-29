"""Select the XGBoost compute device from the COMPUTE_DEVICE environment variable."""
import os


def select_device():
    """Return 'cpu' or 'cuda' per COMPUTE_DEVICE (auto, cpu, cuda; default auto).

    auto picks 'cuda' only if torch is importable and sees a GPU. An explicit
    'cuda' without CUDA raises RuntimeError; unknown values raise ValueError.
    """
    mode = (os.environ.get("COMPUTE_DEVICE") or "auto").strip().lower()
    if mode not in ("auto", "cpu", "cuda"):
        raise ValueError(f"COMPUTE_DEVICE must be auto, cpu or cuda, got {mode!r}")
    if mode == "cpu":
        return "cpu"
    try:
        import torch
        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if mode == "cuda" and not available:
        raise RuntimeError("COMPUTE_DEVICE=cuda but CUDA is not available")
    return "cuda" if available else "cpu"
