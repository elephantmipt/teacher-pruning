import os

from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint

from torch.nn import Module
from pathlib import Path


def load_model_from_path(model: Module, path: Path):
    if os.path.isdir(path):
        load_model_from_path(model, path / "best.pth")
    model_sd = load_checkpoint(path)
    try:
        unpack_checkpoint(model_sd, model=model)
    except KeyError:
        model_sd.load_state_dict(model_sd)


__all__ = ["load_model_from_path"]
