"""Shared checkpoint loading and crop embedding utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import letterbox
from model import CharacterReIDModel


def load_model(checkpoint_path: Path, device: torch.device) -> CharacterReIDModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CharacterReIDModel(str(checkpoint["config"]["backbone"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def embed_instance(model: CharacterReIDModel, image: Image.Image, face_box: list[float] | None, body_box: list[float] | None, device: torch.device, size: int = 224) -> np.ndarray:
    tensors, masks = {}, {}
    for kind, box in (("face", face_box), ("body", body_box)):
        masks[kind] = box is not None
        crop = image.crop(tuple(box)) if box is not None else Image.new("RGB", (size, size), "white")
        tensors[kind] = letterbox(crop, size, False).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensors["face"], tensors["body"], torch.tensor([masks["face"]], device=device), torch.tensor([masks["body"]], device=device))
    return output["embedding"][0].cpu().numpy()
