from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_dialogue_text import atomic_save, cache_is_valid, encode_texts


class FakeTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        lengths = [max(1, min(5, len(text))) for text in texts]
        width = max(lengths)
        input_ids = torch.zeros(len(texts), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, length in enumerate(lengths):
            input_ids[index, :length] = torch.arange(1, length + 1)
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeModel(torch.nn.Module):
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        basis = torch.arange(6, device=input_ids.device, dtype=torch.float32)
        hidden = input_ids.float().unsqueeze(-1) + basis
        return SimpleNamespace(last_hidden_state=hidden)


def test_encode_and_atomic_cache() -> None:
    embeddings = encode_texts(
        ["短い", "もう少し長い", "終わり"],
        FakeTokenizer(),
        FakeModel(),
        torch.device("cpu"),
        batch_size=2,
        max_length=8,
        prefix="query: ",
    )
    assert embeddings.shape == (3, 6)
    np.testing.assert_allclose(
        np.linalg.norm(embeddings.astype(np.float32), axis=1), 1.0, atol=5e-4
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "page.npz"
        atomic_save(path, embeddings, ["a", "b", "c"])
        assert cache_is_valid(path, ["a", "b", "c"])
        assert not cache_is_valid(path, ["a", "c", "b"])


if __name__ == "__main__":
    test_encode_and_atomic_cache()
    print("text cache tests passed")
