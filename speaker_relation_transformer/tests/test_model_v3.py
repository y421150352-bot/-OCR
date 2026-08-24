from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import GeometryTextPageDataset, geometry_text_page_batch_collate
from model_v3 import SpeakerGeometryTextGraphTransformer


def make_page(dialogues: int = 3, candidates: int = 4) -> dict[str, object]:
    torch.manual_seed(17 + dialogues + candidates)
    text_dim = 12
    text_context = torch.randn(dialogues, 3, text_dim)
    text_context_mask = torch.ones(dialogues, 3, dtype=torch.bool)
    text_context_mask[0, 0] = False
    text_context_mask[-1, 2] = False
    text_context[~text_context_mask] = 0
    labels = torch.zeros(dialogues, candidates, dtype=torch.bool)
    labels[:, 0] = True
    return {
        "key": f"page-{dialogues}-{candidates}",
        "geometry": torch.randn(dialogues, candidates, 45),
        "labels": labels,
        "text_context": text_context,
        "text_context_mask": text_context_mask,
    }


def make_model() -> SpeakerGeometryTextGraphTransformer:
    torch.manual_seed(23)
    model = SpeakerGeometryTextGraphTransformer(
        text_dim=12,
        hidden_dim=32,
        heads=4,
        layers=2,
        dropout=0.0,
        attention_dropout=0.0,
        geometry_bias_hidden=16,
    )
    return model


def page_scores(
    model: SpeakerGeometryTextGraphTransformer, page: dict[str, object]
) -> torch.Tensor:
    return torch.stack(
        model.forward_page(
            page["geometry"], page["text_context"], page["text_context_mask"]
        )
    )


def test_forward_shapes_and_backward() -> None:
    model = make_model().train()
    page = make_page()
    scores = page_scores(model, page)
    assert scores.shape == (3, 4)
    scores.sum().backward()
    assert model.geometry_projection[0].weight.grad is not None
    assert model.text_slot_projection[0].weight.grad is not None
    assert model.graph_layers[0].dialogue_attention.out_proj.weight.grad is not None
    assert torch.isfinite(model.scorer[-1].weight.grad).all()


def test_candidate_permutation_equivariance() -> None:
    model = make_model().eval()
    page = make_page()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(page)
    permuted["geometry"] = page["geometry"][:, permutation]
    with torch.inference_mode():
        original_scores = page_scores(model, page)
        permuted_scores = page_scores(model, permuted)
    torch.testing.assert_close(
        permuted_scores, original_scores[:, permutation], atol=1e-5, rtol=1e-5
    )


def test_masked_boundary_text_is_ignored() -> None:
    model = make_model().eval()
    page = make_page()
    changed = dict(page)
    changed_context = page["text_context"].clone()
    changed_context[~page["text_context_mask"]] = 1000.0
    changed["text_context"] = changed_context
    with torch.inference_mode():
        expected = page_scores(model, page)
        actual = page_scores(model, changed)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_neighbor_text_changes_current_dialogue_scores() -> None:
    model = make_model().eval()
    page = make_page()
    changed = dict(page)
    changed_context = page["text_context"].clone()
    # Dialogue 1 slot 0 is its previous dialogue embedding.
    changed_context[1, 0, 0] += 4.0
    changed_context[1, 0, 3] -= 2.0
    changed["text_context"] = changed_context
    with torch.inference_mode():
        expected = page_scores(model, page)
        actual = page_scores(model, changed)
    assert not torch.allclose(actual[1], expected[1])


def test_padded_batch_matches_individual_pages() -> None:
    model = make_model().eval()
    pages = [make_page(3, 4), make_page(2, 2)]
    batch = geometry_text_page_batch_collate(pages)
    with torch.inference_mode():
        individual = [page_scores(model, page) for page in pages]
        batched = model.forward_batch(
            batch["geometry"],
            batch["text_context"],
            batch["text_context_mask"],
            batch["dialogue_mask"],
            batch["candidate_mask"],
        )
    for index, expected in enumerate(individual):
        actual = batched[index, batch["dialogue_mask"][index]][
            :, batch["candidate_mask"][index]
        ]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_geometry_bias_starts_neutral() -> None:
    model = make_model().eval()
    geometry = torch.randn(2, 3, 4, 45)
    with torch.inference_mode():
        bias = model._geometry_bias(geometry)
    assert bias.shape == (2, 3, 4, 4, 4)
    torch.testing.assert_close(bias, torch.zeros_like(bias))


def test_dataset_builds_exact_prev_current_next_context() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        cache_dir = root / "text_cache"
        (data_dir / "packs" / "train").mkdir(parents=True)
        (cache_dir / "train").mkdir(parents=True)
        record = {
            "key": "Book/001",
            "pack": "packs/train/Book__001.npz",
            "text_ids": ["t1", "t2", "t3"],
        }
        (data_dir / "train_pages.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            data_dir / record["pack"],
            geometry=np.zeros((3, 2, 45), dtype=np.float32),
            labels=np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.uint8),
        )
        embeddings = np.arange(36, dtype=np.float32).reshape(3, 12)
        np.savez_compressed(
            cache_dir / "train" / "Book__001.npz",
            embeddings=embeddings,
            text_ids=np.asarray(record["text_ids"], dtype=np.str_),
        )
        page = GeometryTextPageDataset(data_dir, cache_dir, "train")[0]
        context = page["text_context"]
        mask = page["text_context_mask"]
        torch.testing.assert_close(context[1, 0], torch.from_numpy(embeddings[0]))
        torch.testing.assert_close(context[1, 1], torch.from_numpy(embeddings[1]))
        torch.testing.assert_close(context[1, 2], torch.from_numpy(embeddings[2]))
        assert mask.tolist() == [
            [False, True, True],
            [True, True, True],
            [True, True, False],
        ]


if __name__ == "__main__":
    test_forward_shapes_and_backward()
    test_candidate_permutation_equivariance()
    test_masked_boundary_text_is_ignored()
    test_neighbor_text_changes_current_dialogue_scores()
    test_padded_batch_matches_individual_pages()
    test_geometry_bias_starts_neutral()
    test_dataset_builds_exact_prev_current_next_context()
    print("model_v3 tests passed")
