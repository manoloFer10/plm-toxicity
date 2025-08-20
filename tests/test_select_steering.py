import importlib
import math
import sys
import types

import pytest
import torch


@pytest.fixture
def ss(monkeypatch):
    """Provide select_steering module with lightweight stubs for heavy deps."""
    def dummy_score_toxicity(seqs, batch_size=100):
        return [0.1] * len(seqs), [0.9] * len(seqs)

    def dummy_calculate_perplexity(seqs, model, tokenizer_fn):
        return torch.ones(len(seqs))

    monkeypatch.setitem(
        sys.modules,
        "utils.models.gPLM",
        types.SimpleNamespace(gPLM=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.scoring",
        types.SimpleNamespace(
            score_toxicity=dummy_score_toxicity,
            calculatePerplexity=dummy_calculate_perplexity,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.models.protgpt2",
        types.SimpleNamespace(clean_protgpt2_generation=lambda x: x),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.visualizations",
        types.SimpleNamespace(plot_tox_scores=lambda **_: None),
    )

    module = importlib.import_module("utils.select_steering")
    importlib.reload(module)
    return module


def test_get_last_position_logits_shape_and_hooks(ss):
    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size=5):
            super().__init__()
            self.vocab_size = vocab_size
            self.device = torch.device("cpu")
            self.forward_calls = 0

        def forward(self, input_ids=None, attention_mask=None):
            self.forward_calls += 1
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, self.vocab_size)
            return types.SimpleNamespace(logits=logits)

    model = DummyModel()

    def tokenizer_fn(instructions):
        max_len = max(len(s) for s in instructions)
        ids = torch.zeros((len(instructions), max_len), dtype=torch.int64)
        attn = torch.ones_like(ids)
        return types.SimpleNamespace(input_ids=ids, attention_mask=attn)

    hook_calls = []

    def hook(module, input, output):
        hook_calls.append(True)
        return output

    instructions = ["AA", "BBB", "C"]
    logits = ss.get_last_position_logits(
        model,
        instructions,
        tokenize_instructions_fn=tokenizer_fn,
        fwd_hooks=[(model, hook)],
        batch_size=2,
    )

    assert logits.shape == (len(instructions), model.vocab_size)
    assert len(hook_calls) == math.ceil(len(instructions) / 2)


def test_kl_div_fn_matches_manual(ss):
    logits_a = torch.tensor([[[0.0, math.log(3)]]])
    logits_b = torch.tensor([[[0.0, 0.0]]])
    result = ss.kl_div_fn(logits_a, logits_b, epsilon=1e-6)

    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    manual = torch.sum(probs_a * (torch.log(probs_a) - torch.log(probs_b)), dim=-1).mean().to(torch.float64)

    assert torch.isclose(result, manual)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"ablation_tox_score": math.nan, "steering_tox_score": 0.1, "kl_div_score": 0.0, "layer": 0, "n_layer": 10}, True),
        ({"ablation_tox_score": 0.1, "steering_tox_score": 0.1, "kl_div_score": 0.0, "layer": 9, "n_layer": 10, "prune_layer_percentage": 0.1}, True),
        ({"ablation_tox_score": 0.1, "steering_tox_score": 0.1, "kl_div_score": 0.2, "layer": 0, "n_layer": 10, "kl_threshold": 0.1}, True),
        ({"ablation_tox_score": 0.3, "steering_tox_score": 0.1, "kl_div_score": 0.0, "layer": 0, "n_layer": 10, "baseline_tox_score": 0.2}, True),
        ({"ablation_tox_score": 0.1, "steering_tox_score": 0.1, "kl_div_score": 0.0, "layer": 0, "n_layer": 10, "baseline_tox_score": 0.2}, True),
        ({"ablation_tox_score": 0.1, "steering_tox_score": 0.3, "kl_div_score": 0.0, "layer": 0, "n_layer": 10, "baseline_tox_score": 0.2, "kl_threshold": 0.5}, False),
    ],
)
def test_filter_fn_conditions(ss, kwargs, expected):
    assert ss.filter_fn(**kwargs) is expected