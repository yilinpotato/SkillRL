"""CPU regression for actual actor KL block and the production metric reducer."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def kl_block():
    tree = ast.parse((ROOT / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8"))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "self.config.use_kl_loss"
             and any(isinstance(x, ast.Assign) and any(
                 isinstance(t, ast.Subscript) and ast.unparse(t) == "metrics['actor/kl_loss']"
                 for t in x.targets) for x in n.body)]
    assert len(nodes) == 1
    return compile(ast.Module(body=nodes, type_ignores=[]), "<actual actor KL block>", "exec")


@pytest.mark.parametrize("enabled", [False, True])
def test_microbatch_average_preserves_legacy_and_loss_gradients(enabled):
    append = function("verl/utils/py_functional.py", "append_to_dict", {"Dict": Dict})
    reduce = function("verl/utils/metric/utils.py", "reduce_metrics",
                      {"Dict": Dict, "List": List, "Any": Any, "np": np})
    metrics = {}
    config = SimpleNamespace(use_kl_loss=enabled, kl_loss_type="kl", kl_loss_coef=.25)
    # Unequal lengths distinguish microbatch mean 2 from pooled-token mean 2.5.
    for values in ([1.], [3., 3., 3.]):
        log_prob = torch.tensor([values], requires_grad=True)
        base = log_prob.sum() * 2
        env = dict(self=SimpleNamespace(config=config), data={"ref_log_prob": torch.zeros_like(log_prob)},
                   log_prob=log_prob, response_mask=torch.ones_like(log_prob),
                   loss_agg_mode="token-mean", policy_loss=base, metrics=metrics,
                   append_to_dict=append,
                   kl_penalty=lambda logprob, ref_logprob, kl_penalty: logprob-ref_logprob,
                   agg_loss=lambda loss_mat, loss_mask, loss_agg_mode: (loss_mat*loss_mask).sum()/loss_mask.sum())
        exec(kl_block(), env)
        expected = base + (.25 * log_prob.mean() if enabled else 0)
        assert torch.equal(env["policy_loss"], expected)
        actual_grad = torch.autograd.grad(env["policy_loss"], log_prob, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, log_prob)[0]
        assert torch.equal(actual_grad, expected_grad)
    if enabled:
        assert metrics["actor/kl_loss"] == 3.
        assert metrics["actor/kl_loss_microbatch_mean"] == [1., 3.]
        reduced = reduce(metrics)
        assert reduced["actor/kl_loss_microbatch_mean"] == 2.
        assert reduced["actor/kl_loss"] == 3.
        assert reduced["actor/kl_coef"] == .25
    else:
        assert metrics == {}
