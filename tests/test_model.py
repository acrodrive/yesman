"""planner 모델의 구조 확인 (8단계). python -m pytest tests/test_model.py"""

import torch

from yesman.decision import LAT_ACTIONS
from yesman.model import DEC_IN, ModelConfig, Planner, encode_decisions


def _model(**kw):
    torch.manual_seed(0)
    return Planner(ModelConfig(**kw), torch.zeros(8, 3), torch.ones(8, 3)).eval()


def _inputs(B=4):
    g = torch.Generator().manual_seed(1)
    kv, qo = torch.randn(B, 65, 256, generator=g), torch.randn(B, 31, 256, generator=g)
    dec = encode_decisions(torch.zeros(B, 2), torch.full((B, 2), 0.5), torch.zeros(B, 2), torch.zeros(B, 2))
    return kv, qo, dec


def test_encode_decisions():
    lat = torch.tensor([[LAT_ACTIONS.index("turn_left"), LAT_ACTIONS.index("keep_lane")]])
    d = encode_decisions(torch.tensor([[0, 1]]), torch.tensor([[0.3, 0.7]]), lat, torch.tensor([[0.6, 0.0]]))
    assert d.shape == (1, 2, DEC_IN)
    assert d[0, 0, 0] == 1 and d[0, 1, 1] == 1  # go, stop
    assert d[0, 0, 2 + LAT_ACTIONS.index("turn_left")] == 1
    assert torch.allclose(d[0, :, 9], torch.tensor([0.3, 0.7])) and torch.allclose(d[0, :, 10], torch.tensor([0.6, 0.0]))
    assert d[0, 0, 11] == 1 and d[0, 1, 12] == 1  # 구간 번호


def test_judge_does_not_depend_on_noise():
    m = _model(judge=True)
    kv, qo, dec = _inputs()
    a = m.sample(kv, qo, dec, generator=torch.Generator().manual_seed(0))
    b = m.sample(kv, qo, dec, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(a["poses"], b["poses"])  # 경로는 노이즈에 따라 다르다
    assert torch.equal(a["flag_logit"], b["flag_logit"]) and torch.equal(a["reason_logit"], b["reason_logit"])


def test_decision_changes_output_and_b1_ignores_it():
    kv, qo, dec = _inputs()
    dec2 = dec.clone()
    dec2[:, :, 9] = 0.9  # 앞뒤 세기만 바꾼다
    m = _model()
    g = lambda: torch.Generator().manual_seed(0)  # noqa: E731
    assert not torch.allclose(m.sample(kv, qo, dec, generator=g())["poses"], m.sample(kv, qo, dec2, generator=g())["poses"])
    b1 = _model(use_decision=False)
    assert torch.equal(b1.sample(kv, qo, None, generator=g())["poses"], b1.sample(kv, qo, None, generator=g())["poses"])


def test_ddim_returns_x0_of_perfect_denoiser():
    """denoise가 항상 같은 x0를 내면 DDIM은 그 x0(정규화 해제)를 돌려줘야 한다."""
    m = Planner(ModelConfig(), torch.full((8, 3), 2.0), torch.full((8, 3), 3.0)).eval()
    x0 = torch.randn(4, 8, 3)
    m.denoise = lambda z, t, mem: x0
    kv, qo, dec = _inputs()
    for n in (1, 2, 10):
        assert torch.allclose(m.sample(kv, qo, dec, n_steps=n)["poses"], x0 * 3 + 2, atol=1e-5)


def test_loss_runs_and_backprops():
    m = _model(judge=True).train()
    kv, qo, dec = _inputs()
    out = m.loss(kv, qo, dec, torch.randn(4, 8, 3), torch.tensor([1.0, 0, 1, 0]), torch.tensor([[1.0, 0, 0]] * 4))
    assert set(out) == {"traj", "flag", "reason"}
    sum(out.values()).backward()
    assert m.judge_query.grad is not None and m.dec_in[0].weight.grad is not None
