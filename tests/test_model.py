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
    m.denoise = lambda z, t, mem, **kw: x0
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


# ---- 14단계: DiT 내부 표현에서 판단 (새 Ours) ----

def test_layerwise_denoise_matches_transformer_decoder():
    """층을 하나씩 돌리는 denoise가 nn.TransformerDecoder의 forward와 같은 값을 낸다(기존 체크포인트 호환)."""
    m = _model(judge=True)
    kv, qo, dec = _inputs()
    mem = m.memory(kv, qo, dec)
    z, t = torch.randn(4, 8, 3), torch.tensor([0, 10, 500, 999])
    temb = m.time_mlp(__import__("yesman.model", fromlist=["x"]).timestep_embedding(t, m.cfg.d_model))[:, None]
    ref = m.pose_out(m.traj_dec(m.pose_in(z) + m.pose_pos + temb, mem))
    assert torch.allclose(m.denoise(z, t, mem), ref, atol=1e-5)
    out, h = m.denoise(z, t, mem, return_hidden=True)
    assert torch.allclose(out, ref, atol=1e-5) and h.shape == (4, 8, 256)
    _, h2 = m.denoise(z, t, mem, return_hidden=True, hidden_layer=2)
    assert not torch.allclose(h, h2)  # 중간 층의 표현은 마지막 층과 다르다


def test_dit_judge_outputs_and_depends_on_noise():
    for layer in (0, 2):
        m = _model(judge=True, judge_from="dit", judge_layer=layer)
        assert not hasattr(m, "judge_query")  # 병렬 판단 토큰이 없다
        kv, qo, dec = _inputs()
        a = m.sample(kv, qo, dec, generator=torch.Generator().manual_seed(0))
        b = m.sample(kv, qo, dec, generator=torch.Generator().manual_seed(1))
        assert a["flag_logit"].shape == (4,) and a["reason_logit"].shape == (4, 3)
        assert not torch.allclose(a["flag_logit"], b["flag_logit"])  # 판단이 그 샘플의 DiT 표현에서 나온다


def test_dit_judge_loss_joint_vs_detach():
    kv, qo, dec = _inputs()
    flag, reason = torch.tensor([1.0, 0, 1, 0]), torch.tensor([[1.0, 0, 0]] * 4)
    for detach in (False, True):
        m = _model(judge=True, judge_from="dit", judge_detach=detach).train()
        out = m.loss(kv, qo, dec, torch.randn(4, 8, 3), flag, reason, traj=False)
        assert set(out) == {"flag", "reason"}  # 경로 손실 없이 판단 손실만
        (out["flag"] + out["reason"]).backward()
        assert m.dit_head[1].weight.grad is not None
        g = m.traj_dec.layers[0].linear1.weight.grad
        assert (g is None) == detach  # 함께 학습하면 판단 손실이 DiT까지 간다. 고정이면 가지 않는다
    m = _model(judge=True, judge_from="dit").train()
    assert set(m.loss(kv, qo, dec, torch.randn(4, 8, 3), flag, reason)) == {"traj", "flag", "reason"}


def test_sample_vote_picks_a_sample_that_agrees_with_the_vote():
    m = _model(judge=True, judge_from="dit")
    kv, qo, dec = _inputs()
    k = 6
    z0 = torch.randn(4, k, 8, 3, generator=torch.Generator().manual_seed(3))
    each = m.sample(kv.repeat_interleave(k, 0), qo.repeat_interleave(k, 0), dec.repeat_interleave(k, 0),
                    z0=z0.reshape(4 * k, 8, 3))
    p_each = torch.sigmoid(each["flag_logit"]).reshape(4, k)
    tau = float(p_each.median())  # 장면마다 ACCEPT와 REJECT가 섞이도록
    for mm in (1, 3, k):
        v = m.sample_vote(kv, qo, dec, k=k, m=mm, tau=tau, z0=z0)
        assert torch.allclose(v["p"], p_each, atol=1e-6)
        assert torch.equal(v["n_reject"], (p_each < tau).sum(1)) and torch.equal(v["accept"], v["n_reject"] < mm)
        for b in range(4):
            c = int(v["chosen"][b])
            assert (p_each[b, c] >= tau) == bool(v["accept"][b])  # 고른 샘플의 판단이 최종 판단과 같다
            assert torch.allclose(v["poses"][b], each["poses"][b * k + c], atol=1e-5)  # 경로도 그 샘플의 것


def test_decision_kind_ignores_strength():
    from yesman.decision import decision_kind
    t = {"d_seg1_lon": ["go", "go", "stop"], "d_seg1_lat": ["turn_left", "turn_left", "keep_lane"],
         "d_seg2_lon": ["go", "go", "stop"], "d_seg2_lat": ["keep_lane", "keep_lane", "keep_lane"]}
    k = decision_kind(t)
    assert k[0] == k[1] == "go/turn_left|go/keep_lane" and k[2] == "stop/keep_lane|stop/keep_lane"
