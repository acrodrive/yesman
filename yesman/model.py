"""planner 모델 (yesman.md 8.3 M2, M3). 장면 특징은 7단계에서 미리 계산한 LTF 토큰을 쓴다(M1, 학습하지 않음).

구조
- 장면 토큰: LTF keyval(65개) + query_out(31개), 각각 256차원. LayerNorm + 선형층 + 종류 임베딩.
- M2 결정 인코더: 구간마다 [앞뒤 행동 one-hot(2), 좌우 행동 one-hot(7), 앞뒤 세기, 좌우 세기, 구간 one-hot(2)]를
  MLP로 바꿔 토큰 하나를 만든다(2구간 → 토큰 2개 = 조건 c). B1은 결정 토큰을 넣지 않는다.
- M3 경로 생성 (diffusion): waypoint 8개가 토큰 하나씩이다. 노이즈가 섞인 경로 + 시간 임베딩을 입력으로,
  self-attention(waypoint끼리)과 cross-attention(장면 토큰 + 결정 토큰)을 거쳐 깨끗한 경로(x0)를 예측한다.
- M3 판단 토큰: 학습되는 쿼리 하나가 장면 토큰 + 결정 토큰에만 cross-attention한다(따로 된 디코더).
  노이즈를 걷어 내는 도중의 경로는 구조적으로 보지 않으므로, 시작 노이즈가 달라도 flag가 바뀌지 않는다.
  flag head: 실행 가능 logit 하나 (p = sigmoid). 근거 head: 충돌 / 도로 이탈 / 역주행 logit (여러 개일 수 있음).

diffusion: cosine schedule(T=1000), x0 예측(cfg.pred, 10단계에서 v 예측 시험), 학습은 L1(cfg.loss), 샘플링은 DDIM(eta=0, 기본 10단계). 경로는 학습 세트
목표 경로의 (시점, 축)별 평균과 표준편차로 정규화한다. 앵커는 쓰지 않는다(14절 "작게 시작").
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from yesman.decision import LAT_ACTIONS, LON_ACTIONS

N_POSES, POSE_DIM, N_SEG = 8, 3, 2
DEC_IN = len(LON_ACTIONS) + len(LAT_ACTIONS) + 2 + N_SEG  # 13
REASONS = ("collision", "off_road", "wrong_way")
BEV_CLASSES, BEV_PATCH = 7, 16  # LTF BEV 지도 분할의 범주 수(배경 포함), 조각 크기 [픽셀]
OBJ_DIM = {"ltf": 7, "gt": 7, "gtvel": 9}  # x/32, y/32, cos, sin, 길이/5, 너비/5, 있음(LTF는 확률) [+ vx/10, vy/10]


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4  # 경로 디코더 층 수
    n_judge_layers: int = 2  # 판단 토큰 디코더 층 수
    d_ffn: int = 1024
    dropout: float = 0.0
    feat_dim: int = 256  # LTF 토큰 차원
    use_query_out: bool = True  # 장면 토큰에 LTF 디코더 출력(31개)을 더한다
    use_decision: bool = True  # False: B1 (결정을 보지 않는다)
    judge: bool = False  # True: 판단 토큰과 head를 쓴다 (제안 방법)
    T: int = 1000  # diffusion 학습 시간 단계 수
    use_bev_sem: bool = False  # 장면 토큰에 LTF가 예측한 BEV 지도 분할(128x256, 7범주)의 16x16 조각 토큰 128개를 더한다
    pred: str = "x0"  # 디코더가 예측하는 것: x0 (깨끗한 경로) / v (v = sqrt(ac)·eps − sqrt(1−ac)·x0, 10단계 시험)
    loss: str = "l1"  # 경로 손실: l1 / mse
    obj: str = ""  # 12단계 B (진단): 물체 토큰 30개를 장면 토큰에 더한다. ltf = LTF 검출 결과, gt = 정답 위치, gtvel = 정답 위치 + 속도

    def to_dict(self):
        return asdict(self)


def obj_features(kind: str, agent_states=None, agent_logits=None, gt=None) -> torch.Tensor:
    """물체 토큰 입력 (B, 30, OBJ_DIM[kind]). ltf: 7단계에 저장한 LTF 검출 결과(agent_states (B,30,5), agent_logits (B,30)).
    gt / gtvel: scripts/gt_objects.py의 obj_gt (B,30,8: x y heading l w vx vy valid). 빈 자리는 0."""
    if kind == "ltf":
        s, p = agent_states.float(), torch.sigmoid(agent_logits.float())[..., None]
    else:
        s, p = gt[..., :5].float(), gt[..., 7:8].float()
    f = [s[..., :2] / 32, torch.cos(s[..., 2:3]), torch.sin(s[..., 2:3]), s[..., 3:5] / 5, p]
    if kind == "gtvel":
        f.append(gt[..., 5:7].float() / 10)
    return torch.cat(f, -1) * (p > 0 if kind != "ltf" else torch.ones_like(p))


def encode_decisions(lon, lon_s, lat, lat_s) -> torch.Tensor:
    """결정 → M2 입력. 인자는 모두 (B, N_SEG) 텐서(lon, lat는 행동 번호). 출력 (B, N_SEG, DEC_IN)."""
    B = lon.shape[0]
    seg = torch.eye(N_SEG, device=lon.device).expand(B, N_SEG, N_SEG)
    return torch.cat([F.one_hot(lon.long(), len(LON_ACTIONS)).float(), F.one_hot(lat.long(), len(LAT_ACTIONS)).float(),
                      lon_s[..., None].float(), lat_s[..., None].float(), seg], -1)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, T, T + 1, dtype=torch.float64)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()  # (T,), t=0이 가장 덜 섞인 단계


def _decoder(cfg: ModelConfig, n_layers: int) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ffn, cfg.dropout, batch_first=True,
                                       norm_first=True, activation="gelu")
    return nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(cfg.d_model))


class Planner(nn.Module):
    def __init__(self, cfg: ModelConfig, pose_mean: torch.Tensor, pose_std: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.register_buffer("pose_mean", pose_mean.reshape(N_POSES, POSE_DIM).float())
        self.register_buffer("pose_std", pose_std.reshape(N_POSES, POSE_DIM).float())
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(cfg.T))
        # 장면 토큰
        self.scene_in = nn.Sequential(nn.LayerNorm(cfg.feat_dim), nn.Linear(cfg.feat_dim, D))
        n_scene = 65 + (31 if cfg.use_query_out else 0)
        self.scene_pos = nn.Parameter(torch.zeros(1, n_scene, D))
        if cfg.use_bev_sem:  # 조각 하나 = 16x16 픽셀(4 m x 4 m)의 범주 one-hot을 펼친 것
            self.bev_in = nn.Sequential(nn.Linear(BEV_CLASSES * BEV_PATCH * BEV_PATCH, D), nn.LayerNorm(D))
            self.bev_pos = nn.Parameter(torch.zeros(1, (128 // BEV_PATCH) * (256 // BEV_PATCH), D))
        if cfg.obj:  # 물체 하나 = 토큰 하나. 위치는 값 안에 있으므로 위치 임베딩 대신 종류 임베딩 하나만 둔다
            self.obj_in = nn.Sequential(nn.Linear(OBJ_DIM[cfg.obj], D), nn.GELU(), nn.Linear(D, D), nn.LayerNorm(D))
            self.obj_type = nn.Parameter(torch.zeros(1, 1, D))
        # M2 결정 인코더
        if cfg.use_decision:
            self.dec_in = nn.Sequential(nn.Linear(DEC_IN, D), nn.GELU(), nn.Linear(D, D), nn.LayerNorm(D))
        # M3 경로 생성
        self.pose_in = nn.Linear(POSE_DIM, D)
        self.pose_pos = nn.Parameter(torch.zeros(1, N_POSES, D))
        self.time_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.traj_dec = _decoder(cfg, cfg.n_layers)
        self.pose_out = nn.Linear(D, POSE_DIM)
        # M3 판단 토큰
        if cfg.judge:
            self.judge_query = nn.Parameter(torch.zeros(1, 1, D))
            self.judge_dec = _decoder(cfg, cfg.n_judge_layers)
            self.flag_head = nn.Linear(D, 1)
            self.reason_head = nn.Linear(D, len(REASONS))
        for p in (self.scene_pos, self.pose_pos) + ((self.bev_pos,) if cfg.use_bev_sem else ()) + \
                ((self.obj_type,) if cfg.obj else ()):
            nn.init.normal_(p, std=0.02)
        if cfg.judge:
            nn.init.normal_(self.judge_query, std=0.02)

    # ---- 정규화 ----
    def normalize(self, poses):
        return (poses - self.pose_mean) / self.pose_std

    def denormalize(self, z):
        return z * self.pose_std + self.pose_mean

    # ---- 조건 (장면 토큰 + 결정 토큰) ----
    def bev_tokens(self, bev: torch.Tensor) -> torch.Tensor:
        """BEV 지도 분할 (B, 128, 256) 범주 번호 → 조각 토큰 (B, 128, D)."""
        B, H, W = bev.shape
        x = F.one_hot(bev.long(), BEV_CLASSES).float()  # (B, H, W, C)
        x = x.view(B, H // BEV_PATCH, BEV_PATCH, W // BEV_PATCH, BEV_PATCH, BEV_CLASSES)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, (H // BEV_PATCH) * (W // BEV_PATCH), -1)
        return self.bev_in(x) + self.bev_pos

    def memory(self, keyval, query_out, dec: Optional[torch.Tensor], bev: Optional[torch.Tensor] = None,
               obj: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = [keyval.float()] + ([query_out.float()] if self.cfg.use_query_out else [])
        mem = self.scene_in(torch.cat(feats, 1)) + self.scene_pos
        if self.cfg.use_bev_sem:
            assert bev is not None, "BEV 지도 분할 입력이 필요한 모델이다"
            mem = torch.cat([mem, self.bev_tokens(bev)], 1)
        if self.cfg.obj:
            assert obj is not None, "물체 토큰 입력이 필요한 모델이다"
            mem = torch.cat([mem, self.obj_in(obj.float()) + self.obj_type], 1)
        if self.cfg.use_decision:
            assert dec is not None, "결정 입력이 필요한 모델이다"
            mem = torch.cat([mem, self.dec_in(dec)], 1)
        return mem

    def denoise(self, z_t, t, mem, return_hidden: bool = False):
        """노이즈 섞인 정규화 경로 z_t (B, 8, 3), 시간 t (B,) → 예측한 깨끗한 정규화 경로 (B, 8, 3).
        return_hidden이면 (경로, 디코더 마지막 층의 waypoint 은닉 상태 (B, 8, D))를 낸다 (12단계 probe)."""
        temb = self.time_mlp(timestep_embedding(t, self.cfg.d_model))[:, None]
        h = self.traj_dec(self.pose_in(z_t) + self.pose_pos + temb, mem)
        return (self.pose_out(h), h) if return_hidden else self.pose_out(h)

    def judge_out(self, mem) -> Dict[str, torch.Tensor]:
        h = self.judge_dec(self.judge_query.expand(mem.shape[0], -1, -1), mem)[:, 0]
        return {"flag_logit": self.flag_head(h)[:, 0], "reason_logit": self.reason_head(h)}

    # ---- 학습 ----
    def loss(self, keyval, query_out, dec, target_poses, flag=None, reason=None, traj_weight=None, bev=None,
             obj=None) -> Dict:
        mem = self.memory(keyval, query_out, dec, bev, obj)
        x0 = self.normalize(target_poses)
        B = x0.shape[0]
        t = torch.randint(0, self.cfg.T, (B,), device=x0.device)
        ac = self.alphas_cumprod[t][:, None, None]
        eps = torch.randn_like(x0)
        z_t = ac.sqrt() * x0 + (1 - ac).sqrt() * eps
        pred = self.denoise(z_t, t, mem)
        target = x0 if self.cfg.pred == "x0" else ac.sqrt() * eps - (1 - ac).sqrt() * x0
        diff = pred - target
        l_traj = (diff.abs() if self.cfg.loss == "l1" else diff ** 2).mean((1, 2))
        w = traj_weight if traj_weight is not None else torch.ones_like(l_traj)
        out = {"traj": (l_traj * w).sum() / w.sum().clamp(min=1e-6)}
        if self.cfg.judge and flag is not None:
            j = self.judge_out(mem)
            out["flag"] = F.binary_cross_entropy_with_logits(j["flag_logit"], flag.float())
            rej = flag < 0.5  # 근거는 REJECT 샘플에서만 학습한다
            out["reason"] = (F.binary_cross_entropy_with_logits(j["reason_logit"][rej], reason[rej].float())
                             if rej.any() else j["reason_logit"].sum() * 0)
        return out

    # ---- 샘플링 ----
    @torch.no_grad()
    def sample(self, keyval, query_out, dec=None, n_steps: int = 10, generator=None,
               z0: Optional[torch.Tensor] = None, bev: Optional[torch.Tensor] = None,
               obj: Optional[torch.Tensor] = None, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        """z0: 시작 노이즈 (B, 8, 3). 주지 않으면 generator로 뽑는다. return_hidden이면 마지막 단계의 waypoint
        은닉 상태(out["hidden"], (B, 8, D))도 낸다."""
        mem = self.memory(keyval, query_out, dec, bev, obj)
        B = mem.shape[0]
        z = z0 if z0 is not None else torch.randn(B, N_POSES, POSE_DIM, device=mem.device, generator=generator)
        ts = torch.linspace(self.cfg.T - 1, 0, n_steps, device=mem.device).round().long()
        for i, t in enumerate(ts):
            tt = t.expand(B)
            last = i == len(ts) - 1
            out = self.denoise(z, tt, mem, return_hidden=return_hidden and last)
            if return_hidden and last:
                out, hidden = out
            ac = self.alphas_cumprod[t]
            x0 = out if self.cfg.pred == "x0" else ac.sqrt() * z - (1 - ac).sqrt() * out
            if i == len(ts) - 1:
                z = x0
                break
            ac_next = self.alphas_cumprod[ts[i + 1]]
            eps = (z - ac.sqrt() * x0) / (1 - ac).sqrt()
            z = ac_next.sqrt() * x0 + (1 - ac_next).sqrt() * eps  # DDIM, eta = 0
        out = {"poses": self.denormalize(z)}
        if return_hidden:
            out["hidden"] = hidden
        if self.cfg.judge:
            out.update(self.judge_out(mem))
        return out
