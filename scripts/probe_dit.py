"""12단계 D (사후 탐색): 판단을 DiT 안에서 읽으면 더 잘 가르고, 말과 행동이 더 맞는가.

비교하는 판단 점수 (높을수록 '실행할 수 없음'):
- head: 기존 판단 head (1 - p)
- probe: 학습된 모델은 그대로 두고, 노이즈를 걷어 내는 마지막 단계의 waypoint 은닉 상태(8 x 256)를 펼쳐
  2층 MLP로 flag를 학습한 것. 학습은 그 모델의 학습 묶음(보이지 않는 원인 CF⁻ 제외, 클래스 균형 BCE), 검증 묶음 AUC로 고름
- spread: 시작 노이즈 10개로 뽑은 경로가 평균 경로에서 떨어진 거리의 평균 [m]
- combo: probe logit과 log(spread)를 검증 묶음에서 로지스틱 회귀로 합친 것
기준값: 기존과 같다(검증 묶음에서 실행할 수 있는 샘플을 거부하는 비율이 5% 이하가 되는 가장 큰 값).
지표: AUC (규칙 결정 = D2 대 D1, 실제 VLM 3개 = 불가능 대 가능, 판정 불가와 사람과 같은 불가능은 뺌),
말과 행동 일치 = 실제 VLM 결정에서 (REJECT) == (시드 0 경로가 결정을 따르지 않음, L1)의 비율.

사용법: python scripts/probe_dit.py --runs Ours_syn_seed0 Ours_syn_seed1 ...
출력: exp/step12/probe/<run>.json, logs/step12/probe.md
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from predict_planner import load_model, set_table, token_noise  # noqa: E402

from yesman.model import encode_decisions  # noqa: E402
from yesman.train_data import SAMPLE_TYPES, decision_arrays, load_bundle, load_features  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
E = ROOT / "exp/eval"
OUT = ROOT / "exp/step12/probe"
BUNDLE = {"Ours_syn": "_synth", "Ours_WVbalgtvel": "_WV", "Ours_plaus": "_plaus", "Ours_WV": "_WV"}
VLMS = ("gemma31b", "gemma12b", "qwen8b")
N_NOISE = 10


@torch.no_grad()
def run(model, feats, df, noise_seeds=(0,), bs=2048):
    """시드 0의 은닉 상태(펼침, fp16), head p, 시드별 경로 (S, N, 8, 2)."""
    d = decision_arrays(df)
    t = lambda a: torch.as_tensor(a, device="cuda")  # noqa: E731
    dec = encode_decisions(t(d["lon"]), t(d["lon_s"]), t(d["lat"]), t(d["lat_s"]))
    rows = t(df.token.map(feats["row"]).to_numpy())
    hid, p, poses = [], [], []
    for s in noise_seeds:
        z0 = token_noise(df.token.tolist(), s)
        ps = []
        for lo in range(0, len(df), bs):
            sl = slice(lo, lo + bs)
            r = rows[sl]
            o = model.sample(feats["keyval"][r], feats["query_out"][r], dec[sl], z0=z0[sl],
                             obj=feats["obj"][r] if model.cfg.obj else None, return_hidden=s == noise_seeds[0])
            ps.append(o["poses"][..., :2].cpu())
            if s == noise_seeds[0]:
                hid.append(o["hidden"].flatten(1).half())
                p.append(torch.sigmoid(o["flag_logit"]).cpu())
        poses.append(torch.cat(ps))
    return torch.cat(hid), torch.cat(p).numpy(), torch.stack(poses).numpy()


def spread(poses):
    return np.linalg.norm(poses - poses.mean(0), axis=-1).mean((0, 2))


def train_probe(xtr, ytr, xva, yva, steps=3000, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.LayerNorm(xtr.shape[1]), nn.Linear(xtr.shape[1], 256), nn.GELU(), nn.Linear(256, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    ytr_t = torch.as_tensor(ytr, device="cuda", dtype=torch.float)
    pos_w = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), device="cuda")  # 1 = 실행할 수 없음
    best, state = -1, None
    for i in range(1, steps + 1):
        idx = torch.randint(0, len(xtr), (1024,), device="cuda")
        loss = F.binary_cross_entropy_with_logits(net(xtr[idx].float())[:, 0], ytr_t[idx], pos_weight=pos_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 250 == 0:
            with torch.no_grad():
                a = roc_auc_score(yva, logits(net, xva))
            if a > best:
                best, state = a, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, best


@torch.no_grad()
def logits(net, x, bs=8192):
    return torch.cat([net(x[i:i + bs].float())[:, 0] for i in range(0, len(x), bs)]).cpu().numpy()


def tau_for(score_feasible, target=0.05):
    """실행할 수 있는 샘플의 거부 비율이 target 이하가 되는 기준값 (score가 이 값보다 크면 REJECT)."""
    return float(np.quantile(score_feasible, 1 - target))


def one_run(name):
    t0 = time.time()
    model = load_model(name)
    base = name.rsplit("_seed", 1)[0]
    feats = load_features("navtrain", obj=model.cfg.obj)
    split = {}
    for part in ("train", "val"):
        s = load_bundle(part + BUNDLE[base], feats, SAMPLE_TYPES)
        df = s.df[s.df.visibility.fillna("") != "unseen"].reset_index(drop=True)
        if part == "train" and len(df) > 80000:  # 속도: CF⁻는 모두, 나머지는 무작위로 줄인다
            neg = df[df.sample_type == "cf_neg"]
            df = pd.concat([neg, df[df.sample_type != "cf_neg"].sample(80000 - len(neg), random_state=0)]).reset_index(drop=True)
        h, p, poses = run(model, feats, df, noise_seeds=tuple(range(N_NOISE)) if part == "val" else (0,))
        split[part] = dict(h=h, y=(df.flag != "ACCEPT").to_numpy().astype(int), p=p, spread=spread(poses) if part == "val" else None)
    net, val_auc = train_probe(split["train"]["h"], split["train"]["y"], split["val"]["h"], split["val"]["y"])
    va = split["val"]
    lv = logits(net, va["h"])
    combo = LogisticRegression().fit(np.c_[lv, np.log(va["spread"] + 1e-3)], va["y"])
    feas_v = va["y"] == 0
    taus = {"probe": tau_for(lv[feas_v]),
            "head": -json.load(open(E / f"{name}/flag_threshold.json"))["tau"]}  # head 점수 = -p
    res = {"run": name, "probe_val_auc": val_auc, "sets": {}}
    del feats
    feats = load_features("navtest", obj=model.cfg.obj)
    # 규칙 결정: D2(보이지 않는 원인 제외) 대 D1
    d2 = set_table("D2")
    d2 = d2[d2.visibility.fillna("") != "unseen"]
    rule = pd.concat([d2, set_table("D1")], ignore_index=True)
    y = np.r_[np.ones(len(d2), int), np.zeros(len(rule) - len(d2), int)]
    sets = [("rule", rule, y, None)]
    for v in VLMS:
        g = set_table(f"D3x_{v}")
        k = ((g.status == "infeasible") & ~g.same_as_human.fillna(False)) | (g.status == "feasible")
        fol = pd.read_parquet(E / f"{name}/D3x_{v}_l1.parquet").follow.to_numpy().astype(bool)[k.to_numpy()]
        g = g[k].reset_index(drop=True)
        sets.append((v, g, (g.status == "infeasible").to_numpy().astype(int), fol))
    for sname, df, y, fol in sets:
        h, p, poses = run(model, feats, df, noise_seeds=tuple(range(N_NOISE)))
        sc = {"head": -p, "probe": logits(net, h), "spread": spread(poses)}
        sc["combo"] = combo.decision_function(np.c_[sc["probe"], np.log(sc["spread"] + 1e-3)])
        r = {"n_inf": int(y.sum()), "n_fea": int((1 - y).sum()), "auc": {k: roc_auc_score(y, v_) for k, v_ in sc.items()}}
        if fol is not None:
            for k in ("head", "probe"):
                rej = sc[k] > taus[k]
                r[f"{k}_reject_inf"] = float(rej[y == 1].mean())
                r[f"{k}_reject_fea"] = float(rej[y == 0].mean())
                r[f"{k}_agree"] = float((rej == ~fol).mean())
                r[f"{k}_agree_inf"] = float((rej == ~fol)[y == 1].mean())
        res["sets"][sname] = r
        print(f"[{name}] {sname}: " + ", ".join(f"{k} {v_:.3f}" for k, v_ in r["auc"].items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1)
    print(f"[{name}] probe val AUC {val_auc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return res


def summary(runs):
    L = ["# DiT 안에서 판단 읽기 (12단계 D, 사후 탐색, 시드 평균)", "",
         "AUC: 규칙 = D2 대 D1, VLM = 불가능 대 가능. 일치 = (REJECT) == (경로가 결정을 따르지 않음), 실제 VLM 결정 전체 / 불가능만.", ""]
    by = {}
    for r in runs:
        by.setdefault(r["run"].rsplit("_seed", 1)[0], []).append(r)
    for base, rs in by.items():
        L += [f"## {base} (시드 {len(rs)}개, probe 검증 AUC {np.mean([r['probe_val_auc'] for r in rs]):.3f})", "",
              "| 집합 | head | probe | spread | combo | 일치 head / probe | 불가능에서 일치 head / probe | 거부(불가능) head / probe | 거부(가능) head / probe |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in ["rule", *VLMS]:
            m = lambda k: np.mean([r["sets"][s]["auc"][k] for r in rs])  # noqa: E731
            row = f"| {s} | {m('head'):.3f} | {m('probe'):.3f} | {m('spread'):.3f} | {m('combo'):.3f} |"
            if s != "rule":
                g = lambda k: np.mean([r["sets"][s][k] for r in rs])  # noqa: E731
                row += (f" {g('head_agree'):.0%} / {g('probe_agree'):.0%} | {g('head_agree_inf'):.0%} / {g('probe_agree_inf'):.0%} |"
                        f" {g('head_reject_inf'):.0%} / {g('probe_reject_inf'):.0%} | {g('head_reject_fea'):.1%} / {g('probe_reject_fea'):.1%} |")
            else:
                row += " | | | |"
            L.append(row)
        L.append("")
    (ROOT / "logs/step12/probe.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    res = []
    for r in args.runs:
        if not (ROOT / f"exp/train/{r}/model.pt").exists() or not (E / f"{r}/flag_threshold.json").exists():
            print(f"skip {r} (모델 또는 기준값 없음)", flush=True)
            continue
        res.append(one_run(r))
    summary(res)


if __name__ == "__main__":
    main()
