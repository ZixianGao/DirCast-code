#!/usr/bin/env python3
"""Pick decision thresholds on the full validation split (never on test).

One teacher-forced parallel pass per val clip gives, per second, P(cut), the target pointer's choice
and P(speak).  tau_cut maximizes cut-event F1 (+-1 s, strict LIVE seconds), tau_speak maximizes the
Proact-VL per-second speaking F1 (macro over clips).

  torchrun --nproc_per_node 8 -m dircast.tune --ckpt runs/dircast_v2/ckpt_ep2.pt
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .data import ClipDataset
from .evaluate import load_model
from .metrics import _cut_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    model, targs = load_model(a.ckpt, dev)
    model.attn_impl = "flex_attention"
    model.lm.base_model.model.model.language_model.config._attn_implementation = "flex_attention"
    ds = ClipDataset(a.split, cam_mask_key="strict_mask", broadcast=targs.get("broadcast", False),
                     events_v2=targs.get("events_v2", False), events_board=targs.get("events_board", False),
                     kill_feats=targs.get("kill_feats", False))
    recs = []
    with torch.no_grad():
        for i in range(rank, len(ds), world):
            it = ds[i]
            cam = it["cam"].to(dev)
            onair = cam.clone()
            for t in range(len(onair)):
                if onair[t] < 0:
                    onair[t] = onair[t - 1] if t > 0 else 0
            T, K = it["frames"].shape[:2]
            extras = model.pick_glimpses(onair.tolist(), it["visible"], K, torch.Generator().manual_seed(i)) \
                if (model.glimpses > 1 and model.foveate) else None
            seq = model.build(it["context"], T, K, onair.tolist(), it["text"], it["game_clock_start"], it["history"], extras, it.get("events"))
            ids = torch.tensor(seq.ids, device=dev)[None]
            pos = torch.tensor(seq.pos, device=dev)[:, None, :]
            pv, grid = model.pixels(it["frames"], seq, dev)
            model.set_state_injection(seq, it.get("state"), dev)
            h = model.lm.base_model.model.model(input_ids=ids, attention_mask=model.attn_mask(seq, dev), position_ids=pos,
                                                pixel_values=pv, image_grid_thw=grid,
                                                mm_token_type_ids=(ids == 151655).int()).last_hidden_state[0]
            model._inject = None
            prev = torch.cat([torch.tensor([-1], device=dev), onair[:-1]])
            target, cut = model.director_heads(h, seq, prev, it.get("state"))
            tg = target.masked_fill(torch.nn.functional.one_hot(prev.clamp(min=0), K).bool() & (prev >= 0)[:, None], -1e4)
            recs.append({"cam": it["cam"].tolist(), "strict": it["strict"].tolist(), "p_cut": torch.sigmoid(cut).tolist(),
                         "switch_to": tg.argmax(-1).tolist(), "p_speak": torch.sigmoid(model.speak_logits(h, seq)).tolist(),
                         "speak": [bool(s.strip()) for s in it["text"]], "has_audio": bool(it["text_mask"].any())})
    allr = [None] * world
    dist.all_gather_object(allr, recs)
    if rank == 0:
        recs = [r for rr in allr for r in rr]
        best_cut = (0, 0.5)
        for tau in np.arange(0.05, 0.96, 0.025):
            tp = npred = ngt = 0
            for r in recs:
                c, m = r["cam"], r["strict"]
                gt = [t for t in range(1, len(c)) if m[t] and m[t - 1] and c[t] != c[t - 1]]
                pr = [t for t in range(1, len(c)) if m[t] and m[t - 1] and r["p_cut"][t] >= tau]
                x = _cut_f1(pr, gt); tp += x[0]; npred += x[1]; ngt += x[2]
            P, R = tp / max(npred, 1), tp / max(ngt, 1)
            f1 = 2 * P * R / (P + R) if P + R else 0
            if f1 > best_cut[0]:
                best_cut = (f1, float(tau))
        best_sp = (0, 0.5)
        for tau in np.arange(0.05, 0.96, 0.025):
            f1s = []
            for r in recs:
                if not r["has_audio"] or not any(r["speak"]):
                    continue
                lab = {t for t, s in enumerate(r["speak"]) if s}
                pr = {t for t, p in enumerate(r["p_speak"]) if p >= tau}
                tp = len(lab & pr)
                P, R = tp / max(len(pr), 1), tp / len(lab)
                f1s.append(2 * P * R / (P + R) if P + R else 0)
            if np.mean(f1s) > best_sp[0]:
                best_sp = (float(np.mean(f1s)), float(tau))
        n_sw = ok_sw = 0
        for r in recs:
            c, m = r["cam"], r["strict"]
            for t in range(1, len(c)):
                if m[t] and m[t - 1] and c[t] != c[t - 1]:
                    n_sw += 1; ok_sw += r["switch_to"][t] == c[t]
        cut_auc_pairs = [(r["p_cut"][t], r["cam"][t] != r["cam"][t - 1]) for r in recs for t in range(1, len(r["cam"]))
                         if r["strict"][t] and r["strict"][t - 1]]
        ps = np.array([p for p, _ in cut_auc_pairs]); ys = np.array([y for _, y in cut_auc_pairs])
        order = ps.argsort(); ranks = np.empty(len(ps)); ranks[order] = np.arange(1, len(ps) + 1)
        auc = (ranks[ys].sum() - ys.sum() * (ys.sum() + 1) / 2) / max(ys.sum() * (~ys).sum(), 1)
        # rate matching: P(cut) >= tau on the same fraction of seconds as the human switches
        human_rate = float(ys.mean())
        tau_rate = float(np.quantile(ps, 1 - human_rate))
        out = {"tau_cut_f1": best_cut[1], "human_switch_rate": human_rate, "switch_target_acc": ok_sw / max(n_sw, 1), "switch_seconds": n_sw, "cut_auc": float(auc),
               "tau_cut": tau_rate, "val_cut_f1_at_best": best_cut[0], "tau_speak": best_sp[1], "val_speak_f1": best_sp[0],
               "ckpt": a.ckpt, "split": a.split, "clips": len(recs)}
        Path(a.ckpt).with_name("thresholds.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
