#!/usr/bin/env python3
"""Broadcast-quality metrics for directors, on strict LIVE seconds of the test split.

  cut target       at every human cut (both sides labelled), the human's new feed is on air within [t, t+2]
  fight coverage   seconds with a fight (some hero lost hp in the last 3 s with an enemy within 1200) where the
                   on-air hero is a fight participant
  dead feed        seconds whose on-air hero is dead (lower is better)
  flicker          share of shots shorter than 2 s (lower is better)
  correct cut      a predicted cut (on a LIVE second) is correct when it is matched one-to-one to a human cut
                   (both sides labelled and LIVE) within +-2 s that goes to the same feed; precision, recall, F1

  python -m dircast.director_eval_quality --pred NAME=path/director_free.jsonl ...
"""
import argparse
import json
import random

import numpy as np

from dircast.director_eval import load_rows, rule_baselines, window_dir
from dircast.director_eval_extra import shots


def fight_set(st_t):
    """participants of a fight at one second: heroes that lost hp in the last 3 s with an enemy nearby,
    plus the enemies near them."""
    alive = st_t[:, 0] > 0.5
    hurt = (st_t[:, 3] > 0.03) & (st_t[:, 4] > 0) & alive
    return set(np.where(hurt)[0].tolist())


def cut_list(seq, valid=None):
    return [(t, seq[t]) for t in range(1, len(seq)) if seq[t] != seq[t - 1] and (valid is None or (valid[t] and valid[t - 1]))]


def correct_cuts(p, r):
    """-> (matched, predicted cuts, human cuts) for one clip."""
    cam = [c if c is not None and c >= 0 else -1 for c in r["camera_labels"]]
    live = [s == "LIVE" for s in r["vod_state_labels"]]
    hc = cut_list(cam, [c >= 0 and l for c, l in zip(cam, live)])
    pc = [(t, c) for t, c in cut_list(p) if live[t]]
    used = set()
    for t, c in pc:
        for j, (u, d) in enumerate(hc):
            if j not in used and d == c and abs(u - t) <= 2:
                used.add(j); break
    return len(used), len(pc), len(hc)


def evaluate(rows, preds):
    a = {"cuts": 0, "cut_ok": 0, "fight": 0, "fight_ok": 0, "n": 0, "dead": 0, "shots": [], "cm": 0, "cp": 0, "ch": 0}
    for i, r in enumerate(rows):
        if i not in preds:
            continue
        p, cam = preds[i], r["camera_labels"]
        T = len(p)
        st = np.load(window_dir(r) / "state_feats.npy")[int(r["source_window_sec"][0]):][:T]
        live = [s == "LIVE" for s in r["vod_state_labels"]]
        strict = [bool(x) and l and c is not None and c >= 0 for x, l, c in zip(r["strict_mask"], live, cam)]
        for t in range(T):
            if not strict[t]:
                continue
            a["n"] += 1
            a["dead"] += st[t, p[t], 0] < 0.5
            f = fight_set(st[t])
            if f:
                a["fight"] += 1
                a["fight_ok"] += p[t] in f
            if t and cam[t - 1] is not None and cam[t - 1] >= 0 and cam[t] != cam[t - 1]:
                a["cuts"] += 1
                a["cut_ok"] += cam[t] in p[t:t + 3]
        a["shots"] += shots(p)
        m, npc, nhc = correct_cuts(p, r)
        a["cm"] += m; a["cp"] += npc; a["ch"] += nhc
    sh = np.array(a["shots"])
    cp, cr = a["cm"] / max(1, a["cp"]), a["cm"] / max(1, a["ch"])
    return {"cut_P": cp, "cut_R": cr, "cut_F1": 2 * cp * cr / max(1e-9, cp + cr),"cut_target": a["cut_ok"] / max(1, a["cuts"]), "fight_cov": a["fight_ok"] / max(1, a["fight"]),
            "dead_feed": a["dead"] / max(1, a["n"]), "flicker": float((sh < 2).mean()) if len(sh) else 0.0,
            "cuts": a["cuts"], "fight_secs": a["fight"], "seconds": a["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="*", default=[])
    args = ap.parse_args()
    rows = load_rows("test")
    rng = random.Random(0)
    base = {k: {} for k in ("random", "static", "action_director", "human_director")}
    for i, r in enumerate(rows):
        d = rule_baselines(r, rng)
        for k in ("random", "static", "action_director"):
            base[k][i] = d[k]
        base["human_director"][i] = [c if c is not None and c >= 0 else 0 for c in r["camera_labels"]]
    todo = list(base.items())
    for spec in args.pred:
        name, path = spec.split("=", 1)
        todo.append((name, {json.loads(l)["idx"]: json.loads(l)["pred_cam"] for l in open(path)}))
    for name, p in todo:
        res = evaluate(rows, p)
        print(f"{name:16s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))


if __name__ == "__main__":
    main()
