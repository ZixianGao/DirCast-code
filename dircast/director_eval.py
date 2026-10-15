#!/usr/bin/env python3
"""Autonomous-director evaluation (free running, no human history) + rule baselines.

Metrics over strict & LIVE test seconds (all clips):
  top1 / any_visible     agreement with the human director's on-air feed
  cuts_per_min           switching rate (human reference shown alongside)
  kill_coverage          fraction of hero kills (OpenDota kills_log, game clock) whose killer or victim feed is
                         on air within +-1 s of the kill
Baselines (causal, use only the per-feed game state up to t): random switching at the human rate, static first
feed, "action director" (feed maximizing enemies-nearby + recent damage, min hold 3 s), and the human director.

  python -m dircast.director_eval --pred eval/dircast_v3/director_free.jsonl [--pred ...]
"""
import argparse
import json
import random
from functools import lru_cache

import numpy as np

from .data import load_rows, match_of, window_dir
from .match_context import _heroes, _norm

ROOT = None


@lru_cache(maxsize=None)
def kills_by_match(mid):
    """-> list of (game_clock, killer_slot, victim_slot)."""
    from .match_context import META
    meta = json.loads((META / "matches" / f"{mid}.json").read_text())
    by_id, by_norm = _heroes()
    slot_of_hero = {}
    for p in meta["players"]:
        ps = p["player_slot"]; slot = ps if ps < 128 else 5 + ps - 128
        slot_of_hero[_norm(by_id[p["hero_id"]]["name"])] = slot
    out = []
    for p in meta["players"]:
        ps = p["player_slot"]; slot = ps if ps < 128 else 5 + ps - 128
        for k in p.get("kills_log") or []:
            v = slot_of_hero.get(_norm(k["key"]))
            if v is not None:
                out.append((float(k["time"]), slot, v))
    return out


@lru_cache(maxsize=None)
def window_clock(wdir):
    return [json.loads(l)["game_clock_sec"] for l in open(f"{wdir}/labels/labels.jsonl")]


def clip_kills(row):
    """-> list of (t_in_clip, killer, victim) for kills inside the clip."""
    s0, s1 = map(int, row["source_window_sec"])
    clock = window_clock(str(window_dir(row)))[s0:s1]
    out = []
    for gt, killer, victim in kills_by_match(match_of(row)):
        for t, c in enumerate(clock):
            if c is not None and abs(c - gt) < 0.5:
                out.append((t, killer, victim)); break
    return out


def rule_baselines(row, rng):
    T = len(row["camera_labels"])
    st = np.load(window_dir(row) / "state_feats.npy")[int(row["source_window_sec"][0]):][:T]
    action = st[:, :, 4] * 5 + 3 * st[:, :, 3] + 2 * st[:, :, 7] + st[:, :, 12]   # enemies near, damage, death, crowd
    act, hold, cur = [], 0, None
    for t in range(T):
        best = int(action[t].argmax()) if len(action[t]) else 0
        if cur is None or (best != cur and hold >= 3 and action[t, best] > action[t, cur] + 0.5):
            cur, hold = best, 0
        hold += 1; act.append(cur)
    rate = 0.13
    rnd, cur = [], rng.randrange(10)
    for t in range(T):
        if t and rng.random() < rate:
            cur = rng.randrange(10)
        rnd.append(cur)
    return {"random": rnd, "static": [0] * T, "action_director": act}


def score(rows, preds_of):
    agg = {}
    for r in rows:
        cam, m, vis = r["camera_labels"], r["_strict"], r["_visible"]
        kills = clip_kills(r)
        for name, pred in preds_of(r).items():
            a = agg.setdefault(name, {"n": 0, "top1": 0, "vis": 0, "cuts": 0, "secs": 0, "kills": 0, "kcov": 0})
            for t in range(len(cam)):
                if not m[t]:
                    continue
                a["n"] += 1; a["top1"] += pred[t] == cam[t]; a["vis"] += pred[t] == cam[t] or (pred[t] in vis[t])
            a["secs"] += len(pred); a["cuts"] += sum(pred[t] != pred[t - 1] for t in range(1, len(pred)))
            for t, killer, victim in kills:
                if not m[t]:
                    continue
                a["kills"] += 1
                a["kcov"] += any(pred[q] in (killer, victim) for q in range(max(0, t - 1), min(len(pred), t + 2)))
    return {k: {"top1": v["top1"] / max(v["n"], 1), "any_visible": v["vis"] / max(v["n"], 1),
                "cuts_per_min": 60 * v["cuts"] / max(v["secs"], 1), "kill_coverage": v["kcov"] / max(v["kills"], 1),
                "seconds": v["n"], "kills": v["kills"]} for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="*", default=[], help="director_free.jsonl files from dircast.evaluate")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load_rows(a.split)
    for i, r in enumerate(rows):
        live = [s == "LIVE" for s in r["vod_state_labels"]]
        r["_strict"] = [bool(x) and l for x, l in zip(r["strict_mask"], live)]
        r["_visible"] = [set(v or []) for v in (r.get("visible_slots") or [[]] * len(live))]
        r["_idx"] = i
    model_preds = {}
    for f in a.pred:
        name = f.split("/")[-2] if "/" in f else f
        model_preds[name] = {json.loads(l)["idx"]: json.loads(l)["pred_cam"] for l in open(f)}
    rng = random.Random(0)

    def preds_of(r):
        d = rule_baselines(r, rng)
        d["human_director"] = r["camera_labels"]
        for name, mp in model_preds.items():
            if r["_idx"] in mp:
                d[name] = mp[r["_idx"]]
        return d
    res = score(rows, preds_of)
    for k, v in res.items():
        print(f"{k:22s} " + " ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
