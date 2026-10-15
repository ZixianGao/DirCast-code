#!/usr/bin/env python3
"""Additional director metrics on the free-running test predictions (same rows / masks as director_eval.py).

  tol-top1         top-1 agreement allowing the human cut to be off by one second (pred[t] in cam[t-1..t+1])
  cut F1 @2s       precision / recall of cut events against the human cuts, matched within +-2 s (one-to-one)
  shot len         mean shot length (s) and the 1-Wasserstein distance of the shot-length distribution to the human's
  kill latency     for each kill, seconds until a participant (killer or victim) is on air, searched in [t, t+5];
                   median over kills that are reached, and the share reached within 2 s
  mention ground   (joint models only) share of commentary seconds that name a hero whose feed is on air within +-2 s;
                   reference: the human commentary against the human program

  python -m dircast.director_eval_extra --pred NAME=path/director_free.jsonl[:path/commentary_free.jsonl] ...
"""
import argparse
import json
import random
import re

import numpy as np

from .data import load_rows, match_of
from .director_eval import clip_kills, rule_baselines
from .match_context import match_context

ALIAS = {"Keeper of the Light": ["kotl", "keeper"], "Shadow Fiend": ["sf", "shadow fiend", "nevermore"],
         "Terrorblade": ["tb", "terrorblade"], "Outworld Devourer": ["od", "outworld"], "Nature's Prophet": ["prophet", "furion"],
         "Centaur Warrunner": ["centaur"], "Treant Protector": ["treant"]}


def cuts(seq):
    return [t for t in range(1, len(seq)) if seq[t] != seq[t - 1]]


def shots(seq):
    out, n = [], 1
    for t in range(1, len(seq)):
        if seq[t] == seq[t - 1]:
            n += 1
        else:
            out.append(n); n = 1
    out.append(n)
    return out


def match_cuts(p, h, tol=2):
    used, tp = set(), 0
    for c in p:
        best = None
        for j, d in enumerate(h):
            if j not in used and abs(d - c) <= tol and (best is None or abs(d - c) < abs(h[best] - c)):
                best = j
        if best is not None:
            used.add(best); tp += 1
    return tp


def wasserstein(a, b):
    a, b = np.sort(np.array(a, float)), np.sort(np.array(b, float))
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def hero_terms(row):
    terms = {}
    for s in match_context(match_of(row))["slots"]:
        names = [s["hero"].lower()] + ALIAS.get(s["hero"], [])
        terms[s["slot"]] = [n for n in names if len(n) >= 2]
    return terms


def mentions(text, terms):
    t = " " + re.sub(r"[^a-z' ]", " ", text.lower()) + " "
    return {k for k, ns in terms.items() if any(f" {n} " in t for n in ns)}


def evaluate(rows, preds, texts=None):
    agg = {"n": 0, "tol": 0, "pc": 0, "hc": 0, "tp": 0, "shots": [], "hshots": [], "lat": [], "kills": 0,
           "men": 0, "men_ok": 0}
    for i, r in enumerate(rows):
        if i not in preds:
            continue
        p, cam = preds[i], r["camera_labels"]
        live = [s == "LIVE" for s in r["vod_state_labels"]]
        strict = [bool(x) and l and c is not None and c >= 0 for x, l, c in zip(r["strict_mask"], live, cam)]
        T = len(p)
        for t in range(T):
            if strict[t]:
                agg["n"] += 1
                agg["tol"] += p[t] in {cam[q] for q in range(max(0, t - 1), min(T, t + 2))}
        h = [c if c is not None and c >= 0 else -1 for c in cam]
        hv = [t for t in cuts(h) if h[t] >= 0 and h[t - 1] >= 0]
        pc = cuts(p)
        agg["pc"] += len(pc); agg["hc"] += len(hv); agg["tp"] += match_cuts(pc, hv)
        agg["shots"] += shots(p); agg["hshots"] += shots([x for x in h if x >= 0] or [0])
        for t, killer, victim in clip_kills(r):
            if not strict[t]:
                continue
            agg["kills"] += 1
            for q in range(t, min(T, t + 6)):
                if p[q] in (killer, victim):
                    agg["lat"].append(q - t); break
        if texts is not None and i in texts:
            terms = hero_terms(r)
            for t, txt in texts[i].items():
                t = int(t)
                if t >= T or not strict[t]:
                    continue
                m = mentions(txt, terms)
                if m:
                    agg["men"] += 1
                    agg["men_ok"] += bool(m & {p[q] for q in range(max(0, t - 2), min(T, t + 3))})
    prec = agg["tp"] / max(1, agg["pc"]); rec = agg["tp"] / max(1, agg["hc"])
    res = {"tol_top1": agg["tol"] / max(1, agg["n"]),
           "cut_P": prec, "cut_R": rec, "cut_F1": 2 * prec * rec / max(1e-9, prec + rec),
           "shot_len": float(np.mean(agg["shots"])), "shot_W1": wasserstein(agg["shots"], agg["hshots"]),
           "kill_lat_med": float(np.median(agg["lat"])) if agg["lat"] else float("nan"),
           "kill_within2": sum(1 for x in agg["lat"] if x <= 2) / max(1, agg["kills"])}
    if texts is not None:
        res["mention_ground"] = agg["men_ok"] / max(1, agg["men"]); res["mentions"] = agg["men"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="*", default=[])
    a = ap.parse_args()
    rows = load_rows("test")
    rng = random.Random(0)
    base = {k: {} for k in ("random", "static", "action_director", "human_director")}
    for i, r in enumerate(rows):
        d = rule_baselines(r, rng)
        for k in ("random", "static", "action_director"):
            base[k][i] = d[k]
        base["human_director"][i] = [c if c is not None and c >= 0 else 0 for c in r["camera_labels"]]
    human_txt = {i: {str(t): s for t, s in enumerate(r.get("commentary_by_second") or []) if s and s.strip()} for i, r in enumerate(rows)}
    for k, p in base.items():
        res = evaluate(rows, p, human_txt if k == "human_director" else None)
        print(f"{k:24s} " + " ".join(f"{x}={y:.3f}" for x, y in res.items()))
    for spec in a.pred:
        name, paths = spec.split("=", 1)
        dp, _, cp = paths.partition(":")
        preds = {json.loads(l)["idx"]: json.loads(l)["pred_cam"] for l in open(dp)}
        texts = {json.loads(l)["idx"]: json.loads(l)["pred"] for l in open(cp)} if cp else None
        res = evaluate(rows, preds, texts)
        print(f"{name:24s} " + " ".join(f"{x}={y:.3f}" for x, y in res.items()))


if __name__ == "__main__":
    main()
