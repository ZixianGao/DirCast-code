#!/usr/bin/env python3
"""Per-second, per-feed game-state features from the replay trace (what a spectator client shows).

For every window second s (labels.jsonl replay_tick_range [a, b)) and player slot k, the state at the
last tick < b (causal: nothing after the second is used):
   0 alive              1 hp fraction (hp / running max hp)    2 hp lost in the last 1 s (frac)
   3 hp lost in last 3 s 4 enemies within 1200 (/5)            5 allies within 1200 (/4)
   6 nearest enemy distance (/3000, clipped)                    7 died this second
   8 died in the last 5 s 9 distance moved in 1 s (/600)        10 x / 8000   11 y / 8000
   12 heroes (any team) within 1200 (/9)
Writes data/matches/<id>/t<tick>/state_feats.npy  float32 [S, 10, 13].

  python -m dircast.game_state            # all matches
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DIRCAST_DATA", ROOT / "data"))   # dataset root (index/, matches/, meta/)
F = 13
POS = ("CBodyComponent.m_cellX", "CBodyComponent.m_vecX", "CBodyComponent.m_cellY", "CBodyComponent.m_vecY")


def hero_timeline(trace: Path):
    """-> ticks sorted list, per tick list of (slot, x, y, hp, alive) snapshots after applying updates."""
    ents = {}                                        # entity index -> merged fields
    events = []                                      # (tick, slot, x, y, hp, alive, team)
    with open(trace) as f:
        for line in f:
            if '"hero_state"' not in line:
                continue
            d = json.loads(line)
            p = d["payload"]
            e = ents.setdefault(p.get("index"), {})
            e.update({k: v for k, v in p.items() if k not in ("op",)})
            if "m_iPlayerID" not in e or not all(k in e for k in POS):
                continue
            pid = int(e["m_iPlayerID"])
            if pid < 0 or pid > 19:
                continue
            x = float(e["CBodyComponent.m_cellX"]) * 128 + float(e["CBodyComponent.m_vecX"]) - 16384
            y = float(e["CBodyComponent.m_cellY"]) * 128 + float(e["CBodyComponent.m_vecY"]) - 16384
            events.append((int(d["tick"]), pid // 2, x, y, float(e.get("m_iHealth", 0)), int(e.get("m_lifeState", 0)) == 0,
                           int(e.get("m_iTeamNum", 0))))
    events.sort(key=lambda r: r[0])
    return events


def window_features(events, ticks_end):
    """ticks_end: [S] last tick (exclusive) of each second -> [S, 10, F]."""
    S = len(ticks_end)
    state = {k: [0.0, 0.0, 0.0, False, 2 if k < 5 else 3] for k in range(10)}   # x, y, hp, alive, team
    maxhp = np.ones(10)
    out = np.zeros((S, 10, F), np.float32)
    hist_hp, hist_alive, hist_xy = [], [], []
    j = 0
    for s, tb in enumerate(ticks_end):
        while j < len(events) and events[j][0] < tb:
            _, k, x, y, hp, alive, team = events[j]
            state[k] = [x, y, hp, alive, team or state[k][4]]
            if alive:
                maxhp[k] = max(maxhp[k], hp)
            j += 1
        xy = np.array([[state[k][0], state[k][1]] for k in range(10)])
        hp = np.array([state[k][2] for k in range(10)])
        al = np.array([state[k][3] for k in range(10)])
        team = np.array([state[k][4] for k in range(10)])
        hist_hp.append(hp / maxhp); hist_alive.append(al); hist_xy.append(xy)
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        both = al[:, None] & al[None]
        enemy = (team[:, None] != team[None]) & both
        ally = (team[:, None] == team[None]) & both & ~np.eye(10, dtype=bool)
        near = d < 1200
        hp1 = hist_hp[max(0, s - 1)]; hp3 = hist_hp[max(0, s - 3)]
        al1 = hist_alive[max(0, s - 1)]
        died = al1 & ~al
        died5 = np.zeros(10, bool)
        for q in range(max(0, s - 5), s + 1):
            if q > 0:
                died5 |= hist_alive[q - 1] & ~hist_alive[q]
        ne = np.where(enemy, d, 1e9).min(1)
        out[s, :, 0] = al
        out[s, :, 1] = hp / maxhp
        out[s, :, 2] = np.clip(hp1 - hp / maxhp, 0, 1)
        out[s, :, 3] = np.clip(hp3 - hp / maxhp, 0, 1)
        out[s, :, 4] = (enemy & near).sum(1) / 5
        out[s, :, 5] = (ally & near).sum(1) / 4
        out[s, :, 6] = np.clip(ne / 3000, 0, 1)
        out[s, :, 7] = died
        out[s, :, 8] = died5
        out[s, :, 9] = np.clip(np.linalg.norm(xy - hist_xy[max(0, s - 1)], axis=1) / 600, 0, 2) * al
        out[s, :, 10] = xy[:, 0] / 8000
        out[s, :, 11] = xy[:, 1] / 8000
        out[s, :, 12] = ((near & both).sum(1) - 1).clip(0) / 9
    return out


def build_match(mdir: Path):
    events = hero_timeline(mdir / "trace.jsonl")
    n = 0
    for w in sorted(mdir.glob("t*")):
        lab = w / "labels" / "labels.jsonl"
        if not lab.exists():
            continue
        rows = [json.loads(l) for l in open(lab)]
        ends = [r["replay_tick_range"][1] for r in rows]
        np.save(w / "state_feats.npy", window_features(events, ends))
        n += 1
    return mdir.name, n, len(events)


if __name__ == "__main__":
    ms = [DATA / "matches" / m for m in sys.argv[1:]] or sorted(p for p in (DATA / "matches").iterdir() if p.name.isdigit())
    with ProcessPoolExecutor(14) as ex:
        for r in ex.map(build_match, ms):
            print(*r, flush=True)
