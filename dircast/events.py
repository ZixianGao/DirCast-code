"""Causal match-event stream (what the broadcast HUD / spectator client shows), one text line per second.

From OpenDota match details (data/meta/matches/<id>.json), timed by the per-second game clock of the window
(labels.jsonl, pause-aware): kills ("Clockwerk kills Techies"), first blood, buildings, Roshan, and every 10 s a
scoreboard line (team kills so far and the gold lead at the current minute). An event appears at the first second
whose game clock has passed its time, so nothing from the future is used.
"""
import json
import re
from functools import lru_cache

from .data import match_of, window_dir
from .match_context import META, _heroes, _norm, match_context


def _slot(ps):
    return ps if ps < 128 else 5 + ps - 128


def _building(key):
    m = re.match(r"npc_dota_(goodguys|badguys)_(tower|melee_rax|range_rax|fort)(\d)?_?(top|mid|bot)?", key or "")
    if not m:
        return None
    side = "Radiant" if m.group(1) == "goodguys" else "Dire"
    kind = {"tower": f"tier-{m.group(3)} tower", "melee_rax": "melee barracks", "range_rax": "ranged barracks",
            "fort": "Ancient"}[m.group(2)]
    lane = f" {m.group(4)}" if m.group(4) else ""
    return f"{side}{lane} {kind} falls"


@lru_cache(maxsize=None)
def _items():
    it = json.loads((META / "constants" / "items.json").read_text())
    return {k: (v.get("dname") or k, v.get("cost") or 0) for k, v in it.items()}


@lru_cache(maxsize=None)
def match_events(mid, v2=False):
    """-> (events [(time, text)], kill times per team {Radiant: [...], Dire: [...]}, gold_adv per minute)."""
    meta = json.loads((META / "matches" / f"{mid}.json").read_text())
    by_id, _ = _heroes()
    ctx = match_context(mid)
    hero_of = {s["slot"]: s["hero"] for s in ctx["slots"]}
    slot_of_hero = {_norm(by_id[p["hero_id"]]["name"]): _slot(p["player_slot"]) for p in meta["players"]}
    ev, kills = [], {"Radiant": [], "Dire": []}
    for p in meta["players"]:
        k = _slot(p["player_slot"])
        for kl in p.get("kills_log") or []:
            v = slot_of_hero.get(_norm(kl["key"]))
            if v is None:
                continue
            ev.append((float(kl["time"]), f"{hero_of[k]} kills {hero_of[v]}"))
            kills["Radiant" if k < 5 else "Dire"].append(float(kl["time"]))
    for o in meta.get("objectives") or []:
        t, typ = float(o.get("time", 0)), o.get("type", "")
        if typ == "CHAT_MESSAGE_FIRSTBLOOD":
            ev.append((t, "First blood"))
        elif typ == "building_kill":
            b = _building(o.get("key"))
            if b:
                ev.append((t, b))
        elif typ == "CHAT_MESSAGE_ROSHAN_KILL":
            ev.append((t, f"{'Radiant' if o.get('team') == 2 else 'Dire'} kill Roshan"))
        elif typ == "CHAT_MESSAGE_AEGIS":
            k = o.get("slot")
            if k is not None and k in hero_of:
                ev.append((t, f"{hero_of[k]} picks up the Aegis"))
    if v2:   # HUD-level events: major item purchases (>= 1800 gold), buybacks
        items = _items()
        for p in meta["players"]:
            k = _slot(p["player_slot"])
            for pl in p.get("purchase_log") or []:
                name, cost = items.get(pl.get("key"), (pl.get("key"), 0))
                if cost >= 1800 and not str(pl.get("key", "")).startswith("recipe"):
                    ev.append((float(pl["time"]), f"{hero_of[k]} buys {name}"))
            for b in p.get("buyback_log") or []:
                ev.append((float(b["time"]), f"{hero_of[k]} buys back"))
        # (OpenDota teamfights are detected post hoc from later deaths -> not causal, deliberately excluded)
    ev.sort()
    return ev, kills, list(meta.get("radiant_gold_adv") or [])


XP_LEVEL = [0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 5900, 7000, 8200, 9500, 10900, 12400, 14000, 15700,
            17500, 19400, 21400, 23600, 26000, 28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900]


@lru_cache(maxsize=None)
def hero_board(mid):
    """Per slot: (hero, gold_t, xp_t, kill times, death times) for causal scoreboard snapshots."""
    meta = json.loads((META / "matches" / f"{mid}.json").read_text())
    by_id, _ = _heroes()
    ctx = match_context(mid)
    hero_of = {s["slot"]: s["hero"] for s in ctx["slots"]}
    slot_of_hero = {_norm(by_id[p["hero_id"]]["name"]): _slot(p["player_slot"]) for p in meta["players"]}
    board = {}
    deaths = {k: [] for k in range(10)}
    for p in meta["players"]:
        k = _slot(p["player_slot"])
        kt = []
        for kl in p.get("kills_log") or []:
            kt.append(float(kl["time"]))
            v = slot_of_hero.get(_norm(kl["key"]))
            if v is not None:
                deaths[v].append(float(kl["time"]))
        board[k] = [hero_of[k], list(p.get("gold_t") or []), list(p.get("xp_t") or []), kt]
    for k in board:
        board[k].append(deaths[k])
    return board


def board_line(mid, c):
    """Scoreboard at game clock c (values of the last completed minute; kills/deaths up to c)."""
    mi = int(c // 60)
    parts = []
    for k, (hero, gold, xp, kt, dt) in sorted(hero_board(mid).items()):
        lvl = sum(x <= (xp[mi] if 0 <= mi < len(xp) else 0) for x in XP_LEVEL)
        nw = gold[mi] if 0 <= mi < len(gold) else 0
        parts.append(f"{hero} L{lvl} {sum(x <= c for x in kt)}/{sum(x <= c for x in dt)} {nw / 1000:.1f}k")
    return "board: " + ", ".join(parts)


@lru_cache(maxsize=None)
def _window_clock(wdir):
    return [json.loads(l)["game_clock_sec"] for l in open(f"{wdir}/labels/labels.jsonl")]


def kill_feats(row, recent=5.0):
    """Entity-grounded event features per feed [T, 10, 2] (float32): the hero of feed k got a kill this second /
    within the last ``recent`` s of game clock (causal; the victim side is already in state_feats: died now / recently)."""
    import numpy as np
    mid = match_of(row)
    meta = json.loads((META / "matches" / f"{mid}.json").read_text())
    kt = {_slot(p["player_slot"]): [float(k["time"]) for k in (p.get("kills_log") or [])] for p in meta["players"]}
    s0, s1 = map(int, row["source_window_sec"])
    clock = _window_clock(str(window_dir(row)))
    T = len(row["camera_labels"])
    out = np.zeros((T, 10, 2), np.float32)
    for t in range(min(T, s1 - s0)):
        w = s0 + t
        if w >= len(clock) or clock[w] is None:
            continue
        c = clock[w]
        cp = clock[w - 1] if 0 < w <= len(clock) and clock[w - 1] is not None else c - 1
        for k in range(10):
            ts = kt.get(k, [])
            out[t, k, 0] = float(any(cp < x <= c for x in ts))
            out[t, k, 1] = float(any(c - recent < x <= c for x in ts))
    return out


def events_by_second(row, v2=False, board=False):
    """-> list[str] of length T (empty string = nothing new this second)."""
    mid = match_of(row)
    ev, kills, gold = match_events(mid, v2)
    s0, s1 = map(int, row["source_window_sec"])
    clock = _window_clock(str(window_dir(row)))
    out = []
    for t in range(s1 - s0):
        w = s0 + t
        c = clock[w] if w < len(clock) else None
        cp = clock[w - 1] if 0 < w <= len(clock) else None
        parts = []
        if c is not None:
            lo = cp if cp is not None else c - 1
            parts += [txt for (tt, txt) in ev if lo < tt <= c]
            if t % 10 == 0:
                rk = sum(x <= c for x in kills["Radiant"]); dk = sum(x <= c for x in kills["Dire"])
                line = f"score Radiant {rk} - {dk} Dire"
                mi = int(c // 60)
                if 0 <= mi < len(gold) and c >= 0:
                    g = gold[mi]
                    line += f", gold lead {'Radiant' if g >= 0 else 'Dire'} {abs(g) / 1000:.1f}k"
                parts.append(line)
            if board and t % 30 == 0 and c >= 0:
                parts.append(board_line(mid, c))
        out.append("; ".join(parts))
    return out
