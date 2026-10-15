"""Per-match public context: feed slot -> hero / player / team.

Feed k is the hero-chase camera of player slot k (replay ``m_iPlayerID // 2``,
0-4 Radiant, 5-9 Dire).  The hero of each slot is read from the packaged
labels (``target_labels`` at seconds whose ``camera_labels`` is that slot) and
cross-checked against OpenDota ``players[*].hero_id`` (player_slot 0-4 / 128-132).
This is the information a broadcast viewer sees on the HUD (hero portraits,
player names), so it may be given to the model as text.
"""
import collections
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DIRCAST_DATA", ROOT / "data"))   # dataset root (index/, matches/, meta/)
META = DATA / "meta"


def _norm(s: str) -> str:
    return s.lower().replace("npc_dota_hero_", "").replace("_", "").replace(" ", "")


@lru_cache(maxsize=None)
def _heroes():
    h = json.loads((META / "constants" / "heroes.json").read_text())
    by_id = {int(k): v for k, v in h.items()}
    by_norm = {_norm(v["name"]): v for v in by_id.values()}
    return by_id, by_norm


@lru_cache(maxsize=None)
def match_context(match_id: int, index_dir: str = str(DATA / "index")):
    by_id, by_norm = _heroes()
    meta = json.loads((META / "matches" / f"{match_id}.json").read_text())
    od = {}
    for p in meta["players"]:
        ps = p["player_slot"]
        slot = ps if ps < 128 else 5 + (ps - 128)
        od[slot] = p
    # hero per slot from labels
    votes = collections.defaultdict(collections.Counter)
    for sp in ("train", "val", "test"):
        f = Path(index_dir) / f"{sp}.jsonl"
        for line in open(f):
            if f'"{match_id}"' not in line[:200] and str(match_id) not in line[:300]:
                continue
            r = json.loads(line)
            if str(match_id) not in str(r["video_id"]):
                continue
            for c, t in zip(r["camera_labels"], r["target_labels"]):
                if c is not None and c >= 0 and t:
                    votes[c][_norm(t)] += 1
    slots = []
    for k in range(10):
        p = od.get(k, {})
        h_od = by_id.get(p.get("hero_id"))
        lab = votes[k].most_common(1)[0][0] if votes[k] else None
        h_lab = by_norm.get(lab) if lab else None
        hero = h_od or h_lab
        slots.append({
            "slot": k,
            "team": "Radiant" if k < 5 else "Dire",
            "hero": hero["localized_name"] if hero else (lab or "unknown"),
            "player": p.get("name") or p.get("personaname") or f"player {k}",
            "label_agrees": (h_od is not None and h_lab is not None and h_od["id"] == h_lab["id"]),
        })
    return {
        "match_id": match_id,
        "radiant": meta.get("radiant_name") or "Radiant",
        "dire": meta.get("dire_name") or "Dire",
        "league": (meta.get("league") or {}).get("name", ""),
        "slots": slots,
    }


def context_text(ctx) -> str:
    lines = [f"Dota 2 match: {ctx['radiant']} (Radiant) vs {ctx['dire']} (Dire)."]
    if ctx["league"]:
        lines[0] = f"{ctx['league']}. " + lines[0]
    for s in ctx["slots"]:
        lines.append(f"Feed {s['slot']}: {s['hero']} ({s['team']}, {s['player']})")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    ids = [int(x) for x in sys.argv[1:]] or sorted(int(p.name) for p in (DATA / "matches").iterdir() if p.name.isdigit())
    bad = 0
    for m in ids:
        c = match_context(m)
        n = sum(s["label_agrees"] for s in c["slots"])
        bad += n < 10
        print(m, f"agree {n}/10", c["radiant"], "vs", c["dire"], "|", ", ".join(s["hero"] for s in c["slots"]))
    print("matches with slot/hero disagreement:", bad)
