"""Clip dataset for the joint director + commentary model.

One item = one 36 s clip of data/index/{split}.jsonl:
  frames      uint8 [T, K, H, W, 3]  (T=36 s at 1 fps, K=10 hero-chase feeds)
  cam         long  [T]   on-air feed chosen by the human director
  cam_mask    bool  [T]   second supervises the director (verified_mask by default)
  strict      bool  [T]   strict_mask (clean evaluation)
  live        bool  [T]   broadcast is LIVE (only LIVE seconds are director decisions)
  visible     bool  [T, K] feeds whose hero is on the broadcast screen
  text        list[str] commentary of each second ("" = silence)
  text_mask   bool  [T]   second supervises commentary (LIVE and the clip has audio)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .match_context import context_text, match_context

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DIRCAST_DATA", ROOT / "data"))   # dataset root (index/, matches/, meta/)
INDEX = Path(os.environ["DIRCAST_INDEX"]) if os.environ.get("DIRCAST_INDEX") else DATA / "index"   # data/index_polished: LLM-polished commentary


def load_rows(split: str):
    return [json.loads(l) for l in open(INDEX / f"{split}.jsonl")]


HISTORY_SEC = 60


def window_dir(row) -> Path:
    return Path(row["frame_npz"]).parents[1]


def history_text(row, seconds: int = HISTORY_SEC) -> str:
    """Human commentary of the ``seconds`` before the clip (Proact-VL eval gives this as ``history``)."""
    f = window_dir(row) / "labels" / "commentary_by_second.jsonl"
    if not f.exists():
        return ""
    s0 = int(row["source_window_sec"][0])
    words = []
    for line in open(f):
        d = json.loads(line)
        if s0 - seconds <= d["time_sec"] < s0 and d.get("text", "").strip():
            words.append(d["text"].strip())
    return " ".join(words)


def state_feats(row, T):
    """Per-second per-feed game state [T, K, 13] (dircast/game_state.py), zeros if missing."""
    f = window_dir(row) / "state_feats.npy"
    if not f.exists():
        return torch.zeros(T, 10, 13)
    s0 = int(row["source_window_sec"][0])
    a = np.load(f)[s0:s0 + T]
    if len(a) < T:
        a = np.concatenate([a, np.zeros((T - len(a),) + a.shape[1:], a.dtype)])
    return torch.from_numpy(a.astype(np.float32))


def _events(row, v2=False, board=False):
    from .events import events_by_second          # lazy: events imports this module
    try:
        return events_by_second(row, v2, board)
    except Exception:                              # noqa: BLE001  (missing clock / meta -> no events)
        return [""] * len(row["camera_labels"])


def _kill_feats(row, T):
    from .events import kill_feats                  # lazy: events imports this module
    k = kill_feats(row)[:T]
    if len(k) < T:
        k = np.concatenate([k, np.zeros((T - len(k),) + k.shape[1:], k.dtype)])
    return torch.from_numpy(k)


def _kill_set(row, T):
    """[T, 10] bool: feed k shows the killer or the victim of a kill within +-1 s (training target only)."""
    from .director_eval import clip_kills          # lazy: director_eval imports this module
    ks = torch.zeros(T, 10, dtype=torch.bool)
    for t, killer, victim in clip_kills(row):
        for q in range(max(0, t - 1), min(T, t + 2)):
            ks[q, killer] = True; ks[q, victim] = True
    return ks


def match_of(row) -> int:
    return int(str(row["video_id"]).split("_")[-1]) if not str(row["video_id"]).isdigit() else int(row["video_id"])


class ClipDataset(Dataset):
    def __init__(self, split: str, cam_mask_key: str = "verified_mask", rows=None, broadcast=False, events_v2=False,
                 events_board=False, kill_feats=False, kill_targets=False):
        self.kill_targets = kill_targets    # training targets: kill participants (+-1 s) are acceptable on-air feeds
        self.kill_feats = kill_feats        # + entity-grounded event features per feed (got a kill now / recently)
        self.rows = rows if rows is not None else load_rows(split)
        self.split, self.cam_mask_key = split, cam_mask_key
        self.broadcast = broadcast          # single-stream baseline: the human broadcast as the only "feed"
        self.events_v2 = events_v2          # HUD-level events (major items, buybacks)
        self.events_board = events_board    # periodic hero scoreboard (level, K/D, gold)
        if broadcast:                       # clips whose window has no broadcast video (= no audio) carry no commentary
            self.rows = [r for r in self.rows if (window_dir(r) / "broadcast_1fps.npy").exists()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.broadcast:
            s0 = int(r["source_window_sec"][0]); T0 = len(r["camera_labels"])
            full = np.load(window_dir(r) / "broadcast_1fps.npy")
            b = full[s0:s0 + T0]
            if len(b) < T0:                  # broadcast shorter than the window: black frames (no VOD coverage)
                b = np.concatenate([b, np.zeros((T0 - len(b),) + full.shape[1:], full.dtype)])
            frames = torch.from_numpy(b)[:, None]                    # [T,1,H,W,3]
        else:
            z = np.load(r["frame_npz"])
            frames = torch.from_numpy(z["frames"])                   # [T,K,H,W,3]
        T, K = frames.shape[:2]
        live = torch.tensor([s == "LIVE" for s in r["vod_state_labels"]])
        cam = torch.tensor([c if c is not None and c >= 0 else -1 for c in r["camera_labels"]], dtype=torch.long)
        if self.broadcast:
            cam = torch.zeros_like(cam)
        cm = torch.tensor([bool(x) for x in r[self.cam_mask_key]]) & live & (cam >= 0)
        strict = torch.tensor([bool(x) for x in r["strict_mask"]]) & live & (cam >= 0)
        vis = torch.zeros(T, K, dtype=torch.bool)
        for t, vs in enumerate([] if self.broadcast else (r.get("visible_slots") or [[]] * T)):
            for k in vs or []:
                if 0 <= k < K: vis[t, k] = True
        text = [s.strip() if isinstance(s, str) else "" for s in (r.get("commentary_by_second") or [""] * T)]
        has_audio = bool(r.get("commentary_by_second"))
        tm = live & torch.tensor([has_audio] * T)
        mid = match_of(r)
        return {
            "clip_id": r["clip_id"], "match_id": mid, "context": context_text(match_context(mid)),
            "frames": frames, "cam": cam, "cam_mask": cm, "strict": strict, "live": live,
            "visible": vis, "text": text, "text_mask": tm,
            "game_clock_start": float(r.get("game_clock_start") or 0.0),
            "history": history_text(r),
            "state": torch.cat([state_feats(r, T), _kill_feats(r, T)], -1) if self.kill_feats else state_feats(r, T),
            "events": _events(r, self.events_v2, self.events_board),
            **({"kill_set": _kill_set(r, T)} if self.kill_targets else {}),
        }


def collate_single(batch):
    assert len(batch) == 1
    return batch[0]
