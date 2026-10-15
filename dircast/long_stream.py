#!/usr/bin/env python3
"""Long-stream run: DirCast streams a continuous 288 s window (eight back-to-back 36 s test clips of one rendered window)
with the final decoding settings, and logs per stream second the wall time, the KV cache length and the peak GPU
memory, together with its decisions, so that cost and quality can be read against stream position.

  CUDA_VISIBLE_DEVICES=0 DIRCAST_INDEX=data/index_polished python -m dircast.long_stream \
      --ckpt runs/dircast_v23/ckpt_ep2.pt --out eval/long_stream/full [--no-evict] [--windows 8]
--no-evict keeps every feed view in context (program-as-memory switched off at inference).
Two-model pipeline: run the director-only model with --no-text, then the commentator-only model with
--program <its streams.jsonl>, which comments on the director's on-air choices; the pipeline costs the sum.
"""
import argparse
import collections
import json
import time
from pathlib import Path

import torch

from .data import ClipDataset
from .evaluate import load_model
from .stream import Streamer

PER_SECOND = ("frames", "cam", "cam_mask", "strict", "live", "visible", "text_mask", "state")
PER_SECOND_LIST = ("text", "events")


def windows(ds, n):
    """-> list of [clip indices] whose source windows tile [0, 288) without overlap, one list per rendered window."""
    g = collections.defaultdict(dict)
    for i, r in enumerate(ds.rows):
        g[r["clip_id"].rsplit("_", 1)[0]][int(r["source_window_sec"][0])] = i
    out = []
    for w in sorted(g):
        starts = list(range(0, 288, 36))
        if all(s in g[w] for s in starts):
            out.append([g[w][s] for s in starts])
    # spread the selection over matches
    by_match = collections.defaultdict(list)
    for idxs in out:
        by_match[ds.rows[idxs[0]]["video_id"]].append(idxs)
    picked, k = [], 0
    while len(picked) < n and any(by_match.values()):
        for m in sorted(by_match):
            if by_match[m] and len(picked) < n:
                picked.append(by_match[m].pop(k % len(by_match[m])))
        k += 1
    return picked


def concat(items):
    it = dict(items[0])
    for k in PER_SECOND:
        if k in it:
            it[k] = torch.cat([x[k] for x in items], 0)
    for k in PER_SECOND_LIST:
        if k in it:
            it[k] = sum((list(x[k]) for x in items), [])
    for k, v in items[0].items():   # any other per-second tensor (e.g. kill features)
        if k not in it or k in PER_SECOND or k in PER_SECOND_LIST:
            continue
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == items[0]["cam"].shape[0]:
            it[k] = torch.cat([x[k] for x in items], 0)
    it["clip_id"] = items[0]["clip_id"].rsplit("_", 1)[0] + "_long288"
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--horizon", type=int, default=None, help="evict every token older than this many seconds")
    ap.add_argument("--no-text", action="store_true", help="no commentary is generated (director of a pipeline)")
    ap.add_argument("--program", default=None, help="streams.jsonl of a director run: comment on its on-air choices")
    a = ap.parse_args()
    program = {json.loads(l)["window"]: json.loads(l)["pred_cam"] for l in open(a.program)} if a.program else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, targs = load_model(a.ckpt, "cuda")
    if a.no_evict:
        model.memory = False
    thr = json.loads(Path(a.ckpt).with_name("thresholds.json").read_text())
    st = Streamer(model, tau_cut=thr["tau_cut"], tau_speak=thr["tau_speak"], rate_control=True, target_speak=0.85,
                  target_cut=0.135, end_threshold=0.3, speak_rate_init=0.0, horizon=a.horizon)
    ds = ClipDataset("test", cam_mask_key="strict_mask", kill_feats=targs.get("kill_feats", False))
    per_sec, kv, mem = [], [], []
    orig = st._evict

    def timed_evict(t_next):
        torch.cuda.synchronize()
        per_sec.append(time.perf_counter() - timed_evict.t0)
        mem.append(torch.cuda.max_memory_allocated() / 2 ** 30)
        orig(t_next)
        kv.append(st.cache.get_seq_length())
        timed_evict.t0 = time.perf_counter()
    st._evict = timed_evict
    with open(out / "streams.jsonl", "w") as f:
        for w in windows(ds, a.windows):
            item = concat([ds[i] for i in w])
            per_sec.clear(); kv.clear(); mem.clear()
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); timed_evict.t0 = time.perf_counter()
            try:
                run_item = dict(item, cam=torch.tensor(program[item["clip_id"]])) if program is not None else item
                r = st.run_clip(run_item, onair_mode="teacher" if program is not None else "free", gen_text=not a.no_text)
                status = "ok"
            except torch.cuda.OutOfMemoryError:
                r, status = {"pred_cam": [], "texts": {}}, f"oom at second {len(per_sec)}"
                torch.cuda.empty_cache()
            f.write(json.dumps({"window": item["clip_id"], "clips": w, "status": status, "sec_time": per_sec, "kv": kv,
                                "mem_gb": mem, "pred_cam": r["pred_cam"],
                                "pred_text": {str(k): v for k, v in r["texts"].items()},
                                "cam": item["cam"].tolist(), "strict": item["strict"].tolist(),
                                "live": item["live"].tolist(), "text": item["text"]}) + "\n"); f.flush()
            print(f"{item['clip_id']} {status} secs={len(per_sec)} mean={sum(per_sec)/max(1,len(per_sec)):.3f}s "
                  f"kv_end={kv[-1] if kv else 0} peak={max(mem) if mem else 0:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
