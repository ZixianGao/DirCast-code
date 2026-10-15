#!/usr/bin/env python3
"""Full-split streaming evaluation of DirCast.

  torchrun --nproc_per_node 8 -m dircast.evaluate --ckpt runs/dircast_v1/ckpt_ep2.pt --out eval/dircast_v1

Writes (rank 0 merges):
  <out>/director_<mode>.jsonl     per clip: predicted feed, probs, labels/masks
  <out>/commentary_<mode>.jsonl   Proact-VL prediction format: video_id, begin, end, idx, dataset_name, tag,
                                  pred {sec: text}, label {sec: text}
  <out>/metrics.json              director metrics (see dircast/metrics.py)
Commentary metrics are computed by the official Proact-VL scripts (third_party/proactvl_eval).
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .data import ClipDataset
from .metrics import director_metrics, write_label_files
from .model import N_STATE, DirCast
from .stream import Streamer

ROOT = Path(__file__).resolve().parents[1]


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu")
    a = ck["args"]
    m = DirCast(a.get("model", "Qwen/Qwen3-VL-8B-Instruct"), lora_r=a["lora_r"], lo_side=a["lo_side"],
                hi_side=a["hi_side"], lo_window=a["lo_window"], foveate=not a["no_foveate"],
                pointer=not a["no_pointer"], memory=not a["no_memory"], grad_ckpt=False,
                use_state=a.get("use_state", False), state_pointer=a.get("state_pointer", False),
                state_norm=a.get("state_norm", False), glimpses=a.get("glimpses", 1),
                use_events=a.get("use_events", False), n_state=N_STATE + (2 if a.get("kill_feats", False) else 0))
    missing, unexpected = m.load_state_dict(ck["state"], strict=False)
    assert not unexpected, unexpected[:5]
    assert all("lora_" not in k for k in missing), "LoRA weights missing from checkpoint"
    return m.to(dev).eval(), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--modes", nargs="+", default=["teacher", "free"])
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--rep", type=float, default=1.15)
    ap.add_argument("--tau-cut", type=float, default=None, help="default: thresholds.json next to the ckpt (tuned on val)")
    ap.add_argument("--tau-speak", type=float, default=None)
    ap.add_argument("--rate-control", action="store_true", help="rate-controlled speak / cut decisions (targets: train rates)")
    ap.add_argument("--target-speak", type=float, default=0.69)
    ap.add_argument("--target-cut", type=float, default=0.135)
    ap.add_argument("--speak-rate-init", type=float, default=None,
                    help="initial running speak rate of the controller (default: the target; 0 = has not spoken yet)")
    ap.add_argument("--end-threshold", type=float, default=0.0, help="LiveCC-style streaming EOS threshold")
    ap.add_argument("--limit", type=int, default=0, help="smoke test only; reported numbers use the full split")
    ap.add_argument("--program", default=None,
                    help="director_*.jsonl of another model: mode 'program' comments on its on-air choices (two-model pipeline)")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, targs = load_model(a.ckpt, dev)
    thr = Path(a.ckpt).with_name("thresholds.json")
    thr = json.loads(thr.read_text()) if thr.exists() else {}
    a.tau_cut = a.tau_cut if a.tau_cut is not None else thr.get("tau_cut", 0.5)
    a.tau_speak = a.tau_speak if a.tau_speak is not None else thr.get("tau_speak", 0.5)
    if rank == 0:
        print(f"tau_cut={a.tau_cut} tau_speak={a.tau_speak} (thresholds.json: {bool(thr)})", flush=True)
    st = Streamer(model, max_new_tokens=a.max_new_tokens, repetition_penalty=a.rep, tau_cut=a.tau_cut, tau_speak=a.tau_speak,
                  rate_control=a.rate_control, target_speak=a.target_speak, target_cut=a.target_cut,
                  end_threshold=a.end_threshold, speak_rate_init=a.speak_rate_init)
    ds = ClipDataset(a.split, cam_mask_key="strict_mask", broadcast=targs.get("broadcast", False),
                     events_v2=targs.get("events_v2", False), events_board=targs.get("events_board", False),
                     kill_feats=targs.get("kill_feats", False))
    idx = list(range(len(ds)))[: a.limit or None][rank::world]
    program = {json.loads(l)["clip_id"]: json.loads(l)["pred_cam"] for l in open(a.program)} if a.program else None
    assert program is None or a.modes == ["program"], "--program is used with --modes program"
    for mode in a.modes:
        part = out / f".part_{mode}_{rank}.jsonl"
        done = {json.loads(l)["idx"] for l in open(part)} if part.exists() else set()
        with open(part, "a") as f:
            t0 = time.time()
            for n, i in enumerate(idx):
                if i in done:
                    continue
                item = ds[i]
                if mode == "program":   # the on-air feed of every second is the other model's choice
                    run_item = dict(item); run_item["cam"] = torch.tensor(program[item["clip_id"]])
                    r = st.run_clip(run_item, onair_mode="teacher", gen_text=True)
                else:
                    r = st.run_clip(item, onair_mode=mode, gen_text=True)
                f.write(json.dumps({
                    "idx": i, "clip_id": item["clip_id"], "match_id": item["match_id"],
                    "pred_cam": r["pred_cam"], "probs": [[round(x, 4) for x in p] for p in r["probs"].tolist()],
                    "pred_text": {str(k): v for k, v in r["texts"].items()},
                    "p_cut": [round(x, 4) for x in r["p_cut"]], "p_speak": [round(x, 4) for x in r["p_speak"]],
                    "prev_onair": r["pred_cam"] if mode == "free" else program[item["clip_id"]] if mode == "program" else item["cam"].tolist(),
                    "cam": item["cam"].tolist(), "strict": item["strict"].tolist(), "live": item["live"].tolist(),
                    "verified": item["cam_mask"].tolist(), "visible": item["visible"].tolist(),
                    "text": item["text"],
                }) + "\n"); f.flush()
                if rank == 0 and n % 10 == 0:
                    print(f"[{mode}] {n+1}/{len(idx)} {(time.time()-t0)/(n+1):.1f}s/clip", flush=True)
        dist.barrier()
        if rank == 0:
            rows = []
            for r_ in range(world):
                rows += [json.loads(l) for l in open(out / f".part_{mode}_{r_}.jsonl")]
            rows.sort(key=lambda x: x["idx"])
            with open(out / f"director_{mode}.jsonl", "w") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
            write_label_files(rows, out / f"commentary_{mode}.jsonl", out / f"labels_asr_{a.split}.jsonl", model_id=f"dircast_{mode}")
            met = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
            met[mode] = director_metrics(rows)
            met["ckpt"] = a.ckpt; met["tau_cut"] = a.tau_cut; met["tau_speak"] = a.tau_speak
            met["rate_control"] = a.rate_control; met["split"] = a.split
            (out / "metrics.json").write_text(json.dumps(met, indent=1))
            print(mode, json.dumps(met[mode], indent=1), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
