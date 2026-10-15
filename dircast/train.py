#!/usr/bin/env python3
"""Train DirCast (director + commentary) with DDP.

  torchrun --nproc_per_node 8 -m dircast.train --out runs/dircast_v1
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .data import ClipDataset, collate_single
from .model import N_STATE, DirCast

ROOT = Path(__file__).resolve().parents[1]


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lo-side", type=int, default=128)
    ap.add_argument("--hi-side", type=int, default=288)
    ap.add_argument("--lo-window", type=int, default=8)
    ap.add_argument("--lm-weight", type=float, default=1.0)
    ap.add_argument("--vis-weight", type=float, default=0.5)
    ap.add_argument("--speak-weight", type=float, default=1.0)
    ap.add_argument("--cut-pos-weight", type=float, default=2.0)
    ap.add_argument("--no-foveate", action="store_true")
    ap.add_argument("--no-pointer", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--no-commentary", action="store_true", help="director-only ablation")
    ap.add_argument("--no-director", action="store_true", help="commentary-only ablation")
    ap.add_argument("--cam-mask", default="verified_mask")
    ap.add_argument("--use-state", action="store_true", help="game-state tokens per feed")
    ap.add_argument("--ctx-drop", type=float, default=0.0, help="commentary-context dropout rate")
    ap.add_argument("--prog-noise", type=float, default=0.0, help="per-second onset rate of off-program spans (DAgger)")
    ap.add_argument("--kill-weight", type=float, default=0.0, help="event-anchored auxiliary director target (kills)")
    ap.add_argument("--kill-feats", action="store_true", help="entity-grounded event features per feed (got a kill now / recently)")
    ap.add_argument("--state-pointer", action="store_true", help="game state also feeds the pointer / cut gate")
    ap.add_argument("--ctx-span", action="store_true", help="context dropout in contiguous spans")
    ap.add_argument("--glimpses", type=int, default=1, help="high-res feeds shown to the commentator (on-air + extras)")
    ap.add_argument("--use-events", action="store_true", help="causal match-event stream as text each second")
    ap.add_argument("--cf-weight", type=float, default=0.0, help="counterfactual-view contrast weight")
    ap.add_argument("--broadcast", action="store_true", help="single-stream baseline on the human broadcast")
    ap.add_argument("--events-v2", action="store_true", help="HUD-level events: major items, buybacks")
    ap.add_argument("--events-board", action="store_true", help="hero scoreboard snapshot every 30 s")
    ap.add_argument("--cf-margin", type=float, default=0.5)
    ap.add_argument("--state-norm", action="store_true", help="scale state embeddings to the token-embedding norm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def main():
    a = get_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    random.seed(a.seed + rank); np.random.seed(a.seed + rank); torch.manual_seed(a.seed)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    logf = open(out / "log.txt", "a") if rank == 0 else None

    def log(msg):
        if rank == 0:
            line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
            print(line, flush=True); logf.write(line + "\n"); logf.flush()

    model = DirCast(a.model, lora_r=a.lora_r, lo_side=a.lo_side, hi_side=a.hi_side, lo_window=a.lo_window,
                    foveate=not a.no_foveate, pointer=not a.no_pointer, memory=not a.no_memory,
                    use_state=a.use_state, state_pointer=a.state_pointer, state_norm=a.state_norm,
                    glimpses=a.glimpses, use_events=a.use_events, n_state=N_STATE + (2 if a.kill_feats else 0)).to(dev)
    model.train()
    unused = [model.dir_lin, model.dir_lin_cut] if not a.no_pointer else [model.pointer]
    if not a.use_state:
        unused.append(model.state_proj)
    for mod in unused:
        for p in mod.parameters():
            p.requires_grad_(False)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    log(f"trainable: lora {sum(p.numel() for p in lora)/1e6:.1f}M heads {sum(p.numel() for p in heads)/1e6:.2f}M; world {world}")
    params = lora + heads
    for p in params:                                   # identical init on every rank
        dist.broadcast(p.data, 0)

    def allreduce_grads():
        """Average grads across ranks in one flat buffer (no DDP: it clashes with grad checkpointing)."""
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])
        dist.all_reduce(flat); flat /= world
        o = 0
        for p in params:
            n = p.numel()
            p.grad = flat[o:o + n].view_as(p).to(p.dtype); o += n

    tr = ClipDataset("train", cam_mask_key=a.cam_mask, broadcast=a.broadcast, events_v2=a.events_v2,
                     events_board=a.events_board, kill_feats=a.kill_feats, kill_targets=a.kill_weight > 0)
    va = ClipDataset("val", cam_mask_key="strict_mask", broadcast=a.broadcast, events_v2=a.events_v2,
                     events_board=a.events_board, kill_feats=a.kill_feats)
    smp = DistributedSampler(tr, shuffle=True, seed=a.seed)
    dl = DataLoader(tr, batch_size=1, sampler=smp, num_workers=a.workers, collate_fn=collate_single,
                    persistent_workers=True, prefetch_factor=2)
    vsmp = DistributedSampler(va, shuffle=False)
    vdl = DataLoader(va, batch_size=1, sampler=vsmp, num_workers=a.workers, collate_fn=collate_single)

    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": heads, "lr": a.head_lr}], weight_decay=0.0)
    total = a.epochs * math.ceil(len(dl) / a.accum)
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    lm_w = 0.0 if a.no_commentary else a.lm_weight
    step = 0
    for ep in range(a.epochs):
        smp.set_epoch(ep)
        t0, hist = time.time(), []
        for i, item in enumerate(dl):
            if a.no_director:
                item["cam_mask"] = torch.zeros_like(item["cam_mask"])
            sync = (i + 1) % a.accum == 0
            loss, info, _ = model(item, vis_weight=a.vis_weight, lm_weight=lm_w, speak_weight=0.0 if a.no_commentary else a.speak_weight,
                                  cut_pos_weight=a.cut_pos_weight, ctx_drop=a.ctx_drop, ctx_span=a.ctx_span,
                                  cf_weight=a.cf_weight, cf_margin=a.cf_margin, prog_noise=a.prog_noise, kill_weight=a.kill_weight)
            if torch.is_tensor(loss):
                (loss / a.accum).backward()
            hist.append(info)
            if sync:
                allreduce_grads()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if (i + 1) % 20 == 0:
                keys = sorted({k for h in hist[-20:] for k in h})
                msg = " ".join(f"{k}={np.mean([h[k] for h in hist[-20:] if k in h]):.4f}" for k in keys)
                log(f"ep{ep} it{i+1}/{len(dl)} step{step} {msg} lr={sched.get_last_lr()[0]:.2e} "
                    f"{(time.time()-t0)/(i+1):.2f}s/clip")
        # ---- validation (teacher-forced, full val split)
        model.eval()
        acc_n = acc_d = lm_s = lm_n = pers_n = sp_s = sp_n = 0.0
        with torch.no_grad():
            for item in vdl:
                _, info, logits = model(item, vis_weight=a.vis_weight, lm_weight=lm_w)
                cm = item["cam_mask"].clone(); cm[0] = False
                cam = item["cam"]
                if cm.any():
                    acc_n += float((logits.cpu()[cm].argmax(-1) == cam[cm]).sum()); acc_d += float(cm.sum())
                    pers_n += float((cam[:-1][cm[1:]] == cam[1:][cm[1:]]).sum())
                if "lm" in info:
                    lm_s += info["lm"]; lm_n += 1
                if "speak_acc" in info:
                    sp_s += info["speak_acc"]; sp_n += 1
        t = torch.tensor([acc_n, acc_d, lm_s, lm_n, pers_n, sp_s, sp_n], device=dev); dist.all_reduce(t)
        log(f"VAL ep{ep}: dir_acc(strict, teacher-forced, t>=1)={t[0]/max(t[1],1):.4f} persistence={t[4]/max(t[1],1):.4f} "
            f"lm={t[2]/max(t[3],1):.4f} speak_acc={t[5]/max(t[6],1):.4f}")
        model.train()
        if rank == 0:
            torch.save({"state": model.trainable_state(), "args": vars(a), "epoch": ep}, out / f"ckpt_ep{ep}.pt")
        dist.barrier()
    log("done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
