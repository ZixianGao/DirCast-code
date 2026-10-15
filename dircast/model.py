"""DirCast: a streaming Qwen3-VL model that directs (picks the on-air feed) and commentates.

Per second t the stream is

    [t]  0:<lo feed 0> 1:<lo feed 1> ... 9:<lo feed 9>  <DIR>  ->c_t  <hi feed c_t>  <COM>  words_t  <END>

* lo feeds: all K=10 hero-chase views at low resolution (128 px, 16 tokens each).
* <DIR>: a feed-pointer head scores every feed of this second against the
  <DIR> state (``FeedPointer``) plus a hold gate on the previous on-air feed.
* Director-gated foveation ("select-then-speak"): only the chosen feed c_t is
  re-encoded at high resolution (288 px, 81 tokens) and the commentary is
  generated after it, so the director's choice is the commentator's attention.
* Program-as-memory: lo-feed tokens are visible for ``lo_window`` seconds only;
  older context keeps just what went on air (hi feed + text). Training uses
  the same rule as a 4-D attention mask, streaming inference evicts the
  expired KV entries, so memory per second is bounded.

Positions follow Qwen3-VL M-RoPE exactly (``Qwen3VLModel.get_rope_index``), but
are computed here so that streaming chunks with new images can be appended to
a KV cache.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_io import VISION, patchify

DIR_ID, COM_ID, END_ID = 151646, 151648, 151645          # <|object_ref_start|>, <|box_start|>, <|im_end|>
VS_ID, VE_ID, PAD_ID = 151652, 151653, 151655             # vision start / end / image pad
STATE_ID = 151650                                         # <|quad_start|>: placeholder replaced by a game-state embedding
N_STATE = 13
SYSTEM = ("You are the live director and caster of a Dota 2 broadcast. Every second you see the ten "
          "hero-chase feeds, put one feed on air and commentate what happens (or stay silent).")
KIND_TEXT, KIND_LO, KIND_HI, KIND_GL = 0, 1, 2, 3   # GL: extra high-res glimpse (expires like LO)
EXPIRING = (KIND_LO, KIND_GL)


@dataclass
class Seq:
    ids: list
    pos: list            # 3 lists
    sec: list            # second index of each token (-1 = prefix)
    kind: list
    labels: list         # LM labels (-100 = none)
    lab_sec: list        # second of each LM label
    dir_pos: list        # [T]
    lo_tok: list         # [T][K] -> (start, n)
    images: list         # (t, k, 'lo'|'hi') in sequence order
    com_pos: list = None
    state_pos: list = None   # (position, t, k)
    cur: int = 0

    @classmethod
    def new(cls):
        return cls([], [[], [], []], [], [], [], [], [], [], [], [], [])

    def text(self, toks, sec, labels=None):
        n = len(toks)
        self.ids += toks
        for d in range(3):
            self.pos[d] += list(range(self.cur, self.cur + n))
        self.cur += n
        self.sec += [sec] * n
        self.kind += [KIND_TEXT] * n
        self.labels += labels if labels is not None else [-100] * n
        self.lab_sec += [sec] * n

    def image(self, grid, sec, kind):
        g = grid // 2                                       # merged grid side
        self.text([VS_ID], sec)
        start = len(self.ids)
        self.ids += [PAD_ID] * (g * g)
        self.pos[0] += [self.cur] * (g * g)
        self.pos[1] += [self.cur + i for i in range(g) for _ in range(g)]
        self.pos[2] += [self.cur + j for _ in range(g) for j in range(g)]
        self.cur += g
        self.sec += [sec] * (g * g)
        self.kind += [kind] * (g * g)
        self.labels += [-100] * (g * g)
        self.lab_sec += [sec] * (g * g)
        self.text([VE_ID], sec)
        return start, g * g


class FeedPointer(nn.Module):
    """Hold-or-switch director head.

    cut gate   g_t = MLP([h_dir ; h_feed[prev]])       P(switch away from the on-air feed)
    target     s_tk = <q(h_dir), k(h_feed_k) + e_k>    permutation-equivariant pointer over feeds (prev excluded)
    P(c_t = prev) = 1 - sigma(g_t),  P(c_t = k != prev) = sigma(g_t) * softmax_{k != prev}(s_t)
    """

    def __init__(self, d_model, d=512, state_dim=0):
        super().__init__()
        self.state_dim = state_dim
        if state_dim:
            # state-aware pointer: per-feed game state enters the key and the cut gate directly
            self.s_key = nn.Sequential(nn.Linear(state_dim, 128), nn.GELU(), nn.Linear(128, d))
            self.s_cut = nn.Sequential(nn.Linear(3 * state_dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.q = nn.Linear(d_model, d)
        self.k = nn.Linear(d_model, d)
        self.slot = nn.Parameter(torch.zeros(10, d))         # team/slot identity (feeds are ordered by player slot)
        self.cut = nn.Sequential(nn.Linear(2 * d_model, 512), nn.GELU(), nn.Linear(512, 1))
        self.d = d

    def forward(self, h_dir, h_feed, prev, state=None):
        """h_dir [T,D], h_feed [T,K,D], prev [T] long (-1 = unknown), state [T,K,S] -> (target [T,K], cut [T])."""
        q = self.q(h_dir)
        k = self.k(h_feed) + self.slot[: h_feed.shape[1]]
        if self.state_dim and state is not None:
            k = k + self.s_key(state.to(k.dtype))
        target = torch.einsum("td,tkd->tk", q, k) / math.sqrt(self.d)
        has = (prev >= 0)
        h_prev = h_feed[torch.arange(len(prev), device=prev.device), prev.clamp(min=0)] * has[:, None].to(h_feed.dtype)
        cut = self.cut(torch.cat([h_dir, h_prev], -1)).squeeze(-1)
        if self.state_dim and state is not None:
            st = state.to(h_dir.dtype)
            s_prev = st[torch.arange(len(prev), device=prev.device), prev.clamp(min=0)] * has[:, None].to(st.dtype)
            cut = cut + self.s_cut(torch.cat([s_prev, st.max(1).values, st.mean(1)], -1)).squeeze(-1)
        return target, cut


def combine_director(target, cut, prev):
    """-> log P(c_t = k) [T,K] (float32) from the factorized head."""
    target, cut = target.float(), cut.float()
    has = prev >= 0
    idx = prev.clamp(min=0)
    excl = target.scatter(1, idx[:, None], -1e4)
    lp_sw = F.logsigmoid(cut)[:, None] + excl.log_softmax(-1)
    lp_hold = F.logsigmoid(-cut)
    lp = lp_sw.scatter(1, idx[:, None], lp_hold[:, None])
    return torch.where(has[:, None], lp, target.log_softmax(-1))


class DirCast(nn.Module):
    def __init__(self, model_path, lora_r=64, lo_side=128, hi_side=288, lo_window=8,
                 max_words_tokens=24, foveate=True, pointer=True, memory=True, grad_ckpt=True,
                 attn_impl="flex_attention", use_state=False, lora_targets="attn_mlp", state_pointer=False,
                 state_norm=False, glimpses=1, use_events=False, n_state=N_STATE):
        super().__init__()
        import json as _json
        from pathlib import Path as _P
        from transformers import (AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration,
                                  Qwen3VLForConditionalGeneration)
        from peft import LoraConfig, get_peft_model
        self.model_type = _json.loads((_P(model_path) / "config.json").read_text())["model_type"]
        self.patch, self.mean, self.std = VISION[self.model_type]
        cls = {"qwen3_vl": Qwen3VLForConditionalGeneration, "qwen2_vl": Qwen2VLForConditionalGeneration,
               "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration}[self.model_type]
        self.tok = AutoTokenizer.from_pretrained(model_path)
        base = cls.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
        if attn_impl != "sdpa":                         # vision tower stays on sdpa; only the LLM needs the custom mask
            base.config.text_config._attn_implementation = attn_impl
        base.config.vision_config._attn_implementation = "flash_attention_2"   # varlen over the packed images
        for p in base.parameters():
            p.requires_grad_(False)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_r, lora_dropout=0.05,
                         target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        self.lm = get_peft_model(base, cfg)
        if grad_ckpt:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.lm.enable_input_require_grads()
        D = base.config.text_config.hidden_size
        self.state_pointer = use_state and state_pointer
        self.pointer = FeedPointer(D, state_dim=n_state if self.state_pointer else 0).to(torch.bfloat16)
        self.dir_lin = nn.Linear(D, 10).to(torch.bfloat16)      # ablation: plain classifier on <DIR>
        self.dir_lin_cut = nn.Linear(D, 1).to(torch.bfloat16)
        self.speak = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, 1)).to(torch.bfloat16)   # response head at <COM>
        self.use_state = use_state
        self.state_proj = nn.Sequential(nn.Linear(n_state, 512), nn.GELU(), nn.Linear(512, D)).to(torch.bfloat16)
        self._inject = None
        self.state_norm = state_norm
        self.emb_norm = float(base.get_input_embeddings().weight.float().norm(dim=-1).mean())
        self.lm.get_input_embeddings().register_forward_hook(self._inject_hook)
        self.lo_side, self.hi_side, self.lo_window = lo_side, hi_side, lo_window
        self.max_words_tokens = max_words_tokens
        self.foveate, self.use_pointer, self.memory = foveate, pointer, memory
        self.glimpses = glimpses
        self.use_events = use_events
        self._prefix_cache = {}
        self.attn_impl = attn_impl

    def _inject_hook(self, module, inp, out):
        """Replace the embeddings of <STATE> placeholders by projected game-state vectors."""
        if self._inject is None:
            return out
        pos, vec = self._inject
        out = out.clone()
        out[0, pos] = vec.to(out.dtype)
        return out

    def set_state_injection(self, seq, state, device):
        """state [T,K,N_STATE] (indexed by the seq's (t,k)) -> arms the embedding hook for the next forward."""
        if not self.use_state or not seq.state_pos:
            self._inject = None
            return
        pos = torch.tensor([p for p, _, _ in seq.state_pos], device=device)
        tk = torch.tensor([[t, k] for _, t, k in seq.state_pos], device=device)
        feats = state.to(device)[tk[:, 0], tk[:, 1]].to(torch.bfloat16)
        v = self.state_proj(feats)
        if self.state_norm:                               # match the scale of ordinary token embeddings
            v = F.normalize(v.float(), dim=-1).to(v.dtype) * self.emb_norm
        self._inject = (pos, v)

    # ------------------------------------------------------------------ sequence
    def _t(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def prefix_tokens(self, context, history=""):
        if context not in self._prefix_cache:
            self._prefix_cache[context] = self._t("<|im_start|>system\n" + SYSTEM + "\n" + context + "<|im_end|>\n")
        h = self._t("Previous commentary: " + history[-1500:] + "\n") if history else []
        return self._prefix_cache[context] + h

    def second_header(self, seq, t, clock, event=""):
        m, s = divmod(int(round(clock)), 60)
        sign = "-" if clock < 0 else ""
        ev = f" ({event})" if (self.use_events and event) else ""
        seq.text(self._t(f"\n[{sign}{abs(m):02d}:{s:02d}]{ev}"), t)

    def lo_block(self, seq, t, K):
        lo = []
        g = self.lo_side // self.patch
        for k in range(K):
            seq.text(self._t(f" {k}:"), t)
            if self.use_state:
                seq.state_pos.append((len(seq.ids), t, k))
                seq.text([STATE_ID], t)
            lo.append(seq.image(g, t, KIND_LO))
            seq.images.append((t, k, "lo"))
        seq.lo_tok.append(lo)
        seq.text([DIR_ID], t)
        seq.dir_pos.append(len(seq.ids) - 1)

    def onair_block(self, seq, t, c, extra=()):
        """On-air feed c at high resolution (kept in memory) + optional extra glimpses (expire with lo feeds)."""
        seq.text(self._t(f"->{c}"), t)
        if self.foveate:
            seq.image(self.hi_side // self.patch, t, KIND_HI)
            seq.images.append((t, c, "hi"))
            for f in extra:
                seq.text(self._t(f" +{f}:"), t)
                seq.image(self.hi_side // self.patch, t, KIND_GL)
                seq.images.append((t, f, "hi"))

    def pick_glimpses(self, onair, visible, K, generator=None):
        """Training-time extra glimpses: on-screen heroes first, random feeds to fill (glimpses-1 per second)."""
        out = []
        for t in range(len(onair)):
            c = int(onair[t])
            vis = [k for k in torch.nonzero(visible[t]).flatten().tolist() if k != c]
            vis = [vis[i] for i in torch.randperm(len(vis), generator=generator).tolist()]
            rest = [k for k in torch.randperm(K, generator=generator).tolist() if k != c and k not in vis]
            out.append((vis + rest)[: self.glimpses - 1])
        return out

    def words_tokens(self, text):
        w = self._t(" " + text)[: self.max_words_tokens] if text else []
        return w + [END_ID]

    def build(self, context, T, K, onair, texts, clock0, history="", extras=None, events=None):
        seq = Seq.new()
        seq.text(self.prefix_tokens(context, history), -1)
        for t in range(T):
            self.second_header(seq, t, clock0 + t, events[t] if events else "")
            self.lo_block(seq, t, K)
            self.onair_block(seq, t, int(onair[t]), extras[t] if extras else ())
            w = self.words_tokens(texts[t])
            # <COM> predicts the first word (or <END> = silence); each word predicts the next
            seq.com_pos.append(len(seq.ids))
            seq.text([COM_ID] + w[:-1], t, labels=w)
            seq.text([END_ID], t)
        return seq

    def attn_mask(self, seq, device):
        """Causal mask with program-as-memory: lo-feed keys expire after ``lo_window`` seconds."""
        L = len(seq.ids)
        sec = torch.tensor(seq.sec, device=device)
        kind = torch.tensor(seq.kind, device=device)
        lo = (kind == KIND_LO) | (kind == KIND_GL)
        W = self.lo_window if (self.memory and self.lo_window > 0) else 10 ** 9
        if self.attn_impl == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            def mask_mod(b, h, q, kv):
                return (q >= kv) & ~(lo[kv] & (sec[q] - sec[kv] >= W))
            return create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=device)
        m = torch.ones(L, L, dtype=torch.bool, device=device).tril_()
        m &= ~(lo[None, :] & (sec[:, None] - sec[None, :] >= W))
        return m[None, None]

    def pixels(self, frames, seq, device):
        """frames [T,K,H,W,3] uint8 -> pixel_values, grid in sequence order."""
        frames = frames.to(device, non_blocking=True)
        lo_idx = [(t, k) for t, k, r in seq.images if r == "lo"]
        hi_idx = [(t, k) for t, k, r in seq.images if r == "hi"]
        out_pv, out_grid, order = [], [], []
        if lo_idx:
            ii = torch.tensor(lo_idx, device=device)
            pv, gr = patchify(frames[ii[:, 0], ii[:, 1]], self.lo_side, self.patch, self.mean, self.std)
            out_pv.append(pv.view(len(lo_idx), -1, pv.shape[-1])); out_grid.append(gr)
        if hi_idx:
            ii = torch.tensor(hi_idx, device=device)
            pv, gr = patchify(frames[ii[:, 0], ii[:, 1]], self.hi_side, self.patch, self.mean, self.std)
            out_pv.append(pv.view(len(hi_idx), -1, pv.shape[-1])); out_grid.append(gr)
        # interleave back to sequence order
        lo_it, hi_it = iter(range(len(lo_idx))), iter(range(len(hi_idx)))
        pvs, grids = [], []
        for _, _, r in seq.images:
            if r == "lo":
                j = next(lo_it); pvs.append(out_pv[0][j]); grids.append(out_grid[0][j])
            else:
                j = next(hi_it); pvs.append(out_pv[-1][j]); grids.append(out_grid[-1][j])
        return torch.cat(pvs, 0).to(torch.bfloat16), torch.stack(grids).to(device)

    # ------------------------------------------------------------------ heads
    def director_heads(self, h, seq, prev, state=None):
        """h [L,D] last hidden of ``seq``'s tokens -> (target logits [T,K], cut logit [T]) in float32.
        state: [T,K,S] game state of the seq's seconds (used by the state-aware pointer)."""
        h_dir = h[torch.tensor(seq.dir_pos, device=h.device)]
        if not self.use_pointer:
            return self.dir_lin(h_dir).float(), self.dir_lin_cut(h_dir).squeeze(-1).float()
        lo = seq.lo_tok
        n = lo[0][0][1]
        starts = torch.tensor([[s for s, _ in row] for row in lo], device=h.device)       # [T,K]
        idx = starts[..., None] + torch.arange(n, device=h.device)                        # [T,K,n]
        h_feed = h[idx].mean(2)                                                           # [T,K,D]
        target, cut = self.pointer(h_dir, h_feed, prev, state.to(h.device) if (self.state_pointer and state is not None) else None)
        return target.float(), cut.float()

    def director_logits(self, h, seq, prev, state=None):
        """log P(c_t = k) [T,K] (the factorized head's marginal; used for argmax / probabilities)."""
        target, cut = self.director_heads(h, seq, prev, state)
        return combine_director(target, cut, prev)

    def speak_logits(self, h, seq):
        return self.speak(h[torch.tensor(seq.com_pos, device=h.device)]).squeeze(-1).float()

    # ------------------------------------------------------------------ training forward
    def forward(self, item, vis_weight=0.5, lm_weight=1.0, speak_weight=1.0, cut_pos_weight=2.0, ctx_drop=0.0,
                ctx_span=False, cf_weight=0.0, cf_margin=0.5, prog_noise=0.0, kill_weight=0.0):
        dev = next(self.pointer.parameters()).device
        frames = item["frames"]
        T, K = frames.shape[:2]
        cam = item["cam"].to(dev)
        onair = cam.clone()
        # seconds without a label keep the previous on-air feed (feed 0 at clip start)
        for t in range(T):
            if onair[t] < 0:
                onair[t] = onair[t - 1] if t > 0 else 0
        if self.training and prog_noise > 0:
            # on-policy program rollouts (DAgger for directing): the streamed program drifts off the human one for
            # 2-8 s spans (as a free-running director does); the director targets stay the human feed, so the model
            # learns to recover from its own mistakes, and the commentary learns to speak over off-program views
            t = 1
            while t < T:
                if float(torch.rand(1)) < prog_noise:
                    L = int(torch.randint(2, 9, (1,)))
                    k = int(torch.randint(0, K - 1, (1,)))
                    k = k + (k >= int(onair[t]))
                    onair[t:t + L] = k
                    t += L
                else:
                    t += 1
        texts, tm = list(item["text"]), item["text_mask"].clone()
        if self.training and ctx_drop > 0:
            # commentary-context dropout: a dropped second shows silence in the input and is not supervised,
            # so later seconds learn to (re)start speaking after silence (streaming exposure bias)
            if ctx_span:                                   # contiguous silent spans (2-12 s) up to ctx_drop of the clip
                drop = torch.zeros(T, dtype=torch.bool)
                while drop.float().mean() < ctx_drop * float(torch.rand(1)) * 2:
                    L = int(torch.randint(2, 13, (1,))); s0 = int(torch.randint(0, max(1, T - L), (1,)))
                    drop[s0:s0 + L] = True
                idx = drop.nonzero().flatten().tolist()
            else:
                idx = (torch.rand(T) < ctx_drop).nonzero().flatten().tolist()
            for t in idx:
                texts[t] = ""; tm[t] = False
        extras = self.pick_glimpses(onair.tolist(), item["visible"], K) if (self.glimpses > 1 and self.foveate) else None
        seq = self.build(item["context"], T, K, onair.tolist(), texts, item["game_clock_start"],
                         item.get("history", ""), extras, item.get("events"))
        ids = torch.tensor(seq.ids, device=dev)[None]
        pos = torch.tensor(seq.pos, device=dev)[:, None, :]
        pv, grid = self.pixels(frames, seq, dev)
        mask = self.attn_mask(seq, dev)
        mm = (ids == PAD_ID).int()
        self.set_state_injection(seq, item.get("state"), dev)
        try:
            out = self.lm.base_model.model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                                 pixel_values=pv, image_grid_thw=grid, mm_token_type_ids=mm,
                                                 use_cache=False)
        finally:
            self._inject = None
        h = out.last_hidden_state[0]
        prev = torch.cat([torch.tensor([-1], device=dev), onair[:-1]])
        target, cut = self.director_heads(h, seq, prev, item.get("state"))
        logits = combine_director(target, cut, prev)
        cm = item["cam_mask"].to(dev)
        losses = {}
        if cm.any():
            has = prev >= 0
            y_cut = (cam != prev).float()
            mc = cm & has
            l_cut = F.binary_cross_entropy_with_logits(cut[mc], y_cut[mc], pos_weight=torch.tensor(cut_pos_weight, device=dev)) if mc.any() else 0.0
            sw = cm & (~has | (cam != prev))                 # seconds where the target matters
            l_tgt = 0.0
            if sw.any():
                excl = target.masked_fill(F.one_hot(prev.clamp(min=0), target.shape[1]).bool() & has[:, None], -1e4)
                lp = excl.log_softmax(-1)
                l_tgt = F.nll_loss(lp[sw], cam[sw])
                pos_set = item["visible"].to(dev).clone()
                pos_set[torch.arange(T, device=dev), cam.clamp(min=0)] = True
                l_tgt = l_tgt + vis_weight * (-(lp.masked_fill(~pos_set, -1e4).logsumexp(-1))[sw].mean())
            l_kill = 0.0
            if kill_weight > 0 and "kill_set" in item:
                # event-anchored auxiliary target: during a kill (+-1 s) whose participants are off air, cutting to a
                # participant is also an acceptable directing decision (set-valued, in addition to the human feed)
                ks = item["kill_set"].to(dev)
                off = cm & has & ks.any(-1) & ~ks[torch.arange(T, device=dev), prev.clamp(min=0)]
                if off.any():
                    excl_k = target.masked_fill(F.one_hot(prev.clamp(min=0), target.shape[1]).bool(), -1e4).log_softmax(-1)
                    l_kill = F.binary_cross_entropy_with_logits(cut[off], torch.ones_like(cut[off])) \
                        - excl_k.masked_fill(~ks, -1e4).logsumexp(-1)[off].mean()
                    losses["kill"] = l_kill.detach()
            losses["dir"] = l_cut + l_tgt + kill_weight * l_kill
            pred = logits.argmax(-1)
            losses["dir_acc"] = (pred[cm] == cam[cm]).float().mean().detach()
            if mc.any():
                losses["cut_acc"] = ((cut[mc] > 0).float() == y_cut[mc]).float().mean().detach()
        # response head: speak (this second has commentary) vs silent
        tm = tm.to(dev)
        if tm.any() and speak_weight > 0:
            y_sp = torch.tensor([bool(x.strip()) for x in texts], device=dev).float()
            sp = self.speak_logits(h, seq)
            losses["speak"] = F.binary_cross_entropy_with_logits(sp[tm], y_sp[tm])
            losses["speak_acc"] = ((sp[tm] > 0).float() == y_sp[tm]).float().mean().detach()
        lab = torch.tensor(seq.labels, device=dev)
        lsec = torch.tensor(seq.lab_sec, device=dev)
        sel = (lab >= 0) & tm[lsec.clamp(min=0)] & (lsec >= 0)
        if sel.any():
            hs = h[sel]
            lm_logits = self.lm.base_model.model.lm_head(hs).float()
            nll = F.cross_entropy(lm_logits, lab[sel], reduction="none")
            losses["lm"] = nll.mean()
            if cf_weight > 0 and self.foveate and self.training:
                # counterfactual-view contrast: same tokens, but every high-res view shows a different (wrong) feed.
                # The human commentary must be more likely under the true view than under the counterfactual one.
                cf_images = []
                for (t, k, r) in seq.images:
                    if r == "hi":
                        k2 = int(torch.randint(0, K - 1, (1,)))
                        k = k2 + (k2 >= k)
                    cf_images.append((t, k, r))
                seq_cf = Seq(**{**seq.__dict__, "images": cf_images})
                pv_cf, grid_cf = self.pixels(frames, seq_cf, dev)
                self.set_state_injection(seq, item.get("state"), dev)
                # the counterfactual pass is an anchor (no gradient): the loss can only be lowered by making the
                # human commentary more likely under the TRUE view, never by sabotaging the counterfactual one
                try:
                    with torch.no_grad():
                        h_cf = self.lm.base_model.model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                                              pixel_values=pv_cf, image_grid_thw=grid_cf, mm_token_type_ids=mm,
                                                              use_cache=False).last_hidden_state[0]
                        nll_cf = F.cross_entropy(self.lm.base_model.model.lm_head(h_cf[sel]).float(), lab[sel], reduction="none")
                finally:
                    self._inject = None
                ssec = lsec[sel]
                spoken = torch.tensor([bool(x.strip()) for x in texts], device=dev)
                per_t = torch.zeros(T, device=dev).index_add_(0, ssec, nll)
                per_t_cf = torch.zeros(T, device=dev).index_add_(0, ssec, nll_cf)
                cnt = torch.zeros(T, device=dev).index_add_(0, ssec, torch.ones_like(nll))
                use = spoken & (cnt > 0)
                if use.any():
                    d = (per_t[use] - per_t_cf[use]) / cnt[use]          # mean token NLL gap (true - counterfactual)
                    losses["cf"] = F.relu(cf_margin + d).mean()
                    losses["cf_gap"] = (-d).mean().detach()
        total = losses.get("dir", 0.0) + lm_weight * losses.get("lm", 0.0) + speak_weight * losses.get("speak", 0.0) \
            + cf_weight * losses.get("cf", 0.0)
        return total, {k: float(v.detach() if torch.is_tensor(v) else v) for k, v in losses.items()}, logits.detach()

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if "lora_" in k or k.startswith(("pointer.", "dir_lin", "speak.", "state_proj."))}
