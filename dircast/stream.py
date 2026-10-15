"""Causal streaming inference for DirCast (one call per second, KV cache with eviction).

Each second t:
  1. append  [t] 0:<lo 0> ... 9:<lo 9> <DIR>   -> pointer logits -> on-air feed c_t
  2. append  ->c_t <hi c_t> <COM>               -> greedy commentary until <END> (silence = <END> first)
  3. evict lo-feed KV entries older than ``lo_window`` seconds (program-as-memory)

``onair_mode``:
  "free"    the model's own choices form the program (autonomous director);
  "teacher" the human director's previous choices form the program (next-second
            prediction, the setting of the persistence baseline); the model's
            prediction for t is still made before seeing c_t.
"""
import torch
from transformers import DynamicCache

from .model import COM_ID, DIR_ID, END_ID, EXPIRING, KIND_LO, Seq
from .vision_io import patchify


class Streamer:
    def __init__(self, model, max_new_tokens=24, repetition_penalty=1.15, silence_bias=0.0, tau_cut=0.5, tau_speak=0.5,
                 rate_control=False, target_speak=0.69, target_cut=0.135, gain_speak=1.0, gain_cut=2.0, ema=0.1,
                 end_threshold=0.0, speak_rate_init=None, horizon=None):
        self.m = model
        self.lm = model.lm.base_model.model                     # Qwen3VLForConditionalGeneration
        self.core = self.lm.model
        self.dev = next(model.pointer.parameters()).device
        self.max_new_tokens, self.rep = max_new_tokens, repetition_penalty
        self.state = None
        # LiveCC-style streaming EOS threshold: <END> may close a spoken second only once P(<END>) >= end_threshold
        self.end_thr = end_threshold
        # optional context horizon for long streams: every token of a second older than `horizon` seconds leaves the
        # cache (the shared prefix stays), so the cache is bounded; None keeps the program for the whole stream
        self.horizon = horizon
        # running speak rate at stream start: None = the target (legacy); 0.0 = "has not spoken yet" (the model joins
        # a live stream with no own output so far, so the controller starts below target and catches up quickly)
        self.speak_rate_init = speak_rate_init
        self.silence_bias = silence_bias
        self.tau_cut, self.tau_speak = tau_cut, tau_speak
        # rate-controlled decisions: the threshold follows the gap between the running decision rate (EMA)
        # and the human base rate measured on the train split, which breaks silence / no-cut spirals
        self.rc, self.tgt_s, self.tgt_c, self.g_s, self.g_c, self.ema = rate_control, target_speak, target_cut, gain_speak, gain_cut, ema
        self.lm.config.text_config._attn_implementation = "sdpa"   # causal over an (evicted) cache
        self.core.language_model.config._attn_implementation = "sdpa"

    # ---------------------------------------------------------------- low level
    def _forward(self, seq: Seq, images=None):
        ids = torch.tensor(seq.ids, device=self.dev)[None]
        pos = torch.tensor(seq.pos, device=self.dev)[:, None, :]
        kw = {}
        if images is not None:
            pvs, grids = [], []
            for img, side in images:
                pv, g = patchify(img[None], side, self.m.patch, self.m.mean, self.m.std)
                pvs.append(pv); grids.append(g)
            kw = dict(pixel_values=torch.cat(pvs).to(self.dev, torch.bfloat16), image_grid_thw=torch.cat(grids).to(self.dev),
                      mm_token_type_ids=(ids == 151655).int())
        self.m.set_state_injection(seq, self.state, self.dev) if self.state is not None else None
        try:
            out = self.core(input_ids=ids, position_ids=pos, past_key_values=self.cache, use_cache=True, **kw)
        finally:
            self.m._inject = None
        self.kinds += seq.kind; self.secs += seq.sec
        return out.last_hidden_state[0]

    def _new_seq(self):
        s = Seq.new(); s.cur = self.cur
        return s

    def _evict(self, t_next):
        W = self.m.lo_window if self.m.memory else 10 ** 9
        H = self.horizon if self.horizon is not None else 10 ** 9
        keep = [i for i, (k, s) in enumerate(zip(self.kinds, self.secs))
                if not ((k in EXPIRING and t_next - s >= W) or (s >= 0 and t_next - s >= H))]
        if len(keep) == len(self.kinds):
            return
        idx = torch.tensor(keep, device=self.dev)
        for layer in self.cache.layers:
            layer.keys = layer.keys.index_select(2, idx)
            layer.values = layer.values.index_select(2, idx)
        self.kinds = [self.kinds[i] for i in keep]; self.secs = [self.secs[i] for i in keep]

    # ---------------------------------------------------------------- api
    @torch.no_grad()
    def run_clip(self, item, onair_mode="free", gen_text=True, force_text=False):
        """force_text: feed the human commentary instead of generating (exactness tests / teacher forcing)."""
        m = self.m
        frames = item["frames"].to(self.dev)
        T, K = frames.shape[:2]
        self.cache, self.kinds, self.secs, self.cur = DynamicCache(), [], [], 0
        self.state = item.get("state") if m.use_state else None
        seq = self._new_seq()
        seq.text(m.prefix_tokens(item["context"], item.get("history", "")), -1)
        self._forward(seq); self.cur = seq.cur
        prev, preds, probs, texts, cuts, speaks = -1, [], [], {}, [], []
        rate_s, rate_c = (self.tgt_s if self.speak_rate_init is None else self.speak_rate_init), self.tgt_c
        recent = []
        for t in range(T):
            # 1. lo feeds + <DIR>
            seq = self._new_seq()
            m.second_header(seq, t, item["game_clock_start"] + t, (item.get("events") or [""] * T)[t])
            m.lo_block(seq, t, K)
            imgs = [(frames[t, k], m.lo_side) for (_, k, _) in seq.images]
            h = self._forward(seq, imgs); self.cur = seq.cur
            seq_local = Seq.new(); seq_local.dir_pos = [seq.dir_pos[0]]; seq_local.lo_tok = seq.lo_tok
            pv = torch.tensor([prev], device=self.dev)
            target, cut = m.director_heads(h, seq_local, pv, self.state[t:t + 1] if self.state is not None else None)
            from .model import combine_director
            probs.append(combine_director(target, cut, pv)[0].exp().cpu())
            p_cut = float(torch.sigmoid(cut[0]))
            tau_c = min(0.98, max(0.02, self.tau_cut + self.g_c * (rate_c - self.tgt_c))) if self.rc else self.tau_cut
            if prev >= 0 and p_cut < tau_c:
                c = prev
            else:
                tg = target[0].clone()
                if prev >= 0:
                    tg[prev] = -1e4
                c = int(tg.argmax())
            preds.append(c); cuts.append(p_cut)
            if prev >= 0:
                rate_c = (1 - self.ema) * rate_c + self.ema * float(c != prev)
            onair = c if onair_mode == "free" else (int(item["cam"][t]) if int(item["cam"][t]) >= 0 else c)
            # 2. on-air feed (hi) + <COM> + words
            seq = self._new_seq()
            extra = []
            if m.glimpses > 1 and m.foveate:
                tg = target[0].clone(); tg[onair] = -1e9
                extra = tg.topk(m.glimpses - 1).indices.tolist()     # the director's next-best feeds
            m.onair_block(seq, t, onair, extra)
            seq.text([COM_ID], t)
            imgs = [(frames[t, f], m.hi_side) for (_, f, _) in seq.images] if m.foveate else None
            h = self._forward(seq, imgs); self.cur = seq.cur
            words = []
            p_speak = float(torch.sigmoid(m.speak(h[-1:]).float()[0, 0]))
            speaks.append(p_speak)
            if force_text:
                w = m.words_tokens(item["text"][t])
                s2 = self._new_seq(); s2.text(w, t); self._forward(s2); self.cur = s2.cur
            tau_s = min(0.98, max(0.02, self.tau_speak + self.g_s * (rate_s - self.tgt_s))) if self.rc else self.tau_speak
            spoke = (not force_text) and gen_text and p_speak >= tau_s
            rate_s = (1 - self.ema) * rate_s + self.ema * float(spoke)
            if force_text:
                pass
            elif spoke:
                logit = self.lm.lm_head(h[-1:]).float()[0]
                logit[END_ID] = -1e4                          # the response head decided to speak
                for _ in range(self.max_new_tokens + 1):
                    for tok in set(recent[-64:] + words):
                        logit[tok] = logit[tok] / self.rep if logit[tok] > 0 else logit[tok] * self.rep
                    if words and self.end_thr > 0 and float(logit.softmax(-1)[END_ID]) < self.end_thr:
                        logit[END_ID] = -1e4
                    nxt = int(logit.argmax())
                    if len(words) >= self.max_new_tokens:
                        nxt = END_ID
                    s2 = self._new_seq(); s2.text([nxt], t)
                    hh = self._forward(s2); self.cur = s2.cur
                    if nxt == END_ID:
                        break
                    words.append(nxt)
                    logit = self.lm.lm_head(hh[-1:]).float()[0]
            else:                                             # silent second
                s2 = self._new_seq(); s2.text([END_ID], t); self._forward(s2); self.cur = s2.cur
            if words:
                texts[t] = m.tok.decode(words).strip()
                recent += words
            prev = onair if onair_mode == "free" else (int(item["cam"][t]) if int(item["cam"][t]) >= 0 else onair)
            self._evict(t + 1)
        return {"pred_cam": preds, "probs": torch.stack(probs), "texts": texts, "p_cut": cuts, "p_speak": speaks}
