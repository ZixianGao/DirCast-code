"""Director metrics and Proact-VL-format commentary files.

Director (evaluated on strict & LIVE seconds, t >= 1 so the persistence baseline is defined):
  top1          predicted feed == human director's feed
  any_visible   predicted feed shows a hero that is on the broadcast screen (or is the label)
  switch_acc    top1 restricted to seconds where the human cut (cam_t != cam_{t-1})
  hold_acc      top1 restricted to seconds where the human held
  cut_f1        cut events: predicted feed != previous on-air feed vs human cut, matched within +-1 s
  persistence   the same metrics for "repeat the previous on-air feed" (teacher history)
"""
import json

DATASET = "dota2"
TAG = "Solo commentators"   # merged-caster ASR (no diarization) -> Proact-VL solo protocol


def _cut_f1(pred_cuts, gt_cuts, tol=1):
    used, tp = set(), 0
    for p in pred_cuts:
        for j, g in enumerate(gt_cuts):
            if j not in used and abs(p - g) <= tol:
                used.add(j); tp += 1; break
    P = tp / len(pred_cuts) if pred_cuts else 0.0
    R = tp / len(gt_cuts) if gt_cuts else 0.0
    return tp, len(pred_cuts), len(gt_cuts), (2 * P * R / (P + R) if P + R else 0.0)


def director_metrics(rows, mask_key="strict"):
    acc = {"model": [0, 0, 0, 0, 0, 0, 0], "persistence": [0, 0, 0, 0, 0, 0, 0]}   # n, top1, vis, nsw, sw, nhold, hold
    cuts = {"model": [0, 0, 0], "persistence": [0, 0, 0]}
    for r in rows:
        cam, pred, m, vis = r["cam"], r["pred_cam"], r[mask_key], r["visible"]
        T = len(cam)
        gt_cuts = [t for t in range(1, T) if m[t] and m[t - 1] and cam[t] != cam[t - 1]]
        # teacher history: on-air feed before t is the human's; in free mode the model's own previous choice
        model_cuts = [t for t in range(1, T) if m[t] and m[t - 1] and pred[t] != (r.get("prev_onair") or cam)[t - 1]]
        for name, cut_list in (("model", model_cuts), ("persistence", [])):
            tp, npred, ngt, _ = _cut_f1(cut_list, gt_cuts)
            cuts[name][0] += tp; cuts[name][1] += npred; cuts[name][2] += ngt
        for t in range(1, T):
            if not m[t]:
                continue
            for name, p in (("model", pred[t]), ("persistence", cam[t - 1])):
                a = acc[name]
                ok = p == cam[t]
                a[0] += 1; a[1] += ok; a[2] += ok or bool(vis[t][p])
                if cam[t] != cam[t - 1]:
                    a[3] += 1; a[4] += ok
                else:
                    a[5] += 1; a[6] += ok
    out = {}
    for name, a in acc.items():
        tp, npred, ngt = cuts[name]
        P = tp / npred if npred else 0.0
        R = tp / ngt if ngt else 0.0
        out[name] = {"seconds": a[0], "top1": a[1] / max(a[0], 1), "any_visible": a[2] / max(a[0], 1),
                     "switch_acc": a[4] / max(a[3], 1), "switch_seconds": a[3], "hold_acc": a[6] / max(a[5], 1),
                     "cut_precision": P, "cut_recall": R, "cut_f1": 2 * P * R / (P + R) if P + R else 0.0}
    return out


def write_label_files(rows, pred_path, asr_path, model_id="dircast"):
    """Prediction file (with inline labels) for f1_timediff.py, and an ASR file for llm_judge.py."""
    with open(pred_path, "w") as fp, open(asr_path, "w") as fa:
        for r in rows:
            label = {str(t): s for t, s in enumerate(r["text"]) if s.strip()}
            if not label:
                continue                               # no human commentary in this clip: nothing to compare
            T = len(r["text"])
            fp.write(json.dumps({"video_id": r["clip_id"], "begin": 0, "end": T, "idx": r["idx"], "dataset_name": DATASET,
                                 "tag": TAG, "model_id": model_id, "pred": r["pred_text"], "label": label}) + "\n")
            fa.write(json.dumps({"video_path": r["clip_id"], "video_begin": 0, "video_end": T, "idx": r["idx"],
                                 "dataset_name": DATASET, "tag": TAG, "active_speaker": {"name": "caster"},
                                 "annotations": [{"start": int(t), "text": s, "speaker": "caster", "role": "assistant"}
                                                 for t, s in label.items()]}) + "\n")
