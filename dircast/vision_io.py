"""Fast Qwen3-VL image patchification (torch), equivalent to Qwen2VLImageProcessor.

The HF processor takes ~13 s for the 360 thumbnails of one 36 s clip; this
does the same resize -> rescale -> normalize -> patchify on tensors.
Layout matches ``Qwen2VLImageProcessor._preprocess`` (temporal_patch_size=2,
the single frame repeated, merge_size=2).
"""
import torch
import torch.nn.functional as F

PATCH, MERGE, TPS = 16, 2, 2
MEAN = STD = 0.5
# per backbone: (patch, mean, std)
VISION = {"qwen3_vl": (16, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          "qwen2_vl": (14, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
          "qwen2_5_vl": (14, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))}


def patchify(imgs: torch.Tensor, side: int, patch: int = PATCH, mean=(MEAN,) * 3, std=(STD,) * 3):
    """imgs: [N, H, W, 3] uint8 -> (pixel_values [N*g*g, 3*2*p*p] float32, grid [N,3] long).

    Every image is resized to side x side (side % (2*patch) == 0)."""
    PATCH = patch
    assert side % (PATCH * MERGE) == 0
    x = imgs.permute(0, 3, 1, 2).float()
    if x.shape[-1] != side or x.shape[-2] != side:
        x = F.interpolate(x, size=(side, side), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp(0, 255) / 255.0
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1); sd = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    x = (x - m) / sd                                               # [N,3,S,S]
    n, g = x.shape[0], side // PATCH
    x = x.unsqueeze(1).expand(n, TPS, 3, side, side)               # repeat frame in time
    x = x.reshape(n, 1, TPS, 3, g // MERGE, MERGE, PATCH, g // MERGE, MERGE, PATCH)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)                    # n,t,gh,gw,mh,mw,C,tps,ph,pw
    pv = x.reshape(n * g * g, 3 * TPS * PATCH * PATCH)
    grid = torch.tensor([[1, g, g]] * n, dtype=torch.long)
    return pv, grid


def n_tokens(side: int) -> int:
    """LLM tokens per image after 2x2 merge."""
    return (side // PATCH // MERGE) ** 2
