#!/usr/bin/env python3
"""
Saturation profile transfer tool.

Usage:
  python autoedit.py create [--inspo inspo/] [--output profile.json] [--bands 16]
  python autoedit.py apply <input> <output> [--profile profile.json]

Dependencies: numpy, Pillow  (pip install numpy Pillow)
"""

import argparse
import json
import os
import numpy as np
from PIL import Image, ImageOps


# ── HSV helpers ───────────────────────────────────────────────────────────────

def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """arr: float32 HxWx3 in [0,1]. Returns float32 HxWx3 HSV, H in [0,1)."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    v = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    delta = v - mn

    s = np.where(v > 0, delta / np.where(v > 0, v, 1.0), 0.0)

    h = np.zeros_like(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_r = (v == r) & (delta > 0)
        m_g = (v == g) & (delta > 0)
        m_b = (v == b) & (delta > 0)
        h[m_r] = ((g[m_r] - b[m_r]) / delta[m_r]) % 6
        h[m_g] = (b[m_g] - r[m_g]) / delta[m_g] + 2
        h[m_b] = (r[m_b] - g[m_b]) / delta[m_b] + 4
    h /= 6.0

    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Returns float32 HxWx3 RGB in [0,1]."""
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb = np.empty((*h.shape, 3), dtype=np.float32)
    for idx, (rc, gc, bc) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                         (p, q, v), (t, p, v), (v, p, q)]):
        m = i == idx
        rgb[m, 0] = rc[m]
        rgb[m, 1] = gc[m]
        rgb[m, 2] = bc[m]
    return rgb


def load_hsv(path: str) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    arr = np.array(img, dtype=np.float32) / 255.0
    return rgb_to_hsv(arr)


# ── Histogram application ──────────────────────────────────────────────────────

def precompute_ranks(src_vals: np.ndarray, bins: int = 4096) -> np.ndarray:
    """Approximate fractional rank [0,1) for each element via CDF lookup. O(N) vs argsort O(N log N)."""
    flat = src_vals.ravel()
    hist, edges = np.histogram(flat, bins=bins, range=(0.0, 1.0))
    cdf = np.cumsum(hist).astype(np.float32)
    cdf /= cdf[-1]
    idx = np.clip(np.digitize(flat, edges[1:], right=False), 0, bins - 1)
    return cdf[idx].reshape(src_vals.shape)


def ranks_to_histogram(ranks: np.ndarray, ref_hist: np.ndarray) -> np.ndarray:
    """Map precomputed ranks through ref_hist inverse CDF."""
    bins = 256
    centers = (np.arange(bins) + 0.5) / bins

    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    total = ref_cdf[-1]
    if total > 0:
        ref_cdf /= total

    idx = np.searchsorted(ref_cdf, ranks.ravel(), side="left").clip(0, bins - 1)
    return centers[idx].reshape(ranks.shape).astype(np.float32)


# ── Histogram smoothing ────────────────────────────────────────────────────────

def _smooth_hist(h: np.ndarray, sigma: int = 4) -> np.ndarray:
    """Gaussian-smooth a histogram using numpy convolution."""
    size = sigma * 6 + 1
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= kernel.sum()
    return np.convolve(h, kernel, mode="same")


# ── Create profile ─────────────────────────────────────────────────────────────

def cmd_create(inspo_dir: str, output_path: str, n_bands: int):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    images = [
        os.path.join(inspo_dir, f)
        for f in os.listdir(inspo_dir)
        if os.path.splitext(f.lower())[1] in exts
    ]
    if not images:
        print(f"No images found in {inspo_dir!r}")
        return

    band_hists = [np.zeros(256, dtype=np.float64) for _ in range(n_bands)]
    v_hist = np.zeros(256, dtype=np.float64)

    for path in images:
        print(f"  {os.path.basename(path)}")
        hsv = load_hsv(path)
        h_chan, s_chan, v_chan = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        for band in range(n_bands):
            lo, hi = band / n_bands, (band + 1) / n_bands
            mask = (h_chan >= lo) & (h_chan < hi)
            if not mask.any():
                continue
            hist, _ = np.histogram(s_chan[mask], bins=256, range=(0.0, 1.0))
            band_hists[band] += hist

        hist, _ = np.histogram(v_chan, bins=256, range=(0.0, 1.0))
        v_hist += hist

    band_hists = [_smooth_hist(h) for h in band_hists]
    v_hist = _smooth_hist(v_hist)
    profile = {
        "n_bands": n_bands,
        "histograms": [h.tolist() for h in band_hists],
        "v_histogram": v_hist.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(profile, f)
    print(f"Profile saved → {output_path}")


# ── Apply profile ──────────────────────────────────────────────────────────────

def cmd_apply(input_path: str, output_path: str, profile_path: str):
    with open(profile_path) as f:
        profile = json.load(f)

    n_bands = profile["n_bands"]
    band_hists = [np.array(h) for h in profile["histograms"]]

    hsv = load_hsv(input_path)
    h_chan = hsv[..., 0]
    s_chan = hsv[..., 1].copy()
    v_chan = hsv[..., 2]

    # Compute per-band saturation adjustment using Gaussian-feathered hue weights
    # so pixels near band boundaries blend smoothly between adjacent adjustments
    band_width = 1.0 / n_bands
    band_sigma = band_width * 0.5  # feather half a band width on each side

    s_ranks = precompute_ranks(s_chan)

    # Build per-band inverse CDF LUTs: (n_bands, bins)
    bins = 256
    centers = ((np.arange(bins) + 0.5) / bins).astype(np.float32)
    rank_bins = (np.arange(bins) + 0.5) / bins
    all_luts = np.zeros((n_bands, bins), dtype=np.float32)
    has_data = np.zeros(n_bands, dtype=bool)
    for band, ref_hist in enumerate(band_hists):
        if ref_hist.sum() == 0:
            # Hues absent from inspo are left unchanged (weight stays zero → s_chan fallback).
            # If this causes unwanted results, option: interpolate from neighboring bands.
            continue
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1]
        all_luts[band] = centers[np.searchsorted(ref_cdf, rank_bins, side="left").clip(0, bins - 1)]
        has_data[band] = True

    # Precompute 2D LUT: (hue_bins, sat_rank_bins) → s_new
    # Gaussian weights for each hue bin over all bands: (hue_bins, n_bands)
    h_lut_bins = 512
    h_vals = (np.arange(h_lut_bins) + 0.5) / h_lut_bins
    band_centers = (np.arange(n_bands) + 0.5) / n_bands
    diff = np.abs(h_vals[:, None] - band_centers[None, :])
    diff = np.minimum(diff, 1.0 - diff)
    band_weights_2d = np.exp(-0.5 * (diff / band_sigma) ** 2)  # (h_lut_bins, n_bands)
    band_weights_2d *= has_data[None, :]  # zero out missing bands

    weight_sum = band_weights_2d.sum(axis=1, keepdims=True)
    # Weighted blend of all band LUTs: (h_lut_bins, bins)
    lut_2d = np.where(weight_sum > 0, (band_weights_2d @ all_luts) / weight_sum, None)

    # Apply 2D LUT: one lookup per pixel
    h_idx = np.clip((h_chan * h_lut_bins).astype(np.int32), 0, h_lut_bins - 1)
    s_rank_idx = np.clip((s_ranks * bins).astype(np.int32), 0, bins - 1)
    s_new_flat = lut_2d[h_idx.ravel(), s_rank_idx.ravel()]
    # Fall back to s_chan where no band had data (weight_sum == 0)
    no_data = (weight_sum.ravel()[h_idx.ravel()] == 0)
    s_new_flat[no_data] = s_chan.ravel()[no_data]
    s_new = s_new_flat.reshape(s_chan.shape).astype(np.float32)

    # Luminance weighting: fade adjustment to zero below 10% and above 90% V
    weight = np.ones_like(v_chan)
    weight[v_chan < 0.1] = v_chan[v_chan < 0.1] / 0.1
    weight[v_chan > 0.9] = (1.0 - v_chan[v_chan > 0.9]) / 0.1
    weight = np.clip(weight, 0.0, 1.0)

    s_final = np.clip(s_chan + weight * (s_new - s_chan), 0.0, 1.0)

    v_hist = profile.get("v_histogram")
    if v_hist is not None:
        v_ref = np.array(v_hist)
        v_cdf = np.cumsum(v_ref).astype(np.float64)
        v_cdf /= v_cdf[-1]
        rank_bins = (np.arange(bins) + 0.5) / bins
        v_lut = centers[np.searchsorted(v_cdf, rank_bins, side="left").clip(0, bins - 1)]
        v_ranks = precompute_ranks(v_chan)
        v_rank_idx = np.clip((v_ranks * bins).astype(np.int32), 0, bins - 1)
        v_final = v_lut[v_rank_idx]
    else:
        v_final = v_chan

    rgb = hsv_to_rgb(h_chan, s_final, v_final)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(output_path)
    print(f"Saved → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Saturation profile transfer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Build profile from inspo images")
    p_create.add_argument("--inspo", default="inspo", help="Folder of inspiration images")
    p_create.add_argument("--output", default="profile.json", help="Output profile file")
    p_create.add_argument("--bands", type=int, default=16, help="Number of hue bands")

    p_apply = sub.add_parser("apply", help="Apply profile to an image")
    p_apply.add_argument("input", help="Input image path")
    p_apply.add_argument("output", help="Output image path")
    p_apply.add_argument("--profile", default="profile.json", help="Profile file")

    args = parser.parse_args()

    if args.command == "create":
        cmd_create(args.inspo, args.output, args.bands)
    elif args.command == "apply":
        cmd_apply(args.input, args.output, args.profile)


if __name__ == "__main__":
    main()
