"""Build the patch-embedding dataset: pick the cohort, sample, fetch, embed, save.

Three subcommands:

    python src/build.py rank        # lesion area for all 111 tumour slides -> data/lesion_areas.json
    python src/build.py build       # the dataset itself -> data/slides/*.npz
    python src/build.py manifest    # flatten the npz files -> data/manifest.csv

Design decisions, all reported in the README:

  * 224 px patches at 0.5 um/px = a 112 um field of view. Virchow2 was trained
    across 0.25-2.0 um/px, so 0.5 is in-distribution. Camelyon16 is natively
    0.243 um/px, so one patch is a 461 px native crop resized to 224.
  * N = 500 patches per slide, drawn uniformly at random (seeded by slide id)
    from positions whose local tissue density clears a threshold. Random rather
    than top-ranked so the subset the EA picks is spatially free; a
    top-epithelium sampler would hand it a cluster and the spatial term would be
    meaningless.
  * Every patch carries the variables behind the filter (tissue_frac,
    tumor_frac, mean chroma) so a threshold can be revisited without re-cutting
    pixels.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from camelyon import (BASE, RemoteSlide, closest_level, lesion_mask,  # noqa: E402
                      list_slides, page_meta, pick_mask_level, polygons, fetch_xml)

NATIVE_MPP = 0.243          # published Camelyon16 pitch; not stored in the file
TARGET_MPP = 0.5
PATCH_PX = 224
N_PATCHES = 500
TISSUE_MIN = 0.55           # window tissue density required of a candidate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def window_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Mean over every k x k window, via an integral image. No scipy needed."""
    a = arr.astype(np.float32)
    ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = a.shape
    out = np.full((h, w), np.nan, np.float32)
    if h < k or w < k:
        return out
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    out[:h - k + 1, :w - k + 1] = s / (k * k)
    return out


def patch_native_px(native_mpp: float = NATIVE_MPP, target_mpp: float = TARGET_MPP) -> int:
    return int(round(PATCH_PX * target_mpp / native_mpp))


def mask_array(url: str, level: int | None = None) -> tuple[np.ndarray, int]:
    """Read (and cache) a coarse mask level. Returns (array, level)."""
    sid = url.rsplit("/", 1)[-1].replace(".tif", "")
    lv = pick_mask_level(url) if level is None else level
    cp = os.path.join(DATA, "masks", f"{sid}.L{lv}.npy")
    if os.path.exists(cp):
        return np.load(cp), lv
    with RemoteSlide(url, lv) as s:
        arr = s.read_all()
    if arr.ndim == 3:
        arr = arr[..., 0]
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    np.save(cp, arr)
    return arr, lv


def coarse_chroma(sid: str, target_shape: tuple[int, int]) -> np.ndarray:
    """Mean R-G on the tissue-mask grid, from one coarse pyramid level.

    Only used by the optional chroma-ranked sampler; off by default.
    """
    url = f"{BASE}/images/{sid}.tif"
    with RemoteSlide(url, closest_level(url, target_shape)) as s:
        img = s.read_all()
    if img.ndim == 2:
        return np.zeros(target_shape, np.float32)
    ch = img[..., 0].astype(np.float32) - img[..., 1].astype(np.float32)
    if ch.shape != tuple(target_shape):
        ch = np.asarray(Image.fromarray(ch).resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR), dtype=np.float32)
    return ch


def choose_positions(tissue: np.ndarray, scale: float, want: int, seed: int,
                     pnat: int, tissue_min: float = TISSUE_MIN,
                     chroma: np.ndarray | None = None,
                     chroma_pct: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Pick `want` (y, x) native top-lefts on tissue.

    Two filters, both on the same k x k window:
      * tissue density from the provider's mask, which rejects glass;
      * optionally, stain content (chroma R-G) at or above the slide's own
        percentile, which rejects adipocytes. Relative rather than absolute
        because stain intensity drifts between slides. Off by default: at coarse
        resolution the window is diluted by background, so it barely separates
        fat from pale tissue while biasing the pool toward dense cores.
    """
    k = max(1, int(round(pnat / scale)))
    wm = window_mean(tissue, k)
    wm = np.where(np.isfinite(wm), wm, 0.0)

    rng = np.random.default_rng(seed)
    thr = tissue_min
    cand = np.argwhere(wm >= thr)
    while len(cand) < want and thr > 0.15:      # relax on sparse slides
        thr -= 0.1
        cand = np.argwhere(wm >= thr)
    if len(cand) == 0:
        raise RuntimeError("no tissue found")

    if chroma is not None:
        cm = window_mean(np.clip(chroma, 0, None), k)
        cm = np.where(np.isfinite(cm), cm, np.nan)
        floor = np.nanpercentile(cm[cand[:, 0], cand[:, 1]], chroma_pct)
        sub = cand[cm[cand[:, 0], cand[:, 1]] >= floor]
        if len(sub) >= want:
            cand = sub
        else:
            print(f"    chroma filter left {len(sub)} candidates, keeping all")

    pick = rng.choice(len(cand), size=min(want, len(cand)), replace=False)
    cyx = cand[np.sort(pick)]

    half = pnat // 2
    h, w = tissue.shape
    ys = np.clip((cyx[:, 0] * scale).astype(np.int64) + int(k * scale / 2) - half, 0, None)
    xs = np.clip((cyx[:, 1] * scale).astype(np.int64) + int(k * scale / 2) - half, 0, None)
    ys = np.minimum(ys, int(h * scale) - pnat)
    xs = np.minimum(xs, int(w * scale) - pnat)
    return np.stack([ys, xs], 1), wm[cyx[:, 0], cyx[:, 1]]


def frac_at(mask: np.ndarray, ys: np.ndarray, xs: np.ndarray, scale: float,
            pnat: int, value: int | None = None) -> np.ndarray:
    """Mean of `mask` (optionally == value) over each patch footprint."""
    out = np.empty(len(ys), np.float32)
    k = max(1, int(round(pnat / scale)))
    for i, (y, x) in enumerate(zip(ys, xs)):
        win = mask[int(y / scale):int(y / scale) + k, int(x / scale):int(x / scale) + k]
        if win.size == 0:
            out[i] = 0.0
        elif value is None:
            out[i] = float(win.mean())
        else:
            out[i] = float((win == value).mean())
    return out


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
def load_model(device="cuda"):
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.layers import SwiGLUPacked

    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                              mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = model.eval().to(device)
    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    print("  transform config:", cfg)
    torch.set_grad_enabled(False)
    return model, cfg, torch


def embed(model, cfg, torch, patches: np.ndarray, device="cuda", bs: int = 64) -> np.ndarray:
    """patches (N, 224, 224, 3) uint8 -> (N, 2560) float32. Frozen, no grad."""
    from torchvision import transforms

    steps = [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    if cfg["input_size"][-1] != PATCH_PX:          # only resize if the model wants it
        steps.insert(0, transforms.Resize(cfg["input_size"][-1], antialias=True))
    tf = transforms.Compose(steps)

    out = []
    for i in range(0, len(patches), bs):
        x = torch.stack([tf(Image.fromarray(p)) for p in patches[i:i + bs]])
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(x)
        out.append(torch.cat([o[:, 0], o[:, 5:].mean(1)], dim=-1).float().cpu().numpy())
    return np.concatenate(out, 0)


# --------------------------------------------------------------------------- #
# Per-slide build
# --------------------------------------------------------------------------- #
def build_slide(sid: str, label: str, model_bundle, n_patches: int = N_PATCHES) -> dict:
    img_url = f"{BASE}/images/{sid}.tif"
    tis_url = f"{BASE}/background_tissue/{sid}_tissue.tif"

    t0 = time.time()
    tissue, lv_t = mask_array(tis_url)
    img_meta = page_meta(img_url, 0)
    nat_h, nat_w = img_meta.shape
    pnat = patch_native_px()
    sy = nat_h / tissue.shape[0]
    les = lesion_mask(sid, tissue.shape, (nat_h, nat_w))    # tumour, from the XML
    print(f"  native {nat_h}x{nat_w}; tissue L{lv_t} {tissue.shape} scale {sy:.2f}; "
          f"lesion {les.shape} fov {les.mean():.3f}; patch {pnat}px native -> {PATCH_PX}px")

    seed = int.from_bytes(hashlib.blake2b(sid.encode(), digest_size=8).digest(), "big")
    pos, _ = choose_positions(tissue, sy, n_patches, seed, pnat)
    ys, xs = pos[:, 0], pos[:, 1]
    tissue_frac = frac_at(tissue, ys, xs, sy, pnat, value=1) if tissue.max() <= 1 \
        else frac_at(tissue, ys, xs, sy, pnat)
    tumor_frac = frac_at(les, ys, xs, sy, pnat, value=1)

    t1 = time.time()
    with RemoteSlide(img_url, 0) as s:
        crops = s.read_regions([(int(y), int(x), pnat, pnat) for y, x in pos])
    t2 = time.time()

    patches = np.empty((len(crops), PATCH_PX, PATCH_PX, 3), np.uint8)
    for i, c in enumerate(crops):
        patches[i] = np.asarray(Image.fromarray(c).resize((PATCH_PX, PATCH_PX), Image.LANCZOS))
    chroma = (patches[..., 0].astype(np.int16)
              - patches[..., 1].astype(np.int16)).mean((1, 2))

    model, cfg, torch = model_bundle
    t3 = time.time()
    feats = embed(model, cfg, torch, patches)
    t4 = time.time()

    assert feats.shape == (len(patches), 2560), feats.shape
    assert np.isfinite(feats).all(), "non-finite features"

    out = os.path.join(DATA, "slides", f"{sid}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, features=feats, coords=np.stack([ys, xs], 1).astype(np.int32),
        patches=patches,
        tissue_frac=tissue_frac.astype(np.float32),
        tumor_frac=tumor_frac.astype(np.float32),
        chroma=chroma.astype(np.float32),
        slide_id=np.array(sid), label=np.array(label),
        native_shape=np.array([nat_h, nat_w]),
        native_mpp=np.array([NATIVE_MPP], np.float32),
        target_mpp=np.array([TARGET_MPP], np.float32),
        patch_native_px=np.array([pnat], np.int32),
        patch_px=np.array([PATCH_PX], np.int32),
        seed=np.array([seed], np.uint64),
    )
    size = os.path.getsize(out)
    print(f"  {len(patches)} patches | masks {t1-t0:.1f}s | fetch {t2-t1:.1f}s | "
          f"embed {t4-t3:.1f}s | tissue {tissue_frac.mean():.2f} "
          f"tumour {tumor_frac.mean():.2f} | {size/1e6:.1f} MB")
    return {"slide_id": sid, "label": label, "n_patches": int(len(patches)),
            "tissue_frac": float(tissue_frac.mean()),
            "tumor_frac": float(tumor_frac.mean()),
            "tumor_patches": int((tumor_frac > 0.05).sum()),
            "chroma": float(chroma.mean()), "bytes": int(size), "seed": int(seed),
            "native_shape": [int(nat_h), int(nat_w)], "patch_native_px": int(pnat)}


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_rank(_args):
    """Lesion area per tumour slide, from the XML polygons alone (no tiles)."""
    mpp = NATIVE_MPP
    mm2_per_px2 = (mpp * 1e-3) ** 2

    def shoelace(pts):
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    rows = []
    for i in range(1, 112):
        sid = f"tumor_{i:03d}"
        cp = fetch_xml(sid)
        if cp is None:
            print("missing", sid)
            continue
        polys = polygons(cp)
        pos = sum(shoelace(p) for g, p in polys if g in ("tumor", "metastases", "metastasis"))
        neg = sum(shoelace(p) for g, p in polys if g not in ("tumor", "metastases", "metastasis"))
        rows.append({"slide_id": sid,
                     "lesion_mm2": round(float((pos - neg) * mm2_per_px2), 3),
                     "n_polygons": len(polys)})
    rows.sort(key=lambda r: -r["lesion_mm2"])
    areas = np.array([r["lesion_mm2"] for r in rows])
    print(f"{len(rows)} tumour slides; lesion mm2 min {areas.min():.3f} "
          f"median {np.median(areas):.2f} max {areas.max():.2f}")
    print("quantiles 10/25/50/75/90:",
          np.round(np.percentile(areas, [10, 25, 50, 75, 90]), 2).tolist())
    for r in rows[:15]:
        print(f"  {r['slide_id']}  {r['lesion_mm2']:8.2f} mm2  ({r['n_polygons']} polys)")
    os.makedirs(DATA, exist_ok=True)
    json.dump(rows, open(os.path.join(DATA, "lesion_areas.json"), "w"), indent=2)
    print("wrote data/lesion_areas.json")


def cmd_build(args):
    slides = {s["slide_id"]: s for s in list_slides()}
    sel = json.load(open(os.path.join(DATA, "selection.json")))
    todo = [(s, "normal") for s in sel["normal"]] + [(s, "tumour") for s in sel["tumour"]]
    if args.slides:
        todo = [(s, "tumour" if s.startswith("tumor") else "normal") for s in args.slides]

    rows = []
    print("loading Virchow2 ...")
    bundle = load_model()
    for sid, label in todo:
        out = os.path.join(DATA, "slides", f"{sid}.npz")
        if args.skip_existing and os.path.exists(out):
            if len(np.load(out)["coords"]) == args.n_patches:
                print(f"[{sid}] exists ({args.n_patches} patches), skipped")
                continue
        print(f"[{sid}] ({slides[sid]['bytes']/1e9:.2f} GB remote)")
        rows.append(build_slide(sid, label, bundle, args.n_patches))

    mp = os.path.join(DATA, "manifest_rows.json")
    if os.path.exists(mp) and args.skip_existing:
        old = json.load(open(mp))
        seen = {r["slide_id"] for r in rows}
        rows = [r for r in old if r["slide_id"] not in seen] + rows
    json.dump(rows, open(mp, "w"), indent=2)
    print("wrote", mp)


def cmd_manifest(_args):
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "slides", "*.npz"))):
        z = np.load(f)
        rows.append({
            "slide_id": str(z["slide_id"]), "label": str(z["label"]),
            "n_patches": len(z["features"]),
            "mean_tissue_frac": float(z["tissue_frac"].mean()),
            "mean_tumor_frac": float(z["tumor_frac"].mean()),
            "patches_on_tumour_5pct": int((z["tumor_frac"] > 0.05).sum()),
            "mean_chroma": float(z["chroma"].mean()),
            "native_h": int(z["native_shape"][0]), "native_w": int(z["native_shape"][1]),
            "patch_native_px": int(z["patch_native_px"][0]),
            "seed": int(z["seed"][0]),
            "file_mb": round(os.path.getsize(f) / 1e6, 1),
        })
    rows.sort(key=lambda r: (r["label"], r["slide_id"]))
    out = os.path.join(DATA, "manifest.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} slides -> {out}")
    for lab in ("normal", "tumour"):
        sub = [r for r in rows if r["label"] == lab]
        if sub:
            print(f"  {lab:7s} n={len(sub):2d} patches {sum(r['n_patches'] for r in sub):5d} "
                  f"mean tumour_frac {np.mean([r['mean_tumor_frac'] for r in sub]):.3f} "
                  f"size {sum(r['file_mb'] for r in sub)/1000:.2f} GB")


def cmd_export(_args):
    """Write a patch-free copy of each slide for version control.

    The 224 px RGB images are ~95% of each file and no consumer of this dataset
    needs them: the EA works on `features`, and the pixels can be re-cut from the
    remote WSI with `coords` whenever a figure is wanted. Dropping them takes the
    set from 2.3 GB to a size a git repo can carry.
    """
    keep = ("features", "coords", "tissue_frac", "tumor_frac", "chroma",
            "slide_id", "label", "native_shape", "native_mpp", "target_mpp",
            "patch_native_px", "patch_px", "seed")
    out_dir = os.path.join(DATA, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    files = sorted(glob.glob(os.path.join(DATA, "slides", "*.npz")))
    for f in files:
        z = np.load(f)
        dst = os.path.join(out_dir, os.path.basename(f))
        np.savez_compressed(dst, **{k: z[k] for k in keep})
        total += os.path.getsize(dst)
    print(f"{len(files)} slides -> {out_dir} ({total/1e6:.0f} MB, no pixel arrays)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rank").set_defaults(func=cmd_rank)
    b = sub.add_parser("build")
    b.add_argument("--slides", nargs="*", default=None)
    b.add_argument("--n-patches", type=int, default=N_PATCHES)
    b.add_argument("--skip-existing", action="store_true")
    b.set_defaults(func=cmd_build)
    sub.add_parser("manifest").set_defaults(func=cmd_manifest)
    sub.add_parser("export").set_defaults(func=cmd_export)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
