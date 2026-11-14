"""Verification for the built dataset. Six subcommands, all re-runnable.

    python src/qc.py overview                  # integrity + determinism, no network
    python src/qc.py slide --slide tumor_076   # contact sheet, overlay figure, nuclei
    python src/qc.py masks --slide normal_010  # is masks/*_mask.tif really tissue?
    python src/qc.py lesion --slide tumor_020  # XML polygons vs the provider's own raster
    python src/qc.py transfer                  # exact bytes pulled per slide
    python src/qc.py probe                     # mean-pooled linear probe + controls

None of these prove the biology is right. They prove the pipeline is not broken:
the features carry the label, the coordinates reproduce, the geometry is
isotropic, and the lesion ground truth agrees with an independent encoding of
the same annotation.
"""
import argparse
import glob
import hashlib
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from build import N_PATCHES, choose_positions, mask_array, patch_native_px  # noqa: E402
from camelyon import BASE, RemoteSlide, lesion_mask, page_meta  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "qc")


def _slides():
    return sorted(glob.glob(os.path.join(DATA, "slides", "*.npz")))


def _load(sid):
    return np.load(os.path.join(DATA, "slides", f"{sid}.npz"))


# --------------------------------------------------------------------------- #
# overview: integrity + determinism, no network
# --------------------------------------------------------------------------- #
def cmd_overview(_a):
    problems, bad_det, aniso = [], [], []
    total = 0
    files = _slides()
    for f in files:
        z = np.load(f)
        sid, n = str(z["slide_id"]), len(z["features"])
        total += os.path.getsize(f)

        expect = {"features": np.float32, "coords": np.int32, "patches": np.uint8,
                  "tissue_frac": np.float32, "tumor_frac": np.float32,
                  "chroma": np.float32}
        for k, dt in expect.items():
            if z[k].dtype != dt:
                problems.append(f"{sid}:{k} dtype {z[k].dtype}")
            if z[k].shape[0] != n:
                problems.append(f"{sid}:{k} rows {z[k].shape[0]} != {n}")
        if n != N_PATCHES:
            problems.append(f"{sid}: {n} patches")
        if not np.isfinite(z["features"]).all():
            problems.append(f"{sid}: non-finite features")
        if z["patches"].shape[1:] != (224, 224, 3):
            problems.append(f"{sid}: patch shape {z['patches'].shape[1:]}")
        if (z["coords"] < 0).any():
            problems.append(f"{sid}: negative coords")
        if len({tuple(r) for r in z["coords"]}) != n:
            problems.append(f"{sid}: duplicate coords")
        if (str(z["label"]) == "tumour") != sid.startswith("tumor"):
            problems.append(f"{sid}: label {z['label']} disagrees with the name")

        # geometry: one scale per axis, or every window is placed wrong
        m = glob.glob(os.path.join(DATA, f"masks/{sid}_tissue.L*.npy"))
        if not m:
            continue
        tissue = np.load(m[0])
        nat = z["native_shape"]
        sy, sx = nat[0] / tissue.shape[0], nat[1] / tissue.shape[1]
        if abs(sy - sx) > 0.02:
            aniso.append((sid, round(sy, 3), round(sx, 3)))
        seed = int.from_bytes(hashlib.blake2b(sid.encode(), digest_size=8).digest(), "big")
        pos, _ = choose_positions(tissue, sy, len(z["coords"]), seed, patch_native_px())
        if not np.array_equal(pos, z["coords"]):
            bad_det.append(sid)

    print(f"{len(files)} files, {total/1e9:.2f} GB")
    print("integrity problems :", problems or "none")
    print("anisotropic masks  :", aniso or "none")
    print("coords reproduce   :", "all" if not bad_det else bad_det)


# --------------------------------------------------------------------------- #
# slide-level visuals
# --------------------------------------------------------------------------- #
def contact_sheet(sid: str, n: int = 25, cols: int = 5) -> str:
    P = _load(sid)["patches"][:n]
    rows, s = (n + cols - 1) // cols, 224
    canvas = np.full((rows * s, cols * s, 3), 255, np.uint8)
    for i, p in enumerate(P):
        r, c = divmod(i, cols)
        canvas[r * s:(r + 1) * s, c * s:(c + 1) * s] = p
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, f"contact_{sid}.png")
    Image.fromarray(canvas).save(fp)
    return fp


def figure(sid: str, level: int = 6) -> str:
    """Slide thumbnail with the lesion tinted and the sampled positions marked."""
    z = _load(sid)
    coords, tf = z["coords"], z["tumor_frac"]
    nat = tuple(int(v) for v in z["native_shape"])
    with RemoteSlide(f"{BASE}/images/{sid}.tif", level) as s:
        img = s.read_all()
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, 2)

    les = lesion_mask(sid, img.shape[:2], nat) > 0
    out = img.astype(np.float32)
    out[les] = 0.55 * out[les] + 0.45 * np.array([0, 200, 60], np.float32)
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    sy, sx = img.shape[0] / nat[0], img.shape[1] / nat[1]
    d = ImageDraw.Draw(im)
    for (y, x), f in zip(coords, tf):
        cy, cx, r = y * sy, x * sx, 4
        d.ellipse([cx - r, cy - r, cx + r, cy + r], width=2,
                  outline=(220, 0, 0) if f > 0.05 else (40, 40, 40))
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, f"figure_{sid}.png")
    im.save(fp)
    return fp


def nuclei(sid: str, mpp: float = 0.5, n_patches: int = 6) -> dict:
    """Median nucleus diameter over a threshold sweep.

    A 7 um lymphocyte is 14 px at 0.5 um/px, and thresholding isolates only the
    dense chromatin core, so 8-9 px is the expected under-read; ~17 px would mean
    the crop is at native resolution instead.
    """
    from skimage import color, measure, morphology

    P = _load(sid)["patches"][:n_patches]
    res = []
    for p in P:
        g = color.rgb2gray(p)
        for q in np.linspace(10, 40, 7):
            m = morphology.remove_small_objects(g < np.percentile(g, q), 12)
            d = [r.equivalent_diameter_area for r in measure.regionprops(measure.label(m))
                 if 20 < r.area < 3000]
            if d:
                res.append(float(np.median(d)))
    res = np.array(res)
    return {"median_px": float(np.median(res)), "p10": float(np.percentile(res, 10)),
            "p90": float(np.percentile(res, 90)), "n": int(len(res)),
            "median_um": float(np.median(res) * mpp)}


def cmd_slide(a):
    print("contact sheet:", contact_sheet(a.slide))
    print("figure       :", figure(a.slide))
    r = nuclei(a.slide)
    print(f"nucleus: median {r['median_px']:.1f} px ({r['median_um']:.1f} um) "
          f"[{r['p10']:.1f}-{r['p90']:.1f}] over {r['n']} measurements "
          f"(expect ~8-9 px at 0.5 um/px)")


# --------------------------------------------------------------------------- #
# mask semantics
# --------------------------------------------------------------------------- #
def cmd_masks(a):
    """Compare background_tissue/*_tissue.tif with masks/*_mask.tif."""
    sid = a.slide
    tissue, _ = mask_array(f"{BASE}/background_tissue/{sid}_tissue.tif")
    with RemoteSlide(f"{BASE}/masks/{sid}_mask.tif", 6) as s:
        m = s.read_all()
    m = m[..., 0] if m.ndim == 3 else m
    tm, mm = tissue == 1, m == 1
    inter = np.logical_and(tm, mm).sum()
    union = np.logical_or(tm, mm).sum()
    print(f"{sid}: tissue {tm.mean():.4f}  mask==1 {mm.mean():.4f}  "
          f"mask==2 {(m == 2).mean():.4f}")
    print(f"  IoU(tissue, mask==1) {inter/max(union,1):.3f}  "
          f"share of tissue marked 1 {inter/max(tm.sum(),1):.3f}")
    print("  -> mask==1 is tissue, not tumour; a normal slide scores ~0.98 of tissue")


def cmd_lesion(a):
    """XML raster vs the provider's value-2 region. Works on any tumour slide."""
    nat = page_meta(f"{BASE}/images/{a.slide}.tif", 0).shape
    with RemoteSlide(f"{BASE}/masks/{a.slide}_mask.tif", 6) as s:
        m = s.read_all()
    m = m[..., 0] if m.ndim == 3 else m
    x, y = lesion_mask(a.slide, m.shape, nat) == 1, m == 2
    inter = np.logical_and(x, y).sum()
    union = np.logical_or(x, y).sum()
    print(f"{a.slide}: mask==2 {y.mean():.4f}  xml {x.mean():.4f}  "
          f"IoU {inter/max(union,1):.3f}")


# --------------------------------------------------------------------------- #
# transfer size
# --------------------------------------------------------------------------- #
def cmd_transfer(_a):
    """Exact bytes the patch sampling pulled, from the cached tile index."""
    tot_b = tot_n = n = 0
    for f in _slides():
        z = np.load(f)
        if len(z["coords"]) != N_PATCHES:
            continue
        sid = str(z["slide_id"])
        m = page_meta(f"{BASE}/images/{sid}.tif", 0)
        pnat = int(z["patch_native_px"][0])
        idx = set()
        for y, x in z["coords"]:
            for tr in range(y // m.th, (y + pnat - 1) // m.th + 1):
                for tc in range(x // m.tw, (x + pnat - 1) // m.tw + 1):
                    idx.add(tr * m.tiles_across + tc)
        b = int(m.counts[list(idx)].sum())
        tot_b += b
        tot_n += len(idx)
        n += 1
        print(f"{sid:12s} {len(idx):5d} tiles  {b/1e6:6.1f} MB")
    print(f"\n{n} slides: {tot_b/1e9:.2f} GB for patches "
          f"({tot_b/n/1e6:.1f} MB/slide, {tot_n/n:.0f} tiles/slide)")


# --------------------------------------------------------------------------- #
# linear probe
# --------------------------------------------------------------------------- #
def cmd_probe(_a):
    """Does the representation carry the slide label at all? A gate, not a result."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, ids = [], [], []
    for f in _slides():
        z = np.load(f)
        F = z["features"]
        X.append(np.concatenate([F.mean(0), F.std(0)]))
        y.append(1 if str(z["label"]) == "tumour" else 0)
        ids.append(str(z["slide_id"]))
    X, y = np.asarray(X), np.asarray(y)
    if len(np.unique(y)) < 2:
        print("only one class present")
        return

    def loo_auc(xx, yy):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.05, max_iter=5000,
                                               class_weight="balanced"))
        p = cross_val_predict(clf, xx, yy, cv=LeaveOneOut(), method="predict_proba")[:, 1]
        return roc_auc_score(yy, p), p

    auc, p = loo_auc(X, y)
    print(f"{len(y)} slides, {X.shape[1]}-d (mean+std pooling)")
    print(f"features   LOO AUC {auc:.3f}  acc {((p > 0.5).astype(int) == y).mean():.3f}")

    rng = np.random.default_rng(0)
    shuffled = [loo_auc(X, rng.permutation(y))[0] for _ in range(20)]
    print(f"permuted   LOO AUC {np.mean(shuffled):.3f} +- {np.std(shuffled):.3f} "
          f"(20 shuffles)")
    print(f"patch count constant at {N_PATCHES}: cannot carry the label")
    print("\nper-slide p(tumour):")
    for sid, yy, pp in sorted(zip(ids, y, p), key=lambda t: -t[2]):
        print(f"  {sid:12s} true {'tumour' if yy else 'normal':7s} p={pp:.3f}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("overview").set_defaults(func=cmd_overview)
    for name, fn in (("slide", cmd_slide), ("masks", cmd_masks), ("lesion", cmd_lesion)):
        p = sub.add_parser(name)
        p.add_argument("--slide", required=True)
        p.set_defaults(func=fn)
    sub.add_parser("transfer").set_defaults(func=cmd_transfer)
    sub.add_parser("probe").set_defaults(func=cmd_probe)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
