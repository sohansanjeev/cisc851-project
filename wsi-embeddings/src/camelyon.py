"""CAMELYON16 access: slide listing, remote tile reads, tissue and lesion masks.

The slides are 0.3-4 GB each and the local disk holds ~15 GB, so nothing is
downloaded whole. Tiles are fetched with HTTP range requests and the annotation
XML is rasterised locally.

Four quirks of these files, all discovered the hard way:

  * tiles are JPEG with the quantization tables stored once at page level, so a
    decode fails with "Quantization table 0x00 was not defined" unless
    page.jpegtables is passed;
  * tile byte ranges are neither contiguous nor ordered, so a per-tile range
    request is mandatory;
  * tiles that are pure background are stored with byte count 0, and a range
    request for one returns 416 rather than an empty body;
  * downsampled levels are padded to a whole number of tiles (level 3 of a 97792
    px slide is stored 12288 px wide, not 12224), so any native-to-coarse map
    must crop to the declared level size or it will shear.

`masks/*_mask.tif` encodes 1 = tissue and 2 = tumour, NOT a pure tumour mask: on
a normal slide it marks 98% of the tissue. The lesion ground truth used here is
the ASAP XML, which agrees with the value-2 region at IoU 0.80-0.92.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import aiohttp
import fsspec
import numpy as np
import tifffile
from PIL import Image, ImageDraw

BASE = "https://camelyon-dataset.s3.amazonaws.com/CAMELYON16"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CACHE = os.path.join(ROOT, "data", "meta")
XML_CACHE = os.path.join(ROOT, "data", "annotations")


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def list_slides(prefix: str = "CAMELYON16/images/") -> list[dict]:
    """Every slide key with its byte size. One request for all 399 objects."""
    url = (f"https://camelyon-dataset.s3.amazonaws.com/?list-type=2"
           f"&max-keys=1000&prefix={prefix}")
    with urllib.request.urlopen(url, timeout=60) as r:
        xml = r.read().decode()
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    sizes = [int(s) for s in re.findall(r"<Size>(\d+)</Size>", xml)]
    out = [{"slide_id": k.split("/")[-1][:-4], "key": k, "bytes": s}
           for k, s in zip(keys, sizes) if k.endswith(".tif")]
    return sorted(out, key=lambda d: d["slide_id"])


# --------------------------------------------------------------------------- #
# Page metadata
# --------------------------------------------------------------------------- #
@dataclass
class PageMeta:
    shape: tuple[int, int]          # (h, w) as stored; may be padded past the level size
    th: int
    tw: int
    tiles_across: int
    n_tiles: int
    offsets: np.ndarray
    counts: np.ndarray
    jpegtables: bytes | None
    is_tiled: bool
    compression: int
    spp: int = 3                     # samples per pixel

    @property
    def tiles_down(self) -> int:
        return math.ceil(self.shape[0] / self.th)


def describe_page(page) -> PageMeta:
    h, w = page.shape[0], page.shape[1]
    th = page.tilelength if page.is_tiled else h
    tw = page.tilewidth if page.is_tiled else w
    return PageMeta(
        shape=(h, w), th=th, tw=tw,
        tiles_across=math.ceil(w / tw),
        n_tiles=len(page.dataoffsets) if page.is_tiled else 1,
        offsets=np.asarray(page.dataoffsets, dtype=np.int64),
        counts=np.asarray(page.databytecounts, dtype=np.int64),
        jpegtables=page.jpegtables,
        is_tiled=bool(page.is_tiled),
        compression=int(page.compression),
        spp=int(page.samplesperpixel),
    )


def _cache_path(url: str, level: int) -> str:
    tag = url.rsplit("/", 1)[-1].replace(".tif", "")
    return os.path.join(META_CACHE, f"{tag}.L{level}.npz")


def page_meta(url: str, level: int = 0, use_cache: bool = True) -> PageMeta:
    """Metadata for one pyramid level, cached on disk (offsets are ~700 KB)."""
    cp = _cache_path(url, level)
    if use_cache and os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        jt = z["jpegtables"].tobytes() if z["jpegtables"].size else None
        return PageMeta(shape=tuple(int(v) for v in z["shape"]),
                        th=int(z["th"]), tw=int(z["tw"]),
                        tiles_across=int(z["tiles_across"]), n_tiles=int(z["n_tiles"]),
                        offsets=z["offsets"], counts=z["counts"],
                        jpegtables=jt, is_tiled=bool(z["is_tiled"]),
                        compression=int(z["compression"]), spp=int(z["spp"]))
    with fsspec.open(url, "rb", block_size=1 << 20, cache_type="none") as f:
        tif = tifffile.TiffFile(f)
        meta = describe_page(tif.pages[level])
    if use_cache:
        os.makedirs(META_CACHE, exist_ok=True)
        np.savez(cp, shape=np.array(meta.shape), th=meta.th, tw=meta.tw,
                 tiles_across=meta.tiles_across, n_tiles=meta.n_tiles,
                 offsets=meta.offsets, counts=meta.counts,
                 jpegtables=np.frombuffer(meta.jpegtables or b"", dtype=np.uint8),
                 is_tiled=meta.is_tiled, compression=meta.compression, spp=meta.spp)
    return meta


def _levels(url: str) -> list[tuple[int, int]]:
    with fsspec.open(url, "rb", block_size=1 << 20, cache_type="none") as f:
        return [tuple(int(v) for v in lv.shape[:2])
                for lv in tifffile.TiffFile(f).series[0].levels]


def pick_mask_level(url: str, target_max: int = 4096) -> int:
    """Coarsest level whose longest side is <= target_max (keeps tile count low)."""
    for i, shp in enumerate(_levels(url)):
        if max(shp) <= target_max:
            return i
    return 0


def closest_level(url: str, target_shape: tuple[int, int]) -> int:
    """Pyramid level whose area is nearest `target_shape`."""
    tgt = target_shape[0] * target_shape[1]
    return int(np.argmin([abs(h * w - tgt) for h, w in _levels(url)]))


# --------------------------------------------------------------------------- #
# Concurrent tile fetch
# --------------------------------------------------------------------------- #
async def _fetch_all(url, offsets, counts, concurrency=24):
    async with aiohttp.ClientSession() as session:
        return await _fetch(session, url, offsets, counts, concurrency)


async def _fetch(session, url, offsets, counts, concurrency=24):
    sem = asyncio.Semaphore(concurrency)
    out = [None] * len(offsets)

    async def one(i):
        async with sem:
            for attempt in range(4):
                try:
                    async with session.get(
                        url, headers={"Range": f"bytes={offsets[i]}-{offsets[i] + counts[i] - 1}"}
                    ) as r:
                        if r.status != 206:
                            raise RuntimeError(f"status {r.status}")
                        out[i] = await r.read()
                        return
                except Exception:
                    await asyncio.sleep(0.25 * (attempt + 1))
            raise RuntimeError(f"tile {i} failed after retries")

    await asyncio.gather(*(one(i) for i in range(len(offsets))))
    return out


def fetch_tiles(url: str, meta: PageMeta, indices, concurrency: int = 24) -> list[bytes]:
    idx = list(dict.fromkeys(int(i) for i in indices))   # dedupe, keep order
    if not idx:
        return []
    out = [b""] * len(idx)
    live = [j for j, i in enumerate(idx) if meta.counts[i] > 0]   # empty tiles are legit
    if not live:
        return out
    blobs = asyncio.run(_fetch_all(url, meta.offsets[[idx[j] for j in live]],
                                   meta.counts[[idx[j] for j in live]], concurrency))
    for j, b in zip(live, blobs):
        out[j] = b
    return out


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
def decode_tile(page, meta: PageMeta, index: int, blob: bytes) -> np.ndarray:
    if not blob:      # tile stored with byte count 0 == all background
        shape = (meta.th, meta.tw, meta.spp) if meta.spp > 1 else (meta.th, meta.tw)
        return np.zeros(shape, np.uint8)
    kw = {"jpegtables": meta.jpegtables} if meta.compression == 7 else {}
    a = page.decode(blob, int(index), **kw)
    if isinstance(a, tuple):
        a = a[0]
    a = np.asarray(a)
    if a.ndim == 4:          # (1, h, w, c)
        a = a[0]
    return a


class RemoteSlide:
    """Open a slide's pyramid level, fetch tiles, decode with the page object."""

    def __init__(self, url: str, level: int = 0, use_cache: bool = True):
        self.url = url
        self.level = level
        self.meta = page_meta(url, level, use_cache)
        self._fo = fsspec.open(url, "rb", block_size=1 << 20, cache_type="none").open()
        self._tif = tifffile.TiffFile(self._fo)
        self._page = self._tif.pages[level]
        # Downsampled levels are padded to a whole number of tiles, so the stored
        # page is larger than the level it represents. Level 0 is unpadded. Crop
        # to the declared level size everywhere.
        lev = self._tif.series[0].levels
        self.logical = tuple(int(v) for v in lev[level].shape[:2]) if level < len(lev) \
            else self.meta.shape

    def close(self):
        try:
            self._tif.close()
        finally:
            self._fo.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def read_all(self, concurrency: int = 24) -> np.ndarray:
        """Whole level as an array, cropped to the declared size. Coarse use only."""
        m = self.meta
        blobs = fetch_tiles(self.url, m, range(m.n_tiles), concurrency)
        first = decode_tile(self._page, m, 0, blobs[0])
        rgb = first.ndim == 3 and first.shape[-1] >= 3
        H, W = m.tiles_down * m.th, m.tiles_across * m.tw
        canvas = np.zeros((H, W, 3) if rgb else (H, W), np.uint8)
        for i, b in enumerate(blobs):
            a = decode_tile(self._page, m, i, b)
            r, cidx = divmod(i, m.tiles_across)
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[..., 0]
            sl = (slice(r * m.th, r * m.th + a.shape[0]),
                  slice(cidx * m.tw, cidx * m.tw + a.shape[1]))
            if rgb:
                canvas[sl[0], sl[1], :a.shape[2]] = a
            else:
                canvas[sl[0], sl[1]] = a
        out = canvas[:m.shape[0], :m.shape[1]]
        lh, lw = self.logical
        if lh < out.shape[0] or lw < out.shape[1]:
            out = out[:lh, :lw]
        return out

    def read_regions(self, regions, concurrency: int = 24, batch: int = 64) -> list[np.ndarray]:
        """regions: sequence of (y0, x0, h, w) at this level. Returns RGB arrays.

        Decoded tiles are held only for the current batch; 500 patches at once
        would be ~1.5 GB of 512x512 tiles otherwise.
        """
        out: list[np.ndarray] = []
        for start in range(0, len(regions), batch):
            out.extend(self._read_batch(regions[start:start + batch], concurrency))
        return out

    def _read_batch(self, regions, concurrency: int = 24) -> list[np.ndarray]:
        m = self.meta
        need: dict[int, tuple[int, int]] = {}
        for (y0, x0, h, w) in regions:
            tr0, tr1 = y0 // m.th, (y0 + h - 1) // m.th
            tc0, tc1 = x0 // m.tw, (x0 + w - 1) // m.tw
            for tr in range(tr0, tr1 + 1):
                for tc in range(tc0, tc1 + 1):
                    need[tr * m.tiles_across + tc] = (tr, tc)
        idx = list(need)
        blobs = fetch_tiles(self.url, m, idx, concurrency)
        tiles = {i: decode_tile(self._page, m, i, b) for i, b in zip(idx, blobs)}
        out = []
        for (y0, x0, h, w) in regions:
            buf = np.zeros((h, w, 3), np.uint8)
            tr0, tr1 = y0 // m.th, (y0 + h - 1) // m.th
            tc0, tc1 = x0 // m.tw, (x0 + w - 1) // m.tw
            for tr in range(tr0, tr1 + 1):
                for tc in range(tc0, tc1 + 1):
                    t = tiles[tr * m.tiles_across + tc]
                    if t.ndim == 2:
                        t = np.repeat(t[..., None], 3, axis=2)
                    ty, tx = tr * m.th, tc * m.tw
                    oy0, oy1 = max(y0, ty), min(y0 + h, ty + t.shape[0])
                    ox0, ox1 = max(x0, tx), min(x0 + w, tx + t.shape[1])
                    if oy1 <= oy0 or ox1 <= ox0:
                        continue
                    buf[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = \
                        t[oy0 - ty:oy1 - ty, ox0 - tx:ox1 - tx, :3]
            out.append(buf)
        return out


# --------------------------------------------------------------------------- #
# Lesion annotations
# --------------------------------------------------------------------------- #
TUMOUR_GROUPS = ("tumor", "metastases", "metastasis")


def fetch_xml(slide_id: str) -> str | None:
    os.makedirs(XML_CACHE, exist_ok=True)
    cp = os.path.join(XML_CACHE, f"{slide_id}.xml")
    if os.path.exists(cp):
        return cp
    try:
        with urllib.request.urlopen(f"{BASE}/annotations/{slide_id}.xml", timeout=60) as r:
            data = r.read()
    except Exception:
        return None
    open(cp, "wb").write(data)
    return cp


def polygons(xml_path: str) -> list[tuple[str, list[tuple[float, float]]]]:
    root = ET.parse(xml_path).getroot()
    out = []
    for ann in root.iter("Annotation"):
        grp = (ann.get("PartOfGroup") or "").strip().lower()
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in ann.iter("Coordinate")]
        if len(pts) >= 3:
            out.append((grp, pts))
    return out


def rasterise(slide_id: str, level_shape: tuple[int, int],
              native_shape: tuple[int, int]) -> np.ndarray:
    """Lesion mask at `level_shape`, 1 = tumour. Exclusion groups carve holes."""
    h, w = level_shape
    sy, sx = native_shape[0] / h, native_shape[1] / w
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    cp = fetch_xml(slide_id)
    if cp is not None:
        polys = polygons(cp)
        for grp, pts in polys:                     # positives first
            if grp in TUMOUR_GROUPS:
                d.polygon([(x / sx, y / sy) for x, y in pts], fill=1)
        for grp, pts in polys:                     # exclusions carve on top
            if grp not in TUMOUR_GROUPS:
                d.polygon([(x / sx, y / sy) for x, y in pts], fill=0)
    return np.array(img, np.uint8)


def lesion_mask(slide_id: str, level_shape: tuple[int, int],
                native_shape: tuple[int, int]) -> np.ndarray:
    cp = os.path.join(XML_CACHE, f"{slide_id}.lesion_{level_shape[0]}x{level_shape[1]}.npy")
    if os.path.exists(cp):
        return np.load(cp)
    arr = rasterise(slide_id, level_shape, native_shape)
    os.makedirs(XML_CACHE, exist_ok=True)
    np.save(cp, arr)
    return arr
