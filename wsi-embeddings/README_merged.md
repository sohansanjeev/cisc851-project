# CISC 851 Project: Evolutionary Patch Selection for CAMELYON16

This project investigates genetic algorithms for selecting informative image patches from whole-slide histopathology images in the CAMELYON16 dataset. It first produces fixed Virchow2 patch embeddings and spatial coordinates, then searches for subsets of patches using a ridge-classifier fitness function. Selected source patches are also used to find similar patches on held-out slides for normal-versus-tumour classification.

The repository includes both **data preparation and quality checks** and **genetic algorithm experiments, baselines, evaluation, and plotting**. The main EA experiments use the precomputed embeddings in `data/embeddings/`; rebuilding the original pixel dataset is not required to run them.

All paths below are relative to the Git repository root (`cisc851-project/`) unless a section explicitly says to work inside `wsi-embeddings/`.

## Quick start: run the evolutionary algorithm

From the repository root, create and activate a Python virtual environment and install the EA dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r wsi-embeddings/requirements-ea.txt
```

To regenerate the figures from the committed experimental results:

```bash
python wsi-embeddings/src/plot_experiments.py
```

To check that the GA and baselines execute on a smaller run (one seed, one fold, two generations):

```bash
python wsi-embeddings/src/run_experiments.py \
    --k 20 --seeds 851 \
    --generations 2 --fold-limit 1 \
    --output "wsi-embeddings/results/test_run"
```

This is a functionality check, **not** a reproduction of the full results. The output folder must not already exist; choose a different folder name if you have run the test before.

To repeat the three full experiment settings in fresh output folders:

```bash
mkdir -p wsi-embeddings/results/repeat
for k in 10 20 40; do
    python wsi-embeddings/src/run_experiments.py \
        --k "$k" --seeds 851 852 853 \
        --generations 30 --fold-limit 5 \
        --output "wsi-embeddings/results/repeat/k${k}_full"
done
python wsi-embeddings/src/plot_experiments.py --results wsi-embeddings/results/repeat
```

The saved experiment configurations record Python 3.14.6, NumPy 2.5.5 and scikit-learn 1.9.1; plotting used Matplotlib 3.11.2. Dependency ranges for EA runs are listed in `wsi-embeddings/requirements-ea.txt`.

## Data and embeddings

The project uses **40 slides from the CAMELYON16 labelled training set**, with 20 normal and 20 tumour slides.

| class | how chosen |
|---|---|
| normal | `normal_001` … `normal_020` |
| tumour | the 20 slides with the largest annotated lesion area (11.6 – 138.3 mm²) |

Each full slide file is stored as `data/slides/<slide_id>.npz` (when the pixel dataset has been built). The patch-free records in `data/embeddings/<slide_id>.npz` are committed to Git — about 191 MB for 40 slides compared with about 2.3 GB including patch pixels. The EA consumes the patch-free embeddings; patches can be re-cut from the remote slide using `coords` when images are needed.

| key | shape | meaning |
|---|---|---|
| `features` | (500, 2560) float32 | Virchow2 embedding, frozen |
| `coords` | (500, 2) int32 | `(y, x)` top-left of the patch at level 0, native pixels |
| `patches` | (500, 224, 224, 3) uint8 | the RGB patch the embedding came from |
| `tissue_frac` | (500,) float32 | fraction of the footprint the provider's tissue mask calls tissue |
| `tumor_frac` | (500,) float32 | fraction of the footprint inside an annotated metastasis |
| `chroma` | (500,) float32 | mean `R − G` of the patch; low values flag washed-out or glass patches |
| attrs | scalars | `slide_id`, `label`, `native_shape`, `native_mpp`, `target_mpp`, `patch_native_px`, `patch_px`, `seed` |

`data/manifest.csv` is the flat table; `data/selection.json` records the slide choice; `data/lesion_areas.json` is the annotation-area ranking for all 111 tumour slides.

With the precomputed embeddings, read the features and coordinates from the EA input files:

```python
import numpy as np
z = np.load("data/embeddings/tumor_039.npz")
X, xy = z["features"], z["coords"]  # (500, 2560), (500, 2)
```

The full slide files (if available locally) also contain `patches` with shape `(500, 224, 224, 3)`.

For evolutionary search, each slide contributes 500 patch embeddings of dimension 2,560 and their spatial coordinates. Mean and standard deviation pooling of the selected subset gives a 5,120-dimensional feature vector used by a ridge classifier.

## Representation decisions

**Encoder: Virchow2, frozen.** ViT-H/14, 632 M parameters, DINOv2, pretrained on 3.1 M whole-slide images. Output is 2560-d per patch: the class token concatenated with the mean of the 256 spatial patch tokens, with tokens 1–4 dropped because they are registers.

**Scale: 0.5 µm/px, 224 px patches, a 112 µm field of view.** Virchow2 was pretrained at 0.25, 0.5, 1.0 and 2.0 µm/px, so 0.5 is in-distribution. A 112 µm patch spans roughly sixteen lymphocyte diameters, so a few hundred cells: enough to show metastatic carcinoma and the stromal reaction around it, and the field size most 20× Camelyon16 baselines use. It is smaller than a lymph-node follicle, which runs 0.5–1 mm, so the label is carried by cell morphology inside the patch rather than by architecture. Camelyon16 is natively 0.243 µm/px, so one patch is a 461 px native crop downsampled to 224.

**Patches per slide: 500, sampled uniformly at random on tissue.** Not top-ranked by anything. A sampler that takes the most epithelial patches hands the EA a spatially clustered set and the redundancy objective becomes meaningless before the search starts. Random sampling keeps the subset free, which is the whole point of letting an EA choose. C(500, 20) ≈ 3.4 × 10⁴¹, so the search space is genuinely beyond enumeration while one evaluation is still microseconds, and that combination is what makes this a defensible EA problem.

**Coordinates retained.** Without them any spatial term in the fitness is impossible, and a purely feature-based redundancy measure cannot distinguish "twenty patches from one follicle" from "twenty patches across the node".

**Reproducibility.** Patch positions are drawn from `random.default_rng(int(blake2b(slide_id, 8)))`, so the same slide always yields the same 500 patches. The seed is stored in each file.

## Cohort selection

The 111 labelled tumour slides carry ASAP XML lesion annotations whose area spans 0.008 – 138 mm², median 1.4 mm². Most lesions are far too small for a 500-patch sample to land on them: at median lesion size a random tissue patch has a fraction of a percent chance of touching tumour, so `tumor_frac` would be zero almost everywhere and the "did the EA find the metastasis" analysis would have nothing to compare against. The tumour cohort is therefore the 20 largest lesions, 11.6 mm² and up.

## Pixel acquisition and preparation

CAMELYON16 is CC0 and hosted on the AWS Open Data bucket; the 40 slides total 75.8 GB and the local disk had 17 GB free. Nothing is downloaded whole. Every TIFF is tiled at 512×512 and every tile is JPEG-compressed, so the tile index is read from the header once and only the tiles a patch overlaps are fetched over HTTP range requests, 24 concurrent. A 461 px crop straddles up to four tiles, so 500 scattered patches need about 1,700 unique tiles and 89 MB per slide, transferred in roughly 20 s. That is a 21× reduction on the 1.9 GB mean slide, and 3.5 GB for the whole set instead of 75.8 GB. The figure is reproducible with `python src/qc.py transfer`, which sums the cached tile index rather than trusting an estimate.

Three quirks cost real debugging time and are all handled in `src/camelyon.py`.

- The JPEG quantization tables are stored once per page, so a tile decode fails with `Quantization table 0x00 was not defined` unless the page's `jpegtables` are passed explicitly.
- Tiles that are pure background are stored with `byte count == 0`, and a range request for one returns 416 rather than an empty body.
- Downsampled levels are padded to a whole number of tiles: level 3 of a 97792 px-wide slide is stored 12288 px wide rather than 12224, and level 6 of a 114688 px-wide slide is stored 2048 rather than 1792. Level 0 is unpadded, so the patches are unaffected, but anything mapping native coordinates onto a coarse level with a single isotropic scale will shear — which is exactly what happened to the first version of the QC overlays until `RemoteSlide.read_all` was made to crop to the declared level size.

Masks are read at 1/64 resolution on most slides and 1/32 on four of them, which keeps the tile count at 21 or 49, in about 0.5 s.

### Data preparation commands

The following commands operate from the `wsi-embeddings/` directory; rebuilding embeddings involves additional slide-access and model dependencies beyond `requirements-ea.txt`.

```bash
cd wsi-embeddings
python src/build.py rank        # lesion area for all 111 tumour slides
python src/build.py build       # the dataset itself
python src/build.py manifest    # flatten to data/manifest.csv
python src/build.py export      # patch-free copy in data/embeddings/ for git
python src/qc.py overview       # integrity + determinism, no network
python src/qc.py slide --slide tumor_076
python src/qc.py probe
```

## Data quality checks

**Pixel scale.** No `XResolution` tag exists in these files, so 0.243 µm/px is the published figure rather than a measured one. It was verified indirectly: a threshold sweep over nuclei returns a median equivalent diameter of 7.4–8.5 px depending on the slide. A 7 µm lymphocyte is 14 px at 0.5 µm/px, and thresholding isolates only the dense chromatin core, so 7–9 px is the expected under-read, whereas a native-resolution crop would give about 17 px. The measurement separates those two hypotheses, which is all a scale check can do.

**Spatial spread.** `qc/figure_<slide>.png` overlays the 500 positions on a downscaled slide; they cover all tissue fragments and none fall in blank glass.

**Lesion rasterisation.** The XML polygons were cross-checked against the provider's own mask, which encodes 1 = tissue and 2 = tumour, and agree at IoU 0.68–0.92 across six slides — the low end is a 0.02 mm² metastasis, only a handful of pixels wide at 1/64. This is what established that the `masks/*_mask.tif` objects are not pure tumour masks: on a normal slide they mark 98 % of the tissue, so using them as ground truth would have labelled every normal slide as tumour.

**Feature sanity.** All 500 rows are distinct on every slide, the mean pairwise cosine within a slide runs 0.31–0.54 (median 0.41), and there are no non-finite values. The end-to-end gate is mean⊕std pooling of the 500 features into a regularised logistic regression, leave-one-out over the 40 slides, which gives AUROC 0.985 and accuracy 0.925. Controls in the same script: permuted labels give 0.501 ± 0.120, and patch count is constant at 500, so neither the harness nor the sampling step can be carrying the label. Three tumour slides are genuinely hard under this head, and two of them land below 0.5: tumor_011 at p=0.48 and tumor_047 at p=0.12, with tumor_039 at 0.64. That is useful, because the bag-level task is not so saturated that every subset looks the same. Run it with `python src/qc.py probe`.

**A sixth of the patches are mostly fat, left in on purpose.** About 17 % of all patches have mean `R − G < 4`, meaning very little stain, and on two of the normal slides (`normal_006`, `normal_008`) it is 59–67 % because those nodes are mostly adipose and the provider's mask counts fat as tissue. A chroma-ranked sampler was tried and rejected: at coarse resolution the window is diluted by background, so a percentile filter removed only 8 points off the worst slide while biasing the pool toward dense cores. Uniform sampling keeps the pool an honest sample of the slide, and `chroma` ships so the EA team can filter if their fitness rewards cellular patches. This is arguably good material for the project, since a slide whose 500 items span fat and lymphoid tissue is exactly where subset selection has something to do.

**`tissue_frac` means "not background", not "cellular".** The provider's mask counts adipocytes and stroma as tissue, so a patch of fat scores high. If a cellularity filter is wanted, `chroma` is a better lever than `tissue_frac`.

**`tumor_frac` means "inside the annotated outline", not "contains tumour cells".** The pathologist's polygon encloses the metastatic region including fat and stroma within it, which `qc/figure_tumor_076.png` makes visible: green tint, red circles inside it, dark circles outside. A patch can therefore score high on `tumor_frac` while being largely adipose.

## Genetic algorithm and classification method

1. Train a `StandardScaler` and `RidgeClassifier(alpha=10)` on 24 random k-patch bags per classifier-training slide. Keep the model fixed during search and transfer.
2. Represent a candidate as k distinct patch indices referencing the original embedding vectors.
3. Pool those vectors and calculate `fitness = (2 * label - 1) * decision_score`, with normal = 0 and tumour = 1. Larger fitness favours the correct source class. It is not a probability.
4. Search each source slide, then combine the selected vectors from both classes into one prototype bank.
5. On each held-out slide, rank target patches by their highest cosine similarity to any prototype. Pool the k highest-ranking target patches original vectors and classify them. Target labels are used only for evaluation.

The main setting is k = 20. The patch-count comparison also uses k = 10 and k = 40.

| GA setting | Value |
| --- | --- |
| Population | 32 candidates |
| Generations | 30 |
| Parent selection | Tournament of 3 |
| Elites | Best 2 candidates retained |
| Crossover | 80% probability; keep shared patches and fill from the parents remaining patches |
| Mutation | Replace one patch in every offspring |
| Global mutation | Choose any currently unselected patch |
| Local mutation | Choose among the 10 nearest currently unselected patches by spatial coordinates |
| Mixed mutation | Choose local or global replacement with equal probability |

Each search receives `32 + 30 * (32 - 2) = 932` candidate attempts per source slide. Random search and hill climbing use the same budget and the same initial population for a given source and seed. Hill climbing accepts improving single-patch global replacements. Cached duplicate candidates still count towards the budget.

## Evaluation design

Each of five stratified folds uses 16 classifier-training slides, 16 source-search slides, and 8 test slides. The roles do not overlap within a fold. Each slide is tested once across the five folds.

The split seed is 851; search seeds are 851, 852, and 853. Within each k and fold, all methods and seeds share the same ridge model. Accuracy and macro-F1 use the 40 held-out predictions per method and seed. The seeds reuse the same slides and are not independent cohorts.

The references include random search, hill climbing, direct random selection of k target patches, and pooling all 500 target patches.

## Experiment results

Mean held-out accuracy across the three seeds:

| Method | k = 10 | k = 20 | k = 40 |
| --- | ---: | ---: | ---: |
| Global GA | 85.0% | 82.5% | 85.0% |
| Local GA | 84.2% | 82.5% | 83.3% |
| Mixed GA | 85.0% | 83.3% | 83.3% |
| Random search | 81.7% | 81.7% | 83.3% |
| Hill climbing | 85.0% | 80.0% | 81.7% |
| Random target patches | 90.8% | 90.8% | 95.0% |
| All 500 target patches | 90.0% | 95.0% | 95.0% |

Each k uses a ridge model trained with k-patch bags. The all-500 reference uses that setting's model, so its results can differ between columns despite pooling 500 patches.

The searches improve source fitness, but no GA variant exceeds direct random target selection in mean held-out accuracy. Hill climbing gains the most source fitness at each k without the highest held-out accuracy. Better source confidence therefore does not guarantee better transfer. The cause of this gap has not been established.

[The combined summary](results/figures/combined_summary.csv) reports means and sample standard deviations for accuracy and macro-F1. The standard deviations describe seed variation on this cohort. [The figures](results/figures) show search progress and held-out accuracy.

## Project files and outputs

The main code is inside `wsi-embeddings/src/`.

| File | Purpose |
| --- | --- |
| `camelyon.py` | Remote slide access, tile decoding, tissue masks, and ASAP XML lesion rasterisation |
| `build.py` | Cohort ranking, patch sampling, Virchow2 embeddings, and manifest |
| `qc.py` | Dataset integrity, determinism, figures, masks, transfer and linear-probe checks |
| `patch_data.py` | Load slide data and pool patch features |
| `ridge_baseline.py` | Create slide splits and train the ridge classifier |
| `ga_operators.py` | Create populations, select parents, cross over, and mutate |
| `ga_search.py` | Calculate fitness and run the GA |
| `prototype_transfer.py` | Build the prototype bank and select similar target patches |
| `search_baselines.py` | Run random search and hill climbing |
| `run_experiments.py` | Evaluate all methods and save the results |
| `plot_experiments.py` | Combine summaries and create the figures |

Other folders and datasets:

- `data/embeddings/`: compact, committed patch embeddings used by the EA.
- `data/slides/`: full slide records including patch pixels, if built locally.
- `data/manifest.csv`, `data/selection.json`, `data/lesion_areas.json`: cohort metadata.
- `qc/`: quality-control figures and contact sheets.
- `results/`: saved experiments and plot outputs.

Each full run in `results/k10_full`, `results/k20_full`, or `results/k40_full` contains:

- `config.json` and `splits.json`: settings and slide assignments.
- `predictions.csv`: held-out predictions, scores, and selected patch indices.
- `source_results.csv` and `search_history.csv`: source selections, fitness values, and search progress.
- `summary.csv`: accuracy, macro-F1, and confusion-matrix counts for each method and seed.

## Original data preparation environment

The image-processing and embedding pipeline was built in the `dl` conda environment: Python 3.12, torch 2.10.0 (cu128), timm 1.0.29, tifffile 2026.9.15, fsspec 2026.6.0, aiohttp 3.14.3, imagecodecs 2026.8.16 (needed for LZW mask tiles), Pillow 12.1.1, scikit-image 0.26, and scikit-learn 1.9. Only the encoder requires GPU acceleration, using about 4 GB of a 16 GB card. The EA experiments use the separate, smaller dependency list in `requirements-ea.txt`.

The dataset as built contains 40 slides and 20,000 patches, 2.42 GB on disk, split 1.18 GB normal and 1.24 GB tumour, with mean `tumor_frac` of 0.000 and 0.172 by class. Masks, annotations, and tile-index caches add approximately 600 MB and can be deleted after the `.npz` files are built, at the cost of refetching them if rebuilding.

## Licence and citations

CAMELYON16 is CC0. Virchow2 weights are CC-BY-NC-ND-4.0 (research use only). Cite Virchow2 (Vorontsov et al., 2024) and CAMELYON16 (Litjens et al., 2018, *GigaScience*).
