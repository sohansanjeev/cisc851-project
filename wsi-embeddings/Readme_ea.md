# Genetic algorithm for patch selection

Select a small set of patches from a slide using a fixed ridge classifier, then use the selected patch vectors to choose similar patches on unseen slides.

## Data

The files in `data/embeddings/` contain 40 slides: 20 normal and 20 tumour. Each slide has 500 patches with 2,560-value embeddings and spatial coordinates. Mean and standard deviation pooling produces one 5,120-value vector per selected patch set.

The EA uses ridge classification. The original data preparation scripts are src/build.py and src/camelyon.py. The logistic-regression QC baseline is in src/qc.py.

## Method

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

## Evaluation

Each of five stratified folds uses 16 classifier-training slides, 16 source-search slides, and 8 test slides. The roles do not overlap within a fold. Each slide is tested once across the five folds.

The split seed is 851; search seeds are 851, 852, and 853. Within each k and fold, all methods and seeds share the same ridge model. Accuracy and macro-F1 use the 40 held-out predictions per method and seed. The seeds reuse the same slides and are not independent cohorts.

The references include random search, hill climbing, direct random selection of k target patches, and pooling all 500 target patches.

## Run

Run commands from the repository root. To create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r wsi-embeddings/requirements-ea.txt
```

The committed results can be plotted directly:

```bash
python wsi-embeddings/src/plot_experiments.py
```

To repeat all three experiments in fresh output folders:

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

Experiment output folders must not already exist. The saved run configurations record Python 3.14.6, NumPy 2.5.5, and scikit-learn 1.9.1. Plotting used Matplotlib 3.11.2.

## Results

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

## Code and outputs

| File in `src/` | Purpose |
| --- | --- |
| `patch_data.py` | Load slide data and pool patch features |
| `ridge_baseline.py` | Create slide splits and train the ridge classifier |
| `ga_operators.py` | Create populations, select parents, cross over, and mutate |
| `ga_search.py` | Calculate fitness and run the GA |
| `prototype_transfer.py` | Build the prototype bank and select similar target patches |
| `search_baselines.py` | Run random search and hill climbing |
| `run_experiments.py` | Evaluate all methods and save the results |
| `plot_experiments.py` | Combine summaries and create the figures |

Each full run in `results/k10_full`, `results/k20_full`, or `results/k40_full` contains:

- `config.json` and `splits.json`: settings and slide assignments.
- `predictions.csv`: held-out predictions, scores, and selected patch indices.
- `source_results.csv` and `search_history.csv`: source selections, fitness values, and search progress.
- `summary.csv`: accuracy, macro-F1, and confusion-matrix counts for each method and seed.
