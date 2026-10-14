# SPEC — first-three-seconds

This file holds the project's decisions. Agents read it before every task.
**The rules in this file are not changed without explicitly asking and getting approval.**

---

## 1. Goal

From a single station's 3-component recording, estimate an earthquake's magnitude (M) and its uncertainty (σ) using only the first T seconds after the P-wave arrival.

- This is an **early warning / rapid magnitude estimation** project, not earthquake *prediction*.
- Main questions: How does the error change with window length (T) and magnitude? How well does a model trained on global data work on Turkish data?

**Out of scope:** location estimation, multi-station models, P-arrival detection (existing arrivals are used), real-time streaming.

## 2. Data

| Role | Source | Notes |
| --- | --- | --- |
| Train / validation / test | STEAD (via SeisBench) | Raw counts, 100 Hz, 3 components |
| Separate test (transfer) | AFAD (TU and TK networks, FDSN) | Test and fine-tuning only |

- Noise traces are excluded (`trace_category == "earthquake_local"`).
- Traces with missing magnitude are excluded.
- Magnitude range: **M ≥ 2.0**.
- Magnitude type (`source_magnitude_type`) is kept for every trace and reported in the analysis.
- Metadata column names may vary between SeisBench versions; check the actual names before using them.
- Raw data and processed arrays are **never committed** (`data/` is in `.gitignore`).

## 3. Preprocessing

- Sampling rate: **100 Hz** (resample if different).
- Component order: **Z, N, E** (or Z, 1, 2). Same order everywhere.
- Window: **[P − 0.5 s, P + T]**, T ∈ {1, 2, 3, 5}. Main experiments use T = 3.
- Steps: demean → detrend → taper (5% at the edges).
- If filtering is needed, **causal (one-pass) filters only**. Zero-phase filters are forbidden: they leak information before the P arrival.
- Normalization: each window is divided by its maximum absolute value; **log10(peak amplitude)** is stored as a separate scalar input.
- The P-arrival index within the window is saved in the output file (for sanity-check plots).

## 4. Data splitting (critical)

- Split **by event ID** (`source_id`), not by trace.
- Ratios: train 70%, validation 15%, test 15%.
- Fixed seed: **42**.
- The split is done once; event ID lists are saved as files under `splits/` and every experiment uses them.
- Check after every split: the number of shared `source_id`s between sets must be **0**. If not, stop and report.
- AFAD data never enters the STEAD training set.

## 5. Baselines

No result is reported without comparing the DL model against these.

1. **Constant prediction:** mean magnitude of the training set.
2. **Pd regression:** log10(Pd) = a + b·M + c·log10(R), fit on the training set and solved for M.
3. **τc regression:** log10(τc) = a·M + b.

Pd and τc are computed as defined in the pre-study report (0.075 Hz causal high-pass, vertical component, first T seconds).

## 6. Model

- Input: (3, 100·(T + 0.5)) array + log10(peak amplitude) scalar.
- Architecture (v1): 4–5 × [Conv1D(k=7) → BatchNorm → ReLU → MaxPool(2)] → global average pooling → concatenate peak amplitude → Linear → (μ, log σ).
- Loss: `nn.GaussianNLLLoss`.
- Augmentation (training only): shift the P arrival by ±0.2 s, add noise.
- Device: picked automatically in this order: `cuda` → `mps` → `cpu` (see section 10).
- Architecture changes are allowed, but every experiment uses a new config file.

## 7. Evaluation

Every experiment reports:

| Metric | How |
| --- | --- |
| MAE, RMSE | Whole test set |
| Bias | Per magnitude bin: 2–3, 3–4, 4–5, 5–6, 6+ |
| ±1σ coverage | Share of true values within μ ± σ (target ~68%) |
| Window effect | Same metrics for T = 1, 2, 3, 5 |

Every table also shows the sample count per bin. Bins with few samples (< 50) are flagged.

## 8. Required checks

After every stage, the agent runs these and reports the results:

- [ ] Plot of 5 random windows (P arrival marked)
- [ ] Shared events between splits = 0
- [ ] No NaN / Inf in input arrays
- [ ] Label distribution (histogram) for train and test

## 9. Repo structure

```
first-three-seconds/
├── SPEC.md
├── TASKS.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitattributes
├── configs/         # one yaml per experiment
├── data/            # not tracked by git
├── splits/          # event ID lists (tracked by git)
├── src/
│   ├── data.py      # loading, window cutting
│   ├── features.py  # Pd, τc, τp_max
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py     # get_device(), paths, seeding
│   └── smoke_test.py
├── notebooks/       # exploration and plots only
└── results/         # metric tables and figures
```

- Core code lives in `src/`. Notebooks are for exploration and visualization only.
- Dependencies pinned in `requirements.txt`.

## 10. Cross-platform requirements

The project must run unchanged on **macOS, Linux, Windows and Google Colab**.

**Python and packages**
- Developed on Python 3.12; code must also work on **3.10+** (Colab's version can differ).
- `requirements.txt` holds no platform-specific packages or wheel URLs. PyTorch is installed per platform as described in the README (CUDA, MPS or CPU build).
- The project is installable with `pip install -e .` (a minimal `pyproject.toml`), so `src` imports work the same everywhere.

**Paths and files**
- All paths use `pathlib.Path`. No hardcoded absolute paths, no `/` or `\` string joins.
- The data directory comes from config or the `F3S_DATA_DIR` environment variable, defaulting to `./data`. On Colab it can point to Google Drive.
- Files are opened with explicit `encoding="utf-8"`.
- File and folder names: lowercase, no spaces, no non-ASCII characters (case-sensitivity differs between systems).
- `.gitattributes` enforces LF line endings for text files.

**Hardware**
- One helper (`src/utils.py: get_device()`) picks `cuda` → `mps` → `cpu`. No other code hardcodes a device.
- Code must not rely on CUDA-only features. If an op is unsupported on `mps`, fall back to `cpu` for that op instead of failing.

**Processes and scripts**
- Every runnable script has an `if __name__ == "__main__":` guard (Windows and macOS start DataLoader workers with `spawn`).
- `num_workers` is configurable; default `0` on Windows.
- No bash-only scripts. Entry points are Python (`python -m src.train --config configs/x.yaml`).
- No reliance on system tools that are not installed by pip.

**Colab**
- `notebooks/colab_setup.ipynb`: clone the repo, `pip install -e .`, optionally mount Drive and set `F3S_DATA_DIR`, run a smoke test.
- Long jobs save checkpoints regularly so a disconnected session can resume.

**Verification**
- A small smoke test (`python -m src.smoke_test`) runs the full pipeline on ~100 traces in under a minute on CPU.
- Optional: a GitHub Actions matrix (ubuntu, macos, windows) runs the smoke test on every push.

## 11. Agent rules

- **Do not change** the decisions in sections 3, 4 and 7 without asking.
- Every change must keep the section 10 cross-platform rules; run the smoke test before reporting a task as done.
- If a result looks suspiciously good (e.g. MAE < 0.15), suspect data leakage first and check.
- Never invent metadata, column names or data sources; ask when unsure.
- At the end of every task: what you did, which files changed, and the results of the checks.