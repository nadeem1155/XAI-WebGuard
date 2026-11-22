# XAI-WebGuard — Explainable Detection of SQL Injection & XSS

Reference implementation for the M.Tech thesis **"An Explainable Artificial
Intelligence Framework for the Automated Detection of SQL Injection and
Cross-Site Scripting Attacks"** (Fauzia Khatoon, Roll No. 20250501001).

It detects SQL injection (SQLi) and cross-site scripting (XSS) in HTTP request
inputs, separates them from benign traffic, and **explains every alert** by
pointing at the responsible characters of the request and linking them to the
matching CWE and CAPEC entries.

> ⚠️ **The numbers in the thesis and papers are illustrative placeholders.**
> This code is what produces the **real** numbers. Run it on the datasets below
> and replace the placeholder results before you submit.

---

## Table of contents
1. [What you need](#1-what-you-need)
2. [Set up in VS Code (recommended)](#2-set-up-in-vs-code-recommended)
3. [Set up from the terminal](#3-set-up-from-the-terminal)
4. [Step 0 — verify it runs (no datasets)](#4-step-0--verify-it-runs-no-datasets)
5. [Step 1 — get and register the datasets](#5-step-1--get-and-register-the-datasets)
6. [Step 2 — train the models](#6-step-2--train-the-models)
7. [Step 3 — evaluate (produce the real tables)](#7-step-3--evaluate-produce-the-real-tables)
8. [Step 4 — run the prototype service](#8-step-4--run-the-prototype-service)
9. [Where each result goes in the thesis](#9-where-each-result-goes-in-the-thesis)
10. [Troubleshooting](#10-troubleshooting)
11. [Project layout](#11-project-layout)

---

## 1. What you need

- **Python 3.9–3.12**
- A few hundred MB of disk for the datasets
- **For Model A (tree), metrics and the API:** CPU only — the core requirements
- **For Model B (transformer):** `torch` + `transformers` and, realistically, a
  GPU (NVIDIA T4 class or better). Everything except Model B works without a GPU.

Check your Python:

```bash
python --version        # should print 3.9 or newer
```

---

## 2. Set up in VS Code (recommended)

The project ships with a ready-made `.vscode/` folder, so VS Code works out of
the box.

1. **Open the folder.** `File → Open Folder…` and pick the unzipped
   `xai-webguard` folder (open this folder itself, not its parent).
2. **Install the Python extension** if VS Code prompts you — the project already
   recommends it (`.vscode/extensions.json`). Click *Install* on the
   notification, or install **Python** by Microsoft from the Extensions panel.
3. **Create and select an interpreter.**
   - `Ctrl/Cmd + Shift + P` → **Python: Create Environment** → **Venv** → pick
     your Python 3 → tick `requirements.txt` to install into it. VS Code creates
     `.venv/` and installs everything.
   - If you already have an environment, use **Python: Select Interpreter**
     instead and then run the install task below.
4. **Install dependencies** (if not done in step 3):
   `Ctrl/Cmd + Shift + P` → **Tasks: Run Task** → **Install dependencies**.
5. **Run things** — two easy ways:
   - **Run & Debug panel** (the ▷ bug icon on the left): pick a configuration
     from the dropdown and press *F5*. They are numbered in the order you'll use
     them:
     1. Smoke test (no datasets) · 2. Make synthetic data · 3. Train Model A ·
     4. Train Model B · 5. Evaluate detection · 6. Evaluate explanations ·
     7. Serve API.
   - **Tasks** (`Terminal → Run Task…`): *Make synthetic data*, *Run smoke test*,
     *Run tests*.
6. **Run the unit tests visually:** open the **Testing** panel (flask/beaker
   icon). The 12 tests appear automatically — press ▷ to run them.

> The `.env` file sets `PYTHONPATH=src`, so `import xai_webguard` resolves in the
> editor, the debugger and the Test Explorer without any extra setup.

**Fastest VS Code path to confirm everything works:** select interpreter →
Run Task *Install dependencies* → press *F5* on **"1. Smoke test"**. You should
see `SMOKE TEST PASSED`.

---

## 3. Set up from the terminal

```bash
cd xai-webguard

# (recommended) a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To use Model B (the transformer) as well:

```bash
python -m pip install torch transformers
```

Make the package importable for the scripts (the scripts also do this
themselves, but setting it once is cleaner):

```bash
export PYTHONPATH=src                 # Windows PowerShell: $env:PYTHONPATH="src"
```

---

## 4. Step 0 — verify it runs (no datasets)

Before touching real data, confirm the whole pipeline works on a tiny synthetic
corpus. **This is a sanity check, not a result.**

```bash
python scripts/make_synthetic_data.py      # writes data/synthetic.csv
python scripts/smoke_test.py               # trains + classifies + explains
python -m pytest -q                        # 12 unit tests
```

Expected tail of the smoke test:

```
[inspect] 'id=5 OR 1=1--'              -> sqli    conf=1.00 action=block cwe=CWE-89 ...
[inspect] 'q=<img src=x onerror=...>'  -> xss     conf=1.00 action=block cwe=CWE-79 ...
[inspect] 'search=blue shoes'          -> benign  conf=1.00 action=allow ...
SMOKE TEST PASSED — pipeline runs end to end.
```

(The synthetic data is trivially separable, so the score is ~1.0 — that is
expected and means only that the plumbing works.)

---

## 5. Step 1 — get and register the datasets

Nothing is downloaded for you. Obtain each dataset, export it to **CSV**, and put
the files in `data/`:

| Dataset | Source / DOI | Suggested filename |
|---|---|---|
| SQL Injection Attack Dataset v2 | Mendeley `10.17632/mmc4sdmnrc.2` | `data/sql_injection_v2.csv` |
| Large-scale annotated XSS | IEEE DataPort `10.21227/5w2z-3v64` | `data/xss_large.csv` |
| HTTP CSIC 2010 | CSIC (export to CSV) | `data/csic2010.csv` |
| SR-BH 2020 multi-label web attacks | `10.1016/j.cose.2022.102788` | `data/sr_bh_2020.csv` |
| Obfuscated & LLM-augmented XSS | IEEE DataPort `10.21227/qmtc-by33` | `data/xss_obfuscated.csv` |

Then open **`configs/default.yaml`** and, for each source, set:
- `path` — where your CSV is,
- `text_col` — the column holding the request/payload text,
- `label_col` — the column holding the label,
- `label_map` — how that column's values map to `benign` / `sqli` / `xss`.

Labels are harmonised to three classes: `0 = benign, 1 = sqli, 2 = xss`.

**Token-level annotations (for plausibility).** Create
`data/annotations.csv` with columns `text,label,span_start,span_end`, giving the
minimal attack span (character offsets) for ~1,500 malicious samples. Two
annotators should label independently; report Cohen's kappa and adjudicate
disagreements (Section 4.2 of the thesis).

---

## 6. Step 2 — train the models

**Model A — gradient-boosted trees (CPU, fast):**

```bash
python scripts/train_tree.py --config configs/default.yaml --out results/model_a.pkl
```

**Model B — CodeBERT transformer (needs torch + GPU):**

```bash
python scripts/train_transformer.py --config configs/default.yaml --out results/model_b
```

If you have no GPU, skip Model B — the framework runs on Model A alone, and every
evaluation script works without the transformer (it simply reports Model A's
numbers). To use a lighter transformer, set
`transformer.model_name: distilbert-base-uncased` in the config.

---

## 7. Step 3 — evaluate (produce the real tables)

**Detection (Tables 5.1 and 5.2):**

```bash
# Model A only:
python scripts/run_detection_eval.py --tree results/model_a.pkl

# With the transformer and fusion:
python scripts/run_detection_eval.py --tree results/model_a.pkl --transformer results/model_b
```

Writes `results/detection_metrics.json` with per-class precision/recall/F1,
macro-F1, MCC, PR-AUC, ROC-AUC, FPR@99% recall and ECE, for both the
in-distribution test split and the cross-dataset set.

**Explanations (Table 5.3):**

```bash
python scripts/run_explanation_eval.py --tree results/model_a.pkl \
       --transformer results/model_b --annotations data/annotations.csv
```

Writes `results/explanation_metrics.json` with plausibility (token-F1) for
TreeSHAP and Integrated Gradients.

> For the full thesis protocol, repeat training/evaluation over the five seeds
> listed in the config and report the mean with a bootstrap 95% confidence
> interval (Section 4.7).

---

## 8. Step 4 — run the prototype service

```bash
# terminal:
XWG_TREE=results/model_a.pkl uvicorn xai_webguard.serve.app:app --port 8000
# (in VS Code: Run & Debug → "7. Serve API")
```

Then query it:

```bash
curl -s localhost:8000/inspect -H 'content-type: application/json' \
     -d '{"input":"id=1 OR 1=1--","field":"query"}' | python -m json.tool
```

You get back the verdict, confidence, action, the evidence spans and the
CWE/CAPEC mapping. Counterfactuals are **never** returned to a client — they
belong only in an authenticated analyst view.

---

## 9. Where each result goes in the thesis

| Script / output | Fills |
|---|---|
| `run_detection_eval.py` → in_distribution | Table 5.1, Fig. 5.1–5.3 |
| `run_detection_eval.py` → cross_dataset | Table 5.2 |
| `run_explanation_eval.py` | Table 5.3, Fig. 5.6 |
| obfuscation run (point config at the obfuscated set) | Fig. 5.8 |
| `serve/app.py` latency logs | §5.4 cost column |

Replace every illustrative number (and the grey "Note on results" boxes) with
the measured output.

---

## 10. Troubleshooting

- **`ModuleNotFoundError: xai_webguard`** — set `PYTHONPATH=src` (terminal) or
  make sure VS Code picked up `.env` (reopen the folder, reselect the
  interpreter).
- **`No module named pytest`** — `pip install -r requirements.txt` (pytest is in
  it), or `pip install pytest`.
- **`xgboost` wheel fails on Windows** — install the Microsoft Visual C++
  redistributable, or `conda install -c conda-forge xgboost`.
- **Model B errors about torch/transformers** — install them
  (`pip install torch transformers`); they are intentionally optional so Model A
  works on its own.
- **CUDA out of memory** — lower `transformer.batch_size` or `max_len`, or switch
  to `distilbert-base-uncased`.
- **A dataset CSV won't parse** — check `sep`/`encoding` in the source's config
  entry; the loader skips bad lines but needs the right delimiter.

---

## 11. Project layout

```
xai-webguard/
├── .vscode/                 VS Code run/debug/test config (ready to use)
│   ├── settings.json        interpreter, tests, src on the path
│   ├── launch.json          7 numbered run/debug configs
│   ├── tasks.json           install / synthetic data / smoke / tests
│   └── extensions.json      recommended extensions
├── .env                     PYTHONPATH=src (so imports resolve in VS Code)
├── configs/default.yaml     dataset paths, label maps, hyper-parameters
├── src/xai_webguard/        the package
│   ├── canonicalize.py      de-obfuscation + offset map (§3.3)
│   ├── features.py          engineered features (Table 3.1)
│   ├── data.py              load, de-dup, group-aware split
│   ├── pipeline.py          end-to-end inspect() + report
│   ├── models/              tree_model, transformer_model, fusion
│   ├── explain/             tree_shap, integrated_gradients, semantic_mapper
│   ├── evaluate/            detection_metrics, explanation_metrics
│   └── serve/app.py         FastAPI prototype
├── scripts/                 train / evaluate / serve / synthetic / smoke
├── tests/                   pytest unit tests (12)
├── data/                    put datasets here (empty)
├── results/                 models + metric JSON land here (empty)
├── requirements.txt         core deps (+ commented optional ones)
├── pyproject.toml           package + pytest config
├── Makefile                 make install / test / smoke / serve
└── README.md                this file
```

## Ethics

All attack traffic must stay in an isolated lab with intentionally vulnerable
apps (DVWA, OWASP Juice Shop). Do not scan or test third-party systems. The code
contains no evasion payloads.
