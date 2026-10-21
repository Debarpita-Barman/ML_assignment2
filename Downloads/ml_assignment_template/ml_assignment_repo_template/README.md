# Machine Learning Assignment 2 - Decision trees, bagging, boosting (Olivetti faces)

**Name:** Debarpita Barman  **Roll no:** P25AI0202  **Seed:** 25202 (`int("0202")`)
**Final commit hash:** see page 1 of the report (`git rev-parse HEAD`)

## Contents
| path | purpose |
|---|---|
| `q1_trees_bagging.ipynb` | Q1: single tree grid, my own bagging, OOB / disagreement / margin, clean vs masked |
| `q2_boosting.ipynb` | Q2: my own SAMME boosting, toy round, diagnostics, boosting vs bagging, clean vs masked |
| `q3_bias_variance.ipynb` | Q3: 30 resampled refits, bias^2 and variance, depth sweep |
| `split/split_indices.npz`, `.json` | the saved 6/2/2 per-identity split used everywhere |
| `split/q3_resample_indices.npy` | the 30 saved Q3 resampled training sets |
| `configs/config.json` | seed, grids and settings |
| `logs/` | final CSV / JSON logs (all report numbers come from here) |
| `figures/`, `make_figures.py` | report figures and the script that regenerates them from `logs/` |
| `check_submission.py` | pre-submission checks |
| `report/` | CVPR-format LaTeX source and PDF |

## Reproduction (exact commands)
Python version: FILL IN (e.g. 3.11). The first run downloads Olivetti (internet needed once).
```
git clone REPO_URL
cd REPO_FOLDER
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace q1_trees_bagging.ipynb
jupyter nbconvert --to notebook --execute --inplace q2_boosting.ipynb
jupyter nbconvert --to notebook --execute --inplace q3_bias_variance.ipynb
python make_figures.py
python check_submission.py
```
Run the notebooks in this order: Q2 and Q3 read the choices frozen by Q1 (and Q2) from `logs/`.
The split is created once by Q1 and then re-loaded, never re-drawn. Q3 takes a few minutes on a multi-core machine.

## Rules followed
Raw flattened pixels only, no PCA or other dimensionality reduction. Bagging and SAMME boosting are implemented in the notebooks; sklearn is used only for `DecisionTreeClassifier`. Models are trained on clean images only; the masked test set (rows/cols 24-39 set to 0) is used for evaluation only. Test data is never used for selection.
