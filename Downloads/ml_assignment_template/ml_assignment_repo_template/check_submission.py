"""Pre-submission check. Run from the repo root: python check_submission.py"""
import os, subprocess, json
import numpy as np

ok = True
def need(path):
    global ok
    e = os.path.exists(path); ok &= e; print(("OK      " if e else "MISSING ") + path)

for p in ["README.md", "requirements.txt", "configs/config.json", "make_figures.py", "split/split_indices.npz",
          "q1_trees_bagging.ipynb", "q2_boosting.ipynb", "q3_bias_variance.ipynb",
          "logs/q1_final_log.json", "logs/q2_final_log.json", "logs/q3_final_log.json"]:
    need(p)

if os.path.exists("split/split_indices.npz"):
    s = np.load("split/split_indices.npz"); sizes = (len(s["train"]), len(s["val"]), len(s["test"]))
    good = sizes == (240, 80, 80) and len(set(s["train"]) | set(s["val"]) | set(s["test"])) == 400
    ok &= good; print("split sizes", sizes, "-> OK" if good else "-> WRONG")
    if "seed" in s.files and os.path.exists("configs/config.json"):
        same = int(s["seed"]) == json.load(open("configs/config.json"))["seed"]; ok &= same; print("split seed matches config:", same)

try:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print("git HEAD:", head); print("working tree clean:", dirty == "")
    if dirty: print("UNCOMMITTED CHANGES:\n" + dirty); ok = False
except Exception as e:
    print("git not available / not a repo:", e); ok = False
print("\nALL CHECKS PASSED" if ok else "\nFIX THE ITEMS ABOVE BEFORE SUBMITTING")
