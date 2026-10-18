"""
Scientific audit of the FBG-Sensor-PINN repository (phase 1: verification only).

Reads the ORIGINAL raw CSVs (read-only) and reproduces the notebook's PINN set-up
in PyTorch (the notebook used TensorFlow 2.14; architecture, inputs, target,
split, optimiser and epoch count are copied exactly) in order to test:

  E1  the original pipeline as written            -> reproduces the reported ~1 pm error
  E2  a zero-parameter identity "model"           -> shows the target is in the input
  E3  Time-only input, random split               -> temporal-interpolation leakage
  E4  Time-only input, chronological split        -> what generalisation actually looks like
  E5  original set-up, 5 random seeds             -> strain/temperature outputs are not identifiable
  E6  falsification on the single-factor runs     -> predicted T must stay ~constant in the
                                                     strain run and eps ~constant in the temp run
  E7  null-space construction                     -> infinitely many (eps, T) give zero loss
  E8  the notebook's "physics loss" term          -> identically zero by construction

No result in this file is a performance claim for a new model.

Usage:  python -I audit/run_audit.py <repo_root>
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "audit"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

K_E = 1.2    # pm / µε  (value hard-coded in the notebook, cell 13)
K_T = 10.0   # pm / °C  (value hard-coded in the notebook, cell 13)
EPOCHS = 501
torch.set_num_threads(4)

results = {}


def load(name):
    d = pd.read_csv(ROOT / name)
    d.columns = ["Time", "CH1", "CH2", "CH3", "CH4", "Wavelength"]
    d["dl_pm"] = (d["Wavelength"] - d["Wavelength"].iloc[0]) * 1000.0  # identical to notebook cell 4
    return d


strain = load("STRAIN_CSV.csv")
temp = load("TEMP_CSV.csv")
comb = load("TEMP_STRAIN_CSV.csv")
runs = {"strain-only (D1)": strain, "temperature-only (D2)": temp, "combined (D3)": comb}

# ---------------------------------------------------------------- dataset facts
facts = {}
for k, d in runs.items():
    q = np.diff(np.sort(d["Wavelength"].unique())) * 1000
    facts[k] = dict(
        n=len(d), duration_s=round(float(d.Time.iloc[-1]), 1),
        fs_Hz=round(1 / float(np.diff(d.Time).mean()), 3),
        lambda0_nm=float(d.Wavelength.iloc[0]),
        lambda_min_nm=float(d.Wavelength.min()), lambda_max_nm=float(d.Wavelength.max()),
        dl_range_pm=round(float(d.dl_pm.max() - d.dl_pm.min()), 2),
        dl_final_pm=round(float(d.dl_pm.iloc[-50:].mean()), 2),
        active_channels=[c for c in ["CH1", "CH2", "CH3", "CH4"] if d[c].any()],
        noise_pm=round(float(1.4826 * np.median(np.abs(np.diff(d.dl_pm) - np.median(np.diff(d.dl_pm)))) / np.sqrt(2)), 2),
        smallest_wl_increment_pm=round(float(q[q > 0.005].min()), 3),
    )
results["dataset_facts"] = facts


def detect_steps(d, win=50, thr=40.0, min_sep=200):
    """Very simple step detector: difference of means of the next/previous `win` samples."""
    y = d.dl_pm.to_numpy()
    ym = pd.Series(y).rolling(11, center=True, min_periods=1).median().to_numpy()  # suppress load transients
    s = np.full(len(y), 0.0)
    for i in range(win, len(y) - win):
        s[i] = np.median(ym[i:i + win]) - np.median(ym[i - win:i])
    idx = []
    for i in np.argsort(-np.abs(s)):
        if abs(s[i]) < thr:
            break
        if all(abs(i - j) > min_sep for j in idx):
            idx.append(i)
    idx.sort()
    edges = [0] + idx + [len(y)]
    plateaus = []
    for a, b in zip(edges[:-1], edges[1:]):
        a2, b2 = a + min(40, (b - a) // 4), b - min(40, (b - a) // 4)
        seg = y[a2:b2]
        plateaus.append(dict(t_start=round(float(d.Time.iloc[a]), 1), t_end=round(float(d.Time.iloc[b - 1]), 1),
                             mean_pm=round(float(np.median(seg)), 1), std_pm=round(float(seg.std()), 2),
                             slope_pm_per_100s=round(float(np.polyfit(d.Time.iloc[a2:b2], seg, 1)[0] * 100), 2) if len(seg) > 10 else None))
    steps = [round(plateaus[i + 1]["mean_pm"] - plateaus[i]["mean_pm"], 1) for i in range(len(plateaus) - 1)]
    return plateaus, steps


steps_info = {}
for k in ["strain-only (D1)", "combined (D3)"]:
    p, s = detect_steps(runs[k])
    steps_info[k] = dict(plateaus=p, step_sizes_pm=s,
                         implied_step_strain_microstrain_if_KE_1p2=[round(x / K_E, 0) for x in s])
# temperature run: monotone drift -> implied temperature change with nominal K_T
td = temp.dl_pm.rolling(51, center=True).median()
steps_info["temperature-only (D2)"] = dict(
    start_pm=round(float(td.iloc[30:80].mean()), 1), end_pm=round(float(td.iloc[-80:-30].mean()), 1),
    implied_dT_C_if_KT_10=round(float((td.iloc[-80:-30].mean() - td.iloc[30:80].mean()) / K_T), 2),
    implied_dT_C_if_KT_23_bonded_on_steel=round(float((td.iloc[-80:-30].mean() - td.iloc[30:80].mean()) / 23.0), 2),
)
# transients / outliers: samples deviating > 20 pm from an 11-sample running median
for k, d in runs.items():
    med = d.dl_pm.rolling(11, center=True, min_periods=1).median()
    bad = np.abs(d.dl_pm - med) > 20
    t_bad = d.Time[bad].to_numpy()
    clusters = []
    for tb in t_bad:
        if clusters and tb - clusters[-1][1] < 5: clusters[-1][1] = tb
        else: clusters.append([tb, tb])
    steps_info.setdefault(k, {})["transient_samples_gt20pm"] = int(bad.sum())
    steps_info[k]["transient_windows_s"] = [[round(a, 1), round(b, 1)] for a, b in clusters]
    steps_info[k]["max_transient_excursion_pm"] = round(float(np.abs(d.dl_pm - med).max()), 1)
results["structure"] = steps_info

# figure: raw data
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
for a, (k, d) in zip(ax, runs.items()):
    a.plot(d.Time, d.dl_pm, lw=0.7, color="k")
    a.set_title(f"{k}: Bragg-wavelength shift relative to first sample (λ0 = {d.Wavelength.iloc[0]:.5f} nm)")
    a.set_ylabel("Δλ (pm)"); a.grid(alpha=0.3)
ax[-1].set_xlabel("Time (s)")
fig.tight_layout(); fig.savefig(FIG / "01_raw_datasets.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- the notebook model, in torch
class NotebookPINN(torch.nn.Module):
    """Dense(64,relu)-Dense(64,relu)-Dense(32,relu) -> two linear heads (strain, temp). Notebook cell 15/16."""
    def __init__(self, n_in):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(n_in, 64), torch.nn.ReLU(),
                                        torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                        torch.nn.Linear(64, 32), torch.nn.ReLU())
        self.eps = torch.nn.Linear(32, 1)
        self.T = torch.nn.Linear(32, 1)
        for m in self.modules():  # Keras defaults: glorot_uniform weights, zero bias
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight); torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.body(x)
        return self.eps(h), self.T(h)


def train(X, y, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    m = NotebookPINN(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    for _ in range(epochs):  # full-batch, exactly as notebook cell 19
        opt.zero_grad()
        e, T = m(Xt)
        loss = torch.mean((yt - (K_E * e + K_T * T)) ** 2)
        loss.backward(); opt.step()
    return m


def predict(m, X):
    with torch.no_grad():
        e, T = m(torch.tensor(X, dtype=torch.float32))
    e, T = e.numpy().ravel(), T.numpy().ravel()
    return e, T, K_E * e + K_T * T


def metrics(y, p):
    r = y - p
    return dict(MAE_pm=round(float(np.mean(np.abs(r))), 3), RMSE_pm=round(float(np.sqrt(np.mean(r ** 2))), 3),
                R2=round(float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)), 4))


t = comb.Time.to_numpy(); y = comb.dl_pm.to_numpy()
rng = np.random.default_rng(42)
perm = rng.permutation(len(y)); n_te = int(round(0.2 * len(y)))
te_rand, tr_rand = perm[:n_te], perm[n_te:]
cut = int(0.8 * len(y)); tr_chrono, te_chrono = np.arange(cut), np.arange(cut, len(y))

X_orig = np.column_stack([t, y])   # notebook cell 14: X = [Time, delta_lambda_pm]   <-- target inside input
X_time = t.reshape(-1, 1)

exp = {}
m1 = train(X_orig[tr_rand], y[tr_rand], seed=0)
exp["E1_original_inputs_[Time,dl]_random_split"] = metrics(y[te_rand], predict(m1, X_orig[te_rand])[2])
exp["E2_identity_y_hat_equals_input_dl_(no_model)"] = metrics(y[te_rand], X_orig[te_rand, 1])
m3 = train(X_time[tr_rand], y[tr_rand], seed=0, epochs=3000)
exp["E3_Time_only_random_split"] = metrics(y[te_rand], predict(m3, X_time[te_rand])[2])
m4 = train(X_time[tr_chrono], y[tr_chrono], seed=0, epochs=3000)
exp["E4_Time_only_chronological_split_(last_20pct)"] = metrics(y[te_chrono], predict(m4, X_time[te_chrono])[2])
exp["E4b_naive_hold_last_training_value"] = metrics(y[te_chrono], np.full(len(te_chrono), y[tr_chrono][-1]))
m1c = train(X_orig[tr_chrono], y[tr_chrono], seed=0)
exp["E1c_original_inputs_chronological_split"] = metrics(y[te_chrono], predict(m1c, X_orig[te_chrono])[2])
# 1-nearest-neighbour in time: no model, just "copy the closest training sample in time"
from sklearn.neighbors import KNeighborsRegressor
for nm, tr, te in [("random", tr_rand, te_rand), ("chronological", tr_chrono, te_chrono)]:
    knn = KNeighborsRegressor(n_neighbors=1).fit(X_time[tr], y[tr])
    exp[f"E3b_1NN_in_time_{nm}_split"] = metrics(y[te], knn.predict(X_time[te]))
results["leakage_experiments"] = exp

# E5: non-identifiability across seeds (original set-up)
seed_rows, seed_preds = [], {}
for s in range(5):
    ms = train(X_orig[tr_rand], y[tr_rand], seed=s)
    e, T, d = predict(ms, X_orig)
    seed_preds[s] = (e, T)
    seed_rows.append(dict(seed=s, **metrics(y[te_rand], d[te_rand]),
                          strain_range_microstrain=[round(float(e.min()), 1), round(float(e.max()), 1)],
                          temp_range_C=[round(float(T.min()), 1), round(float(T.max()), 1)],
                          strain_share_of_dl_variance=round(float(np.var(K_E * e) / np.var(y)), 3),
                          temp_share_of_dl_variance=round(float(np.var(K_T * T) / np.var(y)), 3)))
results["E5_seed_dependence"] = seed_rows

fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
ax[0].plot(t, y, "k", lw=0.8); ax[0].set_ylabel("Δλ (pm)"); ax[0].set_title("Combined run: measured Δλ (all seeds fit this to ~1 pm)")
for s, (e, T) in seed_preds.items():
    ax[1].plot(t, e, lw=0.8, label=f"seed {s}"); ax[2].plot(t, T, lw=0.8, label=f"seed {s}")
ax[1].set_ylabel("predicted ε (µε)"); ax[1].set_title("Strain output — differs by seed")
ax[2].set_ylabel("predicted T (°C)"); ax[2].set_title("Temperature output — differs by seed")
ax[2].set_xlabel("Time (s)")
for a in ax: a.grid(alpha=0.3)
ax[1].legend(ncol=5, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "02_seed_nonidentifiability.png", dpi=130); plt.close(fig)

# E6: falsification on the single-factor runs (model trained on combined run, original inputs)
fals = {}
for k, d in [("strain-only (D1)", strain), ("temperature-only (D2)", temp)]:
    Xd = np.column_stack([d.Time.to_numpy(), d.dl_pm.to_numpy()])
    e, T, dd = predict(m1, Xd)
    fals[k] = dict(dl_fit=metrics(d.dl_pm.to_numpy(), dd),
                   predicted_strain_range_microstrain=round(float(e.max() - e.min()), 1),
                   predicted_temp_range_C=round(float(T.max() - T.min()), 2),
                   expected=("temperature ~constant (ΔT≈0)" if "strain" in k else "strain ~constant (Δε≈0)"))
results["E6_single_factor_falsification"] = fals

fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
for j, (k, d) in enumerate([("strain-only (D1)", strain), ("temperature-only (D2)", temp)]):
    e, T, _ = predict(m1, np.column_stack([d.Time.to_numpy(), d.dl_pm.to_numpy()]))
    ax[0, j].plot(d.Time, e, "g", lw=0.8); ax[0, j].set_title(f"{k}: PINN strain output"); ax[0, j].set_ylabel("µε")
    ax[1, j].plot(d.Time, T, "r", lw=0.8); ax[1, j].set_title(f"{k}: PINN temperature output"); ax[1, j].set_ylabel("°C")
    ax[1, j].set_xlabel("Time (s)")
for a in ax.ravel(): a.grid(alpha=0.3)
fig.suptitle("Trained on combined run (original set-up). Physically: left-bottom should be flat, right-top should be flat.")
fig.tight_layout(); fig.savefig(FIG / "03_single_factor_falsification.png", dpi=130); plt.close(fig)

# E7: explicit null-space construction — three decompositions with identical, zero data/physics loss
dec = {}
for name, Tfun in {"all_strain (T≡0)": np.zeros_like(y),
                   "all_temperature (ε≡0)": y / K_T,
                   "arbitrary T = 5·sin(2πt/600) °C": 5 * np.sin(2 * np.pi * t / 600)}.items():
    e = (y - K_T * Tfun) / K_E
    dec[name] = dict(physics_residual_max_pm=float(np.max(np.abs(y - (K_E * e + K_T * Tfun)))),
                     strain_range_microstrain=[round(float(e.min()), 1), round(float(e.max()), 1)],
                     temp_range_C=[round(float(Tfun.min()), 2), round(float(Tfun.max()), 2)])
results["E7_null_space_examples"] = dec

# E8: the ablation "physics loss" of notebook cell 24
eps_, T_ = seed_preds[0]
delta_pred = K_E * eps_ + K_T * T_
results["E8_notebook_ablation_phys_loss_value"] = float(np.mean((delta_pred - (K_E * eps_ + K_T * T_)) ** 2))

# Classical baseline of cell 7: units check
results["classical_baseline_cell7_units"] = dict(
    note="regresses Wavelength (nm) on Time; coefficient is nm/s but labelled pm/s; MAE/RMSE are in nm but labelled pm",
    reported_MAE='0.38 "pm"', actual_MAE_in_pm=380.0)

(OUT / "audit_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(results, indent=2, ensure_ascii=False))
