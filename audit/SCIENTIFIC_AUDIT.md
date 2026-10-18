# FBG-Sensor-PINN: Scientific Audit (Phase 1: Verification and Dataset Understanding)

**Scope.** This report checks whether the original work is scientifically valid. It covers the physics, the data, the code and whether the estimation problem is well-posed. It deliberately produces **no new performance claims** and **no improved model**. Every number below can be regenerated with:

```bash
python -I audit/run_audit.py .
```

That command writes `audit/audit_results.json` and `audit/figures/*.png`. The original files are untouched. See `CHANGES.md`.

**Upstream audited:** `anvitha234/FBG-Sensor-PINN` @ `aa7693a` ("Initial commit"). Main artefact: `Internship_PINN.ipynb` (38 cells, TensorFlow 2.14, executed outputs stored).

---

## 0. Executive summary

| # | Finding | Severity | Evidence |
|---|---------|----------|----------|
| F1 | **There is no ground truth for strain or temperature anywhere in the repository.** The CSVs contain only time, a channel flag and one Bragg wavelength. | Critical | §3 |
| F2 | **Target leakage.** The PINN input is `X = [Time, Δλ]` and the target is `y = Δλ`. A zero-parameter identity map scores MAE = 0 pm and R² = 1. The reported R² = 1.000 and MAE ≈ 1 pm only measure how well an MLP can copy one of its inputs. | Critical | §4.1, E1/E2 |
| F3 | **Strain and temperature cannot be identified from a single FBG.** One equation, two unknowns. The loss constrains only `1.2·ε + 10·T`. The ε and T outputs are arbitrary: they change completely between random seeds and violate the experimental design (the model "finds" a ~35 °C swing in the constant-temperature run). | Critical | §5, E5/E6/E7 |
| F4 | **The physics loss is identically zero.** In the λ-ablation (cell 24) `phys_loss = mean((Δλ_pred − (kε·ε + kT·T))²)` with `Δλ_pred ≡ kε·ε + kT·T`. The "best λ = 1.0" result is seed noise. The rest of the PINN is ordinary curve fitting through a fixed linear output layer. | Critical | §4.3, E8 |
| F5 | **Invalid evaluation protocol.** A random shuffle split on a 5 Hz autocorrelated time series means every test sample has training neighbours 0.2 s away. A "copy the nearest-in-time training sample" rule gets R² = 0.999. A chronological split collapses the original model to R² = −283. | Critical | §4.2, E3b/E1c |
| F6 | **Reported numbers are inconsistent or hard-coded.** The headline table (cell 30) hard-codes PINN MAE 1.25 / RMSE 1.76; the computed values are 0.92 / 1.24. The noise-robustness and SHAP values in cell 32 are typed in by hand and differ from cell 27. The frontend hard-codes them again and plots measured Δλ as "Predicted". | High | §4.5 |
| F7 | **The classical baselines are not FBG calibration.** Cell 7 regresses wavelength on *time*, labels nm as pm (reported "MAE 0.38 pm" is really 380 pm) and rescales predictions using the test target's min/max. Cell 9 regresses D3 on D1/D2 waveforms aligned by clock time across independent experiments. The "45× improvement" compares a leaky in-sample PINN with these baselines. | High | §4.4 |
| F8 | **The notebook's only ground-truth test contradicts the headline.** In the simulated bridge (cell 35, known ε and T) the PINN gives MAE 115 µε / 14.5 °C against 6.9 µε / 0.83 °C for a simple low-pass split. The 3-D "PINN" surface (cell 36) is a smoothed copy of the true field, not model output. | High | §4.6 |
| F9 | The measurement equation `Δλ = 1.2·ε + 10·ΔT` (pm, µε, °C) is the right *form*. The coefficients are generic textbook values and were never calibrated on these data. kT = 10 pm/°C is at the low end for bare silica at 1524 nm and could be off by 2–3× if the grating is bonded to a structure. | Medium | §2 |

**Conclusion.** The original pipeline does not estimate strain or temperature, and its reported accuracy is a leakage artefact. The data can support *calibration* of the two sensitivities only if the applied strain and temperature values for D1/D2 are recovered from the lab. They **cannot** support independent strain and temperature estimation in D3 without a second measurement channel, a reference sensor, or explicit assumptions that would then need validating (§5).

---

## 1. Research objective and operating principle

### 1.1 FBG principle

A fibre Bragg grating is a periodic refractive-index modulation of period Λ written into a single-mode fibre core. It reflects a narrow band centred on the Bragg wavelength:

$$\lambda_B = 2\,n_{\text{eff}}\,\Lambda$$

Strain changes Λ (elongation) and n_eff (strain-optic effect). Temperature changes Λ (thermal expansion α) and n_eff (thermo-optic effect ξ). To first order (Hill & Meltz 1997; Kersey et al. 1997; Othonos 1997):

$$\frac{\Delta\lambda_B}{\lambda_B} = (1-p_e)\,\varepsilon + (\alpha_f + \xi)\,\Delta T$$

Here p_e ≈ 0.22 is the effective photo-elastic coefficient of silica, α_f ≈ 0.55×10⁻⁶ /°C, and ξ ≈ 6–9×10⁻⁶ /°C (depends on dopant). In absolute terms:

$$\Delta\lambda_B = K_\varepsilon\,\varepsilon + K_T\,\Delta T$$

### 1.2 Role of a PINN

A PINN (Raissi, Perdikaris & Karniadakis 2019) adds a residual of a known governing equation (usually a PDE) to the training loss. This regularises a network when labelled data are scarce. In this project the only "physics" is the algebraic forward model above. In the notebook, the network outputs (ε̂, T̂), these are pushed through the fixed forward model, and the result is compared with the measured Δλ. The approach can only work if the forward model can be inverted, i.e. if (ε, T) is identifiable from the observations. §5 shows that it cannot be inverted with one grating.

---

## 2. Verification of the measurement equation

### 2.1 What the notebook uses

Cell 13: `k_e = 0.0012 nm/µε → 1.2 pm/µε` and `k_T = 0.01 nm/°C → 10 pm/°C`. The forward model is `Δλ_pred = 1.2·ε̂ + 10·T̂` (pm), with ε̂ in µε and T̂ in °C.

### 2.2 Checks

| Item | Notebook | Literature / derivation at λ_B = 1524.6 nm | Verdict |
|------|----------|--------------------------------------------|---------|
| Functional form | linear, additive, no cross term | first-order model above; cross term ε·ΔT and ξ(T) nonlinearity are negligible for ΔT ≲ 20 °C and ε ≲ 1000 µε | ✅ valid for this range |
| K_ε | 1.2 pm/µε | λ_B(1−p_e) = 1524.6 × 0.78 = **1.19 pm/µε**. Kersey et al. report ≈1.2 pm/µε at 1.55 µm. | ✅ correct as a *nominal* value; not calibrated |
| K_T (bare fibre) | 10 pm/°C | λ_B(α_f+ξ) = 1524.6 × (0.55+6.7…8.6)×10⁻⁶ = **11.0–14.0 pm/°C**. Kersey et al. ≈13.7 pm/°C at 1.55 µm; commercial bare FBGs typically 9.5–11 pm/°C. | ⚠️ plausible lower bound; uncertainty about ±30 % |
| K_T (bonded/embedded) | not considered | Bonded to a substrate of CTE α_s: K_T,eff = λ_B[(α_f+ξ) + (1−p_e)(α_s−α_f)]. For steel (α_s≈12×10⁻⁶) that is **≈ 24–28 pm/°C**; for aluminium (23×10⁻⁶) ≈ 37–41 pm/°C. | ❌ unknown mounting means K_T could be off by 2–3× |
| Units | pm, µε, °C | consistent | ✅ consistent within the PINN cells. ❌ Cell 7 mislabels nm/s and nm as "pm/sec" and "pm". |
| Reference state | Δλ relative to the first sample of each run | assumes the first sample has ε = 0 and the reference T; uses a single noisy sample (σ ≈ 1.5 pm) | ⚠️ unverified. Absolute λ₀ differs by up to 945 pm between runs (§3.3). |
| Strain definition | "strain" | the equation uses *mechanical* strain on the fibre. If bonded, apparent thermal strain of the host appears in K_T,eff. | ⚠️ must be defined for the set-up |
| Applicability to set-up | assumed | needs: same grating in all runs, bare vs bonded, how strain was applied, temperature range | ❓ not documented |

**Bottom line.** The equation is the standard, correct first-order FBG model. The coefficients are generic values that the repository never calibrates. D1 and D2 were designed for that calibration, but the applied strain and temperature values were not recorded in the CSVs. Even with perfect coefficients, the equation is one scalar relation in two unknowns (§5).

---

## 3. Dataset analysis

### 3.1 Files

| File | Rows | Origin | Notes |
|------|------|--------|-------|
| `STRAIN_CSV.csv` | 3 838 | raw interrogator export (D1, strain varied, T nominally constant) | 6 columns |
| `TEMP_CSV.csv` | 3 059 | raw export (D2, T varied, strain nominally constant) | 6 columns |
| `TEMP_STRAIN_CSV.csv` | 9 063 | raw export (D3, both varied) | 6 columns |
| `Week1_*_final.csv` (×3) | same | **derived** by notebook cell 12: raw + renamed headers + `delta_lambda_pm` | no new measurements |
| `frontend/public/Week1_*_final.csv` | same | copies of the above | — |
| `data_dictionary.{csv,md}` | — | generated by cell 3 | describes raw columns only |
| `classical_baseline_results.txt` | — | output of cell 7 | units wrong (§4.4) |

Cell 1 renames `'Unnamed: 5'` → `'Wavelength'`. The original export therefore had a blank sixth header, and the CSVs in the repo were edited afterwards to add `Wavelength`. This is a minor provenance point, but the CSVs are not byte-identical to the instrument export.

### 3.2 Columns

| Column | Unit | Measured / derived | Content |
|--------|------|--------------------|---------|
| `Time (sec)` | s | measured (interrogator clock) | starts at 0.2 s, Δt = 0.2000 s (5 Hz), no gaps |
| `# CH 1` … `# CH 4` | flag | measured (peak-detected on channel) | **CH1 = 1 for every row in all three files; CH2–CH4 always 0.** Only one grating/channel was recorded. |
| `Wavelength` | nm | **measured** (peak wavelength of CH1) | 5 decimals (0.01 pm digits); observed increments are multiples of ≈0.19 pm |
| `delta_lambda_pm` (Week1 files only) | pm | **derived**: (λ − λ_first)·1000 | |

**Directly measured:** time and one Bragg wavelength.
**Derived:** Δλ.
**Interpolated:** none in the CSVs. Notebook cell 9 interpolates D1/D2 onto D3's clock, which is physically meaningless (§4.4).
**Independent references of strain or temperature: none.** There is no load, displacement, micrometer, strain-gauge, thermocouple or RTD column. Every "strain" and "temperature" value in the repository comes either from the network or from the assumed constants.

### 3.3 Experimental conditions inferred from the signal (`figures/01_raw_datasets.png`)

| | D1 strain-only | D2 temperature-only | D3 combined |
|---|---|---|---|
| Duration | 767.6 s | 611.8 s | 1812.6 s |
| λ₀ (first sample) | 1524.60998 nm | 1523.66538 nm | 1524.22429 nm |
| Δλ range | 824 pm (incl. transients) | 143 pm (incl. transients) | 1213 pm (incl. transients) |
| Net Δλ (end − start) | −561 pm | +62 pm | −870 pm |
| Shape | **staircase of 4 downward steps**: −130, −149, −164, −116 pm | **smooth monotonic rise**, ~0 → +61 pm (sigmoid-like, slowing after 400 s) | **staircase of 5 downward steps** −164, −183, −172, −179, −178 pm, then a 1170 s plateau with slow upward drift (+2.5 pm/100 s) |
| Implied size using nominal K | steps ≈ −96…−136 µε each, −465 µε total | ΔT ≈ +6.1 °C (K_T=10) or ≈ +2.7 °C (K_T=23, bonded on steel) | steps ≈ −137…−152 µε each |
| Noise σ (white, from differences) | 1.6 pm | 1.8 pm | 1.4 pm |
| Load-change transients | 4 windows, up to 205 pm excursion, 3–16 s long | **1 window at 458–462 s, ~50–75 pm excursion** | 5 windows, up to 200 pm, 3–10 s |
| Drift within plateaus | −3.6 … +5.7 pm/100 s (≈ flat) | n/a | **+2.5 … +14 pm/100 s, all positive** |

Observations relevant to validity:

1. **The sign is opposite to "stretching".** Tensile strain *increases* λ_B. D1 and D3 show step-wise *decreases*. Possible causes: (a) the fibre was pre-tensioned and the load was released in steps; (b) the grating sits on a surface in compression; or (c) Δλ's reference state is not zero strain. D1's λ₀ is 945 pm above D2's, which is ≈790 µε of pre-strain if it is the same grating at a similar temperature. That points to (a). **This needs confirming by the experimenters.**
2. **D1's temperature looks constant.** The plateaus are flat to ±~5 pm/100 s, i.e. < 0.5 °C/100 s at nominal K_T. This is consistent with the stated design.
3. **D2 contains a mechanical disturbance.** A 3 s, ~70 pm excursion at 459 s is physically impossible for a temperature change of a fibre with thermal mass. It is a strain or peak-tracking artefact, so the "strain-constant" assumption is violated locally.
4. **D3 is strain-dominated.** About −880 pm comes from steps. The candidate thermal component (slow positive drift on every plateau) is only tens of pm (~+30 pm on the final plateau ≈ +3 °C nominal). That makes any temperature estimate from D3 poorly conditioned even with a perfect method. Slow drift can also come from adhesive creep or relaxation, not temperature.
5. **D3 step sizes (≈175 pm) differ from D1 (≈140 pm).** Either the loading differed or temperature changed in step with load. That can't be told apart without reference data.
6. **Transients** (ringing when load is changed) are present in all runs. They are kept in the notebook's training and test sets without comment.
7. **λ₀ differs between runs.** Δλ is re-zeroed per run, so absolute strain or temperature states are not comparable across D1/D2/D3.

---

## 4. Code audit (`Internship_PINN.ipynb`)

### 4.1 Inputs, targets, leakage (cells 13–20)

```python
X      = combined[["Time", "delta_lambda_pm"]].values      # cell 14
y_true = combined["delta_lambda_pm"].values.reshape(-1, 1)  # cell 14
loss   = mean((y - (1.2*eps_hat + 10*T_hat))**2)            # cells 15/17
```

* **The target is an input feature.** The network only has to learn `1.2·ε̂ + 10·T̂ ≈ x₂`. Reproduced in PyTorch with identical architecture and training settings: MAE 0.7–2.0 pm across seeds (notebook: 0.92 pm). **The identity function x₂ gets MAE 0.000 pm, R² 1.000 (E2).** The SHAP result in cell 29 (Δλ ≫ Time) is the signature of this leakage, not a physical insight.
* **ε̂ and T̂ are never supervised.** No labels exist (§3). They are latent variables of an autoencoder whose decoder is the fixed linear map (1.2, 10).
* **Time as a feature** is the elapsed time since that particular run started. It has no physical meaning outside one experiment and lets the network encode the loading schedule. In `figures/03`, the strain output is mostly a linear ramp in Time.
* **No input scaling:** Time goes up to 1812 and Δλ up to ±1000, and the initial loss is 1.1×10⁶. This is minor compared with the issues above.

### 4.2 Data splits and metrics

| Cell | Procedure | Problem |
|------|-----------|---------|
| 15 | 5-fold `KFold(shuffle=True)` | random split of a 5 Hz series; R² computed as **squared Pearson r**, not the coefficient of determination |
| 19–20 | `train_test_split(shuffle=True, test_size=0.2)` | same; test samples sit 0.2 s from training samples |
| 23 | bootstrap over the leaky test predictions | gives a tight CI around a meaningless number |
| 24 | λ-ablation trains **and** evaluates on the full X | no held-out data at all |
| 27 | "noise robustness": adds noise to **y_test (labels)**, not to the model input | measures the injected noise (MAE ≈ 0.8·σ), not the model |
| 28 | "extrapolation": scales Δλ input by 1.5 | no ground truth; only a plot |

Effect of the split (E3b, E1c in `audit_results.json`):

| Predictor | Random split | Chronological split (last 20 %) |
|-----------|-------------|-------------------------------|
| Original PINN set-up `[Time, Δλ]` | MAE 2.0 pm, R² 0.9999 | MAE 37 pm, **R² −283** |
| 1-nearest-neighbour in time (no model) | MAE 2.5 pm, **R² 0.999** | MAE 3.6 pm, R² −1.9 |
| "hold last training value" | — | MAE 3.6 pm |

Even with the target in the input, the original model is ~10× worse than "repeat the last value" once it has to extrapolate in time. The ±1 pm figures come entirely from interpolation plus leakage.

### 4.3 The physics-informed loss

* Training loss (cells 15, 17–19) = MSE(Δλ_meas, 1.2ε̂ + 10T̂). This is a *data* loss through a fixed linear layer. There is no PDE, dynamics, boundary condition or second constraint.
* Ablation (cell 24): `phys_loss = mean((delta_pred - (k_e*eps + k_T*T))**2)` with `delta_pred = k_e*eps + k_T*T` on the line above. It is **identically 0** (E8 reproduces exactly 0.0). Every λ trains the same objective. Differences between λ values (MAE 0.38–0.67) are random-initialisation noise, and "best λ = 1.0" (cells 24/34) is not a finding.
* Cell 31 "physics consistency check" plots `kε·ε̂ + kT·T̂` against itself, so it is on y = x by construction.

### 4.4 Baselines

* **Cell 7 ("classical calibration")**
  * Fits `Wavelength ~ Time` for D1 and D2 and calls the slopes kε and kT ("pm/sec"). They are nm/s and are not sensitivities.
  * Builds "strain" = normalised time and "temperature" = 1 − normalised time, then **rescales predictions to the min/max of the test target** with `np.interp`.
  * Reports MAE "0.38 pm", which is 0.38 nm = **380 pm**, and R² = −1.12. These values are saved to `classical_baseline_results.txt`.
* **Cell 9 ("linear baseline")** interpolates the D1 and D2 Δλ waveforms onto D3's clock and regresses. The three runs are independent experiments, so aligning them by seconds-since-start has no physical meaning. The fitted temperature weight is **−1.85** (unphysical sign) and the evaluation is in-sample (R² 0.923, MAE 56 pm).
* **Cell 30** compares this in-sample, physically meaningless baseline with the leaky PINN and calls it a 45× improvement.

### 4.5 Reporting integrity

* Cell 30 hard-codes `pinn_mae = 1.25, pinn_rmse = 1.76`; cell 20 computed 0.92 / 1.24.
* Cell 32 hard-codes noise-robustness values (2.06 … 8.42) that differ from cell 27 (1.02 … 8.00), and SHAP bars `[800, 5]`.
* Cell 34 prints *"IEEE-grade reproducible"*. There are no TF seeds, paths are Colab's `/content/`, and the outputs come from a local Windows environment.
* Frontend:
  * `ModelComparison.tsx` hard-codes 1.25 pm / 0.999 / "45x improvement".
  * `TimeSeriesPlots.tsx:48` sets `predicted = delta_lambda_pm`, so "Predicted vs Actual" plots the measurement against itself.
  * `ResidualAnalysis.tsx` uses `generateMockData`.
  * The bridge components synthesise data with `Math.random()`.

### 4.6 Simulation cells

* **Cell 35 (bridge simulation, known ε and T):** the PINN gives MAE **115 µε, 14.5 °C**. A low-pass "temperature is slow" split gives 6.9 µε, 0.83 °C. This is the only experiment in the notebook with ground truth for ε and T, and it shows the PINN cannot separate them, which matches §5. The notebook does not discuss it.
* **Cell 36 (3-D animation):** `Z_pred = uniform_filter1d(Z_true, ...)`. The "PINN-predicted" surface is a smoothed copy of the true field and involves no model.

---

## 5. Identifiability assessment

### 5.1 The core problem

At each time step there is one measurement and two unknowns:

$$y(t) = K_\varepsilon\,\varepsilon(t) + K_T\,\Delta T(t) + n(t)$$

The sensitivity matrix **K** = [K_ε, K_T] has rank 1. For **any** function s(t):

$$\varepsilon'(t) = \varepsilon(t) + \tfrac{K_T}{K_\varepsilon}s(t), \qquad \Delta T'(t) = \Delta T(t) - s(t)$$

gives exactly the same y(t). The "physics" term is the source of the degeneracy, so it cannot resolve it, and more network capacity doesn't help. E7 constructs three decompositions of the D3 signal ("all strain", "all temperature", "arbitrary 5 °C sine") with residual < 10⁻¹² pm.

### 5.2 Empirical confirmation

* **Seed dependence (E5, `figures/02`).** Five seeds of the original set-up all fit Δλ to 0.7–2.0 pm. The strain share of Δλ variance ranges over **1 %–72 %**. The ε range runs from [−710, 0] to [0, +88] µε (opposite signs). The minimum predicted T varies from −41 °C to −98 °C.
* **Single-factor falsification (E6, `figures/03`).** A model trained on D3 and applied to D1 (ΔT ≈ 0 by design) predicts a **63 °C** temperature range (≈33 °C between the first and last plateaus; the rest is transients). Applied to D2 (Δε ≈ 0) it predicts a **101 µε** strain range. It fits Δλ to < 1 pm in both cases.

### 5.3 What would make the problem solvable

| Option | What it adds | Notes |
|--------|--------------|-------|
| **A. Independent reference sensors** (thermocouple/RTD beside the grating; load cell, micrometer displacement, or electrical strain gauge) | ground truth for calibration **and** validation | Required for any credible evaluation, whatever the estimator |
| **B. Second grating with a different K_ε/K_T ratio** | a 2×2 sensitivity matrix to invert | Standard solutions: strain-free reference FBG in a loose tube (temperature compensation); FBGs on substrates with different CTE; dual-wavelength or dual-diameter FBGs (Xu et al. 1994); FBG + LPG; FBG in birefringent fibre. Conditioning (det K) sets the noise amplification; Jin et al. 1997 give the error analysis. |
| **C. Calibration from D1/D2** | measured K_ε, K_T for *this* installation | Possible **only** if the applied strain per step (mass and beam geometry, or displacement and gauge length) and the temperature log for D1/D2 are recovered. Calibration alone still does **not** make D3 separable. |
| **D. Structural / temporal assumptions** (e.g. strain piecewise constant, temperature slow and smooth; or strain fast, temperature slow) | a prior that picks one solution from the null space | Makes the problem *conditionally* identifiable. It is the implicit assumption of the cell-35 low-pass method, and D3's step-plus-drift shape fits it. It fails for slow strain (creep, quasi-static loads, thermal expansion of the host), and the assumption must be validated against reference data (option A). Results then become "estimates under assumption X", not measurements. |
| **E. Spectral information beyond the peak** (full reflection spectrum, bandwidth, side-lobes, tilted-FBG cladding modes) | possibly a second independent observable | Needs the raw spectra. Only peak wavelengths were exported. |

### 5.4 Information needed from the experimenters before Phase 2

1. Applied strain for each D1/D3 step: method (weights, micrometer, tensile stage), magnitudes, gauge length or beam geometry, and the timestamps of each change.
2. Temperature log for D2/D3: heater type, set-points, thermocouple/RTD readings and their location relative to the grating.
3. Grating details: same physical FBG in all runs? Nominal λ_B, length, coating, bare or bonded, host material and adhesive.
4. Interrogator: model, wavelength resolution and accuracy, peak-detection method, whether the 4-channel flags mean 4 gratings or 4 ports.
5. Why the shifts are negative under "stretching" (pre-tension release? compression side?).
6. What happened at the transient windows (D1: 126, 249–265, 387–394, 516 s; D2: 458–462 s; D3: 116–122, 241–245, 371–382, 511–516, 638–641 s).
7. In D3: was temperature changed stepwise with the load, or continuously, and by how much?
8. Can a second grating or a reference thermocouple be added to new recordings?

---

## 6. Recommendations for Phase 2 (not carried out here)

1. Recover or record reference ε and T (§5.4) **before** any new modelling. Without them no estimator, PINN or not, can be validated.
2. Calibrate K_ε and K_T from D1/D2 with uncertainties, treating transient windows explicitly (exclude or model them).
3. Choose an identifiability strategy (§5.3 B or D) and state it as an explicit assumption.
4. Evaluation: never put Δλ in the inputs when Δλ is the target. Don't use Time as a feature. Use chronological or leave-one-experiment-out splits. Compute R² properly. Score ε̂ and T̂ against reference values, not Δλ reconstruction. Always report trivial baselines (identity, last value, low-pass split).
5. Remove hard-coded numbers; generate all reported figures from code, with seeds fixed.

---

## References

* K. O. Hill, G. Meltz, "Fiber Bragg grating technology fundamentals and overview," *J. Lightwave Technol.* 15(8), 1263–1276 (1997).
* A. D. Kersey et al., "Fiber grating sensors," *J. Lightwave Technol.* 15(8), 1442–1463 (1997).
* A. Othonos, "Fiber Bragg gratings," *Rev. Sci. Instrum.* 68(12), 4309–4341 (1997).
* Y. J. Rao, "In-fibre Bragg grating sensors," *Meas. Sci. Technol.* 8(4), 355–375 (1997).
* M. G. Xu, J.-L. Archambault, L. Reekie, J. P. Dakin, "Discrimination between strain and temperature effects using dual-wavelength fibre grating sensors," *Electron. Lett.* 30(13), 1085–1087 (1994).
* W. Jin et al., "Simultaneous measurement of strain and temperature: error analysis," *Opt. Eng.* 36(2), 598–609 (1997).
* M. Raissi, P. Perdikaris, G. E. Karniadakis, "Physics-informed neural networks," *J. Comput. Phys.* 378, 686–707 (2019).
* S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, "Leakage in data mining: formulation, detection, and avoidance," *ACM TKDD* 6(4), 15 (2012).
* D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography* 40(8), 913–929 (2017).

*Literature values are quoted from these standard sources as recalled. Check exact figures against the papers before citing them in a publication.*
