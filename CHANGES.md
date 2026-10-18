# Changes relative to upstream

**Upstream:** https://github.com/anvitha234/FBG-Sensor-PINN @ `aa7693a` (Initial commit)

The upstream repository is not modified. This fork keeps every upstream file byte-for-byte identical. All new work lives in new files and directories, on the `scientific-audit` branch.

## Phase 1: scientific audit (verification only, no new model)

| Path | Type | Purpose |
|------|------|---------|
| `audit/SCIENTIFIC_AUDIT.md` | new | Full technical assessment: physics equation, datasets, code audit, identifiability |
| `audit/run_audit.py` | new | Reproducible experiments behind every number in the report (E1–E8). Reads the original raw CSVs read-only. |
| `audit/explore_quick.py` | new | Quick descriptive statistics of the three raw CSVs |
| `audit/audit_results.json` | new (generated) | Output of `run_audit.py` |
| `audit/run_log.txt` | new (generated) | Console log of the last `run_audit.py` run |
| `audit/figures/01_raw_datasets.png` | new (generated) | Raw Δλ of D1/D2/D3 |
| `audit/figures/02_seed_nonidentifiability.png` | new (generated) | Original set-up, 5 seeds: same Δλ fit, different ε/T |
| `audit/figures/03_single_factor_falsification.png` | new (generated) | ε/T outputs on the single-factor runs |
| `CHANGES.md` | new | This file |

### Notes on the reproduction

* The notebook used TensorFlow 2.14. `run_audit.py` reimplements the same model in PyTorch: architecture 64-64-32 ReLU with two linear heads, Glorot-uniform initialisation, inputs `[Time, Δλ]`, target `Δλ`, fixed coefficients 1.2 pm/µε and 10 pm/°C, 80/20 shuffled split (seed 42), full-batch Adam at lr 1e-3 for 501 epochs. Exact numbers differ slightly from the notebook (framework and random state). The qualitative results do not depend on this.
* No upstream file was edited, re-executed or regenerated. The upstream notebook outputs and `classical_baseline_results.txt` are preserved as historical results, including their errors.

### Reproduce

```bash
pip install numpy pandas matplotlib scikit-learn torch
python -I audit/run_audit.py .
```
