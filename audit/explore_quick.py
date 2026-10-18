import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
root = Path(sys.argv[1])
for f in ["STRAIN_CSV.csv","TEMP_CSV.csv","TEMP_STRAIN_CSV.csv"]:
    d = pd.read_csv(root/f); d.columns=["t","c1","c2","c3","c4","wl"]
    dt = np.diff(d.t)
    steps = np.diff(d.wl)*1000
    nz = np.abs(steps[np.abs(steps)>1e-6])
    print(f"== {f}: n={len(d)} t=[{d.t.min():.3f},{d.t.max():.3f}] dt mean={dt.mean():.6f} min={dt.min():.6f} max={dt.max():.6f}")
    print("  channel sums:", d[["c1","c2","c3","c4"]].sum().to_dict(), "nan:", d.isna().sum().sum())
    print(f"  wl min={d.wl.min():.5f} max={d.wl.max():.5f} first={d.wl.iloc[0]:.5f} last={d.wl.iloc[-1]:.5f}")
    print(f"  smallest nonzero step pm: {np.sort(np.unique(np.round(nz,3)))[:8]}")
    print(f"  unique wl values: {d.wl.nunique()}")
    # resolution grid
    u=np.sort(d.wl.unique()); g=np.diff(u)*1000
    print(f"  grid spacing pm: median={np.median(g):.3f}  mode-ish={pd.Series(np.round(g,2)).value_counts().head(4).to_dict()}")
    # coarse profile (every 10% of run)
    idx = np.linspace(0,len(d)-1,21).astype(int)
    seg = [(d.wl.iloc[max(0,i-25):i+25].mean()-d.wl.iloc[:50].mean())*1000 for i in idx]
    print("  profile Δλ pm (21 pts, 50-sample avg):", np.round(seg,1).tolist())
    # noise: std of first-difference / sqrt2 (robust)
    mad = np.median(np.abs(steps-np.median(steps)))*1.4826
    print(f"  noise est (MAD of diff/sqrt2): {mad/np.sqrt(2):.3f} pm; std of diffs {steps.std():.3f} pm; max |jump| {np.abs(steps).max():.1f} pm")
