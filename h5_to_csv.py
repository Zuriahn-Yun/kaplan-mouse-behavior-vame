"""
h5_to_csv.py
Converts all DLC .h5 files in Open-Field-Test/data/raw/ to CSV.
CSVs are saved next to the originals with the same base name.
No data is modified.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\zuria\Kaplan\Vame\Open-Field-Test\data\raw")

h5_files = sorted(RAW_DIR.glob("*.h5"))
print(f"Found {len(h5_files)} .h5 files\n")

for h5_path in h5_files:
    csv_path = h5_path.with_suffix(".csv")

    if csv_path.exists():
        print(f"  [skip] {csv_path.name} already exists")
        continue

    try:
        df = pd.read_hdf(h5_path, key="df_with_missing")
        df.to_csv(csv_path, encoding="utf-8-sig")
        print(f"  [ok]   {h5_path.name}  →  {csv_path.name}  ({len(df):,} rows)")
    except Exception as e:
        print(f"  [ERR]  {h5_path.name}: {e}")

print("\nDone.")
