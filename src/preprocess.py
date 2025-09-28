#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from pathlib import Path

# Resolve project root (always one level up from src/)
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
    if ROOT.name == "src":
        ROOT = ROOT.parent

RAW = ROOT / "data" / "raw" / "zomato_df_final_data.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

# Strip column names
df.columns = [c.strip() for c in df.columns]

# Drop empty + duplicate rows
before = df.shape[0]
df = df.dropna(how="all").drop_duplicates()
after = df.shape[0]
print(f"Dropped {before - after} rows during cleaning")

# Save
clean_path = OUT_DIR / "clean.csv"
df.to_csv(clean_path, index=False)
print("Wrote:", clean_path, "Shape:", df.shape)


# In[ ]:




