#!/usr/bin/env python
# coding: utf-8

# In[7]:


import re, pandas as pd
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
    if ROOT.name == "src":
        ROOT = ROOT.parent

INP = ROOT / "data" / "processed" / "clean.csv"
OUT = ROOT / "data" / "processed" / "features.csv"

df = pd.read_csv(INP)

c_col = next((c for c in ["cuisine","cuisines","Cuisines"] if c in df.columns), None)
if c_col:
    def count_cuisines(s):
        if pd.isna(s): return 0
        return len([x for x in re.split(r"[,/|;]", str(s)) if x.strip()])
    df["cuisine_count"] = df[c_col].apply(count_cuisines).astype(int)

if "cost" in df.columns:
    try:
        df["cost_bin"] = pd.qcut(df["cost"], q=3, labels=["Low","Medium","High"])
    except Exception:
        pass

if "name" in df.columns:
    counts = df["name"].value_counts()
    df["is_chain"] = df["name"].map(lambda x: 1 if counts.get(x, 0) > 1 else 0)

df.to_csv(OUT, index=False)
print("Wrote:", OUT, "Shape:", df.shape)
print("New columns added:", [c for c in ["cuisine_count","cost_bin","is_chain"] if c in df.columns])


# In[ ]:




