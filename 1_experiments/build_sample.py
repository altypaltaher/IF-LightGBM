"""CIC-AAGM2017 (TotalFeatures-ISCXFlowMeter.csv): de-duplicate, drop label-conflicting vectors,
binarise (adware + general malware = 1, benign = 0), and draw a seeded class-balanced sample of 20,000 flows."""
import pandas as pd, numpy as np
d = pd.read_csv("data/TotalFeatures-ISCXFlowMeter.csv", low_memory=False); d.columns = d.columns.str.strip()
d = d.drop_duplicates()
feat = [c for c in d.columns if c != "calss"]
d = d[~d.duplicated(subset=feat, keep=False)]
d["y"] = (d["calss"].str.strip().str.lower() != "benign").astype(int)
print("unique, non-conflicting:", len(d), d["calss"].value_counts().to_dict())
mal = d[d.y == 1].sample(10000, random_state=42)          # proportional to adware / general-malware shares
ben = d[d.y == 0].sample(10000, random_state=42)
s = pd.concat([mal, ben]).sample(frac=1, random_state=42)
print("sample:", s["calss"].value_counts().to_dict())
s.drop(columns="calss").to_csv("data/aagm2017_sample20k.csv", index=False)
