"""
Evaluación del Gain e Importancia Global de 'pct_heuristico_fraude' en LightGBM.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

base = pd.read_parquet("data/processed/dataset_base.parquet")
feats = pd.read_parquet("data/features/features_transaccionales.parquet")

overlap = [c for c in feats.columns if c in base.columns and c != "transaction_id"]
feats_clean = feats.drop(columns=overlap)
merged = base.merge(feats_clean, on="transaction_id")

train = merged[merged["split"] == "train"].copy()
val = merged[merged["split"] == "val"].copy()

exclude_cols = ["transaction_id", "fecha", "split", "fraude", "i", "score_x_monto", "score_heuristico_fraude_10"]
candidate_cols = [c for c in merged.columns if c not in exclude_cols]

cat_cols = ["g", "o", "p", "j", "a", "n"]
for c in cat_cols:
    if c in candidate_cols:
        train[c] = train[c].astype("category")
        val[c] = val[c].astype("category")

X_tr = train[candidate_cols]
y_tr = train["fraude"].values
X_va = val[candidate_cols]
y_va = val["fraude"].values

clf = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1,
)
clf.fit(X_tr, y_tr)

p_va = clf.predict_proba(X_va)[:, 1]
pr, rc, _ = precision_recall_curve(y_va, p_va)
pr_auc = auc(rc, pr)
roc_auc = roc_auc_score(y_va, p_va)
print(f"PR-AUC Val OOT: {pr_auc:.4f} | ROC-AUC Val OOT: {roc_auc:.4f}\n")

gains = pd.Series(clf.booster_.feature_importance(importance_type="gain"), index=candidate_cols).sort_values(ascending=False)
splits = pd.Series(clf.booster_.feature_importance(importance_type="split"), index=candidate_cols)

res = pd.DataFrame({
    "gain": gains.round(1),
    "gain_pct": (gains / gains.sum() * 100).round(2),
    "splits": splits.loc[gains.index],
})

print("=== TOP 25 FEATURES POR GAIN (CONJUNTO TOTAL) ===")
print(res.head(25).to_string())

rank_pct = list(gains.index).index("pct_heuristico_fraude") + 1
gain_val = gains["pct_heuristico_fraude"]
pct_val = res.loc["pct_heuristico_fraude", "gain_pct"]
splits_val = res.loc["pct_heuristico_fraude", "splits"]

print(f"\n=======================================================")
print(f"Posición de pct_heuristico_fraude: #{rank_pct} de {len(candidate_cols)} variables")
print(f"Gain total acumulado: {gain_val:,.1f} ({pct_val}% del total)")
print(f"Splits en los árboles: {splits_val} veces seleccionado como corte")
print(f"=======================================================")
