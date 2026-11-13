"""
Experimento: TabM (Tabular DL Model with Multiple Predictions)
==============================================================
Implementa y entrena la arquitectura TabM con PiecewiseLinearEmbeddings (PLE)
usando los hiperparámetros especificados por el usuario:
- Cabezas (k): 32
- Bloques ocultos (n_blocks): 3
- Ancho de capa (d_block): 512
- Dropout: 0.15
- Embedding numérico: PiecewiseLinearEmbeddings (PLE) con n_bins=48 y d_embedding=16
- Optimizador: AdamW (lr=2e-3, weight_decay=3e-4)
- Batch Size: 512 con Early Stopping de 16 épocas de paciencia
- Features: 13 variables oficiales del Champion de Parsimonia (Challenger v3)
"""

from __future__ import annotations
import copy
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, normalize
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import tabm
import rtdl_num_embeddings
from src.models.supervised import pr_auc, validation_blocks


def build_clustering_features(df_train: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Genera deficit_piso_cluster y distancia_log_cluster ajustados en Train con SVD d=30."""
    print("  -> Generando clustering semántico k=50 (SVD d=30 L2-norm) sobre títulos...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, sublinear_tf=True, strip_accents="unicode")
    X_tr_tfidf = vec.fit_transform(df_train["i"].fillna(""))
    svd = TruncatedSVD(n_components=30, random_state=42)
    X_tr_svd = normalize(svd.fit_transform(X_tr_tfidf))

    kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=2048)
    train_clusters = kmeans.fit_predict(X_tr_svd)

    X_all_tfidf = vec.transform(df_all["i"].fillna(""))
    X_all_svd = normalize(svd.transform(X_all_tfidf))
    all_clusters = kmeans.predict(X_all_svd)

    temp_train = pd.DataFrame({"cluster_i": train_clusters, "monto": df_train["monto"].values})
    cluster_stats = temp_train.groupby("cluster_i")["monto"].agg(["median", "count"])
    global_median = float(df_train["monto"].median())
    m_prior = 10.0
    cluster_stats["bayes_median"] = (
        cluster_stats["count"] * cluster_stats["median"] + m_prior * global_median
    ) / (cluster_stats["count"] + m_prior)
    median_map = cluster_stats["bayes_median"].to_dict()

    all_medians = np.array([median_map.get(c, global_median) for c in all_clusters], dtype=float)
    all_montos = df_all["monto"].values.astype(float)

    distancia_log = (np.log1p(all_montos) - np.log1p(all_medians)).astype(float)
    deficit_piso = np.maximum(0.0, (all_medians - all_montos) / (all_medians + 1e-4)).astype(float)
    return distancia_log, deficit_piso


def main():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    t0 = time.time()

    print("=" * 95)
    print("EXPERIMENTO: ENTRENAMIENTO DE TABM (RTDL_NUM_EMBEDDINGS + ADAMW)")
    print("PARÁMETROS EXACTOS: k=32, n_blocks=3, d_block=512, dropout=0.15, PLE n_bins=48, d_emb=16")
    print(f"DISPOSITIVO: {device} | HILOS CPU: {torch.get_num_threads()}")
    print("=" * 95)

    # 1. Cargar datos
    print("\n1. Cargando datos y generando features de Champion Parsimonia...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")

    # Clustering semántico de Challenger v3
    train_mask_raw = base_df["split"] == "train"
    dist_log, def_piso = build_clustering_features(base_df[train_mask_raw], base_df)
    base_df["distancia_log_cluster"] = dist_log
    base_df["deficit_piso_cluster"] = def_piso

    df = base_df.merge(feat_df, on="transaction_id")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "transaction_id"]).reset_index(drop=True)

    # 2. Definir las 13 variables del Champion de Parsimonia
    cat_cols = ["a", "g", "o"]
    num_cols = [
        "score", "es_madrugada", "ratio_d_m", "ratio_monto_l",
        "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero",
        "deficit_piso_cluster", "distancia_log_cluster"
    ]
    all_feature_cols = num_cols + cat_cols

    print(f"  * Total de variables seleccionadas: {len(all_feature_cols)} ({len(num_cols)} numéricas, {len(cat_cols)} categóricas)")
    print(f"    - Numéricas: {num_cols}")
    print(f"    - Categóricas: {cat_cols}")

    # Splits
    train_df = df.loc[df.split == "train"].copy()
    val_full = df.loc[df.split == "val"].copy()
    test_df = df.loc[df.split == "test"].copy()
    tune_df, cal_df, policy_df = validation_blocks(val_full)

    y_train = train_df.fraude.to_numpy(dtype=np.float32)
    y_tune = tune_df.fraude.to_numpy(dtype=np.float32)
    y_val_full = val_full.fraude.to_numpy(dtype=np.float32)
    y_test = test_df.fraude.to_numpy(dtype=np.float32)

    # 3. Preprocesamiento numérico y categórico
    print("\n2. Preprocesando datos para TabM...")
    # Numéricas: imputación con mediana de train
    imputer = SimpleImputer(strategy="median").fit(train_df[num_cols])
    X_tr_num_raw = imputer.transform(train_df[num_cols]).astype(np.float32)
    X_tu_num_raw = imputer.transform(tune_df[num_cols]).astype(np.float32)
    X_vf_num_raw = imputer.transform(val_full[num_cols]).astype(np.float32)
    X_te_num_raw = imputer.transform(test_df[num_cols]).astype(np.float32)

    # Categóricas: OrdinalEncoder con categoría MISSING para nulos
    for cat in cat_cols:
        df[cat] = df[cat].fillna("__MISSING__").astype(str)
        train_df[cat] = train_df[cat].fillna("__MISSING__").astype(str)
        tune_df[cat] = tune_df[cat].fillna("__MISSING__").astype(str)
        val_full[cat] = val_full[cat].fillna("__MISSING__").astype(str)
        test_df[cat] = test_df[cat].fillna("__MISSING__").astype(str)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(train_df[cat_cols])
    X_tr_cat = enc.transform(train_df[cat_cols]).astype(np.int64)
    X_tu_cat = enc.transform(tune_df[cat_cols]).astype(np.int64)
    X_vf_cat = enc.transform(val_full[cat_cols]).astype(np.int64)
    X_te_cat = enc.transform(test_df[cat_cols]).astype(np.int64)

    # Manejar posibles valores no vistos en test (-1) asignándoles una nueva categoría
    cat_cardinalities = []
    for j in range(len(cat_cols)):
        max_idx = max(int(X_tr_cat[:, j].max()), int(X_tu_cat[:, j].max()), int(X_te_cat[:, j].max()))
        # Si hay -1, reemplazar por max_idx + 1
        mask_tr = X_tr_cat[:, j] == -1
        mask_tu = X_tu_cat[:, j] == -1
        mask_vf = X_vf_cat[:, j] == -1
        mask_te = X_te_cat[:, j] == -1
        if mask_tr.any() or mask_tu.any() or mask_te.any() or mask_vf.any():
            X_tr_cat[mask_tr, j] = max_idx + 1
            X_tu_cat[mask_tu, j] = max_idx + 1
            X_vf_cat[mask_vf, j] = max_idx + 1
            X_te_cat[mask_te, j] = max_idx + 1
            cat_cardinalities.append(max_idx + 2)
        else:
            cat_cardinalities.append(max_idx + 1)

    print(f"  * Cardinalidades de categóricas {cat_cols}: {cat_cardinalities}")

    # 4. Construcción de PiecewiseLinearEmbeddings (PLE)
    print("\n3. Calculando bins para PiecewiseLinearEmbeddings (n_bins=48, d_emb=16)...")
    X_tr_num_tensor = torch.tensor(X_tr_num_raw)
    bins = rtdl_num_embeddings.compute_bins(X_tr_num_tensor, n_bins=48)
    ple = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, d_embedding=16, activation=True, version="B")

    # 5. Instanciación del modelo TabM
    k_heads = 32
    n_blocks = 3
    d_block = 512
    dropout_rate = 0.15
    lr = 2e-3
    weight_decay = 3e-4
    batch_size = 512
    patience = 16
    max_epochs = 40

    print(f"\n4. Instanciando TabM (k={k_heads}, n_blocks={n_blocks}, d_block={d_block}, dropout={dropout_rate})...")
    model = tabm.TabM(
        n_num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        num_embeddings=ple,
        k=k_heads,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout_rate,
        start_scaling_init="random-signs",
    ).to(device)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  * Parámetros entrenables totales: {total_params:,}")

    # 6. Optimizador y Dataloaders
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_num_raw),
        torch.tensor(X_tr_cat),
        torch.tensor(y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Tensores para validación e inferencia
    t_tu_num = torch.tensor(X_tu_num_raw, device=device)
    t_tu_cat = torch.tensor(X_tu_cat, device=device)
    t_vf_num = torch.tensor(X_vf_num_raw, device=device)
    t_vf_cat = torch.tensor(X_vf_cat, device=device)
    t_te_num = torch.tensor(X_te_num_raw, device=device)
    t_te_cat = torch.tensor(X_te_cat, device=device)

    def predict_probs(m: nn.Module, x_num: torch.Tensor, x_cat: torch.Tensor, eval_batch_size: int = 2048) -> np.ndarray:
        m.eval()
        probs_all = []
        n_samples = len(x_num)
        with torch.no_grad():
            for i in range(0, n_samples, eval_batch_size):
                b_num = x_num[i:i + eval_batch_size].to(device)
                b_cat = x_cat[i:i + eval_batch_size].to(device)
                out = m(b_num, b_cat)  # (batch, k, 1)
                # Ensamble de las k cabezas promediando probabilidades sigmoid
                probs_k = torch.sigmoid(out.squeeze(-1))  # (batch, k)
                b_probs = probs_k.mean(dim=-1).cpu().numpy()
                probs_all.append(b_probs)
        return np.concatenate(probs_all)

    print("\n" + "=" * 95)
    print("5. INICIANDO ENTRENAMIENTO DE TABM (MONITOREANDO PR-AUC EN VAL_SEARCH)...")
    print("=" * 95)

    best_val_prauc = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0

    header = f"{'Época':<7s} | {'Loss Train':<12s} | {'PR-AUC Val Tune':<16s} | {'PR-AUC Val Full':<16s} | {'Delta vs Best':<14s} | {'Paciencia'}"
    print(header)
    print("-" * 95)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_t0 = time.time()

        for batch_num, batch_cat, batch_y in train_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_num, batch_cat)  # (batch_size, k, 1)
            # Loss TabM: cada cabeza k calcula su BCE contra el target expandido
            target_expanded = batch_y.unsqueeze(-1).expand(-1, k_heads)  # (batch_size, k)
            loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), target_expanded)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Evaluar en tune (val_search)
        p_tune = predict_probs(model, t_tu_num, t_tu_cat)
        val_prauc_tune = pr_auc(y_tune, p_tune)

        p_val_full = predict_probs(model, t_vf_num, t_vf_cat)
        val_prauc_full = pr_auc(y_val_full, p_val_full)

        delta_best = val_prauc_tune - best_val_prauc

        if val_prauc_tune > best_val_prauc:
            best_val_prauc = val_prauc_tune
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            mark = f"+{delta_best:.5f} (Nuevo Mejor)"
        else:
            patience_counter += 1
            mark = f"{delta_best:.5f}"

        epoch_sec = time.time() - epoch_t0
        print(f"Ep {epoch:<4d} | {avg_loss:<12.5f} | {val_prauc_tune:<16.5f} | {val_prauc_full:<16.5f} | {mark:<14s} | {patience_counter}/{patience} ({epoch_sec:.1f}s)")

        if patience_counter >= patience:
            print(f"\n[!] Early Stopping alcanzado en época {epoch}: sin mejoras durante {patience} épocas.")
            break

    # 7. Evaluación Final con el Mejor Checkpoint
    print("\n" + "=" * 95)
    print(f"6. RESULTADO FINAL DE TABM (MEJOR CHECKPOINT EN ÉPOCA {best_epoch})")
    print("=" * 95)
    model.load_state_dict(best_state)

    final_p_tune = predict_probs(model, t_tu_num, t_tu_cat)
    final_p_val_full = predict_probs(model, t_vf_num, t_vf_cat)
    final_p_test = predict_probs(model, t_te_num, t_te_cat)

    final_prauc_tune = pr_auc(y_tune, final_p_tune)
    final_prauc_val_full = pr_auc(y_val_full, final_p_val_full)
    final_prauc_test = pr_auc(y_test, final_p_test)

    print(f"  * PR-AUC Val Search (tune): {final_prauc_tune:.5f}")
    print(f"  * PR-AUC Val Full (22.5k):  {final_prauc_val_full:.5f}")
    print(f"  * PR-AUC Test (Ciego 22.5k): {final_prauc_test:.5f}")

    # Comparativa contra los modelos líderes
    print("\n" + "=" * 95)
    print("COMPARATIVA DE TABM VS MODELOS LÍDERES DEL PROYECTO")
    print("=" * 95)
    summary_rows = [
        ("Control Ian: LightGBM 11 vars", 0.48136, 0.44435, "+0.0 bps"),
        ("Challenger v3: LightGBM 13 vars", 0.48850, 0.44700, "+26.5 bps"),
        ("Champion Parsimonia: Cuatri-Modelo L1 (C=0.10)", 0.49101, 0.45093, "+65.8 bps"),
        ("Forward Selection Multi-Loss (5 modelos)", 0.49502, 0.45010, "+57.5 bps"),
        ("Deep Learning: TabM (PLE + AdamW + 13 vars)", final_prauc_tune, final_prauc_test, f"{(final_prauc_test - 0.44435)*10000:+.1f} bps"),
    ]

    print(f"{'Arquitectura / Modelo':<48s} | {'val_search':<12s} | {'test final':<12s} | {'Delta vs Ian Test'}")
    print("-" * 95)
    for name, v_sc, t_sc, d_ian in summary_rows:
        print(f"{name:<48s} | {v_sc:.5f}      | {t_sc:.5f}      | {d_ian}")

    print(f"\n[OK] Experimento TabM finalizado en {time.time() - t0:.1f} segundos.")


if __name__ == "__main__":
    main()
