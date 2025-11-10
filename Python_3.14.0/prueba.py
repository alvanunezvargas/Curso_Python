# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# CONFIG
# =========================
INPUT_XLSX      = r"C:\Users\Alvaro\OneDrive\Documentos\Python\Curso Python\propiedades_compuestos.xlsx"
SMILES_COLUMN   = "SMILES"
TARGET_COLUMN   = "XLogP"          # cambia por 'Polar_Area', 'Exact_Mass', etc.
N_BITS          = 1024
RADIUS          = 2                # Morgan radius (2 => ECFP4)
USE_RD_DESC     = True             # añadir descriptores RDKit básicos
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# Para el mapa de calor de similitud:
HEATMAP_SAMPLE  = 120              # nº máximo de moléculas a incluir en el heatmap (para no saturar)
HEATMAP_CMAP    = "viridis"        # colormap matplotlib
HEATMAP_FIGSIZE = (7, 6)

# =========================
# UTILIDADES
# =========================
def smiles_to_mol(s: str):
    """Convierte SMILES a Mol, devolviendo None si no parsea."""
    if pd.isna(s):
        return None
    try:
        return Chem.MolFromSmiles(str(s))
    except Exception:
        return None

def compute_basic_descriptors(mol):
    """Devuelve descriptores RDKit simples o None si mol=None."""
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RingCount": Descriptors.RingCount(mol),
    }

def build_morgan_generator(radius=2, nBits=1024):
    """Crea el generador Morgan (compatible con distintas versiones de RDKit)."""
    try:
        # Nueva API (usada en RDKit >= 2022.09)
        return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits, includeChirality=True)
    except TypeError:
        # API antigua (usaba 'length' en lugar de 'fpSize')
        return rdFingerprintGenerator.GetMorganGenerator(radius=radius, length=nBits, includeChirality=True)


def mol_to_fp_rdkit_and_numpy(mol, gen, nBits):
    """
    A partir de un Mol y un generador Morgan:
    - devuelve (fp_rdkit, arr_np_uint8)
    fp_rdkit: ExplicitBitVect
    arr_np_uint8: numpy array binario (0/1) de longitud nBits
    """
    if mol is None:
        return None, None
    fp = gen.GetFingerprint(mol)                          # RDKit ExplicitBitVect
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)              # convierte a vector numpy 0/1
    return fp, arr

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Limpia serie numérica en string ('3,14', '—', etc.) y la convierte a float (NaN si no se puede)."""
    if s.dtype == object:
        s = (
            s.astype(str)
             .str.replace(",", ".", regex=False)
             .str.replace(r"[^0-9.\-eE+]", "", regex=True)
        )
    return pd.to_numeric(s, errors="coerce")

def tanimoto_similarity_matrix(fp_list):
    """
    Calcula la matriz de similitud de Tanimoto (NxN) a partir de una lista de RDKit fingerprints.
    Usa BulkTanimotoSimilarity para eficiencia.
    """
    n = len(fp_list)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        if i + 1 < n:
            # Similaridades de fp[i] contra fp[i+1:]
            vals = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i+1:])
            sim[i, i+1:] = vals
            sim[i+1:, i] = vals
    return sim

# =========================
# CARGA Y LIMPIEZA
# =========================
df = pd.read_excel(INPUT_XLSX)
print("Columnas:", list(df.columns))
print(df.head(3))

if SMILES_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
    raise ValueError(f"Faltan columnas requeridas: '{SMILES_COLUMN}' y/o '{TARGET_COLUMN}'")

# limpiar objetivo a numérico
y_raw = clean_numeric_series(df[TARGET_COLUMN])

# construir moléculas
mols = df[SMILES_COLUMN].apply(smiles_to_mol)

# crear generador Morgan UNA sola vez
morgan_gen = build_morgan_generator(radius=RADIUS, nBits=N_BITS)

# fingerprints RDKit + arrays numpy
fp_rdkit_list = []
fp_numpy_list = []
for m in mols:
    fp, arr = mol_to_fp_rdkit_and_numpy(m, morgan_gen, N_BITS)
    fp_rdkit_list.append(fp)
    fp_numpy_list.append(arr)

# máscaras de validez
mask_smiles = pd.Series([fp is not None for fp in fp_rdkit_list], index=df.index)
mask_y      = y_raw.notna() & np.isfinite(y_raw.values)
mask        = mask_smiles & mask_y

print(f"Filas totales: {len(df)}")
print(f"SMILES válidos: {mask_smiles.sum()}")
print(f"Objetivo numérico (no NaN/inf): {mask_y.sum()}")
print(f"Filas utilizables (ambos válidos): {mask.sum()}")

if mask.sum() < 3:
    raise RuntimeError("Muy pocas filas válidas tras limpieza. Revisa SMILES/objetivo.")

# filtrar datos válidos
df_valid    = df.loc[mask].reset_index(drop=True)
y           = y_raw.loc[mask].astype(float).values
mols_valid  = [m for (m, keep) in zip(mols, mask) if keep]
fps_rd_valid= [fp for (fp, keep) in zip(fp_rdkit_list, mask) if keep]
fps_np_valid= [arr for (arr, keep) in zip(fp_numpy_list, mask) if keep]

# matriz X a partir de arrays numpy (0/1)
X_fp = np.vstack(fps_np_valid).astype(float)

# añadir descriptores RDKit (opcional)
if USE_RD_DESC:
    desc_rows = [compute_basic_descriptors(m) for m in mols_valid]
    desc_df   = pd.DataFrame(desc_rows).fillna(0.0)
    X = np.hstack([X_fp, desc_df.values.astype(float)])
else:
    X = X_fp

print("X shape:", X.shape, "| y shape:", y.shape)

# =========================
# ENTRENAMIENTO / EVALUACIÓN
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.3f}")

# =========================
# GRÁFICO REAL vs PREDICHO
# =========================
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6)
mn = min(np.min(y_test), np.min(y_pred))
mx = max(np.max(y_test), np.max(y_pred))
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.xlabel(f"Valor real ({TARGET_COLUMN})")
plt.ylabel(f"Valor predicho ({TARGET_COLUMN})")
plt.title(f"Random Forest: {TARGET_COLUMN}")
plt.tight_layout()
plt.show()

# =========================
# HEATMAP DE SIMILITUD (Tanimoto)
# =========================
# Tomamos un subconjunto para que sea legible
n_total = len(fps_rd_valid)
n_use   = min(HEATMAP_SAMPLE, n_total)
fps_sub = fps_rd_valid[:n_use]

print(f"Calculando matriz de similitud Tanimoto para {n_use} moléculas…")
sim_mat = tanimoto_similarity_matrix(fps_sub)  # NxN

plt.figure(figsize=HEATMAP_FIGSIZE)
plt.imshow(sim_mat, interpolation="nearest", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0)
plt.title("Similitud Tanimoto (fingerprints Morgan)")
plt.xlabel("Molécula")
plt.ylabel("Molécula")
cbar = plt.colorbar()
cbar.set_label("Tanimoto")
plt.tight_layout()
plt.show()

# ===========================================
# AGRUPAMIENTO JERÁRQUICO DE MOLÉCULAS SIMILARES
# ===========================================
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# Convertir la matriz de similitud (0–1) en matriz de distancias (1 - similitud)
dist_matrix = 1.0 - sim_mat

# Convertir a formato "condensed" (vector 1D requerido por linkage)
# squareform evita duplicar información simétrica
condensed_dist = squareform(dist_matrix, checks=False)

# Realizar el clustering jerárquico (método 'average' = UPGMA, apropiado para distancias químicas)
linkage_matrix = linkage(condensed_dist, method='average')

# --- Visualizar dendrograma ---
plt.figure(figsize=(10, 6))
dendrogram(
    linkage_matrix,
    color_threshold=0.7,      # ajusta el umbral visual de similitud (0–1)
    leaf_rotation=90,
    leaf_font_size=8,
)
plt.title("Dendrograma químico basado en similitud Tanimoto")
plt.xlabel("Moléculas")
plt.ylabel("Distancia química (1 - similitud)")
plt.tight_layout()
# Etiquetas con nombres o IDs de las moléculas
labels = df_valid["Compound_CID"].iloc[:n_use].astype(str).tolist()

plt.figure(figsize=(12, 6))
dendrogram(
    linkage_matrix,
    labels=labels,          # ← usar CIDs como etiquetas
    color_threshold=0.7,
    leaf_rotation=90,
    leaf_font_size=8,
)
plt.title("Dendrograma químico basado en similitud Tanimoto (etiquetas: Compound_CID)")
plt.xlabel("Compound_CID")
plt.ylabel("Distancia química (1 - similitud)")
plt.tight_layout()
plt.show()

# --- Opcional: asignar clústeres ---
# Por ejemplo, agrupa moléculas con similitud ≥ 0.7 (distancia ≤ 0.3)
threshold = 0.3
clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
print(f"\nNúmero de clústeres encontrados (similitud ≥ {1-threshold:.2f}): {clusters.max()}")

# Añadir los clústeres al dataframe original
df_clusters = df_valid.iloc[:len(clusters)].copy()
df_clusters["Cluster_ID"] = clusters
print(df_clusters[["Name", "SMILES", "Cluster_ID"]].head(10))

# --- (Opcional) Mostrar cuántas moléculas hay por grupo ---
cluster_summary = df_clusters["Cluster_ID"].value_counts().sort_index()
print("\nTamaño de cada clúster:")
print(cluster_summary)
