# -*- coding: utf-8 -*-
"""
Núcleo de cálculo para la comparación de seis algoritmos de agrupamiento.

Contiene: pre-procesamiento único, matriz de Gower (árbitro común), PAM propio
(scikit-learn-extra no importa con numpy 2.x), métricas y estabilidad.

Todo con random_state=42.
"""
import os
import time
import numpy as np
import pandas as pd

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "resultados")

NUMERICAS = [
    "distancia_geografica_km",
    "cpc_match_score",
    "sim_semantica_tfidf",
    "desviacion_presupuesto",
    "actividad_cpc_comprador",
]
CATEGORICA = "modalidad_contratacion"


# --------------------------------------------------------------- datos
def cargar_matriz():
    df = pd.read_csv(os.path.join(RES, "matriz_procesos.csv"))
    return df


def preprocesar(df):
    """Devuelve (num_crudas, num_estandarizadas, cat_codigos, X_codificada, nombres_cols).

    - num_crudas: para Gower (que normaliza por rango, no por z-score).
    - num_estandarizadas: z-score, para K-Means / K-Prototypes / GMM.
    - X_codificada: numéricas estandarizadas + one-hot de la modalidad. Sobre ella
      se calculan Davies-Bouldin y Calinski-Harabasz (asumen euclidiana).
    """
    from sklearn.preprocessing import StandardScaler

    num_crudas = df[NUMERICAS].to_numpy(dtype=np.float64)
    num_std = StandardScaler().fit_transform(num_crudas)

    cat_series = df[CATEGORICA].astype("category")
    cat_cod = cat_series.cat.codes.to_numpy(dtype=np.int32)
    cat_nombres = list(cat_series.cat.categories)

    onehot = np.zeros((len(df), len(cat_nombres)), dtype=np.float64)
    onehot[np.arange(len(df)), cat_cod] = 1.0

    X_cod = np.hstack([num_std, onehot])
    cols = NUMERICAS + [f"mod={c}" for c in cat_nombres]
    return num_crudas, num_std, cat_cod, X_cod, cols, cat_nombres


# --------------------------------------------------------------- Gower
def gower_matrix(num_crudas, cat_cod, chunk=256, dtype=np.float32):
    """Matriz de Gower completa (n x n).

    Numéricas: |xi-xj| / rango_j.  Categórica: 0 si igual, 1 si distinta.
    Distancia = promedio simple sobre las 6 variables (pesos iguales).
    """
    n, p_num = num_crudas.shape
    p = p_num + 1
    rng = num_crudas.max(axis=0) - num_crudas.min(axis=0)
    rng[rng == 0] = 1.0
    Z = (num_crudas / rng).astype(np.float32)
    cat = cat_cod.astype(np.int32)

    D = np.zeros((n, n), dtype=dtype)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        blk = np.abs(Z[i0:i1, None, :] - Z[None, :, :]).sum(axis=2)
        blk += (cat[i0:i1, None] != cat[None, :]).astype(np.float32)
        D[i0:i1] = blk / p
    np.fill_diagonal(D, 0.0)
    return D


def condensada(D):
    """Forma condensada float64 (para scipy.linkage) sin materializar una copia n^2."""
    n = D.shape[0]
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    pos = 0
    for i in range(n - 1):
        m = n - i - 1
        out[pos:pos + m] = D[i, i + 1:]
        pos += m
    return out


# --------------------------------------------------------------- PAM propio
def _build_init(D, k, chunk=512):
    """Inicialización BUILD de PAM: medoides que más reducen el costo total."""
    n = D.shape[0]
    # primer medoide: el de menor suma de distancias a todos
    sumas = np.zeros(n, dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        sumas[i0:i1] = D[i0:i1].sum(axis=1, dtype=np.float64)
    medoides = [int(np.argmin(sumas))]
    dmin = D[medoides[0]].astype(np.float64)

    while len(medoides) < k:
        mejor_c, mejor_g = -1, -np.inf
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            # ganancia de tomar c como medoide = sum_j max(0, dmin_j - D[c,j])
            g = np.maximum(0.0, dmin[None, :] - D[i0:i1]).sum(axis=1, dtype=np.float64)
            g[[m - i0 for m in medoides if i0 <= m < i1]] = -np.inf
            j = int(np.argmax(g))
            if g[j] > mejor_g:
                mejor_g, mejor_c = float(g[j]), i0 + j
        medoides.append(mejor_c)
        dmin = np.minimum(dmin, D[mejor_c])
    return np.array(medoides, dtype=np.int64)


def pam(D, k, max_iter=100, seed=SEED):
    """K-Medoids sobre matriz precalculada.

    Init BUILD de PAM + refinamiento alternante (Voronoi): asignar al medoide más
    cercano y reemplazar cada medoide por el punto de su grupo que minimiza la suma
    de distancias intra-grupo. Los medoides son procesos reales del conjunto.

    NOTA: es la variante alternante, no el SWAP exhaustivo de PAM clásico, que es
    O(k(n-k)^2) por iteración e intratable con n=13.848.
    """
    n = D.shape[0]
    medoides = _build_init(D, k)
    etiquetas = np.argmin(D[medoides], axis=0).astype(np.int32)
    for _ in range(max_iter):
        nuevos = medoides.copy()
        for c in range(k):
            idx = np.flatnonzero(etiquetas == c)
            if idx.size == 0:
                continue
            sub = D[np.ix_(idx, idx)]
            nuevos[c] = idx[int(np.argmin(sub.sum(axis=0, dtype=np.float64)))]
            del sub
        nuevos = np.unique(nuevos)
        if nuevos.size < k:   # medoides colapsados: completar con los más lejanos
            faltan = k - nuevos.size
            dmin = D[nuevos].min(axis=0)
            extra = np.argsort(-dmin)[:faltan]
            nuevos = np.unique(np.concatenate([nuevos, extra]))
        nuevas_etq = np.argmin(D[nuevos], axis=0).astype(np.int32)
        if np.array_equal(nuevos, medoides) and np.array_equal(nuevas_etq, etiquetas):
            medoides, etiquetas = nuevos, nuevas_etq
            break
        medoides, etiquetas = nuevos, nuevas_etq
    return etiquetas, medoides


# --------------------------------------------------------------- métricas
def silueta_gower(D, etiquetas, excluir_ruido=True):
    """Silueta sobre la matriz de Gower (metric='precomputed').

    Devuelve (silueta, n_usados, n_grupos_usados). Si excluir_ruido, quita -1.
    """
    from sklearn.metrics import silhouette_score

    etq = np.asarray(etiquetas)
    if excluir_ruido:
        mask = etq != -1
    else:
        mask = np.ones(len(etq), dtype=bool)
    sub_etq = etq[mask]
    grupos = np.unique(sub_etq)
    if grupos.size < 2 or sub_etq.size < 3:
        return np.nan, int(mask.sum()), int(grupos.size)
    idx = np.flatnonzero(mask)
    if idx.size == D.shape[0]:
        Dm = D
    else:
        Dm = D[np.ix_(idx, idx)]
    s = float(silhouette_score(Dm, sub_etq, metric="precomputed"))
    if Dm is not D:
        del Dm
    return s, int(idx.size), int(grupos.size)


def db_ch(X_cod, etiquetas, excluir_ruido=True):
    """Davies-Bouldin y Calinski-Harabasz sobre la MATRIZ CODIFICADA (euclidiana)."""
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

    etq = np.asarray(etiquetas)
    mask = etq != -1 if excluir_ruido else np.ones(len(etq), dtype=bool)
    sub_etq, sub_X = etq[mask], X_cod[mask]
    if np.unique(sub_etq).size < 2:
        return np.nan, np.nan
    return (float(davies_bouldin_score(sub_X, sub_etq)),
            float(calinski_harabasz_score(sub_X, sub_etq)))


def cobertura(etiquetas):
    """% de procesos asignados a un grupo real (no ruido)."""
    etq = np.asarray(etiquetas)
    return 100.0 * float((etq != -1).sum()) / len(etq)


def dispersion_intra(D, etiquetas):
    """Dispersión intra-grupo W(k) sobre Gower: sum_c (1/(2|C|)) * sum_{i,j in C} d_ij.

    Medida comparable entre algoritmos para la curva del codo (no depende de
    centroides, sólo de la partición y de la distancia árbitro).
    """
    etq = np.asarray(etiquetas)
    total = 0.0
    for c in np.unique(etq):
        if c == -1:
            continue
        idx = np.flatnonzero(etq == c)
        if idx.size < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        total += float(sub.sum(dtype=np.float64)) / (2.0 * idx.size)
        del sub
    return total


# --------------------------------------------------------------- estabilidad
def estabilidad_ari(fit_fn, n, ref_etiquetas, n_rep=20, frac=0.8, seed=SEED):
    """ARI entre la partición de referencia (ajuste completo) y 20 reejecuciones
    sobre submuestras del 80%, comparadas sobre los puntos de cada submuestra.

    fit_fn(idx) -> etiquetas para la submuestra idx.
    Devuelve (media, desviación, lista).
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    ref = np.asarray(ref_etiquetas)
    aris = []
    m = int(round(frac * n))
    for r in range(n_rep):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        etq = fit_fn(idx)
        if etq is None:
            continue
        aris.append(float(adjusted_rand_score(ref[idx], etq)))
    if not aris:
        return np.nan, np.nan, []
    return float(np.mean(aris)), float(np.std(aris)), aris


class Cronometro:
    def __init__(self):
        self.t = {}

    def __call__(self, nombre):
        return _Ctx(self, nombre)


class _Ctx:
    def __init__(self, cron, nombre):
        self.cron, self.nombre = cron, nombre

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.cron.t[self.nombre] = round(time.time() - self.t0, 2)
        return False
