# -*- coding: utf-8 -*-
"""
FASE 2.1 - Paso 3: REVALIDACIÓN DE LA COMPARACIÓN DEL PORTAFOLIO.

Compara, en condiciones idénticas (misma muestra, mismas variables, misma
métrica árbitro), seis familias de agrupamiento sobre la matriz de interacción
proveedor x procesos_activos:

    1) K-Prototypes   (kmodes)            sobre X_mixta
    2) K-Means        (scikit-learn)      sobre X_codificada
    3) Jerárquico aglomerativo (average)  sobre G (Gower precalculada)
    4) K-Medoids / PAM                    sobre G
    5) DBSCAN / HDBSCAN (densidad)        sobre G
    6) Gaussian Mixture                   sobre X_codificada

CAMBIOS DE LA FASE 2.1 respecto de la v1 (resultados/fase2_v1/):
  - Proveedores de referencia por PERFIL COMPETITIVO y de TRES provincias
    distintas, no por volumen bruto (antes salían 3 gigantes de catálogo de
    Pichincha y las variables quedaban degeneradas).
  - cpc_match (binaria, constante en 0) -> cpc_jaccard4 (Jaccard continuo sobre
    prefijos CPC de 4 dígitos).
  - Nueva variable modalidad_afinidad (continua en [0,1]).
  - TF-IDF con sublinear_tf=True y min_df=2.
  - Winsorización al percentil 1/99 antes de estandarizar.
  - Gower PONDERADA: peso 1.0 a cada una de las 7 numéricas y 1/3 a la
    categórica, para que modalidad_norm deje de dominar la distancia.
  - Reglas de descalificación: copia de una columna (ARI>0.9), líneas base de
    control (aleatoria y modalidad trivial), entropía de tamaños y cobertura.

MÉTRICA ÁRBITRO ÚNICA: silhouette_score(G_ponderada, etiquetas,
metric='precomputed'). Davies-Bouldin y Calinski-Harabasz sobre X_codificada
como métricas de referencia.

Los tres proveedores de referencia se anonimizan como P1/P2/P3: nunca se
imprime ni se guarda su RUC ni su razón social.
"""
import json
import math
import os
import time
import unicodedata
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

warnings.filterwarnings("ignore")

SEED = 42
N_MUESTRA = 3000
RANGO_K = list(range(3, 11))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "resultados")

# [FASE 2.2] 6 numéricas. Salieron log_presupuesto (r=1.0000 con
# desviacion_presupuesto) y modalidad_afinidad (eta^2=1.000 contra
# modalidad_norm); entró afinidad_comprador, que mide la relación histórica
# proveedor-comprador y no depende de la modalidad.
NUMERICAS = ["distancia_km", "cpc_jaccard4", "sim_tfidf",
             "desviacion_presupuesto", "actividad_cpc_comprador",
             "afinidad_comprador"]
CATEGORICA = "modalidad_norm"
# log_presupuesto se conserva en el DataFrame como columna de referencia
# (cuartiles y centroides desestandarizados) pero YA NO es variable del modelo.
AUXILIARES = ["log_presupuesto"]

# [FASE 2.1] Gower ponderada: 1.0 por numérica, 1/3 para la categórica
PESO_CATEGORICA = 1.0 / 3.0
PESOS_GOWER = np.array([1.0] * len(NUMERICAS) + [PESO_CATEGORICA])
UMBRAL_CORRELACION = 0.95          # [FASE 2.2] aviso de redundancia

# [FASE 2.1] criterios de elegibilidad, ajustados en la Fase 2.2
MODALIDADES_COMPETITIVAS = {"Subasta Inversa Electrónica", "Menor Cuantía",
                            "Cotización", "Licitación", "Contratacion directa"}
MAX_PCT_CATALOGO = 0.60
MIN_ADJ_COMPETITIVAS = 30          # [FASE 2.2] antes 50
MIN_CPC_DISTINTOS = 5              # [FASE 2.2] descarta monotemáticos
PROVINCIAS_PRIORITARIAS = ["GUAYAS", "PICHINCHA"]
NOMBRE_CONTINUIDAD = "ROCHE ECUADOR S.A."   # [FASE 2.2] corrida de continuidad

# [FASE 2.1] umbrales de descalificación
ARI_MAX_COPIA = 0.90
ENTROPIA_MINIMA = 0.50
COBERTURA_MINIMA = 85.0
PCT_WINSOR = (1.0, 99.0)

# --- capitales de las 24 provincias del Ecuador (lat, lon) -------------------
# claves en MAYÚSCULAS y SIN TILDES, tal como llegan normalizadas desde OCDS
CAPITALES = {
    "AZUAY":                          (-2.9001, -79.0059),   # Cuenca
    "BOLIVAR":                        (-1.5905, -79.0007),   # Guaranda
    "CANAR":                          (-2.7396, -78.8484),   # Azogues
    "CARCHI":                         (0.8117, -77.7178),    # Tulcan
    "CHIMBORAZO":                     (-1.6635, -78.6546),   # Riobamba
    "COTOPAXI":                       (-0.9333, -78.6167),   # Latacunga
    "EL ORO":                         (-3.2581, -79.9554),   # Machala
    "ESMERALDAS":                     (0.9592, -79.6539),    # Esmeraldas
    "GALAPAGOS":                      (-0.9020, -89.6100),   # Pto. Baquerizo Moreno
    "GUAYAS":                         (-2.1709, -79.9224),   # Guayaquil
    "IMBABURA":                       (0.3517, -78.1223),    # Ibarra
    "LOJA":                           (-3.9931, -79.2042),   # Loja
    "LOS RIOS":                       (-1.8022, -79.5344),   # Babahoyo
    "MANABI":                         (-1.0546, -80.4545),   # Portoviejo
    "MORONA SANTIAGO":                (-2.3086, -78.1170),   # Macas
    "NAPO":                           (-0.9938, -77.8129),   # Tena
    "ORELLANA":                       (-0.4625, -76.9868),   # Pto. Fco. de Orellana
    "PASTAZA":                        (-1.4869, -78.0031),   # Puyo
    "PICHINCHA":                      (-0.1807, -78.4678),   # Quito
    "SANTA ELENA":                    (-2.2267, -80.8583),   # Santa Elena
    "SANTO DOMINGO DE LOS TSACHILAS": (-0.2542, -79.1750),   # Santo Domingo
    "SUCUMBIOS":                      (0.0869, -76.8934),    # Nueva Loja
    "TUNGURAHUA":                     (-1.2417, -78.6197),   # Ambato
    "ZAMORA CHINCHIPE":               (-4.0692, -78.9567),   # Zamora
}

NOMBRES_ALGORITMOS = {
    "kprototypes": "K-Prototypes",
    "kmeans": "K-Means",
    "jerarquico": "Jerárquico (average)",
    "kmedoids": "K-Medoids (PAM)",
    "densidad": "DBSCAN/HDBSCAN",
    "gmm": "Gaussian Mixture",
}
ORDEN_ALGORITMOS = ["kprototypes", "kmeans", "jerarquico",
                    "kmedoids", "densidad", "gmm"]

INCIDENCIAS = []


def anotar(msg):
    INCIDENCIAS.append(msg)
    print(f"    [!] {msg}", flush=True)


def sin_tildes_mayus(valor):
    if not valor or not isinstance(valor, str):
        return None
    s = unicodedata.normalize("NFD", valor.strip().upper())
    return "".join(c for c in s if unicodedata.category(c) != "Mn") or None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ============================================================ carga de datos
def cargar():
    perfiles = pd.read_parquet(os.path.join(RES, "perfiles_proveedores.parquet"))
    procesos = pd.read_parquet(os.path.join(RES, "procesos_activos.parquet"))
    actividad = pd.read_parquet(os.path.join(RES, "actividad_buyer_cpc.parquet"))
    prov_buyer = pd.read_parquet(os.path.join(RES, "proveedor_buyer.parquet"))
    with open(os.path.join(RES, "categorias_modalidad.json"), encoding="utf-8") as f:
        categorias = json.load(f)
    return perfiles, procesos, actividad, prov_buyer, categorias


def indice_proveedor_buyer(prov_buyer):
    """[FASE 2.2] proveedor_id -> buyer_id -> nº de adjudicaciones históricas."""
    idx = defaultdict(dict)
    for p, b, n in zip(prov_buyer["proveedor_id"].to_numpy(),
                       prov_buyer["buyer_id"].to_numpy(),
                       prov_buyer["n_awards"].to_numpy()):
        idx[p][b] = int(n)
    return idx


def es_catalogo(modalidad):
    if not modalidad:
        return False
    s = unicodedata.normalize("NFD", str(modalidad).strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return s.startswith("catalogo")


def normalizar_modalidad(valor, conservadas):
    """MISMA regla que extraer_activos.py, aplicada al histórico del proveedor."""
    if es_catalogo(valor):
        return "Catalogo Electronico"
    v = valor if valor else "Otros"
    return v if v in conservadas else "Otros"


def indice_actividad(actividad):
    """buyer_id -> cpc -> conjunto de award_uid (evita contar dos veces un award)."""
    idx = defaultdict(lambda: defaultdict(set))
    for b, c, u in zip(actividad["buyer_id"].to_numpy(),
                       actividad["cpc"].to_numpy(),
                       actividad["award_uid"].to_numpy()):
        idx[b][c].add(int(u))
    return idx


def elegir_proveedores(perfiles, conservadas, n=3):
    """[FASE 2.1] Selección por PERFIL COMPETITIVO y diversidad geográfica.

    ELEGIBLE: <60 % de sus adjudicaciones en Catálogo Electrónico, >=30
    adjudicaciones en modalidades competitivas, >=5 códigos CPC distintos en el
    historial (evita monotemáticos como el P1 de la Fase 2.1, que tenía 208
    adjudicaciones con sólo 2 CPC), provincia válida y corpus no vacío. Se toma
    el mejor elegible (por adjudicaciones competitivas) de tres PROVINCIAS
    DISTINTAS, priorizando GUAYAS y PICHINCHA.
    """
    filas = []
    for idx, r in perfiles.iterrows():
        if (r["provincia"] not in CAPITALES or r["len_corpus"] <= 0
                or pd.isna(r["monto_promedio_ganado"])
                or r["n_cpc_historicos"] < MIN_CPC_DISTINTOS):
            continue
        cruda = json.loads(r["modalidades_hist_json"])
        total = sum(cruda.values())
        if not total:
            continue
        n_catalogo = sum(v for k, v in cruda.items() if es_catalogo(k))
        n_comp = sum(v for k, v in cruda.items()
                     if k in MODALIDADES_COMPETITIVAS)
        pct_cat = n_catalogo / total
        if pct_cat >= MAX_PCT_CATALOGO or n_comp < MIN_ADJ_COMPETITIVAS:
            continue
        norm = Counter()
        for k, v in cruda.items():
            norm[normalizar_modalidad(k, conservadas)] += v
        filas.append({"idx": idx, "provincia": r["provincia"],
                      "adj_competitivas": n_comp, "pct_catalogo": pct_cat,
                      "num_adjudicaciones": int(r["num_adjudicaciones"]),
                      "modalidades_norm": dict(norm), "total_hist": total})

    elegibles = pd.DataFrame(filas)
    if elegibles.empty:
        return elegibles, 0
    elegibles = elegibles.sort_values("adj_competitivas", ascending=False)

    escogidos, usadas = [], set()
    for prov in PROVINCIAS_PRIORITARIAS:               # GUAYAS y PICHINCHA primero
        sub = elegibles[elegibles["provincia"] == prov]
        if len(sub) and len(escogidos) < n:
            escogidos.append(sub.iloc[0])
            usadas.add(prov)
    for _, fila in elegibles.iterrows():               # resto, provincias nuevas
        if len(escogidos) >= n:
            break
        if fila["provincia"] not in usadas:
            escogidos.append(fila)
            usadas.add(fila["provincia"])

    sel = pd.DataFrame(escogidos).sort_values("adj_competitivas",
                                              ascending=False)
    sel = sel.reset_index(drop=True)
    sel = sel.join(perfiles.loc[sel["idx"]].reset_index(drop=True)
                   .drop(columns=["provincia", "num_adjudicaciones"]))
    return sel, len(elegibles)


# =================================================== matriz de interacción
def construir_matriz(perfil, procesos, idx_act, idx_pb):
    n = len(procesos)

    # --- distancia_km (Haversine entre capitales provinciales) --------------
    lat_p, lon_p = CAPITALES[perfil["provincia"]]
    d = np.full(n, np.nan)
    provs = procesos["provincia_buyer"].to_numpy()
    for i, pr in enumerate(provs):
        coord = CAPITALES.get(pr) if isinstance(pr, str) else None
        if coord is not None:
            d[i] = haversine_km(lat_p, lon_p, coord[0], coord[1])
    conocidas = ~np.isnan(d)
    mediana = float(np.median(d[conocidas])) if conocidas.any() else 0.0
    n_desconocidas = int((~conocidas).sum())
    d[~conocidas] = mediana

    # --- cpc_jaccard4 [FASE 2.1] -------------------------------------------
    # Jaccard entre prefijos CPC de 4 dígitos. Reemplaza al cpc_match binario,
    # que resultó constante en 0 porque exigía coincidencia de código completo.
    cpc_hist = set(str(perfil["cpc_historicos"]).split("|")) - {""}
    pref_hist = {c[:4] for c in cpc_hist if len(c) >= 4}
    cpc_jac = np.zeros(n, dtype=float)
    listas_cpc = []
    for i, s in enumerate(procesos["cpc_tender"].to_numpy()):
        cs = set(str(s).split("|")) - {""} if s else set()
        listas_cpc.append(cs)
        pref_proc = {c[:4] for c in cs if len(c) >= 4}
        if pref_hist and pref_proc:
            union = len(pref_hist | pref_proc)
            cpc_jac[i] = len(pref_hist & pref_proc) / union if union else 0.0

    # --- sim_tfidf ----------------------------------------------------------
    textos = list(procesos["texto_items"].fillna("").astype(str))
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True, min_df=2)
    M = vec.fit_transform(textos + [str(perfil["corpus_items"])])
    sim = cosine_similarity(M[-1], M[:-1]).ravel().astype(float)

    # --- presupuesto --------------------------------------------------------
    log_pres = np.log1p(procesos["presupuesto"].to_numpy(dtype=float))
    log_medio_prov = float(np.log1p(float(perfil["monto_promedio_ganado"])))
    desv = log_pres - log_medio_prov

    # --- actividad_cpc_comprador -------------------------------------------
    act = np.zeros(n, dtype=float)
    for i, (b, cs) in enumerate(zip(procesos["buyer_id"].to_numpy(), listas_cpc)):
        por_cpc = idx_act.get(b)
        if not por_cpc or not cs:
            continue
        awards = set()
        for c in cs:
            s = por_cpc.get(c)
            if s:
                awards |= s
        act[i] = len(awards)

    # --- afinidad_comprador [FASE 2.2] -------------------------------------
    # log1p del nº de adjudicaciones históricas (24+25) del PROVEEDOR con ESE
    # comprador. Sustituye a modalidad_afinidad: mide la relación bilateral y
    # no es función de modalidad_norm.
    por_buyer = idx_pb.get(perfil["proveedor_id"], {})
    afin_comp = np.array([np.log1p(por_buyer.get(b, 0))
                          for b in procesos["buyer_id"].to_numpy()], dtype=float)

    X = pd.DataFrame({
        "distancia_km": d,
        "cpc_jaccard4": cpc_jac,
        "sim_tfidf": sim,
        "desviacion_presupuesto": desv,
        "actividad_cpc_comprador": act,
        "afinidad_comprador": afin_comp,
        "log_presupuesto": log_pres,
        CATEGORICA: procesos[CATEGORICA].to_numpy(),
    })
    meta = {
        "provincias_desconocidas": n_desconocidas,
        "mediana_distancia": mediana,
        "pct_cpc_jaccard_no_cero": 100.0 * float((cpc_jac > 0).mean()),
        "cpc_jaccard_medio": float(cpc_jac.mean()),
        "sim_media": float(sim.mean()),
        # afinidad_comprador [FASE 2.2]
        "pct_afin_comp_no_cero": 100.0 * float((afin_comp > 0).mean()),
        "afin_comp_media": float(afin_comp.mean()),
        "afin_comp_max": float(afin_comp.max()),
        "n_compradores_conocidos": int((afin_comp > 0).sum()),
    }
    return X, meta


def winsorizar(num, pcts=PCT_WINSOR):
    """[FASE 2.1] Recorta cada columna a sus percentiles 1/99 antes de escalar.

    Devuelve (matriz recortada, nº de recortes por columna, límites por columna).
    Los límites se guardan en modelo_ganador.pkl: la Fase 3 debe aplicar el
    MISMO recorte a los datos nuevos antes de estandarizar.
    """
    salida = num.copy()
    recortes, limites = [], []
    for j in range(num.shape[1]):
        lo, hi = np.percentile(num[:, j], pcts)
        recortes.append(int(((num[:, j] < lo) | (num[:, j] > hi)).sum()))
        limites.append((float(lo), float(hi)))
        salida[:, j] = np.clip(num[:, j], lo, hi)
    return salida, recortes, limites


def matriz_correlacion(X, alias, umbral=UMBRAL_CORRELACION):
    """[FASE 2.2] Correlación de Pearson entre las numéricas + aviso de redundancia."""
    M = X[NUMERICAS].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(M, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    avisos = []
    for j in range(len(NUMERICAS)):
        for l in range(j + 1, len(NUMERICAS)):
            if abs(C[j, l]) >= umbral:
                avisos.append(f"{alias}: '{NUMERICAS[j]}' y '{NUMERICAS[l]}' "
                              f"con |r|={abs(C[j, l]):.4f} >= {umbral}: "
                              f"esa dimensión pesa de más en la Gower")
    return C, avisos


def imprimir_correlacion(C, alias):
    ancho = max(len(c) for c in NUMERICAS) + 1
    print(f"    matriz de correlación de las numéricas ({alias}):")
    print(" " * (ancho + 6) + " ".join(f"{i + 1:>7}" for i in range(len(NUMERICAS))))
    for i, c in enumerate(NUMERICAS):
        fila = " ".join(f"{C[i, j]:7.3f}" for j in range(len(NUMERICAS)))
        print(f"      {i + 1}. {c:<{ancho}} {fila}")


def preparar_vistas(X):
    """X_mixta (num estandarizadas + categórica), X_codificada (num + one-hot), G."""
    crudas = X[NUMERICAS].to_numpy(dtype=float)
    crudas_w, recortes, limites = winsorizar(crudas)
    escalador = StandardScaler()
    num_std = escalador.fit_transform(crudas_w)
    cat = X[CATEGORICA].astype(str).to_numpy()

    X_mixta = np.empty((len(X), len(NUMERICAS) + 1), dtype=object)
    X_mixta[:, :len(NUMERICAS)] = num_std
    X_mixta[:, len(NUMERICAS)] = cat

    categorias = sorted(pd.unique(cat))
    onehot = np.zeros((len(X), len(categorias)), dtype=float)
    pos = {c: j for j, c in enumerate(categorias)}
    for i, c in enumerate(cat):
        onehot[i, pos[c]] = 1.0
    X_cod = np.hstack([num_std, onehot])

    G = matriz_gower(num_std, cat)
    return (escalador, num_std, cat, X_mixta, X_cod, categorias, G,
            crudas_w, recortes, limites)


def matriz_gower(num_std, cat, pesos=PESOS_GOWER):
    """[FASE 2.1] Gower PONDERADA: 1.0 por numérica, 1/3 para la categórica.

    gower 0.1.2 acepta `weight` y normaliza por la suma de pesos; se verificó
    que _gower_propio reproduce su salida con y sin pesos (dif. máx. 3e-8, que
    es la precisión de float32). Si la librería falla se usa el respaldo propio.
    """
    datos = np.empty((num_std.shape[0], num_std.shape[1] + 1), dtype=object)
    datos[:, :num_std.shape[1]] = num_std
    datos[:, num_std.shape[1]] = cat
    mascara = np.array([False] * num_std.shape[1] + [True])
    try:
        import gower
        G = gower.gower_matrix(datos, weight=np.asarray(pesos, dtype=float),
                               cat_features=mascara)
        G = np.asarray(G, dtype=np.float32)
    except Exception as e:            # pragma: no cover
        anotar(f"gower.gower_matrix falló ({type(e).__name__}: {e}); "
               f"se usa la implementación propia ponderada de respaldo")
        G = _gower_propio(num_std, cat, pesos)
    G = 0.5 * (G + G.T)
    np.fill_diagonal(G, 0.0)
    return G


def rangos_gower(num_std):
    """[FASE 3] Rangos por columna con los que Gower normaliza las numéricas.

    Se guardan en modelo_ganador.pkl: para asignar procesos nuevos hay que usar
    LOS MISMOS rangos del entrenamiento, no los de los datos nuevos.
    """
    r = num_std.max(axis=0) - num_std.min(axis=0)
    r = np.asarray(r, dtype=float)
    r[r == 0] = 1.0
    return r


def distancia_gower_a_referencias(num_std, cat, ref_num, ref_cat, rangos,
                                  pesos=PESOS_GOWER):
    """[FASE 3] Distancia de Gower ponderada de cada fila a cada fila de
    referencia (los medoides). Misma fórmula que _gower_propio, verificada
    contra gower.gower_matrix con y sin pesos.

    Devuelve una matriz (n_filas x n_referencias).
    """
    w = np.asarray(pesos, dtype=float)
    p = num_std.shape[1]
    ref_num = np.atleast_2d(ref_num)
    ref_cat = np.atleast_1d(ref_cat)
    D = np.zeros((num_std.shape[0], ref_num.shape[0]), dtype=float)
    for j in range(ref_num.shape[0]):
        d = (np.abs(num_std - ref_num[j]) / rangos * w[:p]).sum(axis=1)
        d += (np.asarray(cat) != ref_cat[j]).astype(float) * w[p]
        D[:, j] = d / w.sum()
    return D


def _gower_propio(num, cat, pesos=PESOS_GOWER, chunk=256):
    """Gower ponderada de respaldo: sum_j w_j * d_j / sum_j w_j."""
    n, p_num = num.shape
    w = np.asarray(pesos, dtype=np.float32)
    rng = num.max(axis=0) - num.min(axis=0)
    rng[rng == 0] = 1.0
    Z = (num / rng).astype(np.float32)
    codigos = pd.Categorical(cat).codes.astype(np.int32)
    D = np.zeros((n, n), dtype=np.float32)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        blk = (np.abs(Z[i0:i1, None, :] - Z[None, :, :]) * w[:p_num]).sum(axis=2)
        blk += (codigos[i0:i1, None] != codigos[None, :]).astype(np.float32) * w[p_num]
        D[i0:i1] = blk / w.sum()
    return D


# ================================================================ métricas
def silueta_gower(G, etiquetas):
    """Silueta árbitro. Excluye ruido (-1). Devuelve (silueta, n_grupos, pct_ruido)."""
    etq = np.asarray(etiquetas)
    mask = etq != -1
    pct_ruido = 100.0 * float((~mask).sum()) / len(etq)
    sub = etq[mask]
    grupos = np.unique(sub)
    if grupos.size < 2 or pct_ruido > 50.0:
        return np.nan, int(grupos.size), pct_ruido
    if mask.all():
        Gm = G
    else:
        i = np.flatnonzero(mask)
        Gm = G[np.ix_(i, i)]
    s = float(silhouette_score(Gm, sub, metric="precomputed"))
    return s, int(grupos.size), pct_ruido


def db_ch(X_cod, etiquetas):
    etq = np.asarray(etiquetas)
    mask = etq != -1
    sub, Xs = etq[mask], X_cod[mask]
    if np.unique(sub).size < 2:
        return np.nan, np.nan
    return (float(davies_bouldin_score(Xs, sub)),
            float(calinski_harabasz_score(Xs, sub)))


def entropia_normalizada(etiquetas):
    """[FASE 2.1] Entropía de la distribución de tamaños, en [0,1].

    1 = todos los grupos del mismo tamaño; ~0 = casi todo en un solo grupo.
    Se calcula sobre los grupos reales (el ruido no cuenta como grupo).
    """
    etq = np.asarray(etiquetas)
    sub = etq[etq != -1]
    _, cuentas = np.unique(sub, return_counts=True)
    if cuentas.size < 2:
        return 0.0
    p = cuentas / cuentas.sum()
    return float(-(p * np.log(p)).sum() / math.log(cuentas.size))


def cobertura(etiquetas):
    """[FASE 2.1] % de procesos asignados a un grupo real (excluye ruido)."""
    etq = np.asarray(etiquetas)
    return 100.0 * float((etq != -1).sum()) / len(etq)


class Evaluador:
    """[FASE 2.1] Evalúa una partición y aplica las reglas de descalificación.

    Líneas base de control (protocolo del documento): todo candidato debe
    SUPERAR la silueta de (i) etiquetas aleatorias con el mismo nº de grupos y
    (ii) el agrupamiento trivial por modalidad_norm. Las aleatorias se
    precalculan una vez por proveedor para cada nº de grupos posible.
    """

    def __init__(self, alias, G, X_cod, cat, log_presupuesto):
        self.alias = alias
        self.G, self.X_cod, self.cat = G, X_cod, cat
        # partición de referencia por cuartil de presupuesto
        self.cuartil = pd.qcut(log_presupuesto, 4, labels=False,
                               duplicates="drop").astype(int)
        cod_mod = pd.Categorical(cat).codes
        self.base_modalidad, _, _ = silueta_gower(G, cod_mod)
        rng = np.random.default_rng(SEED)
        self.bases_aleatorias = {}
        for k in range(2, 21):
            etq = rng.integers(0, k, size=len(cat))
            if np.unique(etq).size < 2:
                continue
            s, _, _ = silueta_gower(G, etq)
            self.bases_aleatorias[k] = s

    def base_aleatoria(self, n_grupos):
        if n_grupos in self.bases_aleatorias:
            return self.bases_aleatorias[n_grupos]
        disponibles = [k for k in self.bases_aleatorias
                       if not np.isnan(self.bases_aleatorias[k])]
        if not disponibles:
            return np.nan
        cercano = min(disponibles, key=lambda k: abs(k - n_grupos))
        return self.bases_aleatorias[cercano]

    def registrar(self, etq, nota, segundos, extra=None):
        etq = np.asarray(etq, dtype=int)
        s, ng, ruido = silueta_gower(self.G, etq)
        db, ch = db_ch(self.X_cod, etq)
        ari_mod = ari_contra(etq, self.cat)
        ari_cua = ari_contra(etq, self.cuartil)
        ent = entropia_normalizada(etq)
        cob = cobertura(etq)
        base_al = self.base_aleatoria(ng)

        motivos = []
        if not np.isnan(ari_mod) and ari_mod > ARI_MAX_COPIA:
            motivos.append(f"copia modalidad_norm (ARI={ari_mod:.3f})")
        if not np.isnan(ari_cua) and ari_cua > ARI_MAX_COPIA:
            motivos.append(f"copia cuartil de presupuesto (ARI={ari_cua:.3f})")
        if np.isnan(s):
            motivos.append("silueta no calculable (<2 grupos o >50 % ruido)")
        else:
            if not np.isnan(base_al) and s <= base_al:
                motivos.append(f"no supera la línea base aleatoria "
                               f"({s:.4f} <= {base_al:.4f})")
            if (not np.isnan(self.base_modalidad)
                    and s <= self.base_modalidad):
                motivos.append(f"no supera la línea base modalidad "
                               f"({s:.4f} <= {self.base_modalidad:.4f})")
        if ent < ENTROPIA_MINIMA:
            motivos.append(f"entropía de tamaños {ent:.3f} < {ENTROPIA_MINIMA}")
        if cob < COBERTURA_MINIMA:
            motivos.append(f"cobertura {cob:.1f} % < {COBERTURA_MINIMA} %")

        r = {"silueta": s, "db": db, "ch": ch, "n_grupos": ng,
             "ruido": ruido, "notas": nota, "seg": segundos,
             "ari_mod": ari_mod, "ari_cuartil": ari_cua,
             "entropia": ent, "cobertura": cob,
             "base_aleatoria": base_al, "base_modalidad": self.base_modalidad,
             "descalificado": bool(motivos), "motivos": "; ".join(motivos),
             "etq": etq}
        if extra:
            r.update(extra)
        return r


# ============================================================== algoritmos
def correr_kprototypes(X_mixta, k):
    from kmodes.kprototypes import KPrototypes
    idx_cat = [X_mixta.shape[1] - 1]
    modelo = KPrototypes(n_clusters=k, init="Cao", n_init=10, verbose=0,
                         random_state=SEED, n_jobs=1)
    etq = modelo.fit_predict(X_mixta, categorical=idx_cat)
    return np.asarray(etq, dtype=int), f"gamma={modelo.gamma:.4f}", modelo


def correr_kmeans(X_cod, k):
    m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
    return m.fit_predict(X_cod), "", m


def correr_jerarquico(G, k):
    m = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                linkage="average")
    return m.fit_predict(G.astype(np.float64)), "", m


def correr_kmedoids(G, k):
    """scikit-learn-extra no importa con numpy 2.x (ABI rota); se usa el paquete
    `kmedoids` (FasterPAM, PAM exacto sobre matriz precalculada).

    Devuelve también los índices de los medoides: son procesos REALES de la
    muestra, así que la Fase 2.2 puede reportar su ocid.
    """
    import kmedoids as km
    r = km.KMedoids(n_clusters=k, method="fasterpam", metric="precomputed",
                    random_state=SEED, max_iter=300)
    r.fit(G.astype(np.float64))
    medoides = np.asarray(r.medoid_indices_, dtype=int)
    return np.asarray(r.labels_, dtype=int), "FasterPAM", medoides, r


def correr_dbscan(G):
    """Barrido de eps en los percentiles 5/10/15/20 de la distancia al 10º vecino."""
    Gd = G.astype(np.float64)
    k_vecino = 10
    d_knn = np.sort(Gd, axis=1)[:, k_vecino]
    resultados = []
    for pct in (5, 10, 15, 20):
        eps = float(np.percentile(d_knn, pct))
        if eps <= 0:
            resultados.append((f"eps=p{pct}({eps:.4f})", None,
                               "eps=0, configuración descartada"))
            continue
        etq = DBSCAN(eps=eps, min_samples=10, metric="precomputed").fit_predict(Gd)
        resultados.append((f"DBSCAN eps=p{pct}={eps:.4f}", etq, ""))
    return resultados


def correr_hdbscan(G):
    Gd = G.astype(np.float64)
    try:
        import hdbscan as hdb
        m = hdb.HDBSCAN(metric="precomputed", min_cluster_size=30)
        etq = np.asarray(m.fit_predict(Gd), dtype=int)
        if np.unique(etq[etq != -1]).size >= 2:
            return [("HDBSCAN(paquete hdbscan) mcs=30", etq, "")]
        anotar("hdbscan (paquete) devolvió <2 grupos sobre Gower precalculada; "
               "se prueba sklearn.cluster.HDBSCAN")
    except Exception as e:
        anotar(f"paquete hdbscan falló ({type(e).__name__}: {e}); "
               f"se usa sklearn.cluster.HDBSCAN")
    from sklearn.cluster import HDBSCAN as SKHDBSCAN
    m = SKHDBSCAN(metric="precomputed", min_cluster_size=30, copy=True)
    etq = np.asarray(m.fit_predict(Gd), dtype=int)
    return [("HDBSCAN(sklearn) mcs=30", etq, "")]


def correr_gmm(X_cod, k):
    m = GaussianMixture(n_components=k, covariance_type="full",
                        random_state=SEED, n_init=1)
    etq = m.fit_predict(X_cod)
    return etq, m.bic(X_cod), m


def ari_contra(etq, referencia):
    """ARI entre la partición y una partición de referencia (columna conocida).

    Diagnóstico clave de la Fase 2: un ARI cercano a 1 significa que el modelo
    no aporta información más allá de esa columna. En la v1, el ganador daba
    ARI=1.000 contra modalidad_norm.
    """
    e = np.asarray(etq)
    m = e != -1
    if np.unique(e[m]).size < 2:
        return np.nan
    return float(adjusted_rand_score(np.asarray(referencia)[m], e[m]))


def eta_cuadrado(v, cat):
    """Ratio de correlación η²: fracción de la varianza de v explicada por cat.

    η²=1 significa que la numérica es una función determinista de la
    categórica, es decir, una RE-CODIFICACIÓN de modalidad_norm.
    """
    v = np.asarray(v, dtype=float)
    total = float(((v - v.mean()) ** 2).sum())
    if total == 0:
        return np.nan
    entre = 0.0
    for g in np.unique(cat):
        sub = v[np.asarray(cat) == g]
        entre += sub.size * (sub.mean() - v.mean()) ** 2
    return float(entre / total)


def diagnosticar_variables(X, alias):
    """Detecta numéricas degeneradas y numéricas que son una re-codificación
    de la categórica (lo cual reintroduce modalidad_norm por la puerta de
    atrás, anulando en parte la ponderación 1/3 de la Gower)."""
    avisos = []
    cat = X[CATEGORICA].to_numpy()
    for c in NUMERICAS:
        v = X[c].to_numpy(dtype=float)
        rango = float(v.max() - v.min())
        if rango == 0:
            avisos.append(f"{alias}: '{c}' es CONSTANTE ({v[0]:.4g}); no aporta "
                          f"nada a la distancia de Gower")
            continue
        # proporción de la masa concentrada en el valor más frecuente
        vals, cuentas = np.unique(np.round(v, 6), return_counts=True)
        dominante = float(cuentas.max()) / len(v)
        if dominante >= 0.98:
            avisos.append(f"{alias}: '{c}' es casi constante "
                          f"({100 * dominante:.1f} % de los procesos comparten el "
                          f"valor {vals[int(np.argmax(cuentas))]:.4g}); su aporte a "
                          f"Gower es despreciable")
            continue
        e2 = eta_cuadrado(v, cat)
        if not np.isnan(e2) and e2 >= 0.99:
            avisos.append(f"{alias}: '{c}' es una RE-CODIFICACIÓN de "
                          f"modalidad_norm (eta^2={e2:.3f}, {vals.size} valores "
                          f"distintos): aporta peso 1.0 a la distancia pero la "
                          f"información es la misma de la categórica, que pesa "
                          f"1/3. En la práctica modalidad_norm pesa más de lo "
                          f"que indica PESOS_GOWER")
    return avisos


def tabla_eta2(X):
    """η² de cada numérica contra modalidad_norm (evidencia para el diseñador)."""
    cat = X[CATEGORICA].to_numpy()
    return {c: eta_cuadrado(X[c].to_numpy(dtype=float), cat) for c in NUMERICAS}


def diagnosticar_colinealidad(X, alias, umbral=0.999):
    """Detecta numéricas redundantes entre sí.

    Importa porque Gower da peso 1.0 a CADA numérica: dos columnas casi
    idénticas hacen que esa dimensión pese el doble de lo previsto.
    """
    avisos, pares = [], []
    M = X[NUMERICAS].to_numpy(dtype=float)
    for j in range(M.shape[1]):
        for l in range(j + 1, M.shape[1]):
            a, b = M[:, j], M[:, l]
            if a.std() == 0 or b.std() == 0:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if abs(r) >= umbral:
                pares.append((NUMERICAS[j], NUMERICAS[l], r))
                avisos.append(
                    f"{alias}: '{NUMERICAS[j]}' y '{NUMERICAS[l]}' son "
                    f"REDUNDANTES (r={r:.4f}). Gower les da peso 1.0 a cada "
                    f"una, así que esa dimensión pesa el doble que las demás")
    return avisos, pares


def ejecutar_algoritmo(algo, k, G, X_mixta, X_cod):
    """[FASE 2.2] Despachador: corre un candidato concreto y devuelve
    (etiquetas, nota, objeto, medoides|None). Lo usa la corrida de continuidad."""
    if algo == "kprototypes":
        etq, nota, obj = correr_kprototypes(X_mixta, k)
        return etq, nota, obj, None
    if algo == "kmeans":
        etq, nota, obj = correr_kmeans(X_cod, k)
        return etq, nota, obj, None
    if algo == "jerarquico":
        etq, nota, obj = correr_jerarquico(G, k)
        return etq, nota, obj, None
    if algo == "kmedoids":
        etq, nota, med, obj = correr_kmedoids(G, k)
        return etq, nota, obj, med
    if algo == "gmm":
        etq, bic, obj = correr_gmm(X_cod, k)
        return etq, f"BIC={bic:,.0f}", obj, None
    if algo == "densidad":
        cands = correr_hdbscan(G)
        nombre, etq, _ = cands[0]
        return np.asarray(etq, dtype=int), nombre, None, None
    raise ValueError(f"algoritmo desconocido: {algo}")


# ============================================================ evaluación
def evaluar_proveedor(alias, X, G, X_mixta, X_cod, ev):
    """Devuelve dict algoritmo -> {k: {silueta, db, ch, n_grupos, ruido, notas, etq}}."""
    salida = {a: {} for a in ORDEN_ALGORITMOS}
    bics = {}

    for k in RANGO_K:
        # 1) K-Prototypes
        try:
            t = time.time()
            etq, nota, obj = correr_kprototypes(X_mixta, k)
            salida["kprototypes"][k] = ev.registrar(
                etq, nota, time.time() - t, extra={"objeto": obj})
        except Exception as e:
            anotar(f"{alias} K-Prototypes k={k}: {type(e).__name__}: {e}")

        # 2) K-Means
        try:
            t = time.time()
            etq, nota, obj = correr_kmeans(X_cod, k)
            salida["kmeans"][k] = ev.registrar(
                etq, nota, time.time() - t, extra={"objeto": obj})
        except Exception as e:
            anotar(f"{alias} K-Means k={k}: {type(e).__name__}: {e}")

        # 3) Jerárquico
        try:
            t = time.time()
            etq, nota, obj = correr_jerarquico(G, k)
            salida["jerarquico"][k] = ev.registrar(
                etq, nota, time.time() - t, extra={"objeto": obj})
        except Exception as e:
            anotar(f"{alias} Jerárquico k={k}: {type(e).__name__}: {e}")

        # 4) K-Medoids
        try:
            t = time.time()
            etq, nota, medoides, obj = correr_kmedoids(G, k)
            salida["kmedoids"][k] = ev.registrar(
                etq, nota, time.time() - t,
                extra={"medoides": medoides, "objeto": obj})
        except Exception as e:
            anotar(f"{alias} K-Medoids k={k}: {type(e).__name__}: {e}")

        # 6) GMM
        try:
            t = time.time()
            etq, bic, obj = correr_gmm(X_cod, k)
            r = ev.registrar(etq, f"BIC={bic:,.0f}", time.time() - t,
                             extra={"objeto": obj})
            r["bic"] = bic
            bics[k] = bic
            salida["gmm"][k] = r
        except Exception as e:
            anotar(f"{alias} GMM k={k}: {type(e).__name__}: {e}")

        print(f"    k={k} listo", flush=True)

    # 5) Densidad: DBSCAN (barrido de eps) + HDBSCAN; se queda el mejor
    candidatos = []
    try:
        candidatos += correr_dbscan(G)
    except Exception as e:
        anotar(f"{alias} DBSCAN: {type(e).__name__}: {e}")
    try:
        candidatos += correr_hdbscan(G)
    except Exception as e:
        anotar(f"{alias} HDBSCAN: {type(e).__name__}: {e}")

    mejor = None
    detalle_densidad = []
    for nombre, etq, err in candidatos:
        if etq is None:
            detalle_densidad.append((nombre, np.nan, 0, 100.0, err))
            continue
        s, ng, ruido = silueta_gower(G, etq)
        detalle_densidad.append((nombre, s, ng, ruido, err))
        if not np.isnan(s) and (mejor is None or s > mejor[1]):
            mejor = (nombre, s, etq, ng, ruido)
    if mejor is not None:
        nombre, s, etq, ng, ruido = mejor
        r = ev.registrar(etq, f"{nombre}", 0.0)
        salida["densidad"][ng] = r
    else:
        anotar(f"{alias} densidad: ninguna configuración superó los filtros "
               f"(>=2 grupos y <=50 % de ruido)")

    return salida, bics, detalle_densidad




# ================================================================ figuras
def figura_curvas(alias, res, ruta, ev=None):
    """[FASE 2.1] Curvas + líneas base de control; los puntos descalificados
    se dibujan con una X hueca para que se vea a simple vista cuáles no
    pueden ganar."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = plt.get_cmap("tab10")
    for i, algo in enumerate(ORDEN_ALGORITMOS):
        datos = res[algo]
        if not datos:
            continue
        if algo == "densidad":
            k_det, r = list(datos.items())[0]
            if not np.isnan(r["silueta"]):
                etiqueta = (f"{NOMBRES_ALGORITMOS[algo]} ({k_det} grupos, "
                            f"{r['ruido']:.0f} % ruido)")
                if r["descalificado"]:
                    etiqueta += " [DESCAL.]"
                ax.axhline(r["silueta"], color=colores(i), ls="--", lw=2,
                           label=etiqueta)
            continue
        ks = sorted(datos)
        ys = [datos[k]["silueta"] for k in ks]
        ax.plot(ks, ys, marker="o", lw=2, color=colores(i),
                label=NOMBRES_ALGORITMOS[algo], zorder=3)
        kd = [k for k in ks if datos[k]["descalificado"]
              and not np.isnan(datos[k]["silueta"])]
        if kd:
            ax.scatter(kd, [datos[k]["silueta"] for k in kd], s=110,
                       marker="X", facecolors="none", edgecolors="black",
                       linewidths=1.4, zorder=4)

    if ev is not None:
        if not np.isnan(ev.base_modalidad):
            ax.axhline(ev.base_modalidad, color="black", ls=":", lw=2,
                       label=f"línea base: modalidad trivial "
                             f"({ev.base_modalidad:.3f})", zorder=2)
        ks_b = [k for k in RANGO_K if k in ev.bases_aleatorias
                and not np.isnan(ev.bases_aleatorias[k])]
        if ks_b:
            ax.plot(ks_b, [ev.bases_aleatorias[k] for k in ks_b], color="gray",
                    ls=":", lw=2, label="línea base: etiquetas aleatorias",
                    zorder=2)

    ax.set_xlabel("número de grupos k")
    ax.set_ylabel("silueta sobre distancia de Gower PONDERADA (precomputed)")
    ax.set_title(f"Proveedor {alias} — silueta-Gower vs k (6 algoritmos)\n"
                 f"X negra = configuración descalificada")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend(fontsize=7.5, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def figura_barras(resumen, ganador, ruta):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    algos = [NOMBRES_ALGORITMOS[a] for a in resumen["algoritmo"]]
    vals = resumen["silueta_promedio"].to_numpy(dtype=float)
    colores, tramas = [], []
    for a, estado in zip(resumen["algoritmo"], resumen["estado"]):
        if a == ganador:
            colores.append("#c62828"); tramas.append("")
        elif estado != "apto":
            colores.append("#cfd8dc"); tramas.append("//")
        else:
            colores.append("#90a4ae"); tramas.append("")
    barras = ax.bar(algos, vals, color=colores, edgecolor="black", lw=0.6,
                    hatch=tramas)
    for b, v, k, estado in zip(barras, vals, resumen["mejor_k"],
                               resumen["estado"]):
        if not np.isfinite(v):
            continue
        txt = f"{v:.3f}\n(k={'n/d' if pd.isna(k) else int(k)})"
        if estado != "apto":
            txt += "\nDESCAL."
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.01 if v >= 0 else -0.03), txt, ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.set_ylabel("silueta-Gower ponderada promedio (P1, P2, P3)")
    ax.set_title("Comparación del portafolio — ganador en rojo, "
                 "descalificados rayados")
    finitos = vals[np.isfinite(vals)]
    if finitos.size:
        ax.set_ylim(min(0.0, float(finitos.min()) * 1.25),
                    float(finitos.max()) * 1.22)
    ax.axhline(0, color="gray", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


# ==================================================================== main
def main():
    t_inicio = time.time()
    os.makedirs(RES, exist_ok=True)
    generados = []

    perfiles, procesos, actividad, prov_buyer, cat_json = cargar()
    conservadas = set(cat_json["conservadas"])
    idx_act = indice_actividad(actividad)
    idx_pb = indice_proveedor_buyer(prov_buyer)
    print(f"[comparar] perfiles={len(perfiles):,}  procesos activos={len(procesos):,}",
          flush=True)

    muestreado = len(procesos) > N_MUESTRA
    if muestreado:
        procesos = procesos.sample(n=N_MUESTRA, random_state=SEED)
        procesos = procesos.sort_index().reset_index(drop=True)
    n_proc = len(procesos)
    print(f"[comparar] muestra usada por TODOS los algoritmos: {n_proc:,} "
          f"(muestreado={'sí' if muestreado else 'no'}, seed={SEED})", flush=True)

    elegidos, n_elegibles = elegir_proveedores(perfiles, conservadas, 3)
    if len(elegidos) < 3:
        raise RuntimeError(
            f"sólo {len(elegidos)} proveedores cumplen los criterios de "
            f"elegibilidad de la Fase 2.1 en provincias distintas")
    alias = ["P1", "P2", "P3"]

    print(f"\n[comparar] proveedores ELEGIBLES: {n_elegibles} de {len(perfiles):,}")
    print("[comparar] proveedores de referencia (anonimizados):")
    for a, (_, fila) in zip(alias, elegidos.iterrows()):
        print(f"  {a}: provincia={fila['provincia']} | "
              f"adjudicaciones competitivas={fila['adj_competitivas']} | "
              f"catálogo={100 * fila['pct_catalogo']:.1f} % | "
              f"total histórico={fila['num_adjudicaciones']} | "
              f"CPC históricos={fila['n_cpc_historicos']} | "
              f"monto promedio={fila['monto_promedio_ganado']:,.0f} USD", flush=True)

    resultados, bics_todos, densidad_todos, contexto = {}, {}, {}, {}
    for a, (_, fila) in zip(alias, elegidos.iterrows()):
        print(f"\n[comparar] === {a} ({fila['provincia']}) ===", flush=True)
        t = time.time()
        X, meta = construir_matriz(fila, procesos, idx_act, idx_pb)
        print(f"    matriz de interacción lista "
              f"(cpc_jaccard4>0 en {meta['pct_cpc_jaccard_no_cero']:.1f} % "
              f"(media {meta['cpc_jaccard_medio']:.4f}), "
              f"afinidad_comprador>0 en {meta['pct_afin_comp_no_cero']:.1f} % "
              f"({meta['n_compradores_conocidos']} procesos, media "
              f"{meta['afin_comp_media']:.4f}, máx {meta['afin_comp_max']:.3f}), "
              f"sim_tfidf media={meta['sim_media']:.4f}, "
              f"provincias desconocidas={meta['provincias_desconocidas']})",
              flush=True)
        # [FASE 2.2] correlación ANTES de correr nada
        C, avisos_corr = matriz_correlacion(X, a)
        imprimir_correlacion(C, a)
        for aviso in avisos_corr:
            anotar(aviso)
        (esc, num_std, cat, X_mixta, X_cod, categorias, G,
         crudas_w, recortes, limites) = preparar_vistas(X)
        print(f"    winsorización p1/p99: "
              f"{dict(zip(NUMERICAS, recortes))}", flush=True)
        print(f"    Gower ponderada lista {G.shape} en {time.time() - t:.1f} s "
              f"(pesos: 1.0 x{len(NUMERICAS)} numéricas, "
              f"{PESO_CATEGORICA:.3f} categórica)", flush=True)
        for aviso in diagnosticar_variables(X, a):
            anotar(aviso)
        avisos_col, pares_col = diagnosticar_colinealidad(X, a)
        for aviso in avisos_col:
            anotar(aviso)
        ev = Evaluador(a, G, X_cod, cat, X["log_presupuesto"].to_numpy())
        print(f"    líneas base -> modalidad trivial={ev.base_modalidad:.4f} | "
              f"aleatoria k=3..10: "
              f"{[round(ev.bases_aleatorias.get(k, float('nan')), 4) for k in RANGO_K]}",
              flush=True)
        res, bics, det_dens = evaluar_proveedor(a, X, G, X_mixta, X_cod, ev)
        resultados[a] = res
        bics_todos[a] = bics
        densidad_todos[a] = det_dens
        contexto[a] = {"X": X, "G": G, "X_cod": X_cod, "esc": esc, "ev": ev,
                       "categorias": categorias, "meta": meta, "cat": cat,
                       "provincia": fila["provincia"], "recortes": recortes,
                       "eta2": tabla_eta2(X), "colineales": pares_col,
                       "limites_winsor": limites, "corr": C,
                       "X_mixta": X_mixta}
        ruta = os.path.join(RES, f"curvas_{a}.png")
        figura_curvas(a, res, ruta, ev)
        generados.append(ruta)
        print(f"    {a} completado en {time.time() - t:.1f} s", flush=True)

    # -------------------------------- resumen del portafolio [FASE 2.1]
    # Para cada algoritmo se busca el k que maximiza la silueta promedio entre
    # P1-P3 ENTRE LOS k NO DESCALIFICADOS. Un (algoritmo, k) queda descalificado
    # si incumple alguna regla en AL MENOS UN proveedor.
    def claves_candidato(algo):
        if algo == "densidad":
            return sorted({ng for a in alias for ng in resultados[a][algo]})
        return RANGO_K

    def fila_candidato(algo, k):
        """Agrega los proveedores disponibles para un (algoritmo, k).

        Si un proveedor no produjo partición para esa configuración, el
        candidato NO puede ganar (no hay promedio comparable entre P1-P3) pero
        igual se reporta con lo que sí se obtuvo.
        """
        rs, presentes, faltan = [], [], []
        for a in alias:
            datos = resultados[a][algo]
            if algo == "densidad":
                if not datos:
                    faltan.append(a)
                    continue
                rs.append(list(datos.values())[0])
            else:
                if k not in datos:
                    faltan.append(a)
                    continue
                rs.append(datos[k])
            presentes.append(a)
        if not rs:
            return None
        motivos = [f"{a}: sin partición válida" for a in faltan]
        for a, r in zip(presentes, rs):
            if r["descalificado"]:
                motivos.append(f"{a}: {r['motivos']}")
        sils = [r["silueta"] for r in rs]
        return {
            "algoritmo": algo, "k": k,
            "silueta_promedio": float(np.mean(sils)) if not any(
                np.isnan(v) for v in sils) else np.nan,
            "db": float(np.nanmean([r["db"] for r in rs])),
            "ch": float(np.nanmean([r["ch"] for r in rs])),
            "ari_mod": float(np.nanmean([r["ari_mod"] for r in rs])),
            "ari_cuartil": float(np.nanmean([r["ari_cuartil"] for r in rs])),
            "entropia": float(np.mean([r["entropia"] for r in rs])),
            "cobertura": float(np.mean([r["cobertura"] for r in rs])),
            "base_aleatoria": float(np.nanmean([r["base_aleatoria"] for r in rs])),
            "base_modalidad": float(np.nanmean([r["base_modalidad"] for r in rs])),
            "n_grupos": float(np.mean([r["n_grupos"] for r in rs])),
            "n_proveedores": len(rs),
            "descalificado": bool(motivos),
            "motivos": " | ".join(motivos),
        }

    candidatos = []
    for algo in ORDEN_ALGORITMOS:
        for k in claves_candidato(algo):
            f = fila_candidato(algo, k)
            if f is not None:
                candidatos.append(f)
    df_cand = pd.DataFrame(candidatos)
    ruta_cand = os.path.join(RES, "candidatos_evaluados.csv")
    df_cand.to_csv(ruta_cand, index=False, encoding="utf-8-sig")
    generados.append(ruta_cand)

    resumen_filas = []
    for algo in ORDEN_ALGORITMOS:
        sub = df_cand[df_cand["algoritmo"] == algo]
        aptos = sub[(~sub["descalificado"]) & sub["silueta_promedio"].notna()]
        if len(aptos):
            mejor = aptos.loc[aptos["silueta_promedio"].idxmax()]
            estado = "apto"
            motivos = ""
        else:
            # ningún apto: se reporta el mejor por silueta, pero no puede ganar.
            # Los 6 candidatos SIEMPRE aparecen en la tabla, incluso los que no
            # produjeron ninguna partición utilizable.
            conv = sub[sub["silueta_promedio"].notna()] if len(sub) else sub
            if not len(conv):
                motivo = (" | ".join(sorted(set(sub["motivos"]))) if len(sub)
                          else "ninguna configuración calculable")
                resumen_filas.append({"algoritmo": algo, "mejor_k": np.nan,
                                      "silueta_promedio": np.nan, "db": np.nan,
                                      "ch": np.nan, "ari": np.nan,
                                      "ari_cuartil": np.nan, "entropia": np.nan,
                                      "cobertura": np.nan,
                                      "estado": "SIN SALIDA VÁLIDA",
                                      "motivos": motivo[:300]})
                continue
            mejor = conv.loc[conv["silueta_promedio"].idxmax()]
            estado = "DESCALIFICADO"
            motivos = mejor["motivos"]
        resumen_filas.append({
            "algoritmo": algo, "mejor_k": mejor["k"],
            "silueta_promedio": mejor["silueta_promedio"], "db": mejor["db"],
            "ch": mejor["ch"], "ari": mejor["ari_mod"],
            "ari_cuartil": mejor["ari_cuartil"], "entropia": mejor["entropia"],
            "cobertura": mejor["cobertura"], "estado": estado,
            "motivos": motivos if estado == "DESCALIFICADO" else "",
        })

    resumen = pd.DataFrame(resumen_filas)
    resumen = resumen.set_index("algoritmo").reindex(ORDEN_ALGORITMOS).reset_index()
    resumen["estado"] = resumen["estado"].fillna("SIN SALIDA VÁLIDA")
    resumen["motivos"] = resumen["motivos"].fillna(
        "el algoritmo no produjo ninguna partición evaluable")

    aptos = resumen[(resumen["estado"] == "apto")
                    & resumen["silueta_promedio"].notna()]
    aptos = aptos.sort_values(["silueta_promedio", "db"], ascending=[False, True])
    if not len(aptos):
        raise RuntimeError("ningún candidato superó las reglas de descalificación; "
                           "revisar candidatos_evaluados.csv")
    ganador = aptos.iloc[0]["algoritmo"]
    k_ganador = aptos.iloc[0]["mejor_k"]
    sil_ganador = aptos.iloc[0]["silueta_promedio"]
    segundo = aptos.iloc[1]["algoritmo"] if len(aptos) > 1 else None
    sil_segundo = aptos.iloc[1]["silueta_promedio"] if len(aptos) > 1 else np.nan

    # ------------------------------------------------- tabla comparativa
    filas_csv = []
    for algo in ORDEN_ALGORITMOS:
        k_algo = resumen.loc[resumen["algoritmo"] == algo, "mejor_k"]
        k_algo = None if not len(k_algo) or pd.isna(k_algo.iloc[0]) else k_algo.iloc[0]
        for a in alias:
            datos = resultados[a][algo]
            if algo == "densidad":
                r = list(datos.values())[0] if datos else None
                k_usa = list(datos.keys())[0] if datos else np.nan
            else:
                r = datos.get(k_algo) if k_algo is not None else None
                k_usa = k_algo
            if r is None:
                filas_csv.append({"algoritmo": NOMBRES_ALGORITMOS[algo],
                                  "proveedor": a, "mejor_k": np.nan,
                                  "silueta_gower": None, "davies_bouldin": None,
                                  "calinski_harabasz": None,
                                  "ari_vs_modalidad": None, "ari_vs_cuartil": None,
                                  "entropia": None, "cobertura": None,
                                  "descalificado": None,
                                  "notas": "sin configuración válida"})
                continue
            filas_csv.append({
                "algoritmo": NOMBRES_ALGORITMOS[algo], "proveedor": a,
                "mejor_k": k_usa,
                "silueta_gower": None if np.isnan(r["silueta"]) else round(r["silueta"], 4),
                "davies_bouldin": None if np.isnan(r["db"]) else round(r["db"], 4),
                "calinski_harabasz": None if np.isnan(r["ch"]) else round(r["ch"], 1),
                "ari_vs_modalidad": None if np.isnan(r["ari_mod"]) else round(r["ari_mod"], 4),
                "ari_vs_cuartil": None if np.isnan(r["ari_cuartil"]) else round(r["ari_cuartil"], 4),
                "entropia": round(r["entropia"], 4),
                "cobertura": round(r["cobertura"], 2),
                "descalificado": r["descalificado"],
                "notas": (f"{r['notas']} | ruido={r['ruido']:.1f} % | "
                          f"{r['seg']:.1f} s"
                          + (f" | DESCALIFICADO: {r['motivos']}"
                             if r["descalificado"] else "")),
            })

    for f in filas_csv:
        f["ganador_portafolio"] = (f["algoritmo"] == NOMBRES_ALGORITMOS[ganador])
    for _, r in resumen.iterrows():
        filas_csv.append({
            "algoritmo": NOMBRES_ALGORITMOS[r["algoritmo"]],
            "proveedor": "PROMEDIO_P1_P3",
            "mejor_k": r["mejor_k"],
            "silueta_gower": (None if pd.isna(r["silueta_promedio"])
                              else round(r["silueta_promedio"], 4)),
            "davies_bouldin": None if pd.isna(r["db"]) else round(r["db"], 4),
            "calinski_harabasz": None if pd.isna(r["ch"]) else round(r["ch"], 1),
            "ari_vs_modalidad": None if pd.isna(r["ari"]) else round(r["ari"], 4),
            "ari_vs_cuartil": None if pd.isna(r["ari_cuartil"]) else round(r["ari_cuartil"], 4),
            "entropia": None if pd.isna(r["entropia"]) else round(r["entropia"], 4),
            "cobertura": None if pd.isna(r["cobertura"]) else round(r["cobertura"], 2),
            "descalificado": r["estado"] != "apto",
            "notas": f"{r['estado']}. {r['motivos']}".strip(),
            "ganador_portafolio": r["algoritmo"] == ganador,
        })

    ruta_csv = os.path.join(RES, "comparacion_modelos.csv")
    pd.DataFrame(filas_csv).to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    generados.append(ruta_csv)

    # líneas base de control, como evidencia separada
    filas_base = []
    for a in alias:
        ev = contexto[a]["ev"]
        filas_base.append({"proveedor": a, "provincia": contexto[a]["provincia"],
                           "linea_base": "modalidad_norm (trivial)",
                           "n_grupos": len(set(contexto[a]["cat"])),
                           "silueta_gower": round(float(ev.base_modalidad), 4)})
        for k, s in sorted(ev.bases_aleatorias.items()):
            filas_base.append({"proveedor": a, "provincia": contexto[a]["provincia"],
                               "linea_base": "aleatoria", "n_grupos": k,
                               "silueta_gower": (None if np.isnan(s)
                                                 else round(float(s), 4))})
    ruta_base = os.path.join(RES, "lineas_base.csv")
    pd.DataFrame(filas_base).to_csv(ruta_base, index=False, encoding="utf-8-sig")
    generados.append(ruta_base)

    # η² de cada numérica contra modalidad_norm (cuánta de su información ya
    # está en la categórica)
    filas_eta = []
    for a in alias:
        for c, e2 in contexto[a]["eta2"].items():
            filas_eta.append({"proveedor": a, "provincia": contexto[a]["provincia"],
                              "variable": c,
                              "eta2_vs_modalidad": (None if pd.isna(e2)
                                                    else round(float(e2), 4))})
    ruta_eta = os.path.join(RES, "eta2_variables.csv")
    pd.DataFrame(filas_eta).to_csv(ruta_eta, index=False, encoding="utf-8-sig")
    generados.append(ruta_eta)

    # detalle por k (evidencia adicional)
    det = []
    for a in alias:
        for algo in ORDEN_ALGORITMOS:
            for k, r in resultados[a][algo].items():
                det.append({"proveedor": a, "algoritmo": NOMBRES_ALGORITMOS[algo],
                            "k_o_grupos": k, "silueta_gower": r["silueta"],
                            "davies_bouldin": r["db"], "calinski_harabasz": r["ch"],
                            "n_grupos": r["n_grupos"], "pct_ruido": r["ruido"],
                            "ari_vs_modalidad": r["ari_mod"],
                            "ari_vs_cuartil_presupuesto": r["ari_cuartil"],
                            "entropia_tamanos": r["entropia"],
                            "cobertura": r["cobertura"],
                            "base_aleatoria": r["base_aleatoria"],
                            "base_modalidad": r["base_modalidad"],
                            "descalificado": r["descalificado"],
                            "motivos_descalificacion": r["motivos"],
                            "segundos": round(r["seg"], 2), "notas": r["notas"]})
    ruta_det = os.path.join(RES, "comparacion_detalle_k.csv")
    pd.DataFrame(det).to_csv(ruta_det, index=False, encoding="utf-8-sig")
    generados.append(ruta_det)

    ruta_barras = os.path.join(RES, "comparacion_barras.png")
    figura_barras(resumen, ganador, ruta_barras)
    generados.append(ruta_barras)

    # ------------------------------------------ perfil del ganador + atípicos
    mejor_prov, mejor_sil, k_usado = None, -np.inf, None
    for a in alias:
        datos = resultados[a][ganador]
        if ganador == "densidad":
            if not datos:
                continue
            ng, r = list(datos.items())[0]
            cand_k = ng
        else:
            if int(k_ganador) not in datos:
                continue
            r = datos[int(k_ganador)]
            cand_k = int(k_ganador)
        if not np.isnan(r["silueta"]) and r["silueta"] > mejor_sil:
            mejor_prov, mejor_sil, k_usado = a, r["silueta"], cand_k

    ctx = contexto[mejor_prov]
    if ganador == "densidad":
        etq = list(resultados[mejor_prov][ganador].values())[0]["etq"]
    else:
        etq = resultados[mejor_prov][ganador][k_usado]["etq"]

    # [FASE 2.2] medoides: si gana PAM, cada grupo tiene un proceso REAL que lo
    # representa; se reporta su ocid.
    r_gan = (list(resultados[mejor_prov][ganador].values())[0]
             if ganador == "densidad"
             else resultados[mejor_prov][ganador][k_usado])
    medoides_idx = r_gan.get("medoides")
    ocid_medoide = {}
    if medoides_idx is not None:
        ocids = procesos["ocid"].to_numpy()
        for idx_m in medoides_idx:
            ocid_medoide[int(etq[int(idx_m)])] = str(ocids[int(idx_m)])

    X_orig = ctx["X"]
    X_cod = ctx["X_cod"]
    centro = []
    for g in sorted(set(int(v) for v in etq)):
        m = etq == g
        fila = {"grupo": ("ruido" if g == -1 else g), "tamano": int(m.sum()),
                "pct": round(100.0 * m.sum() / len(etq), 2)}
        for c in NUMERICAS:
            fila[c] = round(float(X_orig.loc[m, c].mean()), 4)
        fila["presupuesto_medio_usd"] = round(
            float(np.expm1(X_orig.loc[m, "log_presupuesto"]).mean()), 2)
        fila["modalidad_dominante"] = X_orig.loc[m, CATEGORICA].mode().iloc[0]
        fila["pct_modalidad_dominante"] = round(
            100.0 * float((X_orig.loc[m, CATEGORICA]
                           == fila["modalidad_dominante"]).mean()), 1)
        fila["ocid_medoide"] = ocid_medoide.get(g, "")
        centro.append(fila)
    df_centro = pd.DataFrame(centro).sort_values("tamano", ascending=False)
    ruta_centro = os.path.join(RES, "perfil_centroides_ganador.csv")
    df_centro.to_csv(ruta_centro, index=False, encoding="utf-8-sig")
    generados.append(ruta_centro)

    # IsolationForest: global = DIAGNÓSTICO, por grupo al 5 % = PRODUCTO
    iso_global = IsolationForest(contamination=0.05, random_state=SEED)
    pred_global = iso_global.fit_predict(X_cod)
    iso_por_grupo = {}
    filas_at = []
    for g in sorted(set(int(v) for v in etq)):
        m = etq == g
        n_g = int(m.sum())
        if n_g >= 10:
            iso = IsolationForest(contamination=0.05, random_state=SEED)
            pred = iso.fit_predict(X_cod[m])
            iso_por_grupo[g] = iso
            pct_intra = 100.0 * float((pred == -1).mean())
            nota = ""
        else:
            pct_intra = np.nan
            nota = "grupo demasiado pequeño para IsolationForest"
        filas_at.append({
            "grupo": ("ruido" if g == -1 else g), "tamano": n_g,
            "pct_atipicos_intra_grupo": (None if np.isnan(pct_intra)
                                         else round(pct_intra, 2)),
            "pct_atipicos_modelo_global": round(
                100.0 * float((pred_global[m] == -1).mean()), 2),
            "notas": nota,
        })
    df_at = pd.DataFrame(filas_at)
    ruta_at = os.path.join(RES, "atipicos_resumen.csv")
    df_at.to_csv(ruta_at, index=False, encoding="utf-8-sig")
    generados.append(ruta_at)

    # grupos minúsculos distorsionan el rango: sólo se miran los de tamaño >= 30
    intra = df_at.loc[df_at["tamano"] >= 30, "pct_atipicos_intra_grupo"].dropna()
    if len(intra) >= 3 and float(intra.max() - intra.min()) < 2.0:
        anotar("IsolationForest ajustado POR GRUPO con contamination=0.05 "
               "devuelve ~5 % de atípicos en todos los grupos por construcción "
               "(la contaminación es un parámetro, no una medición). La columna "
               "pct_atipicos_modelo_global (un solo bosque sobre toda la muestra) "
               "sí discrimina y es la que conviene leer")

    # ------------------------------------------ modelo_ganador.pkl [FASE 2.2]
    # Todo lo que la Fase 3 necesita para reproducir la asignación sobre datos
    # nuevos: objeto entrenado, variables, límites de winsorización, parámetros
    # de estandarización, categorías del one-hot y pesos de la Gower.
    import joblib
    paquete = {
        "fase": "2.2",
        "seed": SEED,
        "algoritmo": ganador,
        "algoritmo_nombre": NOMBRES_ALGORITMOS[ganador],
        "k": int(k_usado),
        "proveedor_alias": mejor_prov,
        "provincia_proveedor": ctx["provincia"],
        # OJO: son dos números distintos. El modelo se entrena sobre el mejor
        # proveedor; el promedio es el criterio con el que ganó el portafolio.
        "silueta_del_proveedor_entrenado": float(mejor_sil),
        "silueta_promedio_portafolio": float(sil_ganador),
        "variables_numericas": list(NUMERICAS),
        "variable_categorica": CATEGORICA,
        "variables_auxiliares": list(AUXILIARES),
        "pesos_gower": PESOS_GOWER.tolist(),
        "peso_categorica": PESO_CATEGORICA,
        "winsor_percentiles": PCT_WINSOR,
        "winsor_limites": {c: lim for c, lim in
                           zip(NUMERICAS, ctx["limites_winsor"])},
        "escalador_media": ctx["esc"].mean_.tolist(),
        "escalador_escala": ctx["esc"].scale_.tolist(),
        "escalador": ctx["esc"],
        # [FASE 3] rangos con los que Gower normalizó durante el entrenamiento
        "gower_rangos": rangos_gower(
            ctx["esc"].transform(
                winsorizar(ctx["X"][NUMERICAS].to_numpy(dtype=float))[0])).tolist(),
        # vectores de los medoides en el espacio estandarizado del entrenamiento
        "medoides_num_std": (
            None if medoides_idx is None else
            ctx["esc"].transform(
                winsorizar(ctx["X"][NUMERICAS].to_numpy(dtype=float))[0]
            )[np.asarray(medoides_idx, dtype=int)].tolist()),
        "medoides_modalidad": (
            None if medoides_idx is None else
            [str(ctx["cat"][int(i)]) for i in np.asarray(medoides_idx, dtype=int)]),
        "categorias_onehot": list(ctx["categorias"]),
        "modelo": r_gan.get("objeto"),
        "medoides_indices": (None if medoides_idx is None
                             else np.asarray(medoides_idx).tolist()),
        "medoides_ocid": ocid_medoide,
        "etiquetas": etq.tolist(),
        "iso_global": iso_global,
        "iso_por_grupo": iso_por_grupo,
        "ocids_muestra": procesos["ocid"].tolist(),
        "nota": ("PAM se ajusta sobre una matriz de distancias precalculada: "
                 "para asignar un proceso nuevo hay que winsorizar, estandarizar "
                 "con escalador_media/escala, calcular su distancia de Gower "
                 "ponderada a los medoides y tomar el más cercano."),
    }
    ruta_pkl = os.path.join(RES, "modelo_ganador.pkl")
    joblib.dump(paquete, ruta_pkl)
    generados.append(ruta_pkl)

    ruta_etq = os.path.join(RES, "etiquetas_ganador.npz")
    np.savez_compressed(ruta_etq, etiquetas=etq,
                        modalidad=np.asarray(ctx["cat"], dtype=object),
                        proveedor=np.array([mejor_prov]),
                        algoritmo=np.array([ganador]),
                        k=np.array([k_usado]))
    generados.append(ruta_etq)

    # ---------------------------------------------- diagnóstico del ganador
    ari_gan = ari_contra(etq, ctx["cat"])
    if not np.isnan(ari_gan) and ari_gan >= 0.60:
        anotar(f"El ganador ({NOMBRES_ALGORITMOS[ganador]}) sigue bastante "
               f"alineado con modalidad_norm (ARI={ari_gan:.3f} en "
               f"{mejor_prov}), aunque por debajo del umbral de "
               f"descalificación ({ARI_MAX_COPIA})")

    # ------------------------------ corrida de continuidad [FASE 2.2]
    continuidad = corrida_continuidad(perfiles, procesos, idx_act, idx_pb,
                                      conservadas, ganador, k_usado, RES)
    if continuidad and continuidad.get("ruta_csv"):
        generados.append(continuidad["ruta_csv"])

    # -------------------------------------------------------------- reporte
    print("\n[comparar] archivos generados:")
    for r in generados:
        print(f"  {r}")

    dur = time.time() - t_inicio
    _reporte(perfiles, procesos, muestreado, n_proc, resumen, resultados,
             densidad_todos, bics_todos, alias, ganador, k_ganador, sil_ganador,
             segundo, sil_segundo, mejor_prov, k_usado, df_centro, df_at, dur,
             elegidos, contexto, n_elegibles, df_cand, continuidad, ruta_pkl)


def comparar_rondas(resumen):
    """[FASE 2.2] Lee las filas PROMEDIO_P1_P3 de las rondas guardadas."""
    def leer(sub):
        ruta = os.path.join(RES, sub, "comparacion_modelos.csv")
        if not os.path.exists(ruta):
            return None
        df = pd.read_csv(ruta)
        df = df[df["proveedor"] == "PROMEDIO_P1_P3"]
        out = {}
        for _, r in df.iterrows():
            estado = ""
            if "descalificado" in df.columns and not pd.isna(r.get("descalificado")):
                estado = " DESCAL." if bool(r["descalificado"]) else " apto"
            out[r["algoritmo"]] = (r["silueta_gower"], r["mejor_k"], estado)
        return out

    v1, v21 = leer("fase2_v1"), leer("fase2_v21")
    if v1 is None and v21 is None:
        return None

    def fmt(d, nombre):
        if d is None or nombre not in d:
            return "n/d"
        s, k, est = d[nombre]
        if pd.isna(s):
            return "n/d"
        kk = "n/d" if pd.isna(k) else int(k)
        return f"{s:.4f} k={kk}{est}"

    lineas = []
    for algo in ORDEN_ALGORITMOS:
        nombre = NOMBRES_ALGORITMOS[algo]
        r = resumen[resumen["algoritmo"] == algo]
        if len(r) and not pd.isna(r.iloc[0]["silueta_promedio"]):
            est = " apto" if r.iloc[0]["estado"] == "apto" else " DESCAL."
            actual = (f"{r.iloc[0]['silueta_promedio']:.4f} "
                      f"k={int(r.iloc[0]['mejor_k'])}{est}")
        else:
            actual = "n/d"
        lineas.append(f"{nombre:<22} | {fmt(v1, nombre):>22} | "
                      f"{fmt(v21, nombre):>22} | {actual:>28}")
    return lineas


def corrida_continuidad(perfiles, procesos, idx_act, idx_pb, conservadas,
                        ganador, k, res_dir, nombre=NOMBRE_CONTINUIDAD):
    """[FASE 2.2] Aplica el algoritmo ganador al proveedor de la Fase 1.

    Sirve de puente con el trabajo previo: ese proveedor NO es elegible bajo los
    criterios de la Fase 2.2, así que no participa de la comparación, pero
    conviene ver cómo se comporta el modelo escogido sobre él.
    """
    m = perfiles["nombre"].fillna("").str.upper().str.contains(
        nombre.upper().replace(" S.A.", "").strip(), regex=False)
    if not m.any():
        anotar(f"corrida de continuidad: no se encontró '{nombre}' en los "
               f"perfiles; se omite")
        return None
    fila = perfiles[m].iloc[0].copy()
    if fila["provincia"] not in CAPITALES or pd.isna(fila["monto_promedio_ganado"]):
        anotar(f"corrida de continuidad: '{nombre}' sin provincia válida o sin "
               f"monto promedio; se omite")
        return None

    cruda = json.loads(fila["modalidades_hist_json"])
    total = sum(cruda.values()) or 1
    n_cat = sum(v for kk, v in cruda.items() if es_catalogo(kk))
    n_comp = sum(v for kk, v in cruda.items() if kk in MODALIDADES_COMPETITIVAS)
    norm = Counter()
    for kk, v in cruda.items():
        norm[normalizar_modalidad(kk, conservadas)] += v
    fila["modalidades_norm"] = dict(norm)

    print(f"\n[comparar] === CONTINUIDAD: {nombre} ===", flush=True)
    try:
        X, meta = construir_matriz(fila, procesos, idx_act, idx_pb)
        (esc, num_std, cat, X_mixta, X_cod, categorias, G,
         crudas_w, recortes, limites) = preparar_vistas(X)
        ev = Evaluador("PC", G, X_cod, cat, X["log_presupuesto"].to_numpy())
        etq, nota, obj, medoides = ejecutar_algoritmo(ganador, int(k), G,
                                                      X_mixta, X_cod)
        r = ev.registrar(etq, nota, 0.0)
    except Exception as e:
        anotar(f"corrida de continuidad falló ({type(e).__name__}: {e})")
        return None

    ocids = procesos["ocid"].to_numpy()
    filas = []
    for g in sorted(set(int(v) for v in np.asarray(etq))):
        sel = np.asarray(etq) == g
        f = {"grupo": ("ruido" if g == -1 else g), "tamano": int(sel.sum()),
             "pct": round(100.0 * sel.sum() / len(etq), 2)}
        for c in NUMERICAS:
            f[c] = round(float(X.loc[sel, c].mean()), 4)
        f["presupuesto_medio_usd"] = round(
            float(np.expm1(X.loc[sel, "log_presupuesto"]).mean()), 2)
        f["modalidad_dominante"] = X.loc[sel, CATEGORICA].mode().iloc[0]
        f["ocid_medoide"] = ""
        filas.append(f)
    if medoides is not None:
        for idx_m in medoides:
            g = int(np.asarray(etq)[int(idx_m)])
            for f in filas:
                if f["grupo"] == g:
                    f["ocid_medoide"] = str(ocids[int(idx_m)])
    df = pd.DataFrame(filas)
    ruta = os.path.join(res_dir, "continuidad_perfil_grupos.csv")
    df.to_csv(ruta, index=False, encoding="utf-8-sig")

    print(f"    silueta={r['silueta']:.4f} | grupos={r['n_grupos']} | "
          f"ARI vs modalidad={r['ari_mod']:.3f} | entropía={r['entropia']:.3f} | "
          f"cobertura={r['cobertura']:.1f} %", flush=True)

    return {"nombre": nombre, "provincia": fila["provincia"],
            "adjudicaciones": int(fila["num_adjudicaciones"]),
            "competitivas": n_comp, "pct_catalogo": 100.0 * n_cat / total,
            "n_cpc": int(fila["n_cpc_historicos"]),
            "elegible": (n_comp >= MIN_ADJ_COMPETITIVAS
                         and fila["n_cpc_historicos"] >= MIN_CPC_DISTINTOS
                         and n_cat / total < MAX_PCT_CATALOGO),
            "resultado": r, "perfil": df, "ruta_csv": ruta, "meta": meta,
            "base_modalidad": ev.base_modalidad}


def _reporte(perfiles, procesos, muestreado, n_proc, resumen, resultados,
             densidad_todos, bics_todos, alias, ganador, k_ganador, sil_ganador,
             segundo, sil_segundo, mejor_prov, k_usado, df_centro, df_at, dur,
             elegidos, contexto, n_elegibles, df_cand, continuidad, ruta_pkl):
    dist = procesos["modalidad_norm"].value_counts()
    dist_txt = ", ".join(f"{k}={v}" for k, v in dist.items())

    print("\n\n=== REPORTE FASE 2.2 — PARA EL DISEÑADOR ===")
    print(f"ESTADO: {'OK' if not INCIDENCIAS else 'OK con incidencias'}")
    print(f"PROVEEDORES ÚNICOS (24+25): {len(perfiles):,} | "
          f"ELEGIBLES tras los criterios de la Fase 2.2 "
          f"(>={MIN_ADJ_COMPETITIVAS} competitivas, >={MIN_CPC_DISTINTOS} CPC, "
          f"<{100 * MAX_PCT_CATALOGO:.0f} % catálogo): {n_elegibles}")
    print("PROVEEDORES DE REFERENCIA (anonimizados):")
    for a, (_, f) in zip(alias, elegidos.iterrows()):
        print(f"  {a}: provincia={f['provincia']} | "
              f"adjudicaciones competitivas={f['adj_competitivas']} | "
              f"catálogo={100 * f['pct_catalogo']:.1f} % | "
              f"total histórico={f['num_adjudicaciones']}")
    print(f"PROCESOS ACTIVOS: {n_proc:,} "
          f"(muestreados: {'sí' if muestreado else 'no'}) | "
          f"MODALIDAD_NORM: {dist_txt}")

    print(f"VARIABLES DEL MODELO ({len(NUMERICAS)} numéricas + modalidad_norm "
          f"con peso {PESO_CATEGORICA:.3f}): {', '.join(NUMERICAS)}")
    print("  (fuera desde la Fase 2.2: log_presupuesto por r=1.0000 con "
          "desviacion_presupuesto; modalidad_afinidad por eta^2=1.000)")
    print("VARIABLES POR PROVEEDOR:")
    for a in alias:
        m = contexto[a]["meta"]
        print(f"  {a}: cpc_jaccard4>0 en {m['pct_cpc_jaccard_no_cero']:.1f} % "
              f"(media {m['cpc_jaccard_medio']:.4f}) | "
              f"sim_tfidf media={m['sim_media']:.4f} | "
              f"recortes winsor p1/p99={sum(contexto[a]['recortes'])}")
    print("AFINIDAD_COMPRADOR (variable nueva):")
    for a in alias:
        m = contexto[a]["meta"]
        print(f"  {a}: >0 en {m['pct_afin_comp_no_cero']:.1f} % de los procesos "
              f"({m['n_compradores_conocidos']} de {len(contexto[a]['X']):,} con "
              f"comprador ya conocido) | media={m['afin_comp_media']:.4f} | "
              f"máx={m['afin_comp_max']:.3f} | "
              f"eta^2 vs modalidad={contexto[a]['eta2']['afinidad_comprador']:.3f}")

    print("DEPENDENCIA DE LAS NUMÉRICAS RESPECTO DE modalidad_norm "
          "(eta^2; 1.000 = re-codificación de la categórica):")
    for c in NUMERICAS:
        vals = " | ".join(f"{a}={contexto[a]['eta2'][c]:.3f}"
                          if not pd.isna(contexto[a]["eta2"][c]) else f"{a}=n/d"
                          for a in alias)
        print(f"  {c:<26} {vals}")

    pares = contexto[alias[0]]["colineales"]
    if pares:
        print("NUMÉRICAS REDUNDANTES ENTRE SÍ (|r| >= 0.999):")
        for v1, v2, r in pares:
            print(f"  {v1} ~ {v2}: r={r:.4f} (esa dimensión pesa el doble "
                  f"en la Gower)")

    print("LÍNEAS BASE DE CONTROL (silueta-Gower ponderada):")
    for a in alias:
        ev = contexto[a]["ev"]
        als = [ev.bases_aleatorias.get(k, np.nan) for k in RANGO_K]
        als = [v for v in als if not np.isnan(v)]
        rango = (f"{min(als):.4f}..{max(als):.4f}" if als else "n/d")
        print(f"  {a}: modalidad trivial={ev.base_modalidad:.4f} | "
              f"aleatoria (k=3..10)={rango}")

    print("TABLA COMPARATIVA (silueta-Gower ponderada promedio P1-P3 | mejor k | "
          "DB | CH | ARI-mod | ARI-cuartil | entropía | cobertura | estado):")
    for _, r in resumen.iterrows():
        s = "n/d" if pd.isna(r["silueta_promedio"]) else f"{r['silueta_promedio']:.4f}"
        db = "n/d" if pd.isna(r["db"]) else f"{r['db']:.3f}"
        ch = "n/d" if pd.isna(r["ch"]) else f"{r['ch']:.1f}"
        ar = "n/d" if pd.isna(r["ari"]) else f"{r['ari']:.3f}"
        ac = "n/d" if pd.isna(r["ari_cuartil"]) else f"{r['ari_cuartil']:.3f}"
        en = "n/d" if pd.isna(r["entropia"]) else f"{r['entropia']:.3f}"
        co = "n/d" if pd.isna(r["cobertura"]) else f"{r['cobertura']:.1f}%"
        k = "n/d" if pd.isna(r["mejor_k"]) else int(r["mejor_k"])
        print(f"  {NOMBRES_ALGORITMOS[r['algoritmo']]:<22} | {s:>8} | k={k:<4} | "
              f"DB={db:>7} | CH={ch:>8} | ARIm={ar:>6} | ARIc={ac:>6} | "
              f"H={en:>5} | cob={co:>6} | {r['estado']}")
        if r["estado"] != "apto" and r["motivos"]:
            print(f"      motivo: {r['motivos'][:150]}")
    print("  (ARIm/ARIc = Adjusted Rand Index contra modalidad_norm y contra el "
          "cuartil de log_presupuesto; >0.90 descalifica por copia de columna)")
    txt_segundo = (f"{NOMBRES_ALGORITMOS[segundo]}, silueta={sil_segundo:.4f}"
                   if segundo else "n/d (ningún otro candidato quedó apto)")
    print(f"GANADOR: {NOMBRES_ALGORITMOS[ganador]}, k={int(k_ganador)}, "
          f"silueta={sil_ganador:.4f} | SEGUNDO: {txt_segundo}")
    n_desc = int(df_cand["descalificado"].sum())
    print(f"DESCALIFICACIONES: {n_desc} de {len(df_cand)} configuraciones "
          f"(algoritmo x k) incumplen alguna regla; detalle en "
          f"candidatos_evaluados.csv")

    # ---------------------------------- revalidación de K-Medoids [FASE 2.2]
    fila_pam = resumen[resumen["algoritmo"] == "kmedoids"]
    if len(fila_pam) and fila_pam.iloc[0]["estado"] == "apto":
        fp = fila_pam.iloc[0]
        gano = ganador == "kmedoids"
        print(f"¿K-MEDOIDS REVALIDÓ POR TERCERA VEZ?: "
              f"{'sí' if gano else 'no'} — "
              f"{'gana el portafolio' if gano else 'apto pero no gana'}, "
              f"silueta={fp['silueta_promedio']:.4f}, k={int(fp['mejor_k'])}")
    elif len(fila_pam):
        fp = fila_pam.iloc[0]
        print(f"¿K-MEDOIDS REVALIDÓ POR TERCERA VEZ?: no — {fp['estado']}"
              + (f", silueta={fp['silueta_promedio']:.4f}, k={int(fp['mejor_k'])}"
                 if not pd.isna(fp["silueta_promedio"]) else "")
              + f" | motivo: {fp['motivos'][:120]}")
    else:
        print("¿K-MEDOIDS REVALIDÓ POR TERCERA VEZ?: no — sin salida evaluable")

    # ---------------------------------- comparación entre rondas [FASE 2.2]
    print("COMPARACIÓN ENTRE RONDAS (silueta-Gower promedio P1-P3 | k | estado):")
    rondas = comparar_rondas(resumen)
    if rondas is None:
        print("  no se pudieron leer las rondas previas en fase2_v1/ y fase2_v21/")
    else:
        print(f"  {'algoritmo':<22} | {'v1 (Fase 2)':>22} | "
              f"{'v2.1':>22} | {'v2.2 (esta)':>28}")
        for linea in rondas:
            print(f"  {linea}")
        print("  NOTA: v1 usaba Gower SIN pesos y 6 variables con cpc_match "
              "binaria; v2.1 Gower ponderada y 7 variables; v2.2 Gower "
              "ponderada y 6 variables sin redundancias. Las siluetas NO son "
              "directamente comparables entre rondas (cambió la distancia); "
              "sí lo son el orden relativo y el estado.")

    print("DBSCAN/HDBSCAN:")
    for a in alias:
        for nombre, s, ng, ruido, err in densidad_todos[a]:
            st = "n/d (descartada)" if np.isnan(s) else f"silueta={s:.4f}"
            print(f"  {a} {nombre}: grupos={ng}, ruido={ruido:.1f} %, {st} {err}")
        d = resultados[a]["densidad"]
        if d:
            ng, r = list(d.items())[0]
            print(f"  {a} -> USADA: {r['notas']} (grupos={ng}, "
                  f"ruido={r['ruido']:.1f} %, silueta={r['silueta']:.4f})")

    print("GMM:")
    for a in alias:
        b = bics_todos[a]
        if not b:
            continue
        k_bic = min(b, key=b.get)
        sils = {k: v["silueta"] for k, v in resultados[a]["gmm"].items()
                if not np.isnan(v["silueta"])}
        k_sil = max(sils, key=sils.get) if sils else None
        txt_sil = (f"mejor k por silueta={k_sil} ({sils[k_sil]:.4f})"
                   if k_sil is not None else "mejor k por silueta=n/d")
        print(f"  {a}: mejor k por BIC={k_bic} (BIC={b[k_bic]:,.0f}) | {txt_sil}")

    print(f"PERFIL DE CENTROIDES DEL GANADOR "
          f"({NOMBRES_ALGORITMOS[ganador]}, k={k_usado}, proveedor {mejor_prov}):")
    cols = ["grupo", "tamano", "pct", "distancia_km", "cpc_jaccard4",
            "sim_tfidf", "desviacion_presupuesto", "actividad_cpc_comprador",
            "afinidad_comprador", "presupuesto_medio_usd",
            "modalidad_dominante", "ocid_medoide"]
    cols = [c for c in cols if c in df_centro.columns]
    print(df_centro[cols].head(8).to_string(index=False))

    print("ATÍPICOS (ganador):")
    for _, r in df_at.iterrows():
        pi = ("n/d" if r["pct_atipicos_intra_grupo"] is None
              or pd.isna(r["pct_atipicos_intra_grupo"])
              else f"{r['pct_atipicos_intra_grupo']:.1f} %")
        print(f"  grupo {r['grupo']} (n={r['tamano']}): intra-grupo={pi} | "
              f"modelo global={r['pct_atipicos_modelo_global']:.1f} % {r['notas']}")

    print(f"CORRIDA DE CONTINUIDAD ({NOMBRE_CONTINUIDAD}):")
    if not continuidad:
        print("  no se pudo ejecutar (ver PROBLEMAS/HALLAZGOS)")
    else:
        c, rc = continuidad, continuidad["resultado"]
        print(f"  perfil: provincia={c['provincia']} | "
              f"adjudicaciones={c['adjudicaciones']} "
              f"(competitivas={c['competitivas']}, "
              f"catálogo={c['pct_catalogo']:.1f} %) | CPC={c['n_cpc']} | "
              f"ELEGIBLE bajo Fase 2.2: {'sí' if c['elegible'] else 'no'}")
        print(f"  {NOMBRES_ALGORITMOS[ganador]} k={int(k_usado)}: "
              f"silueta={rc['silueta']:.4f} | grupos={rc['n_grupos']} | "
              f"ARI vs modalidad={rc['ari_mod']:.3f} | "
              f"entropía={rc['entropia']:.3f} | "
              f"cobertura={rc['cobertura']:.1f} % | "
              f"línea base modalidad={c['base_modalidad']:.4f} | "
              f"{'PASA' if not rc['descalificado'] else 'NO PASA'} las 4 reglas")
        if rc["descalificado"]:
            print(f"    motivo: {rc['motivos'][:160]}")
        cols_c = [x for x in ["grupo", "tamano", "pct", "cpc_jaccard4",
                              "sim_tfidf", "afinidad_comprador",
                              "presupuesto_medio_usd", "modalidad_dominante",
                              "ocid_medoide"] if x in c["perfil"].columns]
        print(c["perfil"][cols_c].head(8).to_string(index=False))

    print(f"MODELO GUARDADO: {ruta_pkl}")
    print(f"  contiene: objeto entrenado, {len(NUMERICAS)} variables numéricas, "
          f"límites de winsorización p1/p99, media y escala del StandardScaler, "
          f"categorías del one-hot, pesos de la Gower, medoides e "
          f"IsolationForest global y por grupo")

    print(f"TIEMPOS: comparar_modelos.py = {dur / 60:.1f} min "
          f"({dur:.0f} s) para 3 proveedores x 6 algoritmos x k=3..10 "
          f"+ corrida de continuidad")
    print("PROBLEMAS/HALLAZGOS:")
    if INCIDENCIAS:
        for i in INCIDENCIAS:
            print(f"  - {i}")
    else:
        print("  - sin incidencias durante el entrenamiento")
    print("=== FIN REPORTE ===")


if __name__ == "__main__":
    main()
