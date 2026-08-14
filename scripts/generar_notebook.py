# -*- coding: utf-8 -*-
"""Genera notebooks/comparacion_portafolio.ipynb (sin ejecutar)."""
import os
import nbformat as nbf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "notebooks", "comparacion_portafolio.ipynb")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C = []   # celdas


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


# ============================================================== 0
md(r"""
# Comparación de seis algoritmos de agrupamiento
### Segmentación de compatibilidad proveedor–licitación · SERCOP (OCDS)
**Proyecto final de Inteligencia Artificial — ESPOL, CCPG1044, Grupo #3**

Este notebook implementa y ejecuta el protocolo de comparación fijado de antemano.
No rediseña el análisis: sólo lo ejecuta y reporta los resultados reales.

**Decisiones registradas antes de modelar**

| Punto | Decisión |
|---|---|
| Datos | `data/{2024,2025,2026}.jsonl.gz` en disco. Ninguna API consumida. |
| Proveedor consultante | `EC-RUC-1790475689001-5192` — ROCHE ECUADOR S.A. (elegido tras el diagnóstico de `scripts/diag_proveedor.py`) |
| Historial del proveedor | Construido **sólo** con procesos `tender.status = "complete"`, para no filtrar información desde las filas que se agrupan |
| Semilla | `random_state = 42` en todo |
| `scikit-learn-extra` | No importa con numpy 2.x → PAM implementado sobre la matriz de Gower |
""")

code(r"""
import os, sys, json, time, gc, warnings, platform
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(BASE, "scripts"))
import nucleo as N

RES = os.path.join(BASE, "resultados")
os.makedirs(RES, exist_ok=True)
SEED = 42
np.random.seed(SEED)

import sklearn, scipy, matplotlib, kmodes
VERSIONES = {
    "python": platform.python_version(),
    "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__, "kmodes": kmodes.__version__,
    "matplotlib": matplotlib.__version__, "so": f"{platform.system()} {platform.release()}",
}
for k, v in VERSIONES.items():
    print(f"{k:>14}: {v}")

try:
    from sklearn_extra.cluster import KMedoids
    SKEXTRA = "OK"
except Exception as e:
    SKEXTRA = f"{type(e).__name__}: {e}"
print(f"\nscikit-learn-extra -> {SKEXTRA}")

cron = N.Cronometro()
""")

# ============================================================== 1 PASO 0
md(r"""
## PASO 0 — Inventario del corte local

Cifras producidas por `scripts/construir_matriz.py` sobre los tres `.jsonl.gz`.
""")

code(r"""
inv = json.load(open(os.path.join(RES, "paso0_inventario.json"), encoding="utf-8"))

print(f"Líneas leídas ................ {inv['lineas_leidas']:,}")
print(f"ocid únicos .................. {inv['ocid_unicos']:,}  (duplicados colapsados: {inv['duplicados_colapsados']})")
print()
print("Reparto por tender.status:")
for k, v in sorted(inv["status"].items(), key=lambda kv: -kv[1]):
    print(f"   {k:<12} {v:>7,}")
print()
print(f"(a) PROCESOS ACTIVOS ......... {inv['activos_tender_status']:,}")
print(f"    umbral del enunciado ..... 1.000  ->  {'SE CONTINÚA' if inv['activos_tender_status'] >= 1000 else 'DETENER'}")
print()
print("Descartes por datos faltantes (sin imputar):")
for k, v in inv["descartes_por_falta_de_datos"].items():
    print(f"   {k:<28} {v:>6,}")
print()
m = inv["matriz"]
print(f"(b) MATRIZ ................... {m['filas']:,} filas x {m['columnas_semanticas']} columnas")
print(f"    codificada (one-hot) ..... {m['filas']:,} x {m['columnas_codificadas_onehot']}  ({m['modalidades_distintas']} modalidades)")
print()
g = inv["gower_n2"]
print(f"(c) MATRIZ DE GOWER (n^2) .... {g['celdas']:,} celdas")
print(f"    float64 .................. {g['float64_GB']} GB")
print(f"    float32 (la que se usa) .. {g['float32_GB']} GB")
print(f"    condensada float64 ....... {g['condensada_float64_GB']} GB")
print()
p = inv["proveedor_consultante"]
print(f"Proveedor consultante: {p['nombre']}  ({p['id']})")
print(f"   procesos históricos={p['procesos_historicos']}  provincia={p['provincia']}  "
      f"CPC distintos={p['cpc_distintos']}  mediana monto=${p['mediana_monto_historico']:,.2f}")
""")

code(r"""
df = N.cargar_matriz()
print(f"Matriz cargada: {df.shape}")
display(df[N.NUMERICAS + [N.CATEGORICA]].head(8))
display(df[N.NUMERICAS].describe(percentiles=[.5, .9, .99]).T)
print("\nModalidades (categórica única, convenios de Catálogo Electrónico colapsados):")
print(df[N.CATEGORICA].value_counts().to_string())
""")

# ============================================================== 2 preprocesamiento
md(r"""
## Pre-procesamiento único

Los seis algoritmos reciben exactamente la misma matriz y el mismo pre-procesamiento.

- `log(1+x)` sobre los montos: ya incorporado en `desviacion_presupuesto`
  = `log1p(monto_proceso) − log1p(mediana_histórica_del_proveedor)`.
- Estandarización (z-score) de las cinco numéricas → usada por K-Means, K-Prototypes y GMM.
- `modalidad_contratacion` es la única categórica; one-hot para K-Means/GMM y nativa para K-Prototypes.
- Gower normaliza por **rango**, no por z-score: recibe las numéricas crudas (es su definición).
""")

code(r"""
with cron("preprocesamiento"):
    num, num_std, cat, X_cod, cols_cod, cat_nombres = N.preprocesar(df)
n = len(df)
print(f"n = {n:,}")
print(f"numéricas crudas      {num.shape}   (para Gower)")
print(f"numéricas z-score     {num_std.shape}  media={num_std.mean():.2e} std={num_std.std():.4f}")
print(f"matriz codificada     {X_cod.shape}  (5 numéricas z + {len(cat_nombres)} one-hot)")
print(f"modalidades           {len(cat_nombres)}")
""")

# ============================================================== 3 Gower
md(r"""
## Árbitro único: matriz de distancias de Gower

Una sola matriz, compartida por todos. La silueta se evalúa **siempre** sobre ella con
`metric='precomputed'`, aunque cada algoritmo se ajuste en el espacio que le corresponde.
Sin ese árbitro común las siluetas no serían comparables.

Distancia = promedio simple de las 6 variables: numéricas `|xi−xj|/rango`, categórica `0/1`.
Se guarda en **float32** (0.71 GB en vez de 1.43 GB) por la memoria disponible en la máquina.
""")

code(r"""
with cron("gower"):
    D = N.gower_matrix(num, cat)
print(f"D: {D.shape}  dtype={D.dtype}  {D.nbytes/1024**3:.3f} GB  ({cron.t['gower']} s)")
print(f"   min={D.min():.4f}  max={D.max():.4f}  media={D.mean():.4f}  diagonal={np.abs(np.diag(D)).max():.1e}")
print(f"   simétrica: {np.allclose(D[:500,:500], D[:500,:500].T)}")
""")

# ============================================================== 4 evaluador
md(r"""
## Evaluador común

- **Silueta**: sobre Gower, `metric='precomputed'`. Excluye el ruido (−1) cuando lo hay.
- **Davies-Bouldin / Calinski-Harabasz**: sobre la **matriz codificada** (euclidiana). Sus
  implementaciones no aceptan matriz precalculada — quedan como criterio **secundario**.
- **Cobertura**: % de procesos asignados a un grupo real.
""")

code(r"""
RESULTADOS = []       # filas de la tabla 6
ETIQUETAS = {}        # nombre -> etiquetas del ajuste final
CODO = {}             # nombre -> {k: W(k)}
DETALLE = {}          # notas por algoritmo

def evaluar(nombre, etq, k_reportado, guardar=True, nota=""):
    sil, n_usados, n_grupos = N.silueta_gower(D, etq)
    db, ch = N.db_ch(X_cod, etq)
    cob = N.cobertura(etq)
    fila = {
        "Algoritmo": nombre,
        "k usado/obtenido": k_reportado,
        "Silueta (Gower)": sil,
        "Davies-Bouldin": db,
        "Calinski-Harabasz": ch,
        "Cobertura": cob,
        "n evaluados (silueta)": n_usados,
        "Nota": nota,
    }
    if guardar:
        RESULTADOS.append(fila)
        ETIQUETAS[nombre] = np.asarray(etq)
    return fila

def barrer_k(nombre, fit_fn, ks=range(3, 11)):
    # Barre k y devuelve (df_barrido, mejor_k, mejor_etiquetas) por silueta sobre Gower.
    filas, etqs = [], {}
    CODO[nombre] = {}
    for k in ks:
        t0 = time.time()
        etq = fit_fn(k)
        dt = time.time() - t0
        sil, nu, ng = N.silueta_gower(D, etq)
        w = N.dispersion_intra(D, etq)
        CODO[nombre][k] = w
        filas.append({"k": k, "Silueta (Gower)": sil, "W(k) intra": w,
                      "grupos": ng, "seg": round(dt, 1)})
        etqs[k] = etq
        print(f"   k={k}  silueta={sil:.4f}  W={w:8.1f}  ({dt:.1f}s)")
        gc.collect()
    t = pd.DataFrame(filas)
    mejor_k = int(t.loc[t["Silueta (Gower)"].idxmax(), "k"])
    return t, mejor_k, etqs[mejor_k]
""")

# ============================================================== 5 K-Prototypes
md(r"""
## Candidato 1 — K-Prototypes (`kmodes`)

`gamma` automático, `init='Cao'`, `n_init=10`. Barrido k = 3…10.
""")

code(r"""
from kmodes.kprototypes import KPrototypes

Xkp = np.hstack([num_std.astype(object), cat.reshape(-1, 1).astype(object)])
IDX_CAT = [num_std.shape[1]]
avisos_kp = []

def fit_kproto(k, X=Xkp, idx=None):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        mdl = KPrototypes(n_clusters=k, init="Cao", n_init=10, gamma=None,
                          random_state=SEED, n_jobs=1, verbose=0)
        etq = mdl.fit_predict(X, categorical=IDX_CAT)
        for w in ws:
            msg = str(w.message)
            if msg not in avisos_kp:
                avisos_kp.append(msg)
    fit_kproto.ultimo = mdl
    return etq

print("K-Prototypes:")
with cron("kprototypes_barrido"):
    tab_kp, k_kp, etq_kp = barrer_k("K-Prototypes", fit_kproto)
print(f"\ngamma automático del último ajuste: {fit_kproto.ultimo.gamma:.4f}")
for a in avisos_kp:
    print("AVISO kmodes:", a)
display(tab_kp)
print(f"Mejor k por silueta sobre Gower: {k_kp}")
""")

code(r"""
nota_kp = f"gamma auto={fit_kproto.ultimo.gamma:.4f}; init=Cao"
if any("n_init" in a for a in avisos_kp):
    nota_kp += "; kmodes forzó n_init=1 (Cao es determinista)"
evaluar("K-Prototypes", etq_kp, k_kp, nota=nota_kp)
print(nota_kp)
print("Tamaño de grupos:", np.bincount(etq_kp))
""")

# ============================================================== 6 K-Means
md(r"""
## Candidato 2 — K-Means

One-hot de la modalidad, `k-means++`, `n_init=10`. Se ajusta sobre la matriz codificada;
la silueta se mide sobre Gower.
""")

code(r"""
from sklearn.cluster import KMeans

def fit_kmeans(k, X=None):
    X = X_cod if X is None else X
    return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED).fit_predict(X)

print("K-Means:")
with cron("kmeans_barrido"):
    tab_km, k_km, etq_km = barrer_k("K-Means", fit_kmeans)
display(tab_km)
print(f"Mejor k por silueta sobre Gower: {k_km}")
evaluar("K-Means", etq_km, k_km, nota="one-hot modalidad; k-means++; n_init=10")
print("Tamaño de grupos:", np.bincount(etq_km))
""")

# ============================================================== 7 Jerárquico
md(r"""
## Candidato 3 — Jerárquico aglomerativo

Sobre la matriz de Gower precalculada, **enlace promedio**. El `linkage` se calcula una
sola vez y se corta con `fcluster` para cada k.
""")

code(r"""
from scipy.cluster.hierarchy import linkage, fcluster

with cron("linkage_average"):
    cond = N.condensada(D)
    Zlink = linkage(cond, method="average")
    del cond; gc.collect()
print(f"linkage average calculado en {cron.t['linkage_average']} s")

def fit_jerar(k):
    return fcluster(Zlink, k, criterion="maxclust").astype(np.int32) - 1

print("Jerárquico (enlace promedio sobre Gower):")
with cron("jerarquico_barrido"):
    tab_jr, k_jr, etq_jr = barrer_k("Jerárquico (promedio)", fit_jerar)
display(tab_jr)
print(f"Mejor k por silueta sobre Gower: {k_jr}")
evaluar("Jerárquico (promedio)", etq_jr, k_jr, nota="Gower precomputada; enlace promedio")
print("Tamaño de grupos:", np.bincount(etq_jr))
""")

# ============================================================== 8 PAM
md(r"""
## Candidato 4 — K-Medoids / PAM

`scikit-learn-extra` no importa con numpy 2.x (ver primera celda), así que PAM está
implementado en `scripts/nucleo.py` sobre la matriz de Gower: inicialización **BUILD**
de PAM + refinamiento alternante. Los medoides son **procesos reales** del conjunto.

> El SWAP exhaustivo de PAM clásico es O(k(n−k)²) por iteración e intratable con n = 13 848;
> se usa la variante alternante. Queda constancia.
""")

code(r"""
MEDOIDES = {}

def fit_pam(k):
    etq, med = N.pam(D, k)
    MEDOIDES[k] = med
    return etq

print("K-Medoids / PAM:")
with cron("pam_barrido"):
    tab_pam, k_pam, etq_pam = barrer_k("K-Medoids (PAM)", fit_pam)
display(tab_pam)
print(f"Mejor k por silueta sobre Gower: {k_pam}")
evaluar("K-Medoids (PAM)", etq_pam, k_pam, nota="BUILD + alternante; medoides = procesos reales")
print("Tamaño de grupos:", np.bincount(etq_pam))
print("\nMedoides (procesos reales):")
display(df.iloc[MEDOIDES[k_pam]][["ocid", "modalidad_contratacion", "provincia_comprador",
                                  "cpc_match_score", "sim_semantica_tfidf", "monto"]])
""")

# ============================================================== 9 DBSCAN
md(r"""
## Candidato 5 — DBSCAN sobre Gower

`min_samples = 10`. `eps` se elige con el gráfico de k-distancias (distancia al
10.º vecino más cercano, ordenada). El número de grupos es **salida**, no entrada:
se barre `eps` y se reporta cuántos grupos salen y qué cobertura dan.
""")

code(r"""
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 110

MIN_SAMPLES = 10
with cron("k_distancias"):
    kd = np.partition(D, MIN_SAMPLES, axis=1)[:, MIN_SAMPLES]
    kd_ord = np.sort(kd)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(kd_ord, lw=1.4, color="#1f4e79")
for q, c in [(90, "#c00000"), (95, "#e08214"), (99, "#7b3294")]:
    v = np.percentile(kd, q)
    ax.axhline(v, ls="--", lw=1, color=c, label=f"p{q} = {v:.4f}")
ax.set_xlabel(f"procesos ordenados"); ax.set_ylabel(f"distancia de Gower al {MIN_SAMPLES}º vecino")
ax.set_title(f"Gráfico de k-distancias (min_samples={MIN_SAMPLES})")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_kdistancias_dbscan.png"), dpi=150)
plt.show()

print(f"k-distancia: min={kd.min():.4f}  mediana={np.median(kd):.4f}  "
      f"p90={np.percentile(kd,90):.4f}  p99={np.percentile(kd,99):.4f}  max={kd.max():.4f}")
""")

code(r"""
from sklearn.cluster import DBSCAN

EPS_GRID = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100]
MS_GRID = [5, 10, 20]

filas_db = []
etq_db_por_config = {}
with cron("dbscan_barrido"):
    for ms in MS_GRID:
        for eps in EPS_GRID:
            etq = DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit_predict(D)
            ng = int(len(set(etq.tolist())) - (1 if -1 in etq else 0))
            cob = N.cobertura(etq)
            sil = np.nan
            if ng >= 2:
                sil, _, _ = N.silueta_gower(D, etq)
            filas_db.append({"min_samples": ms, "eps": eps, "grupos": ng,
                             "Cobertura": cob, "Silueta (Gower)": sil})
            etq_db_por_config[(ms, eps)] = etq
            gc.collect()

tab_db = pd.DataFrame(filas_db)
display(tab_db)
""")

code(r"""
# Selección: entre las configuraciones con >=2 grupos, la de mayor silueta sobre Gower.
val = tab_db.dropna(subset=["Silueta (Gower)"])
val = val[val["grupos"] >= 2]
if len(val):
    mejor = val.loc[val["Silueta (Gower)"].idxmax()]
    ms_b, eps_b = int(mejor["min_samples"]), float(mejor["eps"])
    etq_dbscan = etq_db_por_config[(ms_b, eps_b)]
    ng_b = int(mejor["grupos"])
    print(f"Mejor DBSCAN: min_samples={ms_b}  eps={eps_b}  -> {ng_b} grupos, "
          f"cobertura={mejor['Cobertura']:.2f}%, silueta={mejor['Silueta (Gower)']:.4f}")
    nota_db = (f"eps={eps_b} (k-dist), min_samples={ms_b}; k es SALIDA; "
               f"silueta medida sobre el subconjunto sin ruido "
               f"({int((etq_dbscan!=-1).sum()):,}/{n:,} procesos)")
    evaluar("DBSCAN", etq_dbscan, ng_b, nota=nota_db)
    CODO["DBSCAN"] = {ng_b: N.dispersion_intra(D, etq_dbscan)}
else:
    print("Ninguna configuración de DBSCAN produjo >=2 grupos.")
    etq_dbscan = None
""")

# ============================================================== 10 GMM
md(r"""
## Candidato 6 — Mezclas gaussianas (GMM)

Covarianza **completa**, selección de k por **BIC**. Se ajusta sobre la matriz codificada.
""")

code(r"""
from sklearn.mixture import GaussianMixture

filas_gmm, etq_gmm_por_k = [], {}
CODO["GMM"] = {}
with cron("gmm_barrido"):
    for k in range(3, 11):
        t0 = time.time()
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED,
                            n_init=3, reg_covar=1e-4, max_iter=300).fit(X_cod)
        etq = g.predict(X_cod)
        sil, _, ng = N.silueta_gower(D, etq)
        w = N.dispersion_intra(D, etq)
        CODO["GMM"][k] = w
        filas_gmm.append({"k": k, "BIC": g.bic(X_cod), "AIC": g.aic(X_cod),
                          "Silueta (Gower)": sil, "W(k) intra": w,
                          "convergió": g.converged_, "seg": round(time.time()-t0, 1)})
        etq_gmm_por_k[k] = etq
        print(f"   k={k}  BIC={g.bic(X_cod):,.0f}  silueta={sil:.4f}  ({time.time()-t0:.1f}s)")
        gc.collect()

tab_gmm = pd.DataFrame(filas_gmm)
display(tab_gmm)
k_gmm = int(tab_gmm.loc[tab_gmm["BIC"].idxmin(), "k"])
etq_gmm = etq_gmm_por_k[k_gmm]
k_gmm_sil = int(tab_gmm.loc[tab_gmm["Silueta (Gower)"].idxmax(), "k"])
print(f"\nk elegido por BIC (criterio del enunciado): {k_gmm}")
print(f"(k que maximizaría la silueta sería {k_gmm_sil} — se respeta el BIC)")
evaluar("GMM", etq_gmm, k_gmm,
        nota=f"covarianza completa; k por BIC (min BIC={tab_gmm['BIC'].min():,.0f}); reg_covar=1e-4")
print("Tamaño de grupos:", np.bincount(etq_gmm))
""")

# ============================================================== 11 líneas base
md(r"""
## Líneas base de control

Dos referencias triviales, medidas con exactamente las mismas métricas:
1. **Etiquetas aleatorias** (mismo k que el mejor candidato prototípico).
2. **Agrupamiento por `modalidad_contratacion`** (la categórica sola).
""")

code(r"""
BASE_FILAS = []

def evaluar_base(nombre, etq, k_rep, nota=""):
    sil, nu, ng = N.silueta_gower(D, etq)
    db, ch = N.db_ch(X_cod, etq)
    fila = {"Algoritmo": nombre, "k usado/obtenido": k_rep, "Silueta (Gower)": sil,
            "Davies-Bouldin": db, "Calinski-Harabasz": ch,
            "Cobertura": N.cobertura(etq), "n evaluados (silueta)": nu, "Nota": nota}
    BASE_FILAS.append(fila)
    ETIQUETAS[nombre] = np.asarray(etq)
    return fila

rng = np.random.default_rng(SEED)
k_ref = k_km
etq_rand = rng.integers(0, k_ref, size=n).astype(np.int32)
evaluar_base("LÍNEA BASE: aleatoria", etq_rand, k_ref, nota=f"k={k_ref} (igual que K-Means); semilla 42")

etq_mod = cat.copy()
evaluar_base("LÍNEA BASE: por modalidad", etq_mod, len(cat_nombres),
             nota=f"{len(cat_nombres)} modalidades tras colapsar Catálogo Electrónico")

tabla7 = pd.DataFrame(BASE_FILAS)
display(tabla7)
""")

# ============================================================== 12 estabilidad
md(r"""
## Estabilidad — 20 reejecuciones sobre submuestras del 80 %

Para cada candidato, en su k seleccionado: se reajusta sobre 20 submuestras aleatorias
del 80 % y se compara con la partición de referencia (ajuste completo) restringida a esos
mismos puntos. Se reporta el **ARI promedio** y su **desviación**.
""")

code(r"""
def sub_D(idx):
    return D[np.ix_(idx, idx)]

def est_kproto(idx):
    return fit_kproto(k_kp, X=Xkp[idx])

def est_kmeans(idx):
    return fit_kmeans(k_km, X=X_cod[idx])

def est_jerar(idx):
    Ds = sub_D(idx)
    c = N.condensada(Ds); del Ds; gc.collect()
    Zs = linkage(c, method="average"); del c; gc.collect()
    return fcluster(Zs, k_jr, criterion="maxclust") - 1

def est_pam(idx):
    Ds = sub_D(idx)
    e, _ = N.pam(Ds, k_pam); del Ds; gc.collect()
    return e

def est_gmm(idx):
    return GaussianMixture(n_components=k_gmm, covariance_type="full", random_state=SEED,
                           n_init=3, reg_covar=1e-4, max_iter=300).fit_predict(X_cod[idx])

def est_dbscan(idx):
    Ds = sub_D(idx)
    e = DBSCAN(eps=eps_b, min_samples=ms_b, metric="precomputed").fit_predict(Ds)
    del Ds; gc.collect()
    return e

PLAN_EST = [
    ("K-Prototypes", est_kproto, etq_kp),
    ("K-Means", est_kmeans, etq_km),
    ("Jerárquico (promedio)", est_jerar, etq_jr),
    ("K-Medoids (PAM)", est_pam, etq_pam),
    ("DBSCAN", est_dbscan, etq_dbscan),
    ("GMM", est_gmm, etq_gmm),
]

ARI = {}
with cron("estabilidad"):
    for nombre, fn, ref in PLAN_EST:
        if ref is None:
            ARI[nombre] = (np.nan, np.nan, [])
            print(f"{nombre:<24} -> no ejecutado")
            continue
        t0 = time.time()
        media, desv, lista = N.estabilidad_ari(fn, n, ref, n_rep=20, frac=0.8, seed=SEED)
        ARI[nombre] = (media, desv, lista)
        print(f"{nombre:<24} ARI = {media:.4f} ± {desv:.4f}   ({time.time()-t0:.0f}s, {len(lista)} reps)")
        gc.collect()
""")

code(r"""
# Estabilidad de las líneas base, con las mismas métricas
def est_rand(idx):
    return np.random.default_rng(SEED + len(idx)).integers(0, k_ref, size=len(idx))

def est_mod(idx):
    return cat[idx]

for nombre, fn, ref in [("LÍNEA BASE: aleatoria", est_rand, etq_rand),
                        ("LÍNEA BASE: por modalidad", est_mod, etq_mod)]:
    media, desv, _ = N.estabilidad_ari(fn, n, ref, n_rep=20, frac=0.8, seed=SEED)
    ARI[nombre] = (media, desv, [])
    print(f"{nombre:<28} ARI = {media:.4f} ± {desv:.4f}")
""")

# ============================================================== 13 tablas 6 y 7
md(r"""
## Tabla 6 — Comparación de los seis candidatos
""")

code(r"""
tabla6 = pd.DataFrame(RESULTADOS)
tabla6["ARI"] = tabla6["Algoritmo"].map(lambda a: ARI.get(a, (np.nan,))[0])
tabla6["ARI desv."] = tabla6["Algoritmo"].map(lambda a: ARI.get(a, (np.nan, np.nan))[1])

COLS6 = ["Algoritmo", "k usado/obtenido", "Silueta (Gower)", "Davies-Bouldin",
         "Calinski-Harabasz", "Cobertura", "ARI", "ARI desv.", "n evaluados (silueta)", "Nota"]
tabla6 = tabla6[COLS6]
tabla6.to_csv(os.path.join(RES, "tabla6_comparacion.csv"), index=False, encoding="utf-8")

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
display(tabla6.style.format({"Silueta (Gower)": "{:.4f}", "Davies-Bouldin": "{:.4f}",
                             "Calinski-Harabasz": "{:.1f}", "Cobertura": "{:.2f}%",
                             "ARI": "{:.4f}", "ARI desv.": "{:.4f}"}))
print("\nDavies-Bouldin y Calinski-Harabasz están calculados sobre la MATRIZ CODIFICADA")
print("(euclidiana), no sobre Gower. Criterio SECUNDARIO.")
""")

code(r"""
tabla7 = pd.DataFrame(BASE_FILAS)
tabla7["ARI"] = tabla7["Algoritmo"].map(lambda a: ARI.get(a, (np.nan,))[0])
tabla7["ARI desv."] = tabla7["Algoritmo"].map(lambda a: ARI.get(a, (np.nan, np.nan))[1])
tabla7 = tabla7[COLS6]
tabla7.to_csv(os.path.join(RES, "tabla7_lineas_base.csv"), index=False, encoding="utf-8")
display(tabla7.style.format({"Silueta (Gower)": "{:.4f}", "Davies-Bouldin": "{:.4f}",
                             "Calinski-Harabasz": "{:.1f}", "Cobertura": "{:.2f}%",
                             "ARI": "{:.4f}", "ARI desv.": "{:.4f}"}))

print("\n¿Algún candidato NO supera las líneas base triviales?")
sil_rand = float(tabla7.loc[tabla7["Algoritmo"].str.contains("aleatoria"), "Silueta (Gower)"].iloc[0])
sil_mod = float(tabla7.loc[tabla7["Algoritmo"].str.contains("modalidad"), "Silueta (Gower)"].iloc[0])
print(f"   línea base aleatoria: silueta = {sil_rand:.4f}")
print(f"   línea base modalidad: silueta = {sil_mod:.4f}")
for _, r in tabla6.iterrows():
    marcas = []
    if not (r["Silueta (Gower)"] > sil_rand): marcas.append("NO supera la aleatoria")
    if not (r["Silueta (Gower)"] > sil_mod): marcas.append("NO supera la de modalidad")
    print(f"   {r['Algoritmo']:<24} {r['Silueta (Gower)']:.4f}  "
          f"{'; '.join(marcas) if marcas else 'supera ambas'}")
""")

# ============================================================== 14 regla de decisión
md(r"""
## Regla de decisión (fijada de antemano)

1. Se descarta todo candidato con **cobertura < 90 %** o **ARI de estabilidad < 0.60**.
2. Entre los que pasan, gana la **mayor silueta sobre Gower**.
3. Diferencia < 0.02 entre los dos mejores = **empate técnico**, se resuelve por
   interpretabilidad de los centroides.
""")

code(r"""
UMBRAL_COB, UMBRAL_ARI, UMBRAL_EMPATE = 90.0, 0.60, 0.02

descartados, admitidos = [], []
for _, r in tabla6.iterrows():
    razones = []
    if pd.isna(r["Silueta (Gower)"]):
        razones.append("sin silueta (no produjo >=2 grupos)")
    if pd.notna(r["Cobertura"]) and r["Cobertura"] < UMBRAL_COB:
        razones.append(f"cobertura {r['Cobertura']:.2f}% < 90%")
    if pd.isna(r["ARI"]) or r["ARI"] < UMBRAL_ARI:
        razones.append(f"ARI {r['ARI']:.4f} < 0.60" if pd.notna(r["ARI"]) else "ARI no disponible")
    (descartados if razones else admitidos).append((r, razones))

print("=" * 78)
print("DESCARTADOS")
print("=" * 78)
if not descartados:
    print("   (ninguno)")
for r, razones in descartados:
    print(f"   {r['Algoritmo']:<24} -> {'; '.join(razones)}")

print()
print("=" * 78)
print("ADMITIDOS (ordenados por silueta sobre Gower)")
print("=" * 78)
adm = sorted(admitidos, key=lambda x: -x[0]["Silueta (Gower)"])
for r, _ in adm:
    print(f"   {r['Algoritmo']:<24} silueta={r['Silueta (Gower)']:.4f}  "
          f"cobertura={r['Cobertura']:.2f}%  ARI={r['ARI']:.4f}")

EMPATE = False
if len(adm) == 0:
    GANADOR = None
    print("\nNINGÚN candidato pasa los filtros.")
else:
    GANADOR = adm[0][0]["Algoritmo"]
    if len(adm) >= 2:
        d = adm[0][0]["Silueta (Gower)"] - adm[1][0]["Silueta (Gower)"]
        print(f"\nDiferencia entre los dos mejores: {d:.4f}")
        if d < UMBRAL_EMPATE:
            EMPATE = True
            print(f"EMPATE TÉCNICO (< {UMBRAL_EMPATE}) entre '{adm[0][0]['Algoritmo']}' "
                  f"y '{adm[1][0]['Algoritmo']}' -> se resuelve por interpretabilidad de centroides.")
    print(f"\nGANADOR PROVISIONAL: {GANADOR}")
""")

code(r"""
# Si hay empate técnico, comparar interpretabilidad de los centroides de los dos mejores.
if EMPATE:
    print("Perfiles de los dos empatados (medias por grupo, variables originales):\n")
    for cand, _ in adm[:2]:
        nom = cand["Algoritmo"]; e = ETIQUETAS[nom]
        p = df.assign(_g=e).groupby("_g")[N.NUMERICAS].mean()
        p["procesos"] = df.assign(_g=e).groupby("_g").size()
        p["modalidad dominante"] = df.assign(_g=e).groupby("_g")[N.CATEGORICA].agg(
            lambda s: s.value_counts().index[0])
        print(f"--- {nom} ---")
        display(p.round(3))
        # dispersión relativa entre centroides: cuanto mayor, más separados/legibles
        z = (p[N.NUMERICAS] - p[N.NUMERICAS].mean()) / p[N.NUMERICAS].std(ddof=0).replace(0, 1)
        print(f"    separación media de centroides (|z| medio): {z.abs().to_numpy().mean():.3f}")
        print(f"    modalidades dominantes distintas: {p['modalidad dominante'].nunique()} "
              f"de {len(p)} grupos\n")
else:
    print("Sin empate técnico: gana directamente la mayor silueta.")
""")

# ============================================================== 15 tabla 8
md(r"""
## Tabla 8 — Perfiles de los grupos del ganador

La columna **Lectura de negocio** se deja vacía a propósito.
""")

code(r"""
etq_gan = ETIQUETAS[GANADOR]
dfg = df.assign(_g=etq_gan)
dfg = dfg[dfg["_g"] != -1]

perf = dfg.groupby("_g").agg(
    Procesos=("ocid", "size"),
    **{"Similitud semántica": ("sim_semantica_tfidf", "mean"),
       "Coincid. CPC": ("cpc_match_score", "mean"),
       "Desv. presupuesto": ("desviacion_presupuesto", "mean"),
       "Distancia (km)": ("distancia_geografica_km", "mean")}
).reset_index().rename(columns={"_g": "Grupo"})

dom = dfg.groupby("_g")[N.CATEGORICA].agg(lambda s: s.value_counts().index[0])
perf["Modalidad dominante"] = perf["Grupo"].map(dom)
perf["Lectura de negocio"] = ""

tabla8 = perf[["Grupo", "Procesos", "Similitud semántica", "Coincid. CPC",
               "Desv. presupuesto", "Distancia (km)", "Modalidad dominante",
               "Lectura de negocio"]]
tabla8.to_csv(os.path.join(RES, "tabla8_perfiles.csv"), index=False, encoding="utf-8")
print(f"Ganador: {GANADOR}   (k = {len(tabla8)} grupos)")
display(tabla8.style.format({"Similitud semántica": "{:.4f}", "Coincid. CPC": "{:.4f}",
                             "Desv. presupuesto": "{:.4f}", "Distancia (km)": "{:.1f}"}))
""")

# ============================================================== 16 tabla 9
md(r"""
## Tabla 9 — Atípicos por grupo (Isolation Forest, `contamination=0.05`)
""")

code(r"""
from sklearn.ensemble import IsolationForest

filas9 = []
with cron("isolation_forest"):
    for g in sorted(dfg["_g"].unique()):
        sub = dfg[dfg["_g"] == g]
        Xg = sub[N.NUMERICAS].to_numpy(dtype=np.float64)
        if len(sub) < 10:
            filas9.append({"Grupo": g, "Procesos": len(sub), "Atípicos": np.nan,
                           "%": np.nan, "Variable que más se aleja": "grupo < 10 procesos"})
            continue
        iso = IsolationForest(contamination=0.05, random_state=SEED, n_estimators=200)
        pred = iso.fit_predict(Xg)
        out = pred == -1
        # variable que más se aleja: mayor |media_atípicos - media_grupo| en unidades sigma
        mu, sd = Xg.mean(0), Xg.std(0)
        sd[sd == 0] = 1.0
        desv = np.abs((Xg[out].mean(0) - mu) / sd)
        var = N.NUMERICAS[int(np.argmax(desv))]
        filas9.append({"Grupo": g, "Procesos": len(sub), "Atípicos": int(out.sum()),
                       "%": round(100.0 * out.sum() / len(sub), 2),
                       "Variable que más se aleja": f"{var} ({desv.max():.2f}σ)"})

tabla9 = pd.DataFrame(filas9)
tabla9.to_csv(os.path.join(RES, "tabla9_atipicos.csv"), index=False, encoding="utf-8")
display(tabla9)
""")

# ============================================================== 17 figuras
md(r"""
## Figuras
""")

code(r"""
# --- Curva del codo por algoritmo (dispersión intra-grupo W(k) sobre Gower)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
colores = {"K-Prototypes": "#1f4e79", "K-Means": "#c00000", "Jerárquico (promedio)": "#2e7d32",
           "K-Medoids (PAM)": "#e08214", "GMM": "#7b3294", "DBSCAN": "#795548"}
for nom, dd in CODO.items():
    if len(dd) < 2:
        continue
    ks = sorted(dd); axes[0].plot(ks, [dd[k] for k in ks], "o-", label=nom,
                                  color=colores.get(nom), lw=1.8, ms=5)
axes[0].set_xlabel("k"); axes[0].set_ylabel("W(k): dispersión intra-grupo (Gower)")
axes[0].set_title("Curva del codo por algoritmo"); axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)

for nom, tab in [("K-Prototypes", tab_kp), ("K-Means", tab_km),
                 ("Jerárquico (promedio)", tab_jr), ("K-Medoids (PAM)", tab_pam)]:
    axes[1].plot(tab["k"], tab["Silueta (Gower)"], "o-", label=nom, color=colores.get(nom), lw=1.8, ms=5)
axes[1].plot(tab_gmm["k"], tab_gmm["Silueta (Gower)"], "o-", label="GMM", color=colores["GMM"], lw=1.8, ms=5)
axes[1].axhline(sil_rand, ls=":", color="gray", label="línea base aleatoria")
axes[1].axhline(sil_mod, ls="--", color="gray", label="línea base modalidad")
axes[1].set_xlabel("k"); axes[1].set_ylabel("silueta (Gower, precomputed)")
axes[1].set_title("Silueta sobre Gower vs k"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_codo_por_algoritmo.png"), dpi=150)
plt.show()
""")

code(r"""
# --- Silueta por grupo del ganador
from sklearn.metrics import silhouette_samples

mask = etq_gan != -1
idx = np.flatnonzero(mask)
Dg = D if idx.size == n else D[np.ix_(idx, idx)]
sv = silhouette_samples(Dg, etq_gan[mask], metric="precomputed")
if Dg is not D:
    del Dg; gc.collect()
sil_gan = float(sv.mean())

fig, ax = plt.subplots(figsize=(7.5, 6))
y = 10
for g in sorted(np.unique(etq_gan[mask])):
    vals = np.sort(sv[etq_gan[mask] == g])
    ax.fill_betweenx(np.arange(y, y + len(vals)), 0, vals, alpha=.8)
    ax.text(-0.045, y + len(vals) / 2, f"G{g}\n(n={len(vals)})", fontsize=8, va="center")
    y += len(vals) + 40
ax.axvline(sil_gan, color="#c00000", ls="--", lw=1.5, label=f"media = {sil_gan:.4f}")
ax.set_xlabel("coeficiente de silueta (Gower)"); ax.set_yticks([])
ax.set_title(f"Silueta por grupo — {GANADOR}"); ax.legend(); ax.grid(alpha=.3, axis="x")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_silueta_por_grupo_ganador.png"), dpi=150)
plt.show()
print(f"silueta media del ganador = {sil_gan:.4f}")
""")

code(r"""
# --- Proyección 2D: MDS sobre la matriz de Gower.
# MDS/SMACOF es O(n^2) por iteración; con n=13.848 no es viable en tiempo razonable.
# Se usa una SUBMUESTRA ESTRATIFICADA por grupo del ganador. Es sólo una figura:
# ninguna métrica de la tabla 6 depende de ella.
from sklearn.manifold import MDS

N_MDS = 2000
rng = np.random.default_rng(SEED)
idx_all = np.flatnonzero(etq_gan != -1)
etq_all = etq_gan[idx_all]
sel = []
for g in np.unique(etq_all):
    ig = idx_all[etq_all == g]
    cuota = max(20, int(round(N_MDS * len(ig) / len(idx_all))))
    cuota = min(cuota, len(ig))
    sel.append(rng.choice(ig, size=cuota, replace=False))
sel = np.sort(np.concatenate(sel))
print(f"submuestra MDS: {len(sel):,} de {len(idx_all):,} procesos "
      f"({100*len(sel)/len(idx_all):.1f}%), estratificada por grupo")

Dm = D[np.ix_(sel, sel)].astype(np.float64)
with cron("mds"):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
              n_init=4, max_iter=300, normalized_stress=False)
    P = mds.fit_transform(Dm)
del Dm; gc.collect()
print(f"MDS: stress={mds.stress_:.2f}  ({cron.t['mds']} s)")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for g in sorted(np.unique(etq_gan[sel])):
    m = etq_gan[sel] == g
    ax.scatter(P[m, 0], P[m, 1], s=9, alpha=.55, label=f"G{g} (n={m.sum()})")
ax.set_title(f"Proyección MDS sobre Gower — grupos de {GANADOR}\n"
             f"(submuestra estratificada de {len(sel):,} procesos)")
ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2"); ax.legend(fontsize=8, markerscale=2)
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_mds_grupos_ganador.png"), dpi=150)
plt.show()
""")

# ============================================================== 18 asignabilidad
md(r"""
## Asignabilidad — cómo se asignaría un proveedor nuevo

Prototipo (K-Prototypes, K-Means, GMM) y medoide (K-Medoids) son directos.
Jerárquico y DBSCAN no producen prototipo: se usa el **centroide empírico** del grupo.
""")

code(r"""
ASIGNA = {
 "K-Prototypes": "Directa por PROTOTIPO: distancia mixta (euclidiana en las numéricas + "
                 "desacuerdo x gamma en la modalidad) al prototipo de cada grupo; gana el mínimo.",
 "K-Means": "Directa por PROTOTIPO: distancia euclidiana al centroide en el espacio codificado "
            "(numéricas z-score + one-hot de modalidad).",
 "Jerárquico (promedio)": "SIN prototipo nativo. Se calcula el CENTROIDE EMPÍRICO de cada grupo "
                          "(media de las numéricas + modalidad modal) y se asigna por Gower al más cercano.",
 "K-Medoids (PAM)": "Directa por MEDOIDE: distancia de Gower a cada medoide, que es un proceso real "
                    "del conjunto; gana el mínimo.",
 "DBSCAN": "SIN prototipo nativo y con ruido. Se calcula el CENTROIDE EMPÍRICO de cada grupo real "
           "y se asigna por Gower; si la distancia supera eps, el proceso queda como ruido.",
 "GMM": "Directa por PROTOTIPO probabilístico: se evalúa la densidad de cada componente y se asigna "
        "al de mayor probabilidad posterior (predict_proba).",
}
asig = pd.DataFrame([{"Algoritmo": a, "Asignación de un proveedor nuevo": t} for a, t in ASIGNA.items()])
asig.to_csv(os.path.join(RES, "asignabilidad.csv"), index=False, encoding="utf-8")
pd.set_option("display.max_colwidth", 200)
display(asig)
""")

# ============================================================== 19 log
md(r"""
## Log de ejecución
""")

code(r"""
lineas = []
lineas.append("LOG DE EJECUCIÓN — comparación de seis algoritmos de agrupamiento")
lineas.append("Proyecto final de IA — ESPOL CCPG1044 Grupo #3 — SERCOP/OCDS")
lineas.append(f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}")
lineas.append("")
lineas.append("VERSIONES DE LIBRERÍAS")
for k, v in VERSIONES.items():
    lineas.append(f"   {k:<16} {v}")
lineas.append(f"   {'scikit-learn-extra':<16} {SKEXTRA}")
lineas.append("")
lineas.append("SEMILLA: random_state = 42 en todos los ajustes")
lineas.append("")
lineas.append("DATOS")
lineas.append(f"   archivos ................ data/2024.jsonl.gz, 2025.jsonl.gz, 2026.jsonl.gz")
lineas.append(f"   procesos leídos ......... {inv['lineas_leidas']:,}")
lineas.append(f"   procesos activos ........ {inv['activos_tender_status']:,}")
lineas.append(f"   filas de la matriz ...... {n:,} x {len(N.NUMERICAS)+1}")
lineas.append(f"   matriz codificada ....... {X_cod.shape[0]:,} x {X_cod.shape[1]}")
lineas.append(f"   Gower ................... {D.shape[0]:,}^2 float32 = {D.nbytes/1024**3:.3f} GB")
lineas.append(f"   proveedor consultante ... {inv['proveedor_consultante']['nombre']} "
              f"({inv['proveedor_consultante']['id']})")
lineas.append("")
lineas.append("TIEMPOS (segundos)")
for k, v in cron.t.items():
    lineas.append(f"   {k:<24} {v:>9.2f}")
lineas.append(f"   {'TOTAL medido':<24} {sum(cron.t.values()):>9.2f}")
lineas.append("")
lineas.append("k SELECCIONADO POR ALGORITMO")
lineas.append(f"   K-Prototypes ............ k={k_kp} (máx. silueta Gower)")
lineas.append(f"   K-Means ................. k={k_km} (máx. silueta Gower)")
lineas.append(f"   Jerárquico .............. k={k_jr} (máx. silueta Gower)")
lineas.append(f"   K-Medoids (PAM) ......... k={k_pam} (máx. silueta Gower)")
lineas.append(f"   DBSCAN .................. {ng_b} grupos OBTENIDOS (eps={eps_b}, min_samples={ms_b})")
lineas.append(f"   GMM ..................... k={k_gmm} (mín. BIC)")
lineas.append("")
lineas.append("DESVIACIONES Y LIMITACIONES DECLARADAS")
lineas.append("   1. scikit-learn-extra no importa con numpy 2.x:")
lineas.append(f"      {SKEXTRA}")
lineas.append("      -> PAM implementado en scripts/nucleo.py sobre la matriz de Gower")
lineas.append("         (init BUILD + refinamiento alternante; el SWAP exhaustivo de PAM clásico")
lineas.append("          es O(k(n-k)^2) por iteración e intratable con n=13.848).")
lineas.append("   2. La matriz de Gower se guarda en float32 (0.71 GB) en vez de float64 (1.43 GB)")
lineas.append("      por la memoria física disponible en la máquina.")
lineas.append("   3. La proyección MDS se calcula sobre una submuestra estratificada de "
              f"{len(sel):,} procesos:")
lineas.append("      SMACOF es O(n^2) por iteración y no es viable con n=13.848. Es sólo una figura;")
lineas.append("      ninguna métrica de la tabla 6 depende de ella.")
lineas.append("   4. Davies-Bouldin y Calinski-Harabasz se calculan sobre la matriz CODIFICADA")
lineas.append("      (euclidiana), no sobre Gower. Criterio secundario, así etiquetado.")
lineas.append("   5. 812 procesos activos (5,5%) quedaron fuera de la matriz por datos faltantes")
lineas.append("      (508 sin ítems, 304 sin provincia del comprador). No se imputó nada.")
lineas.append("   6. 'Catálogo Electrónico' no aparece entre los procesos activos: la regla de")
lineas.append("      colapsar convenios se aplicó pero resulta inocua sobre esta matriz.")
for a in avisos_kp:
    lineas.append(f"   7. Aviso de kmodes: {a}")
lineas.append("")
lineas.append("SALIDAS")
for f in ["tabla6_comparacion.csv", "tabla7_lineas_base.csv", "tabla8_perfiles.csv",
          "tabla9_atipicos.csv", "asignabilidad.csv", "matriz_procesos.csv",
          "fig_codo_por_algoritmo.png", "fig_silueta_por_grupo_ganador.png",
          "fig_mds_grupos_ganador.png", "fig_kdistancias_dbscan.png"]:
    p = os.path.join(RES, f)
    lineas.append(f"   {'OK ' if os.path.exists(p) else 'FALTA'} resultados/{f}")
lineas.append("")
lineas.append("RESULTADO")
lineas.append(f"   GANADOR: {GANADOR}  (silueta Gower = "
              f"{float(tabla6.loc[tabla6['Algoritmo']==GANADOR,'Silueta (Gower)'].iloc[0]):.4f})")
for r, razones in descartados:
    lineas.append(f"   DESCARTADO: {r['Algoritmo']} -> {'; '.join(razones)}")

txt = "\n".join(lineas)
with open(os.path.join(RES, "log_ejecucion.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
print(txt)
""")

md(r"""
---
Todas las tablas y figuras quedan en `resultados/`. Las cifras de este notebook son las
producidas por esta ejecución; no hay valores estimados ni rellenados a mano.
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3.11 (.venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.15"},
}
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"notebook escrito: {OUT}  ({len(C)} celdas)")
