# -*- coding: utf-8 -*-
"""
Figuras por modelo. Reajusta los seis algoritmos en su k seleccionado (mismos
parámetros y semilla que el notebook), verifica que la silueta reproduce la de
tabla6, y dibuja cada modelo sobre UNA MISMA incrustación MDS para que sean
comparables entre sí.

Salidas en resultados/:
  etiquetas_modelos.npz            etiquetas de los 6 + 2 líneas base
  fig_modelo_<slug>.png            una figura de 3 paneles por modelo
  fig_panel_comparativo.png        rejilla 2x3 con los seis en el mismo espacio
"""
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nucleo as N

RES = N.RES
SEED = 42
np.random.seed(SEED)

# k seleccionados en la ejecución del notebook (ver resultados/log_ejecucion.txt)
K_SEL = {"K-Prototypes": 3, "K-Means": 5, "Jerárquico (promedio)": 3,
         "K-Medoids (PAM)": 10, "GMM": 10}
DBSCAN_EPS, DBSCAN_MS = 0.015, 20

SLUG = {"K-Prototypes": "kprototypes", "K-Means": "kmeans",
        "Jerárquico (promedio)": "jerarquico", "K-Medoids (PAM)": "pam",
        "DBSCAN": "dbscan", "GMM": "gmm"}
ORDEN = ["K-Medoids (PAM)", "DBSCAN", "GMM", "Jerárquico (promedio)",
         "K-Means", "K-Prototypes"]

t0 = time.time()
df = N.cargar_matriz()
num, num_std, cat, X_cod, cols_cod, cat_nombres = N.preprocesar(df)
n = len(df)
print(f"n={n}", flush=True)

D = N.gower_matrix(num, cat)
print(f"Gower listo ({time.time()-t0:.1f}s)", flush=True)

CACHE_ETQ = os.path.join(RES, "etiquetas_modelos.npz")
USAR_CACHE = os.path.exists(CACHE_ETQ) and "--refit" not in sys.argv

etiquetas = {}

from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.mixture import GaussianMixture
from kmodes.kprototypes import KPrototypes

if USAR_CACHE:
    # Las etiquetas ya se ajustaron con estos mismos parámetros y semilla; se
    # reutilizan para no repetir el barrido (K-Prototypes solo tarda ~12 min).
    # La verificación contra tabla6 más abajo sigue actuando de control.
    _z = np.load(CACHE_ETQ, allow_pickle=True)
    _inv = {v: k for k, v in SLUG.items()}
    for _slug in _z.files:
        etiquetas[_inv.get(_slug, _slug)] = _z[_slug]
    print("etiquetas recuperadas de caché (usa --refit para reajustar)", flush=True)
    rng = np.random.default_rng(SEED)
else:
    etiquetas["K-Means"] = KMeans(n_clusters=K_SEL["K-Means"], init="k-means++",
                                  n_init=10, random_state=SEED).fit_predict(X_cod)
    print("K-Means ok", flush=True)

    cond = N.condensada(D)
    Zlink = linkage(cond, method="average")
    del cond; gc.collect()
    etiquetas["Jerárquico (promedio)"] = (
        fcluster(Zlink, K_SEL["Jerárquico (promedio)"], criterion="maxclust").astype(np.int32) - 1)
    print("Jerárquico ok", flush=True)

    etq_pam, medoides = N.pam(D, K_SEL["K-Medoids (PAM)"])
    etiquetas["K-Medoids (PAM)"] = etq_pam
    print("PAM ok", flush=True)

    etiquetas["DBSCAN"] = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MS,
                                 metric="precomputed").fit_predict(D)
    print("DBSCAN ok", flush=True)

    etiquetas["GMM"] = GaussianMixture(n_components=K_SEL["GMM"], covariance_type="full",
                                       random_state=SEED, n_init=3, reg_covar=1e-4,
                                       max_iter=300).fit_predict(X_cod)
    print("GMM ok", flush=True)

    Xkp = np.hstack([num_std.astype(object), cat.reshape(-1, 1).astype(object)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        etiquetas["K-Prototypes"] = KPrototypes(
            n_clusters=K_SEL["K-Prototypes"], init="Cao", n_init=10, gamma=None,
            random_state=SEED, n_jobs=1).fit_predict(Xkp, categorical=[num_std.shape[1]])
    print(f"K-Prototypes ok ({time.time()-t0:.0f}s total)", flush=True)

    rng = np.random.default_rng(SEED)
    etiquetas["LÍNEA BASE: aleatoria"] = rng.integers(0, K_SEL["K-Means"], size=n).astype(np.int32)
    etiquetas["LÍNEA BASE: por modalidad"] = cat.copy()

# ---- verificación contra tabla6
t6 = pd.read_csv(os.path.join(RES, "tabla6_comparacion.csv")).set_index("Algoritmo")
print("\nVERIFICACIÓN (silueta recalculada vs tabla6):")
ok_todo = True
for a in ORDEN:
    s, _, ng = N.silueta_gower(D, etiquetas[a])
    ref = float(t6.loc[a, "Silueta (Gower)"])
    coincide = abs(s - ref) < 1e-4
    ok_todo &= coincide
    print(f"   {a:<24} {s:.4f} vs {ref:.4f}  {'OK' if coincide else '*** DIFIERE ***'}")
if not ok_todo:
    print("   AVISO: alguna etiqueta no reproduce la tabla6.")

np.savez_compressed(os.path.join(RES, "etiquetas_modelos.npz"),
                    **{SLUG.get(k, k): v for k, v in etiquetas.items()})

# ---- MDS compartido (submuestra estratificada por los grupos del ganador)
from sklearn.manifold import MDS
N_MDS = 2000
gan = etiquetas["K-Medoids (PAM)"]
idx_all = np.arange(n)
# RNG dedicado: si dependiera del `rng` general, la submuestra cambiaría según
# se hayan reajustado los modelos o se hayan leído del caché, y la incrustación
# MDS cacheada dejaría de ser válida.
rng_sel = np.random.default_rng(SEED + 1)
sel = []
for g in np.unique(gan):
    ig = idx_all[gan == g]
    cuota = min(len(ig), max(25, int(round(N_MDS * len(ig) / n))))
    sel.append(rng_sel.choice(ig, size=cuota, replace=False))
sel = np.sort(np.concatenate(sel))
print(f"\nsubmuestra MDS compartida: {len(sel)} procesos", flush=True)

# La incrustación MDS se cachea: es lo más caro (~210 s) y no cambia entre
# re-renderizados de las figuras.
CACHE_MDS = os.path.join(RES, "_cache_mds.npz")
P = None
if os.path.exists(CACHE_MDS):
    with np.load(CACHE_MDS) as c:      # el context manager cierra el handle:
        if np.array_equal(c["sel"], sel):   # sin él, os.remove falla en Windows
            P, stress = c["P"], float(c["stress"])
    if P is not None:
        print(f"MDS recuperado de caché, stress={stress:.1f}", flush=True)
    else:
        os.remove(CACHE_MDS)
        print("caché de MDS obsoleto (la submuestra cambió): se recalcula", flush=True)
if P is None:
    Dm = D[np.ix_(sel, sel)].astype(np.float64)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
              n_init=4, max_iter=300, normalized_stress=False)
    P = mds.fit_transform(Dm)
    stress = float(mds.stress_)
    del Dm; gc.collect()
    np.savez_compressed(CACHE_MDS, P=P, sel=sel, stress=stress)
    print(f"MDS listo, stress={stress:.1f} ({time.time()-t0:.0f}s)", flush=True)

# submatriz de Gower de la submuestra, para las siluetas de los paneles
Dsub = D[np.ix_(sel, sel)]

from sklearn.metrics import silhouette_samples

CMAP = plt.get_cmap("tab20")


def colores_de(etq_u):
    return {g: ("#b0b0b0" if g == -1 else CMAP(i % 20))
            for i, g in enumerate(sorted(etq_u))}


def figura_modelo(nombre):
    etq = etiquetas[nombre]
    e_sub = etq[sel]
    grupos = sorted(np.unique(etq))
    col = colores_de(grupos)
    sil_total = float(t6.loc[nombre, "Silueta (Gower)"]) if nombre in t6.index else np.nan
    cob = float(t6.loc[nombre, "Cobertura"]) if nombre in t6.index else 100.0
    k_rep = int(t6.loc[nombre, "k usado/obtenido"]) if nombre in t6.index else len(grupos)

    # 11.5" de ancho: al escalar a los 6,45" de la caja de texto del .docx el
    # factor es 0.56, así que estas tipografías quedan legibles en papel.
    fig = plt.figure(figsize=(11.5, 4.6))
    # Márgenes explícitos en vez de tight_layout: con GridSpec, tight_layout
    # avisa de incompatibilidad y deja el suptitle encima del título del panel 1.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 0.95], wspace=.30,
                          top=.745, bottom=.135, left=.055, right=.985)

    # --- panel 1: proyección MDS
    ax = fig.add_subplot(gs[0, 0])
    ruido = e_sub == -1
    if ruido.any():
        ax.scatter(P[ruido, 0], P[ruido, 1], s=7, c="#b0b0b0", alpha=.45,
                   label=f"ruido (n={ruido.sum()})", marker="x", linewidths=.7)
    for g in grupos:
        if g == -1:
            continue
        m = e_sub == g
        if not m.any():
            continue
        ax.scatter(P[m, 0], P[m, 1], s=11, color=col[g], alpha=.7,
                   edgecolors="white", linewidths=.25)
    ax.set_title("Proyección MDS sobre Gower\n(misma incrustación en los seis modelos)",
                 fontsize=13)
    ax.set_xlabel("MDS 1", fontsize=11.5); ax.set_ylabel("MDS 2", fontsize=11.5)
    ax.tick_params(labelsize=10); ax.grid(alpha=.25)
    if ruido.any():
        ax.legend(fontsize=10, loc="best", markerscale=1.6)

    # --- panel 2: silueta por grupo (sobre la submuestra)
    ax = fig.add_subplot(gs[0, 1])
    msk = e_sub != -1
    if np.unique(e_sub[msk]).size >= 2:
        ii = np.flatnonzero(msk)
        sv = silhouette_samples(Dsub[np.ix_(ii, ii)], e_sub[msk], metric="precomputed")
        y = 0
        for g in sorted(np.unique(e_sub[msk])):
            vals = np.sort(sv[e_sub[msk] == g])
            ax.fill_betweenx(np.arange(y, y + len(vals)), 0, vals,
                             color=col[g], alpha=.85, linewidth=0)
            if len(vals) > 25:
                ax.text(-0.035, y + len(vals) / 2, f"G{g}", fontsize=9.5, va="center", ha="right")
            y += len(vals) + 12
        ax.axvline(sv.mean(), color="#c00000", ls="--", lw=1.3,
                   label=f"media submuestra {sv.mean():.3f}")
        ax.legend(fontsize=9.5, loc="lower right")
    ax.set_yticks([]); ax.tick_params(labelsize=10)
    ax.set_xlabel("coeficiente de silueta (Gower)", fontsize=11.5)
    ax.set_title("Silueta por grupo", fontsize=13); ax.grid(alpha=.25, axis="x")

    # --- panel 3: tamaño de los grupos
    ax = fig.add_subplot(gs[0, 2])
    tam = pd.Series(etq).value_counts().sort_index()
    etqs = [f"G{g}" if g != -1 else "ruido" for g in tam.index]
    ax.barh(range(len(tam)), tam.values,
            color=[col[g] for g in tam.index], height=.72)
    ax.set_yticks(range(len(tam))); ax.set_yticklabels(etqs, fontsize=10)
    ax.invert_yaxis(); ax.tick_params(labelsize=10)
    for i, v in enumerate(tam.values):
        ax.text(v + n * .012, i, f"{v:,}", va="center", fontsize=9)
    ax.set_xlim(0, tam.max() * 1.22)
    ax.set_xlabel("procesos", fontsize=11.5)
    ax.set_title("Tamaño de los grupos", fontsize=13); ax.grid(alpha=.25, axis="x")

    sub = (f"k = {k_rep}   ·   silueta Gower = {sil_total:.4f}   ·   "
           f"cobertura = {cob:.2f}%")
    if nombre == "DBSCAN":
        sub += "   ·   DESCARTADO por cobertura < 90%"
    fig.suptitle(f"{nombre}", fontsize=16, y=.965, x=.5, fontweight="semibold")
    fig.text(.5, .895, sub, ha="center", fontsize=12, color="#444444")
    p = os.path.join(RES, f"fig_modelo_{SLUG[nombre]}.png")
    fig.savefig(p, dpi=145, facecolor="white")
    plt.close(fig)
    print(f"   {os.path.basename(p)}", flush=True)


print("\nFiguras por modelo:")
for a in ORDEN:
    figura_modelo(a)

# ---- panel comparativo 2x3
# 2 columnas x 3 filas: proporción vertical, para que al encajar en los 6,45"
# de ancho de la página cada panel siga siendo legible.
fig, axes = plt.subplots(3, 2, figsize=(9.5, 12.4))
for ax, a in zip(axes.ravel(), ORDEN):
    etq = etiquetas[a]; e_sub = etq[sel]
    grupos = sorted(np.unique(etq)); col = colores_de(grupos)
    ruido = e_sub == -1
    if ruido.any():
        ax.scatter(P[ruido, 0], P[ruido, 1], s=6, c="#b0b0b0", alpha=.4,
                   marker="x", linewidths=.6)
    for g in grupos:
        if g == -1:
            continue
        m = e_sub == g
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=8, color=col[g], alpha=.72,
                       edgecolors="white", linewidths=.2)
    s = float(t6.loc[a, "Silueta (Gower)"]); c = float(t6.loc[a, "Cobertura"])
    k_rep = int(t6.loc[a, "k usado/obtenido"])
    marca = "  ✗ descartado" if c < 90 else ("  ★ ganador" if a == "K-Medoids (PAM)" else "")
    ax.set_title(f"{a}{marca}\nk={k_rep} · silueta {s:.4f} · cobertura {c:.1f} %", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=.2)
fig.suptitle("Los seis modelos sobre la misma proyección MDS de Gower\n"
             f"submuestra estratificada de {len(sel):,} procesos activos",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, .945])
fig.savefig(os.path.join(RES, "fig_panel_comparativo.png"), dpi=145, facecolor="white")
plt.close(fig)
print("   fig_panel_comparativo.png")
print(f"\nTotal {time.time()-t0:.0f}s")
