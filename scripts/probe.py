# -*- coding: utf-8 -*-
"""Sondeo de tiempos/memoria de las operaciones caras antes de lanzar el notebook."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nucleo as N

def mem():
    import ctypes
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    s = MS(); s.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 1024**3

print(f"RAM disponible inicial: {mem():.2f} GB")
df = N.cargar_matriz()
num, num_std, cat, X_cod, cols, cat_nombres = N.preprocesar(df)
n = len(df)
print(f"n={n}  X_cod={X_cod.shape}  modalidades={len(cat_nombres)}")

t = time.time(); D = N.gower_matrix(num, cat); tg = time.time()-t
print(f"Gower: {tg:.1f}s  dtype={D.dtype}  {D.nbytes/1024**3:.2f} GB  "
      f"min={D.min():.4f} max={D.max():.4f} media={D.mean():.4f}")
print(f"RAM disponible tras Gower: {mem():.2f} GB")

# K-Means k=6 (referencia barata)
from sklearn.cluster import KMeans
t = time.time()
km = KMeans(n_clusters=6, init="k-means++", n_init=10, random_state=N.SEED).fit(X_cod)
print(f"KMeans k=6: {time.time()-t:.1f}s")

t = time.time(); s, nu, ng = N.silueta_gower(D, km.labels_); ts = time.time()-t
print(f"Silueta Gower (n completo): {ts:.1f}s -> {s:.4f}")

t = time.time(); etq, med = N.pam(D, 6); tp = time.time()-t
print(f"PAM k=6: {tp:.1f}s  tam grupos={np.bincount(etq)}")

t = time.time(); w = N.dispersion_intra(D, km.labels_)
print(f"dispersion_intra: {time.time()-t:.1f}s -> {w:.1f}")

# K-Prototypes: sondeo con k=3, n_init=1 para extrapolar
from kmodes.kprototypes import KPrototypes
import warnings
Xkp = np.hstack([num_std.astype(object), cat.reshape(-1,1).astype(object)])
t = time.time()
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    kp = KPrototypes(n_clusters=3, init="Cao", n_init=1, random_state=N.SEED, n_jobs=1)
    lkp = kp.fit_predict(Xkp, categorical=[5])
    avisos = [str(x.message)[:120] for x in ws]
print(f"KPrototypes k=3 n_init=1: {time.time()-t:.1f}s  gamma={kp.gamma:.4f}  iters={kp.n_iter_}")
for a in avisos: print("   aviso:", a)

# linkage jerárquico
t = time.time(); cond = N.condensada(D); tc = time.time()-t
print(f"condensada: {tc:.1f}s  {cond.nbytes/1024**3:.2f} GB   RAM libre={mem():.2f} GB")
from scipy.cluster.hierarchy import linkage
t = time.time(); Zl = linkage(cond, method="average"); tl = time.time()-t
print(f"linkage average: {tl:.1f}s   RAM libre={mem():.2f} GB")
del cond

# DBSCAN precomputed
from sklearn.cluster import DBSCAN
t = time.time()
db = DBSCAN(eps=0.05, min_samples=10, metric="precomputed").fit(D)
ng = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"DBSCAN eps=0.05: {time.time()-t:.1f}s  grupos={ng}  ruido={(db.labels_==-1).mean()*100:.1f}%")
print(f"RAM disponible final: {mem():.2f} GB")
