# -*- coding: utf-8 -*-
"""Empaqueta todo lo que necesita el reporte .docx en un único JSON."""
import json
import os
import re
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "resultados")


def csv(n):
    return pd.read_csv(os.path.join(RES, n))


t6 = csv("tabla6_comparacion.csv")
t7 = csv("tabla7_lineas_base.csv")
t8 = csv("tabla8_perfiles.csv")
t9 = csv("tabla9_atipicos.csv")
asig = csv("asignabilidad.csv")
inv = json.load(open(os.path.join(RES, "paso0_inventario.json"), encoding="utf-8"))
log = open(os.path.join(RES, "log_ejecucion.txt"), encoding="utf-8").read()

UMBRAL_COB, UMBRAL_ARI = 90.0, 0.60
t6 = t6.copy()


def coma(x, d):
    return f"{x:.{d}f}".replace(".", ",")


def motivos(r):
    m = []
    if r["Cobertura"] < UMBRAL_COB:
        m.append(f"cobertura {coma(r['Cobertura'], 2)} %, por debajo del mínimo de 90 %")
    if r["ARI"] < UMBRAL_ARI:
        m.append(f"ARI de estabilidad {coma(r['ARI'], 3)}, por debajo del mínimo de 0,60")
    return m


t6["_motivos"] = t6.apply(motivos, axis=1)
t6["_ok"] = t6["_motivos"].map(lambda x: len(x) == 0)
adm = t6[t6["_ok"]].sort_values("Silueta (Gower)", ascending=False)

# tiempos del log: sólo dentro del bloque TIEMPOS, no de VERSIONES
tiempos = {}
bloque = re.search(r"^TIEMPOS \(segundos\)\n(.*?)(?=\n[A-ZÁÉÍÓÚ][^\n]*\n)", log, re.S | re.M)
if bloque:
    for m in re.finditer(r"^   (\S[\S ]*?)\s{2,}([\d.]+)\s*$", bloque.group(1), re.M):
        tiempos[m.group(1).strip()] = float(m.group(2))

# ---- balance de los grupos por modelo (revela particiones degeneradas)
import numpy as np
z = np.load(os.path.join(RES, "etiquetas_modelos.npz"), allow_pickle=True)
NOM = {"pam": "K-Medoids (PAM)", "dbscan": "DBSCAN", "gmm": "GMM",
       "jerarquico": "Jerárquico (promedio)", "kmeans": "K-Means",
       "kprototypes": "K-Prototypes"}
balance = []
for slug in ["pam", "dbscan", "gmm", "jerarquico", "kmeans", "kprototypes"]:
    e = z[slug]
    real = e[e != -1]
    t = pd.Series(real).value_counts().sort_values(ascending=False)
    p = t / t.sum()
    balance.append({
        "Modelo": NOM[slug],
        "Grupos": int(len(t)),
        "Mayor (%)": round(100 * float(p.iloc[0]), 1),
        "Dos mayores (%)": round(100 * float(p.iloc[:2].sum()), 1),
        "Menor (n)": int(t.iloc[-1]),
        "Grupos <1%": int((p < 0.01).sum()),
        "Entropía norm.": round(float(-(p * np.log(p)).sum() / np.log(len(t))), 3),
    })
tam_jer = pd.Series(z["jerarquico"]).value_counts().sort_index().to_dict()

datos = {
    "inventario": inv,
    "balance": balance,
    "tamanos_jerarquico": {str(k): int(v) for k, v in tam_jer.items()},
    "tabla6": t6.drop(columns=["_motivos", "_ok"]).to_dict("records"),
    "descartados": [{"algoritmo": r["Algoritmo"], "motivos": r["_motivos"]}
                    for _, r in t6.iterrows() if not r["_ok"]],
    "admitidos": [{"algoritmo": r["Algoritmo"], "silueta": r["Silueta (Gower)"],
                   "cobertura": r["Cobertura"], "ari": r["ARI"]}
                  for _, r in adm.iterrows()],
    "ganador": {
        "algoritmo": adm.iloc[0]["Algoritmo"],
        "k": int(adm.iloc[0]["k usado/obtenido"]),
        "silueta": float(adm.iloc[0]["Silueta (Gower)"]),
        "cobertura": float(adm.iloc[0]["Cobertura"]),
        "ari": float(adm.iloc[0]["ARI"]),
        "ari_desv": float(adm.iloc[0]["ARI desv."]),
        "segundo": adm.iloc[1]["Algoritmo"],
        "silueta_segundo": float(adm.iloc[1]["Silueta (Gower)"]),
        "diferencia": float(adm.iloc[0]["Silueta (Gower)"] - adm.iloc[1]["Silueta (Gower)"]),
    },
    "tabla7": t7.to_dict("records"),
    "tabla8": t8.fillna("").to_dict("records"),
    "tabla9": t9.to_dict("records"),
    "asignabilidad": asig.to_dict("records"),
    "lineas_base": {
        "aleatoria": float(t7.loc[t7["Algoritmo"].str.contains("aleatoria"), "Silueta (Gower)"].iloc[0]),
        "modalidad": float(t7.loc[t7["Algoritmo"].str.contains("modalidad"), "Silueta (Gower)"].iloc[0]),
    },
    "tiempos": tiempos,
    "log": log,
}
out = os.path.join(RES, "reporte_datos.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(datos, f, ensure_ascii=False, indent=1)
print(f"escrito {out}")
print("ganador:", datos["ganador"]["algoritmo"], datos["ganador"]["silueta"])
print("descartados:", [d["algoritmo"] for d in datos["descartados"]])
