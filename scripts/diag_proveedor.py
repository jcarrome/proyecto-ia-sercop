# -*- coding: utf-8 -*-
"""
Diagnóstico: qué proveedor consultante produce variables con señal real.
Compara proveedores con más historial ('complete') contra el corpus de procesos activos.
Sólo diagnóstico: no escribe la matriz final.
"""
import gzip, json, os, sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from construir_matriz import (DATA, RES, ANIOS, extraer_release, cpcs_de_items,
                              region_de_party, region_de_rol, prefijo_cpc,
                              colapsar_modalidad, items_del_proceso, texto_de_items,
                              monto_proceso)

hist_txt = defaultdict(list); hist_cpc = defaultdict(Counter)
hist_n = Counter(); hist_reg = defaultdict(Counter); hist_montos = defaultdict(list)
hist_cat = Counter(); nombres = {}
act_textos = []; act_cpcs = []

for anio in ANIOS:
    with gzip.open(os.path.join(DATA, f"{anio}.jsonl.gz"), "rt", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea: continue
            try: rel = extraer_release(json.loads(linea))
            except json.JSONDecodeError: continue
            t = rel.get("tender") if isinstance(rel.get("tender"), dict) else {}
            st = t.get("status")
            mod = colapsar_modalidad(t.get("procurementMethodDetails"))

            if st == "active":
                if mod is None or region_de_rol(rel, ("buyer", "procuringEntity")) is None:
                    continue
                its, _ = items_del_proceso(rel)
                if not its: continue
                txt = texto_de_items(rel, its)
                cp = cpcs_de_items(its)
                m, _ = monto_proceso(rel)
                if not txt.strip() or not cp or m is None: continue
                act_textos.append(txt); act_cpcs.append(cp)
                continue

            if st != "complete": continue
            es_cat = (mod == "Catálogo Electrónico")
            for aw in rel.get("awards") or []:
                if not isinstance(aw, dict): continue
                its = [i for i in (aw.get("items") or []) if isinstance(i, dict)]
                txt = " ".join([str(i.get("description") or "") for i in its] +
                               [str((i.get("classification") or {}).get("description") or "")
                                for i in its]).strip()
                cpcs = cpcs_de_items(its)
                monto = (aw.get("value") or {}).get("amount")
                for s in aw.get("suppliers") or []:
                    if not isinstance(s, dict) or not s.get("id"): continue
                    sid = s["id"]; nombres[sid] = s.get("name") or sid
                    hist_n[sid] += 1
                    if es_cat: hist_cat[sid] += 1
                    if txt: hist_txt[sid].append(txt)
                    for c in cpcs: hist_cpc[sid][c] += 1
                    if isinstance(monto, (int, float)) and monto > 0:
                        hist_montos[sid].append(float(monto))
                    r = region_de_party(rel, sid)
                    if r: hist_reg[sid][r] += 1
    print(f"  leído {anio}", file=sys.stderr)

print(f"procesos activos utilizables: {len(act_textos)}", file=sys.stderr)

cands = [s for s in hist_n if hist_reg.get(s) and hist_txt.get(s)
         and len(hist_cpc.get(s, {})) >= 5 and hist_montos.get(s)]
por_total = sorted(cands, key=lambda s: (-hist_n[s], s))[:8]
por_nocat = sorted(cands, key=lambda s: (-(hist_n[s] - hist_cat[s]), s))[:8]
sel = list(dict.fromkeys(por_total + por_nocat))
print(f"candidatos evaluados: {len(sel)}", file=sys.stderr)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
STOP_ES = ["de","la","el","los","las","y","en","para","con","por","del","al","un","una",
           "unos","unas","a","o","que","se","su","sus","e","u","servicio","servicios",
           "adquisicion","adquisición"]

docs = [" ".join(hist_txt[s]) for s in sel] + act_textos
vec = TfidfVectorizer(max_features=5000, stop_words=STOP_ES, lowercase=True,
                      strip_accents="unicode", sublinear_tf=True)
X = vec.fit_transform(docs)
k = len(sel)
S = cosine_similarity(X[:k], X[k:])   # k x n_activos

# cpc_match_score por candidato
def cpc_scores(sid):
    pref = {n: {prefijo_cpc(c, n) for c in hist_cpc[sid]} - {None} for n in (1,2,3,4,5)}
    out = np.empty(len(act_cpcs))
    for j, cps in enumerate(act_cpcs):
        sc = []
        for c in cps:
            mejor = 0
            for n in (1,2,3,4,5):
                p = prefijo_cpc(c, n)
                if p is not None and p in pref[n]: mejor = n
                else: break
            sc.append(mejor/5.0)
        out[j] = np.mean(sc) if sc else 0.0
    return out

filas = []
for i, sid in enumerate(sel):
    sim = S[i]; cm = cpc_scores(sid)
    filas.append({
        "id": sid,
        "nombre": (nombres.get(sid) or "")[:38],
        "hist": hist_n[sid],
        "hist_nocat": hist_n[sid] - hist_cat[sid],
        "%cat": round(100*hist_cat[sid]/hist_n[sid], 1),
        "prov": hist_reg[sid].most_common(1)[0][0][:12],
        "med_monto": round(float(np.median(hist_montos[sid])), 1),
        "sim_med": round(float(sim.mean()), 4),
        "sim_p90": round(float(np.percentile(sim, 90)), 4),
        "sim_max": round(float(sim.max()), 4),
        "sim_cv": round(float(sim.std()/sim.mean()) if sim.mean() > 0 else 0, 2),
        "cpc_med": round(float(cm.mean()), 4),
        "cpc_std": round(float(cm.std()), 4),
        "cpc>0.6": round(float((cm > 0.6).mean()*100), 1),
    })

out = pd.DataFrame(filas).sort_values("hist", ascending=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(out.to_string(index=False))
out.to_csv(os.path.join(RES, "diag_proveedores.csv"), index=False, encoding="utf-8")
