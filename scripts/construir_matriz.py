# -*- coding: utf-8 -*-
"""
PASO 0 + construcción de la matriz de comparación.

Lee los OCDS locales (data/*.jsonl.gz), deduplica por ocid, elige el proveedor
consultante de forma determinista, y arma la matriz de procesos ACTIVOS descritos
por variables de interacción con ese proveedor.

Salidas:
  resultados/matriz_procesos.csv        matriz cruda (una fila por proceso activo)
  resultados/paso0_inventario.json      cifras del inventario

Uso:  .venv\\Scripts\\python.exe scripts/construir_matriz.py
"""
import gzip
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "resultados")
ANIOS = ["2024", "2025", "2026"]
SEED = 42

# ---------------------------------------------------------------- geografía
# Centroides aproximados (capital provincial) de las 24 provincias del Ecuador.
# Fuente: coordenadas públicas de las capitales; se usan sólo para la haversine.
PROVINCIAS = {
    "AZUAY": (-2.900, -79.005),
    "BOLIVAR": (-1.594, -79.001),
    "CANAR": (-2.554, -78.939),
    "CARCHI": (0.812, -77.717),
    "CHIMBORAZO": (-1.664, -78.654),
    "COTOPAXI": (-0.933, -78.616),
    "EL ORO": (-3.259, -79.961),
    "ESMERALDAS": (0.968, -79.652),
    "GALAPAGOS": (-0.744, -90.314),
    "GUAYAS": (-2.190, -79.889),
    "IMBABURA": (0.350, -78.122),
    "LOJA": (-3.993, -79.204),
    "LOS RIOS": (-1.803, -79.535),
    "MANABI": (-1.055, -80.454),
    "MORONA SANTIAGO": (-2.308, -78.117),
    "NAPO": (-0.994, -77.813),
    "ORELLANA": (-0.462, -76.987),
    "PASTAZA": (-1.487, -78.002),
    "PICHINCHA": (-0.225, -78.512),
    "SANTA ELENA": (-2.227, -80.859),
    "SANTO DOMINGO DE LOS TSACHILAS": (-0.253, -79.175),
    "SUCUMBIOS": (0.087, -76.888),
    "TUNGURAHUA": (-1.241, -78.620),
    "ZAMORA CHINCHIPE": (-4.068, -78.955),
}

ALIAS_PROV = {
    "SANTO DOMINGO": "SANTO DOMINGO DE LOS TSACHILAS",
    "SANTO DOMINGO DE LOS TSACHILA": "SANTO DOMINGO DE LOS TSACHILAS",
    "STO DOMINGO DE LOS TSACHILAS": "SANTO DOMINGO DE LOS TSACHILAS",
    "SUCUMBIOS ": "SUCUMBIOS",
    "CANIAR": "CANAR",
    "MANABI ": "MANABI",
}


def sin_tildes(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_prov(s):
    """Normaliza el nombre de provincia a una clave de PROVINCIAS o None."""
    if not isinstance(s, str) or not s.strip():
        return None
    k = sin_tildes(s).upper().strip()
    k = re.sub(r"\s+", " ", k)
    k = ALIAS_PROV.get(k, k)
    return k if k in PROVINCIAS else None


def haversine_km(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


# Matriz de distancias entre provincias, precalculada
DIST_PROV = {
    (p, q): haversine_km(PROVINCIAS[p], PROVINCIAS[q])
    for p in PROVINCIAS for q in PROVINCIAS
}

# ---------------------------------------------------------------- utilidades OCDS


def extraer_release(obj):
    if "releases" in obj and isinstance(obj["releases"], list) and obj["releases"]:
        return obj["releases"][0]
    if "compiledRelease" in obj:
        return obj["compiledRelease"]
    return obj


def region_de_rol(rel, roles_buscados):
    for p in rel.get("parties") or []:
        if not isinstance(p, dict):
            continue
        roles = p.get("roles") or []
        if any(r in roles for r in roles_buscados):
            reg = norm_prov((p.get("address") or {}).get("region"))
            if reg:
                return reg
    return None


def region_de_party(rel, party_id):
    for p in rel.get("parties") or []:
        if isinstance(p, dict) and p.get("id") == party_id:
            return norm_prov((p.get("address") or {}).get("region"))
    return None


def items_del_proceso(rel):
    """Ítems de tender; si no hay, cae a los ítems de awards. Devuelve (lista, origen)."""
    t = rel.get("tender") if isinstance(rel.get("tender"), dict) else {}
    its = [i for i in (t.get("items") or []) if isinstance(i, dict)]
    if its:
        return its, "tender.items"
    its = []
    for a in rel.get("awards") or []:
        if isinstance(a, dict):
            its.extend([i for i in (a.get("items") or []) if isinstance(i, dict)])
    return its, ("awards.items" if its else "sin_items")


def cpcs_de_items(items):
    out = []
    for i in items:
        cid = (i.get("classification") or {}).get("id")
        if isinstance(cid, (str, int)) and str(cid).strip():
            out.append(str(cid).strip())
        for ac in i.get("additionalClassifications") or []:
            if isinstance(ac, dict) and str(ac.get("id") or "").strip():
                out.append(str(ac["id"]).strip())
    return out


def texto_de_items(rel, items):
    partes = []
    t = rel.get("tender") if isinstance(rel.get("tender"), dict) else {}
    for campo in ("description", "title"):
        v = t.get(campo)
        if isinstance(v, str) and v.strip():
            partes.append(v.strip())
    for i in items:
        for campo in ("description",):
            v = i.get(campo)
            if isinstance(v, str) and v.strip():
                partes.append(v.strip())
        cl = i.get("classification") or {}
        if isinstance(cl.get("description"), str) and cl["description"].strip():
            partes.append(cl["description"].strip())
    return " ".join(partes)


def monto_proceso(rel):
    """Monto de referencia: tender.value -> suma de lots -> suma de awards."""
    t = rel.get("tender") if isinstance(rel.get("tender"), dict) else {}
    v = (t.get("value") or {}).get("amount")
    if isinstance(v, (int, float)) and v > 0:
        return float(v), "tender.value"
    s = 0.0
    for lot in t.get("lots") or []:
        if isinstance(lot, dict):
            a = (lot.get("value") or {}).get("amount")
            if isinstance(a, (int, float)):
                s += float(a)
    if s > 0:
        return s, "tender.lots"
    s = 0.0
    for aw in rel.get("awards") or []:
        if isinstance(aw, dict):
            a = (aw.get("value") or {}).get("amount")
            if isinstance(a, (int, float)):
                s += float(a)
    if s > 0:
        return s, "awards.value"
    return None, "sin_monto"


RE_CATALOGO = re.compile(r"cat[aá]logo\s+electr[oó]nico", re.IGNORECASE)


def colapsar_modalidad(m):
    """Colapsa todos los convenios de Catálogo Electrónico en una sola categoría."""
    if not isinstance(m, str) or not m.strip():
        return None
    m = m.strip()
    if RE_CATALOGO.search(sin_tildes(m)) or RE_CATALOGO.search(m):
        return "Catálogo Electrónico"
    return m


def prefijo_cpc(c, n):
    """Primeros n dígitos del CPC, ignorando puntos."""
    d = re.sub(r"\D", "", str(c))
    return d[:n] if len(d) >= n else None


# ---------------------------------------------------------------- lectura


def leer_todo():
    """Devuelve dict ocid -> release deduplicado (se queda con el 'date' más reciente)."""
    por_ocid = {}
    dups = 0
    lineas = 0
    for anio in ANIOS:
        ruta = os.path.join(DATA, f"{anio}.jsonl.gz")
        with gzip.open(ruta, "rt", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                lineas += 1
                try:
                    rel = extraer_release(json.loads(linea))
                except json.JSONDecodeError:
                    continue
                ocid = rel.get("ocid") or rel.get("id")
                if not ocid:
                    continue
                prev = por_ocid.get(ocid)
                if prev is None:
                    por_ocid[ocid] = rel
                else:
                    dups += 1
                    if str(rel.get("date") or "") > str(prev.get("date") or ""):
                        por_ocid[ocid] = rel
        print(f"  leído {anio}", file=sys.stderr)
    return por_ocid, lineas, dups


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    print("Leyendo OCDS...", file=sys.stderr)
    releases, n_lineas, n_dups = leer_todo()
    print(f"  {n_lineas} líneas, {len(releases)} ocid únicos, {n_dups} duplicados colapsados",
          file=sys.stderr)

    # ---- clasificar por estado
    activos, completos = [], []
    status_c = Counter()
    for ocid, rel in releases.items():
        t = rel.get("tender") if isinstance(rel.get("tender"), dict) else {}
        st = t.get("status")
        status_c[st if isinstance(st, str) and st.strip() else "(sin dato)"] += 1
        if st == "active":
            activos.append((ocid, rel))
        elif st == "complete":
            completos.append((ocid, rel))

    print(f"  activos={len(activos)}  completos={len(completos)}", file=sys.stderr)

    # ---- historial de proveedores: SÓLO procesos 'complete' (evita fuga de información
    #      desde las filas que vamos a agrupar)
    hist_textos = defaultdict(list)     # supplier_id -> [texto de ítems adjudicados]
    hist_cpc = defaultdict(Counter)     # supplier_id -> Counter(cpc completo)
    hist_montos = defaultdict(list)     # supplier_id -> [montos adjudicados]
    hist_procesos = Counter()           # supplier_id -> nº de procesos
    hist_region = defaultdict(Counter)  # supplier_id -> Counter(provincia)
    nombre_prov = {}

    for ocid, rel in completos:
        for aw in rel.get("awards") or []:
            if not isinstance(aw, dict):
                continue
            its = [i for i in (aw.get("items") or []) if isinstance(i, dict)]
            txt = " ".join(
                [str(i.get("description") or "") for i in its]
                + [str((i.get("classification") or {}).get("description") or "") for i in its]
            ).strip()
            cpcs = cpcs_de_items(its)
            monto = (aw.get("value") or {}).get("amount")
            for s in aw.get("suppliers") or []:
                if not isinstance(s, dict) or not s.get("id"):
                    continue
                sid = s["id"]
                nombre_prov[sid] = s.get("name") or sid
                hist_procesos[sid] += 1
                if txt:
                    hist_textos[sid].append(txt)
                for c in cpcs:
                    hist_cpc[sid][c] += 1
                if isinstance(monto, (int, float)) and monto > 0:
                    hist_montos[sid].append(float(monto))
                reg = region_de_party(rel, sid)
                if reg:
                    hist_region[sid][reg] += 1

    # ---- proveedor consultante FIJADO (decisión del usuario tras el diagnóstico de
    #      scripts/diag_proveedor.py: es el candidato con variables más discriminantes).
    PROV = "EC-RUC-1790475689001-5192"   # ROCHE ECUADOR S.A.
    if PROV not in hist_procesos:
        print(f"ERROR: el proveedor {PROV} no tiene historial 'complete'", file=sys.stderr)
        sys.exit(1)
    prov_region = hist_region[PROV].most_common(1)[0][0]
    prov_cpc_full = set(hist_cpc[PROV])
    prov_cpc_pref = {n: {prefijo_cpc(c, n) for c in prov_cpc_full} - {None} for n in (1, 2, 3, 4, 5)}
    prov_mediana_monto = float(np.median(hist_montos[PROV]))
    corpus_prov = " ".join(hist_textos[PROV])

    print(f"  proveedor consultante: {PROV} ({nombre_prov.get(PROV)}) "
          f"| {hist_procesos[PROV]} procesos | {prov_region} | "
          f"{len(prov_cpc_full)} CPC distintos", file=sys.stderr)

    # ---- actividad del comprador en los CPC del proveedor (histórico 'complete')
    prov_cpc3 = prov_cpc_pref[3]
    act_comprador = Counter()
    for ocid, rel in completos:
        bid = (rel.get("buyer") or {}).get("id")
        if not bid:
            continue
        its, _ = items_del_proceso(rel)
        c3 = {prefijo_cpc(c, 3) for c in cpcs_de_items(its)} - {None}
        if c3 & prov_cpc3:
            act_comprador[bid] += 1

    # ---- construir filas de procesos activos
    filas = []
    descartes = Counter()
    for ocid, rel in activos:
        t = rel.get("tender") if isinstance(rel.get("tender"), dict) else {}

        modalidad = colapsar_modalidad(t.get("procurementMethodDetails"))
        if modalidad is None:
            descartes["sin_modalidad"] += 1
            continue

        reg_buyer = region_de_rol(rel, ("buyer", "procuringEntity"))
        if reg_buyer is None:
            descartes["sin_provincia_comprador"] += 1
            continue

        items, origen_items = items_del_proceso(rel)
        if not items:
            descartes["sin_items"] += 1
            continue

        texto = texto_de_items(rel, items)
        if not texto.strip():
            descartes["sin_texto"] += 1
            continue

        cpcs = cpcs_de_items(items)
        if not cpcs:
            descartes["sin_cpc"] += 1
            continue

        monto, origen_monto = monto_proceso(rel)
        if monto is None:
            descartes["sin_monto"] += 1
            continue

        # cpc_match_score: profundidad de coincidencia jerárquica normalizada,
        # promediada sobre los CPC del proceso.
        scores = []
        for c in cpcs:
            mejor = 0
            for n in (1, 2, 3, 4, 5):
                p = prefijo_cpc(c, n)
                if p is not None and p in prov_cpc_pref[n]:
                    mejor = n
                else:
                    break
            scores.append(mejor / 5.0)
        cpc_match = float(np.mean(scores)) if scores else 0.0

        bid = (rel.get("buyer") or {}).get("id")
        filas.append({
            "ocid": ocid,
            "buyer_id": bid,
            "modalidad_contratacion": modalidad,
            "provincia_comprador": reg_buyer,
            "distancia_geografica_km": DIST_PROV[(prov_region, reg_buyer)],
            "cpc_match_score": cpc_match,
            "monto": monto,
            "actividad_cpc_comprador": float(act_comprador.get(bid, 0)),
            "texto": texto,
            "origen_items": origen_items,
            "origen_monto": origen_monto,
        })

    df = pd.DataFrame(filas)
    print(f"  filas construidas: {len(df)}  (descartes: {dict(descartes)})", file=sys.stderr)

    # ---- similitud semántica TF-IDF (máx 5000 términos), sin embeddings preentrenados
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    STOP_ES = [
        "de", "la", "el", "los", "las", "y", "en", "para", "con", "por", "del", "al",
        "un", "una", "unos", "unas", "a", "o", "que", "se", "su", "sus", "e", "u",
        "servicio", "servicios", "adquisicion", "adquisición",
    ]
    corpus = [corpus_prov] + df["texto"].tolist()
    vec = TfidfVectorizer(max_features=5000, stop_words=STOP_ES, lowercase=True,
                          strip_accents="unicode", sublinear_tf=True)
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[0], X[1:]).ravel()
    df["sim_semantica_tfidf"] = sims
    print(f"  TF-IDF: {X.shape[1]} términos | sim media={sims.mean():.4f} "
          f"max={sims.max():.4f}", file=sys.stderr)

    # ---- desviación de presupuesto (en escala log, respecto a la mediana histórica)
    df["desviacion_presupuesto"] = np.log1p(df["monto"]) - math.log1p(prov_mediana_monto)

    # ---- guardar matriz cruda
    cols = ["ocid", "buyer_id", "provincia_comprador", "modalidad_contratacion",
            "distancia_geografica_km", "cpc_match_score", "sim_semantica_tfidf",
            "desviacion_presupuesto", "actividad_cpc_comprador", "monto",
            "origen_items", "origen_monto", "texto"]
    df[cols].to_csv(os.path.join(RES, "matriz_procesos.csv"), index=False, encoding="utf-8")

    # ---- PASO 0: inventario
    n = len(df)
    n_modalidades = df["modalidad_contratacion"].nunique()
    # columnas de la matriz codificada (one-hot de la modalidad + 5 numéricas)
    n_cols_onehot = 5 + n_modalidades
    bytes_gower_64 = n * n * 8
    bytes_gower_32 = n * n * 4

    inv = {
        "seed": SEED,
        "lineas_leidas": n_lineas,
        "ocid_unicos": len(releases),
        "duplicados_colapsados": n_dups,
        "status": dict(status_c),
        "activos_tender_status": len(activos),
        "completos": len(completos),
        "proveedor_consultante": {
            "id": PROV,
            "nombre": nombre_prov.get(PROV),
            "procesos_historicos": hist_procesos[PROV],
            "provincia": prov_region,
            "cpc_distintos": len(prov_cpc_full),
            "mediana_monto_historico": prov_mediana_monto,
            "docs_corpus": len(hist_textos[PROV]),
        },
        "descartes_por_falta_de_datos": dict(descartes),
        "matriz": {
            "filas": int(n),
            "columnas_semanticas": 6,
            "columnas_codificadas_onehot": int(n_cols_onehot),
            "modalidades_distintas": int(n_modalidades),
        },
        "gower_n2": {
            "celdas": int(n * n),
            "float64_GB": round(bytes_gower_64 / 1024 ** 3, 3),
            "float32_GB": round(bytes_gower_32 / 1024 ** 3, 3),
            "condensada_float64_GB": round(n * (n - 1) / 2 * 8 / 1024 ** 3, 3),
        },
        "tfidf_terminos": int(X.shape[1]),
        "modalidades_top": df["modalidad_contratacion"].value_counts().head(15).to_dict(),
        "origen_items": df["origen_items"].value_counts().to_dict(),
        "origen_monto": df["origen_monto"].value_counts().to_dict(),
        "segundos": round(time.time() - t0, 1),
    }
    with open(os.path.join(RES, "paso0_inventario.json"), "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=2)
    print(json.dumps(inv, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
