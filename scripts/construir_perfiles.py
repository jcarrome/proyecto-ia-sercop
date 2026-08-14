# -*- coding: utf-8 -*-
"""
FASE 2 - Paso 1: PERFILES DE PROVEEDORES (histórico 2024 + 2025).

Recorre data/2024.jsonl.gz y data/2025.jsonl.gz, agrupa por
awards[].suppliers[].id y construye, por proveedor:

    cpc_historicos          conjunto de awards[].items[].classification.id
    monto_promedio_ganado   media de awards[].value.amount
    num_adjudicaciones      número de awards
    provincia               moda de parties[].address.region con rol "supplier"
    corpus_items            concatenación de awards[].items[].description
    modalidades_hist_json   distribución de tender.procurementMethodDetails
                            (valor CRUDO) de los procesos donde ganó [FASE 2.1]

FASE 2.1: la modalidad se guarda cruda, no normalizada, porque la regla de
normalización ("las 8 no-catálogo más frecuentes, el resto Otros") se define
sobre los procesos ACTIVOS y ese archivo se genera en el paso 2. La
normalización se aplica en comparar_modelos.py, que ya tiene ambos conjuntos.

Salidas
    resultados/perfiles_proveedores.parquet
    resultados/actividad_buyer_cpc.parquet   (auxiliar: tríos buyer/cpc/award
                                              para calcular después la variable
                                              actividad_cpc_comprador sin doble
                                              conteo de un mismo award)
    resultados/proveedor_buyer.parquet       [FASE 2.2] nº de adjudicaciones de
                                              cada proveedor con cada comprador,
                                              insumo de afinidad_comprador
"""
import gzip
import json
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "resultados")
ANIOS_HISTORICOS = ("2024", "2025")


def normalizar_texto_provincia(valor):
    """MAYÚSCULAS sin tildes: 'CAÑAR' -> 'CANAR', 'Manabí' -> 'MANABI'."""
    if not valor or not isinstance(valor, str):
        return None
    s = unicodedata.normalize("NFD", valor.strip().upper())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = " ".join(s.split())
    return s or None


def leer_release(linea):
    """Devuelve el release compilado según el formato de la línea."""
    obj = json.loads(linea)
    if isinstance(obj, dict):
        if "releases" in obj and obj["releases"]:
            return obj["releases"][0]
        if "compiledRelease" in obj:
            return obj["compiledRelease"]
    return obj


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)

    # acumuladores por proveedor
    cpc = defaultdict(set)
    suma_monto = defaultdict(float)
    n_monto = defaultdict(int)
    n_adj = defaultdict(int)
    provincias = defaultdict(Counter)
    corpus = defaultdict(list)
    modalidades = defaultdict(Counter)   # [FASE 2.1] modalidad cruda por proveedor
    nombres = defaultdict(Counter)       # [FASE 2.2] razón social más frecuente
    prov_buyer = Counter()               # [FASE 2.2] (proveedor, comprador) -> awards

    # auxiliar: actividad histórica del comprador por CPC
    filas_actividad = []
    uid_award = 0

    lineas = 0
    awards_totales = 0
    awards_sin_supplier = 0
    awards_sin_monto = 0

    for anio in ANIOS_HISTORICOS:
        ruta = os.path.join(DATA, f"{anio}.jsonl.gz")
        if not os.path.exists(ruta):
            print(f"[ERROR] no existe {ruta}", file=sys.stderr)
            sys.exit(1)
        print(f"[perfiles] leyendo {ruta} ...", flush=True)
        with gzip.open(ruta, "rt", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                rel = leer_release(linea)
                lineas += 1

                # provincia declarada por el proveedor en parties
                region_por_party = {}
                for p in rel.get("parties") or []:
                    roles = p.get("roles") or []
                    if "supplier" in roles and p.get("id"):
                        reg = normalizar_texto_provincia(
                            (p.get("address") or {}).get("region"))
                        if reg:
                            region_por_party[p["id"]] = reg

                buyer_id = (rel.get("buyer") or {}).get("id")
                # [FASE 2.1] modalidad del proceso al que pertenecen los awards
                modalidad_proc = (rel.get("tender") or {}).get(
                    "procurementMethodDetails")

                for a in rel.get("awards") or []:
                    awards_totales += 1
                    uid_award += 1
                    proveedores = [s.get("id") for s in (a.get("suppliers") or [])
                                   if s.get("id")]
                    # [FASE 2.2] razón social declarada en el award
                    for s in a.get("suppliers") or []:
                        if s.get("id") and s.get("name"):
                            nombres[s["id"]][str(s["name"]).strip()] += 1
                    monto = (a.get("value") or {}).get("amount")
                    if monto is None:
                        awards_sin_monto += 1
                    items = a.get("items") or []
                    cpcs_award = set()
                    descripciones = []
                    for it in items:
                        cid = (it.get("classification") or {}).get("id")
                        if cid:
                            cpcs_award.add(str(cid))
                        desc = it.get("description")
                        if desc:
                            descripciones.append(str(desc))

                    if buyer_id:
                        for cid in cpcs_award:
                            filas_actividad.append((buyer_id, cid, uid_award))

                    if not proveedores:
                        awards_sin_supplier += 1
                        continue

                    for pid in proveedores:
                        n_adj[pid] += 1
                        if monto is not None:
                            suma_monto[pid] += float(monto)
                            n_monto[pid] += 1
                        if cpcs_award:
                            cpc[pid].update(cpcs_award)
                        if descripciones:
                            corpus[pid].extend(descripciones)
                        reg = region_por_party.get(pid)
                        if reg:
                            provincias[pid][reg] += 1
                        modalidades[pid][modalidad_proc or "(sin dato)"] += 1
                        if buyer_id:
                            prov_buyer[(pid, buyer_id)] += 1

    print(f"[perfiles] líneas leídas: {lineas:,} | awards: {awards_totales:,}",
          flush=True)

    filas = []
    for pid, n in n_adj.items():
        prov = provincias[pid].most_common(1)[0][0] if provincias[pid] else None
        texto = " ".join(corpus[pid]) if corpus[pid] else ""
        filas.append({
            "proveedor_id": pid,
            "nombre": (nombres[pid].most_common(1)[0][0]
                       if nombres[pid] else None),
            "num_adjudicaciones": n,
            "monto_promedio_ganado": (suma_monto[pid] / n_monto[pid]
                                      if n_monto[pid] else None),
            "provincia": prov,
            "n_cpc_historicos": len(cpc[pid]),
            "cpc_historicos": "|".join(sorted(cpc[pid])),
            "len_corpus": len(texto),
            "corpus_items": texto,
            "modalidades_hist_json": json.dumps(dict(modalidades[pid]),
                                                ensure_ascii=False),
        })

    df = pd.DataFrame(filas).sort_values("num_adjudicaciones", ascending=False)
    salida = os.path.join(RES, "perfiles_proveedores.parquet")
    df.to_parquet(salida, index=False)

    act = pd.DataFrame(filas_actividad,
                       columns=["buyer_id", "cpc", "award_uid"])
    act = act.drop_duplicates()
    salida_act = os.path.join(RES, "actividad_buyer_cpc.parquet")
    act.to_parquet(salida_act, index=False)

    # [FASE 2.2] adjudicaciones de cada proveedor con cada comprador
    pb = pd.DataFrame([{"proveedor_id": p, "buyer_id": b, "n_awards": n}
                       for (p, b), n in prov_buyer.items()])
    salida_pb = os.path.join(RES, "proveedor_buyer.parquet")
    pb.to_parquet(salida_pb, index=False)

    con_prov = int(df["provincia"].notna().sum())
    con_corpus = int((df["len_corpus"] > 0).sum())
    con_monto = int(df["monto_promedio_ganado"].notna().sum())

    print("")
    print("=== PERFILES DE PROVEEDORES (2024+2025) ===")
    print(f"PROVEEDORES ÚNICOS: {len(df):,}")
    print(f"  con provincia válida .......... {con_prov:,}")
    print(f"  con corpus de items no vacío .. {con_corpus:,}")
    print(f"  con monto promedio ............ {con_monto:,}")
    print(f"  awards sin proveedor .......... {awards_sin_supplier:,}")
    print(f"  awards sin monto .............. {awards_sin_monto:,}")
    print(f"ARCHIVO: {salida}")
    print(f"ARCHIVO: {salida_act}  ({len(act):,} tríos buyer/cpc/award)")
    print(f"ARCHIVO: {salida_pb}  ({len(pb):,} pares proveedor/comprador)")
    print(f"TIEMPO: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
