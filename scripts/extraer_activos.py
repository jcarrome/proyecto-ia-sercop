# -*- coding: utf-8 -*-
"""
FASE 2 - Paso 2: PROCESOS VIGENTES (2025 + 2026, tender.status == "active").

Por proceso extrae:
    ocid, buyer_id, provincia_buyer (parties con rol "buyer"),
    presupuesto      = tender.value.amount
    cpc_tender       = lista de tender.items[].classification.id
    texto_items      = concatenación de tender.items[].description
    modalidad_norm   = normalización de tender.procurementMethodDetails

DESVIACIÓN DOCUMENTADA respecto del diseño: en estos datos sólo ~43 % de los
procesos activos traen tender.value.amount; el resto lleva el presupuesto
desglosado en tender.lots[].value.amount. Aplicar la regla al pie de la letra
descartaría más de la mitad del universo vigente, así que cuando falta
tender.value se usa la SUMA de los montos de los lotes y se reporta el origen
en la columna `origen_presupuesto`. No se inventa ningún valor.

Salida
    resultados/procesos_activos.parquet
"""
import gzip
import json
import os
import sys
import time
import unicodedata
from collections import Counter

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "resultados")
ANIOS_VIGENTES = ("2025", "2026")
N_MODALIDADES_CONSERVADAS = 8


def normalizar_texto_provincia(valor):
    if not valor or not isinstance(valor, str):
        return None
    s = unicodedata.normalize("NFD", valor.strip().upper())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = " ".join(s.split())
    return s or None


def leer_release(linea):
    obj = json.loads(linea)
    if isinstance(obj, dict):
        if "releases" in obj and obj["releases"]:
            return obj["releases"][0]
        if "compiledRelease" in obj:
            return obj["compiledRelease"]
    return obj


def es_catalogo(modalidad):
    if not modalidad:
        return False
    s = unicodedata.normalize("NFD", modalidad.strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return s.startswith("catalogo")


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)

    filas = []
    n_activos = 0
    descartes = Counter()

    for anio in ANIOS_VIGENTES:
        ruta = os.path.join(DATA, f"{anio}.jsonl.gz")
        if not os.path.exists(ruta):
            print(f"[ERROR] no existe {ruta}", file=sys.stderr)
            sys.exit(1)
        print(f"[activos] leyendo {ruta} ...", flush=True)
        with gzip.open(ruta, "rt", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                rel = leer_release(linea)
                tender = rel.get("tender") or {}
                if tender.get("status") != "active":
                    continue
                n_activos += 1

                buyer_id = (rel.get("buyer") or {}).get("id")
                if not buyer_id:
                    descartes["sin_buyer"] += 1
                    continue

                # presupuesto: tender.value y, si falta, suma de lotes
                monto = (tender.get("value") or {}).get("amount")
                origen = "tender.value"
                if monto is None:
                    montos_lote = [(l.get("value") or {}).get("amount")
                                   for l in (tender.get("lots") or [])]
                    montos_lote = [m for m in montos_lote if m is not None]
                    if montos_lote:
                        monto = float(sum(montos_lote))
                        origen = "tender.lots"
                if monto is None or float(monto) <= 0:
                    descartes["sin_presupuesto"] += 1
                    continue

                provincia_buyer = None
                for p in rel.get("parties") or []:
                    if "buyer" in (p.get("roles") or []):
                        provincia_buyer = normalizar_texto_provincia(
                            (p.get("address") or {}).get("region"))
                        if provincia_buyer:
                            break

                cpc_tender, descripciones = set(), []
                for it in tender.get("items") or []:
                    cid = (it.get("classification") or {}).get("id")
                    if cid:
                        cpc_tender.add(str(cid))
                    desc = it.get("description")
                    if desc:
                        descripciones.append(str(desc))

                filas.append({
                    "ocid": rel.get("ocid"),
                    "anio_archivo": anio,
                    # [FASE 3] fecha del release, para el "corte de datos" del panel
                    "fecha": (str(rel.get("date"))[:10] if rel.get("date") else None),
                    "buyer_id": buyer_id,
                    "provincia_buyer": provincia_buyer,
                    "presupuesto": float(monto),
                    "origen_presupuesto": origen,
                    "cpc_tender": "|".join(sorted(cpc_tender)),
                    "n_cpc_tender": len(cpc_tender),
                    "texto_items": " ".join(descripciones),
                    "modalidad_cruda": tender.get("procurementMethodDetails"),
                })

    df = pd.DataFrame(filas)
    # un proceso puede aparecer en 2025 y 2026: nos quedamos con la versión más reciente
    antes = len(df)
    df = df.drop_duplicates(subset="ocid", keep="last").reset_index(drop=True)
    descartes["ocid_duplicado"] = antes - len(df)

    # --- modalidad_norm -----------------------------------------------------
    catalogo = df["modalidad_cruda"].map(es_catalogo)
    no_catalogo = df.loc[~catalogo, "modalidad_cruda"].fillna("Otros")
    top8 = list(no_catalogo.value_counts().head(N_MODALIDADES_CONSERVADAS).index)

    def norm_modalidad(valor):
        if es_catalogo(valor):
            return "Catalogo Electronico"
        v = valor if valor else "Otros"
        return v if v in top8 else "Otros"

    df["modalidad_norm"] = df["modalidad_cruda"].map(norm_modalidad)
    df = df.drop(columns=["modalidad_cruda"])

    salida = os.path.join(RES, "procesos_activos.parquet")
    df.to_parquet(salida, index=False)

    # [FASE 2.1] catálogo de categorías, para que comparar_modelos.py normalice
    # las modalidades históricas de los proveedores con la MISMA regla
    salida_cat = os.path.join(RES, "categorias_modalidad.json")
    with open(salida_cat, "w", encoding="utf-8") as f:
        json.dump({"conservadas": top8,
                   "categorias": sorted(df["modalidad_norm"].unique().tolist())},
                  f, ensure_ascii=False, indent=2)

    dist = df["modalidad_norm"].value_counts()
    print("")
    print("=== PROCESOS ACTIVOS (2025+2026) ===")
    print(f"procesos con tender.status='active' leídos: {n_activos:,}")
    for k, v in descartes.items():
        print(f"  descartados por {k}: {v:,}")
    print(f"PROCESOS ACTIVOS FINALES: {len(df):,}")
    print(f"  presupuesto desde tender.value: "
          f"{int((df['origen_presupuesto'] == 'tender.value').sum()):,}")
    print(f"  presupuesto desde tender.lots:  "
          f"{int((df['origen_presupuesto'] == 'tender.lots').sum()):,}")
    print(f"  sin provincia del comprador:    "
          f"{int(df['provincia_buyer'].isna().sum()):,}")
    print(f"  sin items (texto vacío):        "
          f"{int((df['texto_items'].str.len() == 0).sum()):,}")
    print("")
    print("DISTRIBUCIÓN modalidad_norm:")
    for k, v in dist.items():
        print(f"  {k}: {v:,} ({100.0 * v / len(df):.1f} %)")
    print("")
    print(f"ARCHIVO: {salida}")
    print(f"ARCHIVO: {salida_cat}")
    print(f"TIEMPO: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
