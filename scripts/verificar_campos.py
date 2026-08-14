# -*- coding: utf-8 -*-
"""
FASE 1 — Verificación de cobertura de campos en datos OCDS de SERCOP.
Cuenta, por año, en cuántos procesos viene poblado (no nulo, no vacío)
cada campo de interés, y realiza los análisis adicionales solicitados.

Solo stdlib. Uso:  python scripts/verificar_campos.py
"""
import csv
import gzip
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTADOS = os.path.join(BASE, "resultados")

ANIOS = ["2024", "2025", "2026"]

# Orden fijo de los 18 campos a verificar
CAMPOS = [
    "parties[]",
    "parties[].address.region (supplier)",
    "parties[].address.region (buyer)",
    "awards[].suppliers[].id",
    "awards[].value.amount",
    "awards[].items[].classification.id",
    "awards[].items[].description",
    "buyer.id",
    "tender",
    "tender.items[].classification.id",
    "tender.items[].description",
    "tender.value.amount",
    "planning.budget",
    "tender.procurementMethodDetails",
    "tender.status",
    "tender.tenderPeriod.endDate",
    "tender.tenderers[]",
    "bids.details[]",
]


def poblado(v):
    """True si el valor no es nulo ni vacío (cadena vacía, lista/dict vacíos)."""
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True  # números, booleanos


def extraer_release(obj):
    """Devuelve el compiled release según la forma del objeto de la línea."""
    if "releases" in obj and isinstance(obj["releases"], list) and obj["releases"]:
        return obj["releases"][0]
    if "compiledRelease" in obj:
        return obj["compiledRelease"]
    return obj


def region_por_rol(rel, rol):
    """True si alguna party con el rol dado tiene address.region poblado."""
    for p in rel.get("parties") or []:
        if not isinstance(p, dict):
            continue
        roles = p.get("roles") or []
        if rol in roles:
            region = (p.get("address") or {}).get("region")
            if poblado(region):
                return True
    return False


def alguno(lista, fn):
    """True si fn devuelve True para algún elemento de la lista."""
    for x in lista or []:
        if isinstance(x, dict) and fn(x):
            return True
    return False


def verificar_proceso(rel):
    """Devuelve dict campo -> bool (poblado o no) para un proceso."""
    r = {}
    parties = rel.get("parties")
    r["parties[]"] = poblado(parties)
    r["parties[].address.region (supplier)"] = region_por_rol(rel, "supplier")
    r["parties[].address.region (buyer)"] = region_por_rol(rel, "buyer")

    awards = rel.get("awards") or []
    r["awards[].suppliers[].id"] = alguno(
        awards, lambda a: alguno(a.get("suppliers"), lambda s: poblado(s.get("id"))))
    r["awards[].value.amount"] = alguno(
        awards, lambda a: poblado((a.get("value") or {}).get("amount")))
    r["awards[].items[].classification.id"] = alguno(
        awards, lambda a: alguno(a.get("items"),
                                 lambda i: poblado((i.get("classification") or {}).get("id"))))
    r["awards[].items[].description"] = alguno(
        awards, lambda a: alguno(a.get("items"), lambda i: poblado(i.get("description"))))

    r["buyer.id"] = poblado((rel.get("buyer") or {}).get("id"))

    tender = rel.get("tender")
    r["tender"] = poblado(tender)
    t = tender if isinstance(tender, dict) else {}
    r["tender.items[].classification.id"] = alguno(
        t.get("items"), lambda i: poblado((i.get("classification") or {}).get("id")))
    r["tender.items[].description"] = alguno(
        t.get("items"), lambda i: poblado(i.get("description")))
    r["tender.value.amount"] = poblado((t.get("value") or {}).get("amount"))

    budget = (rel.get("planning") or {}).get("budget")
    if isinstance(budget, dict):
        amt = budget.get("amount")
        if isinstance(amt, dict):
            r["planning.budget"] = poblado(amt.get("amount"))
        else:
            r["planning.budget"] = poblado(amt)
    else:
        r["planning.budget"] = False

    r["tender.procurementMethodDetails"] = poblado(t.get("procurementMethodDetails"))
    r["tender.status"] = poblado(t.get("status"))
    r["tender.tenderPeriod.endDate"] = poblado((t.get("tenderPeriod") or {}).get("endDate"))
    r["tender.tenderers[]"] = poblado(t.get("tenderers"))

    bids = rel.get("bids")
    r["bids.details[]"] = poblado((bids or {}).get("details")) if isinstance(bids, dict) else False
    return r


def contar_items(rel):
    """Número total de ítems (tender + awards) para elegir la muestra más pequeña."""
    n = 0
    t = rel.get("tender")
    if isinstance(t, dict):
        n += len(t.get("items") or [])
    for a in rel.get("awards") or []:
        if isinstance(a, dict):
            n += len(a.get("items") or [])
    return n


def main():
    hoy = date.today().isoformat()
    resumen = {}          # anio -> {campo: conteo}, total
    status_por_anio = {}  # anio -> Counter
    modalidades = {}      # anio -> Counter
    tenderers_por_modalidad = Counter()   # (modalidad) -> [con_tenderers, total] solo 2024-2026 juntos
    tenderers_modalidad = defaultdict(lambda: [0, 0])  # modalidad -> [con, total] (todos los años)
    regiones_muestra = []
    vigentes_2026 = 0
    total_2026 = 0
    muestra_min = None    # (n_items, tam_json, rel) del 2026
    otras_rutas = Counter()  # rutas alternativas de presupuesto detectadas

    for anio in ANIOS:
        ruta = os.path.join(DATA, f"{anio}.jsonl.gz")
        conteos = Counter()
        total = 0
        statuses = Counter()
        modas = Counter()

        with gzip.open(ruta, "rt", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    obj = json.loads(linea)
                except json.JSONDecodeError as e:
                    print(f"ADVERTENCIA {anio}: línea inválida ({e})", file=sys.stderr)
                    continue
                rel = extraer_release(obj)
                total += 1
                res = verificar_proceso(rel)
                for campo, ok in res.items():
                    if ok:
                        conteos[campo] += 1

                t = rel.get("tender") if isinstance(rel.get("tender"), dict) else {}
                st = t.get("status")
                statuses[st if poblado(st) else "(sin dato)"] += 1
                mod = t.get("procurementMethodDetails")
                mod_key = mod if poblado(mod) else "(sin dato)"
                modas[mod_key] += 1

                # d) tenderers por modalidad (acumulado en todos los años)
                tenderers_modalidad[mod_key][1] += 1
                if res["tender.tenderers[]"]:
                    tenderers_modalidad[mod_key][0] += 1

                # e) muestra de regiones (primeros 10 valores distintos)
                if len(regiones_muestra) < 10:
                    for p in rel.get("parties") or []:
                        if not isinstance(p, dict):
                            continue
                        reg = (p.get("address") or {}).get("region")
                        if poblado(reg) and reg not in regiones_muestra:
                            regiones_muestra.append(reg)
                            if len(regiones_muestra) >= 10:
                                break

                # rutas alternativas de presupuesto
                planning = rel.get("planning")
                if isinstance(planning, dict):
                    for k in planning.keys():
                        if k not in ("budget",) and "budget" in k.lower():
                            otras_rutas[f"planning.{k}"] += 1
                if isinstance(t, dict):
                    for k in t.keys():
                        if "budget" in k.lower() or "estimat" in k.lower():
                            otras_rutas[f"tender.{k}"] += 1

                if anio == "2026":
                    total_2026 += 1
                    # b) vigencia
                    end = (t.get("tenderPeriod") or {}).get("endDate")
                    if poblado(end) and str(end)[:10] >= hoy:
                        vigentes_2026 += 1
                    # f) proceso más pequeño por número de ítems (desempate: tamaño JSON)
                    clave = (contar_items(rel), len(linea))
                    if muestra_min is None or clave < muestra_min[0]:
                        muestra_min = (clave, rel)

        resumen[anio] = (conteos, total)
        status_por_anio[anio] = statuses
        modalidades[anio] = modas
        print(f"[{anio}] procesados {total} procesos", file=sys.stderr)

        # CSV de cobertura
        csv_path = os.path.join(RESULTADOS, f"cobertura_{anio}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["campo", "procesos_con_dato", "total_procesos", "porcentaje"])
            for campo in CAMPOS:
                n = conteos.get(campo, 0)
                pct = 100.0 * n / total if total else 0.0
                w.writerow([campo, n, total, f"{pct:.2f}"])

    # f) guardar muestra
    if muestra_min:
        with open(os.path.join(RESULTADOS, "muestra_registro.json"), "w", encoding="utf-8") as fm:
            json.dump(muestra_min[1], fm, ensure_ascii=False, indent=2)

    # ---- Salida estructurada (JSON) para armar el reporte ----
    salida = {
        "fecha_hoy": hoy,
        "totales": {a: resumen[a][1] for a in ANIOS},
        "cobertura": {
            campo: {
                a: round(100.0 * resumen[a][0].get(campo, 0) / resumen[a][1], 2)
                if resumen[a][1] else 0.0
                for a in ANIOS
            }
            for campo in CAMPOS
        },
        "status_por_anio": {a: dict(status_por_anio[a]) for a in ANIOS},
        "vigencia_2026": {"vigentes": vigentes_2026, "total": total_2026},
        "top_modalidades": {
            a: modalidades[a].most_common(10) for a in ANIOS
        },
        "tenderers_por_modalidad": {
            mod: {"con_tenderers": c[0], "total": c[1],
                  "pct": round(100.0 * c[0] / c[1], 2) if c[1] else 0.0}
            # top 10 modalidades por volumen total
            for mod, c in sorted(tenderers_modalidad.items(),
                                 key=lambda kv: -kv[1][1])[:10]
        },
        "muestra_regiones": regiones_muestra,
        "otras_rutas_presupuesto": dict(otras_rutas),
    }
    out_path = os.path.join(RESULTADOS, "analisis_fase1.json")
    with open(out_path, "w", encoding="utf-8") as fo:
        json.dump(salida, fo, ensure_ascii=False, indent=2)
    print(f"Resultados escritos en {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
