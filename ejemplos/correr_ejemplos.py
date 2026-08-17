# -*- coding: utf-8 -*-
"""
CONJUNTO DE EJEMPLOS PARA CORRER EL MODELO

Pasa los diez RUCs de `ejemplos_rucs.csv` por el modelo entrenado
(resultados/modelo_ganador.pkl) y compara el resultado con el esperado.

    python ejemplos/correr_ejemplos.py

No necesita los datos crudos ni reentrenar: usa el modelo guardado y los
perfiles ya calculados. Tarda unos 10 segundos en total.

Devuelve código de salida 0 si los diez casos coinciden con lo esperado, y 1
si alguno discrepa (así sirve como prueba de regresión).
"""
import os
import sys
import time
import types

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AQUI = os.path.dirname(os.path.abspath(__file__))

# El motor del panel vive en app/dashboard.py y usa los decoradores de caché de
# Streamlit. Para correrlo desde consola se sustituye Streamlit por un doble
# mínimo: sólo hacen falta los decoradores y set_page_config.
_st = types.ModuleType("streamlit")


def _cache(*a, **k):
    if a and callable(a[0]):
        return a[0]
    return lambda fn: fn


_st.cache_resource = _cache
_st.cache_data = _cache
_st.set_page_config = lambda **k: None


class _ConfigColumnas:
    @staticmethod
    def TextColumn(*a, **k):
        return None

    @staticmethod
    def NumberColumn(*a, **k):
        return None

    @staticmethod
    def ProgressColumn(*a, **k):
        return None


_st.column_config = _ConfigColumnas
sys.modules["streamlit"] = _st

sys.path.insert(0, os.path.join(BASE, "app"))

import pandas as pd            # noqa: E402
import dashboard as panel      # noqa: E402

ANCHO = 96


def main():
    ruta_csv = os.path.join(AQUI, "ejemplos_rucs.csv")
    if not os.path.exists(ruta_csv):
        print(f"[ERROR] no existe {ruta_csv}", file=sys.stderr)
        return 1
    casos = pd.read_csv(ruta_csv, dtype={"ruc": str})

    datos, err = panel.cargar_todo()
    if err:
        print(f"[ERROR] no se pudo cargar el modelo: {err}", file=sys.stderr)
        return 1

    m = datos["modelo"]
    print("=" * ANCHO)
    print("EJEMPLOS PARA CORRER EL MODELO")
    print("=" * ANCHO)
    print(f"  modelo          : {m['algoritmo_nombre']}, k={m['k']} "
          f"(Fase {m['fase']}, semilla {m['seed']})")
    print(f"  silueta-Gower   : {m['silueta_promedio_portafolio']:.4f} "
          f"(promedio de 3 proveedores de referencia)")
    print(f"  procesos activos: {len(datos['procesos']):,}")
    print(f"  proveedores     : {len(datos['perfiles']):,}")
    print(f"  corte de datos  : {datos['fecha_corte']}")
    print(f"  umbrales        : alta >= {panel.UMBRAL_ALTA} | "
          f"media >= {panel.UMBRAL_MEDIA} | baja >= {panel.UMBRAL_BAJA}")
    print()

    cab = (f"{'#':>2}  {'RUC':13}  {'caso':26} {'obtenido':10} "
           f"{'esperado':10} {'oport.':>7} {'compat':>7} {'s':>5}  ok")
    print(cab)
    print("-" * ANCHO)

    fallos = []
    for _, c in casos.iterrows():
        t0 = time.time()
        r = panel.analizar_proveedor(c["ruc"])
        dt = time.time() - t0

        if r is None:
            obtenido, oport, compat = "EXPLORACION", 0, 0.0
        elif r.get("sin_datos_suficientes"):
            obtenido, oport, compat = "EXPLORACION", 0, 0.0
        else:
            obtenido = r["nivel"]
            oport = r["n_oportunidades"]
            compat = r["compat_media_oportunidades"]

        ok = obtenido == str(c["veredicto_esperado"]).strip()
        if not ok:
            fallos.append((c["ruc"], c["caso"], obtenido,
                           c["veredicto_esperado"]))
        print(f"{int(c['n']):>2}  {c['ruc']:13}  {str(c['caso'])[:26]:26} "
              f"{obtenido:10} {str(c['veredicto_esperado']):10} "
              f"{oport:7,} {compat:7.3f} {dt:5.2f}  "
              f"{'OK' if ok else 'XX'}")

    print("-" * ANCHO)
    print(f"\ncasos: {len(casos)}  |  coinciden: {len(casos) - len(fallos)}  "
          f"|  discrepancias: {len(fallos)}")

    if fallos:
        print("\nDISCREPANCIAS:")
        for ruc, caso, obt, esp in fallos:
            print(f"  RUC {ruc} ({caso}): se obtuvo {obt}, se esperaba {esp}")
        return 1

    print("\nTodos los ejemplos coinciden con el resultado esperado.")
    print("\nPara ver estos mismos casos en la interfaz:")
    print("    streamlit run app/dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
