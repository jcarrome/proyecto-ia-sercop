# -*- coding: utf-8 -*-
"""
FASE 3 — DASHBOARD DEL PROVEEDOR (prototipo local, sin APIs).

Materializa los casos de uso del diseño sobre el modelo ganador de la Fase 2.2
(K-Medoids / PAM, k=3). NO reimplementa el cálculo: importa comparar_modelos.py
y reutiliza construir_matriz, winsorizar y la distancia de Gower ponderada.

Ejecución:
    streamlit run app/dashboard.py

Asignación de un proceso a un grupo (mismo pipeline que la Fase 2.2):
    matriz de interacción -> winsorizar con los límites guardados ->
    estandarizar con el escalador guardado -> distancia de Gower ponderada
    contra los 3 medoides -> grupo del medoide más cercano.
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(BASE, "scripts")
RES = os.path.join(BASE, "resultados")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

st.set_page_config(page_title="Compatibilidad proveedor–licitación · SERCOP",
                   page_icon="📊", layout="wide")

PATRON_ID = re.compile(r"(\d{13})")
AVISO_DATOS = ("Los datos provienen de un corte local del portal de datos "
               "abiertos del SERCOP; el sistema no consume servicios externos.")
AVISO_AFINIDAD = ("La afinidad mide la cercanía al perfil del grupo; "
                  "no representa probabilidad de adjudicación.")

ETIQUETAS_VARIABLES = {
    "distancia_km": "Distancia (km)",
    "cpc_jaccard4": "Coincidencia de rubros (CPC)",
    "sim_tfidf": "Similitud textual",
    "desviacion_presupuesto": "Desviación de presupuesto (log)",
    "actividad_cpc_comprador": "Actividad del comprador en el rubro",
    "afinidad_comprador": "Relación previa con el comprador (log)",
}


# =========================================================== carga cacheada
@st.cache_resource(show_spinner=False)
def cargar_todo():
    """Carga modelo y datos una sola vez. Devuelve (datos, error)."""
    import json
    import joblib
    import comparar_modelos as cm

    faltantes = [n for n in ("modelo_ganador.pkl", "perfiles_proveedores.parquet",
                             "procesos_activos.parquet", "actividad_buyer_cpc.parquet",
                             "proveedor_buyer.parquet", "categorias_modalidad.json")
                 if not os.path.exists(os.path.join(RES, n))]
    if faltantes:
        return None, ("Faltan archivos en resultados/: " + ", ".join(faltantes)
                      + ". Ejecute primero scripts/construir_perfiles.py, "
                      "scripts/extraer_activos.py y scripts/comparar_modelos.py.")
    try:
        modelo = joblib.load(os.path.join(RES, "modelo_ganador.pkl"))
        perfiles = pd.read_parquet(os.path.join(RES, "perfiles_proveedores.parquet"))
        procesos = pd.read_parquet(os.path.join(RES, "procesos_activos.parquet"))
        actividad = pd.read_parquet(os.path.join(RES, "actividad_buyer_cpc.parquet"))
        prov_buyer = pd.read_parquet(os.path.join(RES, "proveedor_buyer.parquet"))
        with open(os.path.join(RES, "categorias_modalidad.json"), encoding="utf-8") as f:
            cats = json.load(f)
    except Exception as e:
        return None, f"No se pudieron leer los archivos del modelo: {type(e).__name__}: {e}"

    if modelo.get("medoides_ocid") is None or modelo.get("gower_rangos") is None:
        return None, ("modelo_ganador.pkl no contiene los medoides o los rangos "
                      "de Gower. Vuelva a ejecutar scripts/comparar_modelos.py.")

    perfiles = perfiles.copy()
    perfiles["ruc"] = perfiles["proveedor_id"].str.extract(PATRON_ID)[0]

    datos = {
        "cm": cm,
        "modelo": modelo,
        "perfiles": perfiles,
        "procesos": procesos.reset_index(drop=True),
        "idx_act": cm.indice_actividad(actividad),
        "idx_pb": cm.indice_proveedor_buyer(prov_buyer),
        "conservadas": set(cats["conservadas"]),
        "fecha_corte": str(procesos["fecha"].max()) if "fecha" in procesos.columns else "n/d",
    }
    return datos, None


@st.cache_data(show_spinner=False)
def rucs_de_ejemplo():
    """P1/P2/P3 de la Fase 2.2, recalculados con la MISMA función de selección."""
    datos, err = cargar_todo()
    if err:
        return []
    cm = datos["cm"]
    elegidos, _ = cm.elegir_proveedores(datos["perfiles"], datos["conservadas"], 3)
    salida = []
    for i, (_, r) in enumerate(elegidos.iterrows(), 1):
        m = PATRON_ID.search(str(r["proveedor_id"]))
        if m:
            salida.append({"alias": f"P{i}", "ruc": m.group(1),
                           "provincia": r["provincia"],
                           "competitivas": int(r["adj_competitivas"])})
    return salida


# ====================================================== motor de asignación
@st.cache_data(show_spinner=False, max_entries=32)
def analizar_proveedor(ruc):
    """Aplica el pipeline de la Fase 2.2 al proveedor indicado.

    Devuelve un dict con la tabla de procesos, el perfil de cada grupo y el
    grupo asignado al proveedor, o None si el RUC no tiene historial.
    """
    datos, err = cargar_todo()
    if err:
        return None
    cm, modelo = datos["cm"], datos["modelo"]
    perfiles, procesos = datos["perfiles"], datos["procesos"]

    coincide = perfiles[perfiles["ruc"] == ruc]
    if coincide.empty:
        return None
    fila = coincide.iloc[0].copy()
    if fila["provincia"] not in cm.CAPITALES or pd.isna(fila["monto_promedio_ganado"]):
        return {"sin_datos_suficientes": True,
                "motivo": ("El proveedor existe en el histórico pero no tiene "
                           "provincia reconocida o monto promedio adjudicado, "
                           "que el modelo necesita para ubicarlo.")}

    # modalidades normalizadas del histórico (las usa el perfil, no el modelo)
    import json
    from collections import Counter
    cruda = json.loads(fila["modalidades_hist_json"])
    norm = Counter()
    for k, v in cruda.items():
        norm[cm.normalizar_modalidad(k, datos["conservadas"])] += v
    fila["modalidades_norm"] = dict(norm)

    t0 = time.time()
    # --- MISMO pipeline de la Fase 2.2 -----------------------------------
    X, meta = cm.construir_matriz(fila, procesos, datos["idx_act"], datos["idx_pb"])

    # winsorizar con los LÍMITES GUARDADOS (no recalculados)
    # pandas 3 puede devolver una vista de sólo lectura: copiar antes de recortar
    crudas = np.array(X[cm.NUMERICAS].to_numpy(dtype=float), copy=True)
    lims = modelo["winsor_limites"]
    for j, c in enumerate(cm.NUMERICAS):
        lo, hi = lims[c]
        crudas[:, j] = np.clip(crudas[:, j], lo, hi)
    # estandarizar con el ESCALADOR GUARDADO
    Z = (crudas - np.asarray(modelo["escalador_media"])) / np.asarray(
        modelo["escalador_escala"])
    cat = X[cm.CATEGORICA].astype(str).to_numpy()

    # distancia de Gower ponderada contra los 3 medoides
    grupos_ordenados = sorted(modelo["medoides_ocid"].keys())
    ref_num = np.asarray(modelo["medoides_num_std"], dtype=float)
    ref_cat = np.asarray(modelo["medoides_modalidad"], dtype=object)
    D = cm.distancia_gower_a_referencias(
        Z, cat, ref_num, ref_cat, np.asarray(modelo["gower_rangos"], dtype=float),
        pesos=np.asarray(modelo["pesos_gower"], dtype=float))

    idx_cercano = np.argmin(D, axis=1)
    etiquetas = np.array([grupos_ordenados[i] for i in idx_cercano])
    d_asignada = D[np.arange(len(D)), idx_cercano]

    # afinidad = 1 - distancia normalizada (min-max sobre esta consulta)
    d_min, d_max = float(d_asignada.min()), float(d_asignada.max())
    if d_max > d_min:
        afinidad = 1.0 - (d_asignada - d_min) / (d_max - d_min)
    else:
        afinidad = np.ones_like(d_asignada)

    # atípicos con el IsolationForest GLOBAL guardado
    marca_atipico = np.zeros(len(Z), dtype=bool)
    try:
        categorias = list(modelo["categorias_onehot"])
        onehot = np.zeros((len(Z), len(categorias)), dtype=float)
        pos = {c: j for j, c in enumerate(categorias)}
        for i, c in enumerate(cat):
            if c in pos:
                onehot[i, pos[c]] = 1.0
        X_cod = np.hstack([Z, onehot])
        marca_atipico = modelo["iso_global"].predict(X_cod) == -1
    except Exception:
        pass   # el aviso de atípicos es accesorio; nunca debe romper la consulta

    tabla = pd.DataFrame({
        "ocid": procesos["ocid"].to_numpy(),
        "objeto": procesos["texto_items"].fillna("").astype(str).str.slice(0, 60),
        "modalidad": procesos[cm.CATEGORICA].to_numpy(),
        "presupuesto": procesos["presupuesto"].to_numpy(),
        "afinidad": np.round(afinidad, 4),
        "atipico": marca_atipico,
        "grupo": etiquetas,
    })
    for c in cm.NUMERICAS:
        tabla[c] = X[c].to_numpy()

    # perfil descriptivo de cada grupo, con su proceso representativo
    perfil_grupos = []
    for g in grupos_ordenados:
        sel = tabla["grupo"] == g
        if not sel.any():
            continue
        p = {"grupo": int(g), "tamano": int(sel.sum()),
             "pct": round(100.0 * sel.sum() / len(tabla), 1),
             "afinidad_media": round(float(tabla.loc[sel, "afinidad"].mean()), 4),
             "modalidad_dominante": tabla.loc[sel, "modalidad"].mode().iloc[0],
             "presupuesto_medio": float(tabla.loc[sel, "presupuesto"].mean()),
             "ocid_medoide": modelo["medoides_ocid"][g]}
        for c in cm.NUMERICAS:
            p[c] = round(float(tabla.loc[sel, c].mean()), 4)
        perfil_grupos.append(p)

    # --- ÍNDICE DE COMPATIBILIDAD ------------------------------------------
    # La afinidad al medoide mide TIPICIDAD dentro del grupo, no compatibilidad
    # con el proveedor: en estos datos el grupo más "afín" es el difuso, que no
    # comparte rubros. Para ordenar los grupos se usa un índice explícito sobre
    # las tres variables que sí miden compatibilidad, cada una reescalada entre
    # los grupos para que ninguna domine por su unidad.
    VARS_COMPAT = ["cpc_jaccard4", "sim_tfidf", "afinidad_comprador"]
    for c in VARS_COMPAT:
        vals = [p[c] for p in perfil_grupos]
        lo, hi = min(vals), max(vals)
        for p in perfil_grupos:
            p[f"_n_{c}"] = 0.0 if hi <= lo else (p[c] - lo) / (hi - lo)
    for p in perfil_grupos:
        p["indice_compatibilidad"] = round(
            float(np.mean([p[f"_n_{c}"] for c in VARS_COMPAT])), 4)
    perfil_grupos.sort(key=lambda x: x["indice_compatibilidad"], reverse=True)

    grupo_proveedor = perfil_grupos[0]["grupo"] if perfil_grupos else None

    return {
        "ruc": ruc, "provincia": fila["provincia"],
        "adjudicaciones": int(fila["num_adjudicaciones"]),
        "n_cpc": int(fila["n_cpc_historicos"]),
        "monto_promedio": float(fila["monto_promedio_ganado"]),
        "tabla": tabla, "perfil_grupos": perfil_grupos,
        "grupo_proveedor": grupo_proveedor,
        "segundos": round(time.time() - t0, 2),
        "n_procesos": len(tabla),
    }


@st.cache_data(show_spinner=False)
def perfil_grupos_referencia():
    """Perfil de los 3 grupos según el proveedor con el que se entrenó el modelo.

    Es lo que se muestra en la pantalla de exploración (arranque en frío).
    """
    ej = rucs_de_ejemplo()
    if not ej:
        return []
    r = analizar_proveedor(ej[1]["ruc"] if len(ej) > 1 else ej[0]["ruc"])
    return r["perfil_grupos"] if r and "perfil_grupos" in r else []


def lectura_de_negocio(perfil, es_el_mejor, sin_historial=False):
    """Traduce el perfil numérico de un grupo a una frase entendible.

    Se basa en los valores reales del grupo, no en su posición en la lista.
    `sin_historial` cambia la redacción para el arranque en frío, donde no hay
    historial del consultante al que referirse.
    """
    if es_el_mejor and sin_historial:
        return (f"**Grupo más específico.** Agrupa procesos con rubros CPC y "
                f"descripciones muy parecidos entre sí (coincidencia interna "
                f"{perfil['cpc_jaccard4']:.2f}, similitud textual "
                f"{perfil['sim_tfidf']:.2f}). Si su empresa trabaja en un nicho "
                f"definido, es probable que sus oportunidades estén aquí.")
    if es_el_mejor:
        return (f"**Grupo accionable.** Es donde su historial encaja mejor: "
                f"coincidencia de rubros CPC de {perfil['cpc_jaccard4']:.2f} y "
                f"similitud textual de {perfil['sim_tfidf']:.2f}, muy por encima "
                f"de los otros grupos"
                + (", y con entidades con las que ya ha trabajado"
                   if perfil["afinidad_comprador"] > 0.05 else "")
                + ". Son las oportunidades a revisar primero.")
    if perfil["distancia_km"] < 120:
        return (f"**Cercanía geográfica.** Entidades próximas a su provincia "
                f"({perfil['distancia_km']:.0f} km en promedio), pero con poca "
                f"coincidencia de rubro (CPC {perfil['cpc_jaccard4']:.3f}). "
                f"Revíselas si su ventaja es logística más que técnica.")
    return (f"**Fondo del mercado.** Entidades distantes "
            f"({perfil['distancia_km']:.0f} km en promedio) y rubros que no ha "
            f"trabajado (CPC {perfil['cpc_jaccard4']:.3f}). Es el grueso del "
            f"universo vigente y la compatibilidad esperada es baja.")


# ================================================================ interfaz
def pie_de_pagina(datos):
    st.divider()
    m = datos["modelo"]
    st.caption(
        f"Corte de datos: {datos['fecha_corte']} · "
        f"Modelo: {m['algoritmo_nombre']}, k={m['k']} · "
        f"Selección por comparación de 6 algoritmos "
        f"(silueta-Gower ponderada {m['silueta_promedio_portafolio']:.4f}, "
        f"promedio de 3 proveedores de referencia)"
    )


def pantalla_inicio(datos):
    st.title("Compatibilidad proveedor–licitación")
    st.markdown("#### Consulte qué procesos vigentes se ajustan al perfil de un proveedor")
    st.info(AVISO_DATOS, icon="ℹ️")

    col1, col2 = st.columns([2, 3])
    with col1:
        st.text_input("RUC del proveedor (13 dígitos)", key="entrada_ruc",
                      max_chars=13, placeholder="0000000000001")
        st.button("Consultar", type="primary", key="btn_consultar",
                  on_click=lambda: st.session_state.update(
                      {"consulta": st.session_state.entrada_ruc.strip()}))
    with col2:
        ejemplos = rucs_de_ejemplo()
        if ejemplos:
            st.markdown("**Usar proveedor de ejemplo**")
            st.caption("Los tres proveedores de referencia con los que se "
                       "comparó el portafolio de modelos (Fase 2.2).")
            for e in ejemplos:
                c1, c2 = st.columns([3, 1])
                c1.write(f"`{e['ruc']}` · {e['provincia']} · "
                         f"{e['competitivas']} adjudicaciones competitivas")
                c2.button("Usar", key=f"ej_{e['ruc']}",
                          on_click=lambda r=e["ruc"]: st.session_state.update(
                              {"consulta": r}))
    pie_de_pagina(datos)


def tarjeta_grupo(p, indice, con_boton=True, sin_historial=False):
    mejor = indice == 0
    with st.container(border=True):
        st.markdown(f"### Grupo {p['grupo']}"
                    + ((" · más específico" if sin_historial else " · accionable")
                       if mejor else ""))
        st.caption(lectura_de_negocio(p, mejor, sin_historial))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Procesos", f"{p['tamano']:,}")
        c2.metric("Compatibilidad relativa", f"{p['indice_compatibilidad']:.3f}",
                  help="Coincidencia de rubros, similitud textual y relación "
                       "previa con el comprador, reescaladas ENTRE LOS TRES "
                       "GRUPOS: 1.000 es el más compatible de los tres, no una "
                       "compatibilidad perfecta.")
        c3.metric("Afinidad media", f"{p['afinidad_media']:.3f}",
                  help="Cercanía media al proceso representativo del grupo. "
                       "Mide tipicidad dentro del grupo, no compatibilidad.")
        c4.metric("Presupuesto medio", f"US$ {p['presupuesto_medio']:,.0f}")
        st.markdown("**Modalidad dominante:** " + str(p["modalidad_dominante"]))
        with st.expander("Variables medias del grupo"):
            filas = [{"Variable": ETIQUETAS_VARIABLES.get(c, c), "Media": p[c]}
                     for c in ETIQUETAS_VARIABLES]
            st.dataframe(pd.DataFrame(filas), hide_index=True,
                         use_container_width=True)
        st.caption(f"Proceso representativo (medoide): `{p['ocid_medoide']}`")
        if con_boton:
            st.button(f"Ver procesos del grupo {p['grupo']}",
                      key=f"ver_{p['grupo']}_{indice}",
                      on_click=lambda g=p["grupo"]: st.session_state.update(
                          {"grupo_activo": g}))


def pantalla_resultado(datos, r):
    st.title("Resultado de la consulta")

    perfiles = r["perfil_grupos"]
    principal = perfiles[0]
    st.markdown(
        f"### RUC `{r['ruc']}` · Grupo {principal['grupo']} de {len(perfiles)}")
    st.caption(f"{r['provincia']} · {r['adjudicaciones']} adjudicaciones "
               f"históricas · {r['n_cpc']} rubros CPC · "
               f"{r['n_procesos']:,} procesos vigentes analizados "
               f"en {r['segundos']} s")
    st.markdown(lectura_de_negocio(principal, True))

    lateral, principal_col = st.columns([1, 4.2])
    with lateral:
        st.markdown("#### Perfil del grupo")
        st.metric("Procesos en el grupo", f"{principal['tamano']:,}")
        st.metric("Compatibilidad relativa",
                  f"{principal['indice_compatibilidad']:.3f}",
                  help="Coincidencia de rubros, similitud textual y relación "
                       "previa con el comprador, reescaladas ENTRE LOS TRES "
                       "GRUPOS: 1.000 es el más compatible de los tres, no una "
                       "compatibilidad perfecta.")
        st.metric("Afinidad media al medoide",
                  f"{principal['afinidad_media']:.3f}",
                  help="Tipicidad dentro del grupo; no mide compatibilidad.")
        st.markdown("**Modalidad dominante**")
        st.write(principal["modalidad_dominante"])
        st.markdown("**Variables medias**")
        for c, etiqueta in ETIQUETAS_VARIABLES.items():
            st.markdown(f"<div style='font-size:0.82rem;line-height:1.35;"
                        f"margin-bottom:.35rem'>{etiqueta}<br>"
                        f"<b>{principal[c]:,.3f}</b></div>",
                        unsafe_allow_html=True)
        st.markdown("**Proceso representativo**")
        st.code(principal["ocid_medoide"], language=None)
        st.caption("Es un proceso real del grupo (medoide de PAM), no un promedio.")

    with principal_col:
        etiquetas = [f"Grupo {p['grupo']}" + (" · accionable" if i == 0 else "")
                     for i, p in enumerate(perfiles)]
        pestanas = st.tabs(etiquetas)
        for pes, p in zip(pestanas, perfiles):
            with pes:
                sub = r["tabla"][r["tabla"]["grupo"] == p["grupo"]].copy()
                sub = sub.sort_values("afinidad", ascending=False)
                vista = pd.DataFrame({
                    "OCID": sub["ocid"],
                    "Objeto": sub["objeto"].str.slice(0, 42),
                    "Modalidad": sub["modalidad"],
                    "Presupuesto": sub["presupuesto"].round(2),
                    "Afinidad": sub["afinidad"],
                    "CPC": sub["cpc_jaccard4"].round(3),
                    "Similitud": sub["sim_tfidf"].round(3),
                    "⚠": np.where(sub["atipico"], "⚠", ""),
                })
                st.caption(f"{len(vista):,} procesos vigentes, ordenados por "
                           f"afinidad descendente. ⚠ marca los atípicos "
                           f"detectados por Isolation Forest. «CPC» y "
                           f"«Similitud» permiten juzgar la compatibilidad "
                           f"real de cada proceso.")
                st.dataframe(
                    vista.head(300), hide_index=True,
                    use_container_width=True, height=420,
                    column_config={
                        "OCID": st.column_config.TextColumn(width="medium"),
                        "Objeto": st.column_config.TextColumn(width="medium"),
                        "Modalidad": st.column_config.TextColumn(width="small"),
                        "Presupuesto": st.column_config.NumberColumn(
                            format="$ %.0f", width="small"),
                        "Afinidad": st.column_config.NumberColumn(
                            format="%.3f", width="small"),
                        "CPC": st.column_config.NumberColumn(
                            format="%.3f", width="small"),
                        "Similitud": st.column_config.NumberColumn(
                            format="%.3f", width="small"),
                        "⚠": st.column_config.TextColumn(width="small"),
                    })
                st.download_button(
                    "Descargar resultados (CSV)",
                    data=vista.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"afinidad_{r['ruc']}_grupo{p['grupo']}.csv",
                    mime="text/csv", key=f"dl_{p['grupo']}")

    st.warning(AVISO_AFINIDAD, icon="⚠️")
    st.button("← Nueva consulta", on_click=lambda: st.session_state.update(
        {"consulta": None}))
    pie_de_pagina(datos)


def pantalla_exploracion(datos, ruc, motivo=None):
    st.title("Exploración de grupos")
    if motivo:
        st.warning(motivo, icon="⚠️")
    else:
        st.warning(
            f"El RUC `{ruc}` no registra adjudicaciones en el histórico "
            f"2024–2025 de este corte de datos. Sin historial, el modelo no "
            f"puede calcular la coincidencia de rubros, la similitud textual "
            f"ni la relación previa con los compradores.",
            icon="⚠️")
    st.markdown(
        "Puede ubicarse manualmente: abajo están los tres grupos en que el "
        "modelo divide los procesos vigentes, con sus características y un "
        "proceso representativo de cada uno. Revise cuál se parece más a lo "
        "que su empresa ofrece.")

    perfiles = perfil_grupos_referencia()
    if not perfiles:
        st.error("No se pudo construir el perfil de referencia de los grupos.")
        pie_de_pagina(datos)
        return

    for i, p in enumerate(perfiles):
        tarjeta_grupo(p, i, sin_historial=True)

    g = st.session_state.get("grupo_activo")
    if g is not None:
        ej = rucs_de_ejemplo()
        base = analizar_proveedor(ej[1]["ruc"] if len(ej) > 1 else ej[0]["ruc"])
        if base:
            sub = base["tabla"][base["tabla"]["grupo"] == g].copy()
            sub = sub.sort_values("afinidad", ascending=False)
            st.markdown(f"#### Procesos del grupo {g}")
            st.caption("Perfil de referencia; la afinidad exacta depende del "
                       "proveedor que consulte.")
            st.dataframe(
                pd.DataFrame({
                    "OCID": sub["ocid"], "Objeto": sub["objeto"],
                    "Modalidad": sub["modalidad"],
                    "Presupuesto (US$)": sub["presupuesto"].round(2),
                    "Afinidad": sub["afinidad"],
                }).head(200), hide_index=True, use_container_width=True,
                height=380)

    st.warning(AVISO_AFINIDAD, icon="⚠️")
    st.button("← Nueva consulta", on_click=lambda: st.session_state.update(
        {"consulta": None, "grupo_activo": None}))
    pie_de_pagina(datos)


def main():
    datos, err = cargar_todo()
    if err:
        st.title("Compatibilidad proveedor–licitación")
        st.error(err, icon="🚫")
        st.stop()

    consulta = st.session_state.get("consulta")
    if not consulta:
        pantalla_inicio(datos)
        return

    ruc = re.sub(r"\D", "", str(consulta))
    if len(ruc) != 13:
        pantalla_inicio(datos)
        st.error(f"El RUC debe tener exactamente 13 dígitos; recibí "
                 f"{len(ruc)}. Verifique el número e intente de nuevo.",
                 icon="🚫")
        return

    resultado = analizar_proveedor(ruc)
    if resultado is None:
        pantalla_exploracion(datos, ruc)
    elif resultado.get("sin_datos_suficientes"):
        pantalla_exploracion(datos, ruc, motivo=resultado["motivo"])
    else:
        pantalla_resultado(datos, resultado)


if __name__ == "__main__":
    main()
