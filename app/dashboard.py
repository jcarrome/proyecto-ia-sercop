# -*- coding: utf-8 -*-
"""
FASE 3 — PANEL DEL PROVEEDOR (prototipo local, sin APIs).

Materializa los casos de uso del diseño sobre el modelo ganador de la Fase 2.2
(K-Medoids / PAM, k=3). NO reimplementa el cálculo: importa comparar_modelos.py
y reutiliza construir_matriz, winsorizar y la distancia de Gower ponderada.

Ejecución:
    streamlit run app/dashboard.py

Asignación de un proceso a un grupo (mismo pipeline que la Fase 2.2):
    matriz de interacción -> winsorizar con los límites guardados ->
    estandarizar con el escalador guardado -> distancia de Gower ponderada
    contra los 3 medoides -> grupo del medoide más cercano.

La interfaz trabaja en dos capas: el proveedor ve un veredicto y una lista de
oportunidades en lenguaje de negocio; el detalle metodológico (medoides,
silueta, Isolation Forest, comparación de los seis algoritmos) queda plegado
al final para la sustentación académica.
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
                   page_icon="🎯", layout="wide",
                   initial_sidebar_state="collapsed")

PATRON_ID = re.compile(r"(\d{13})")
AVISO_DATOS = ("Los datos provienen de un corte local del portal de datos "
               "abiertos del SERCOP; el sistema no consume servicios externos.")

ETIQUETAS_VARIABLES = {
    "distancia_km": "Distancia (km)",
    "cpc_jaccard4": "Coincidencia de rubros (CPC)",
    "sim_tfidf": "Similitud textual",
    "desviacion_presupuesto": "Desviación de presupuesto (log)",
    "actividad_cpc_comprador": "Actividad del comprador en el rubro",
    "afinidad_comprador": "Relación previa con el comprador (log)",
}

# --- umbrales del semáforo -------------------------------------------------
# Calibrados sobre la distribución real del corte (no son valores arbitrarios).
# La compatibilidad por proceso es el promedio de las dos variables que miden
# coincidencia de negocio: rubro CPC y similitud textual, ambas en [0, 1].
# Con el corte en 0.15 un proveedor muy diversificado obtiene 0 oportunidades
# —que es la respuesta honesta— y uno de nicho obtiene decenas o cientos.
UMBRAL_ALTA = 0.35
UMBRAL_MEDIA = 0.15
UMBRAL_BAJA = 0.05

NIVELES = {
    "ALTA": {
        "icono": "🟢", "color": "#15803d", "fondo": "#dcfce7", "borde": "#86efac",
        "titulo": "Alta compatibilidad",
    },
    "MEDIA": {
        "icono": "🟡", "color": "#a16207", "fondo": "#fef9c3", "borde": "#fde047",
        "titulo": "Compatibilidad media",
    },
    "BAJA": {
        "icono": "🟠", "color": "#c2410c", "fondo": "#ffedd5", "borde": "#fdba74",
        "titulo": "Compatibilidad baja",
    },
    "NULA": {
        "icono": "⚪", "color": "#475569", "fondo": "#f1f5f9", "borde": "#cbd5e1",
        "titulo": "Sin resultado concluyente",
    },
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


@st.cache_data(show_spinner=False)
def tabla_comparacion_modelos():
    """Comparación de los seis algoritmos (para el panel técnico)."""
    ruta = os.path.join(RES, "comparacion_modelos.csv")
    if not os.path.exists(ruta):
        return None
    try:
        return pd.read_csv(ruta)
    except Exception:
        return None


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

    # afinidad = 1 - distancia normalizada (min-max sobre esta consulta).
    # Mide TIPICIDAD dentro del grupo, no compatibilidad: se conserva para el
    # panel técnico, pero NO ordena la lista que ve el proveedor.
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
        "objeto": procesos["texto_items"].fillna("").astype(str).str.slice(0, 90),
        "provincia_buyer": procesos["provincia_buyer"].fillna("n/d").to_numpy(),
        "modalidad": procesos[cm.CATEGORICA].to_numpy(),
        "presupuesto": procesos["presupuesto"].to_numpy(),
        "afinidad": np.round(afinidad, 4),
        "atipico": marca_atipico,
        "grupo": etiquetas,
    })
    for c in cm.NUMERICAS:
        tabla[c] = X[c].to_numpy()

    # --- COMPATIBILIDAD POR PROCESO ---------------------------------------
    # Escala ABSOLUTA en [0, 1]: promedio de coincidencia de rubro CPC y
    # similitud textual. A diferencia de la afinidad, es comparable entre
    # proveedores y permite decir "no hay nada compatible" cuando así es.
    tabla["compatibilidad"] = np.round(
        0.5 * tabla["cpc_jaccard4"].clip(0, 1) +
        0.5 * tabla["sim_tfidf"].clip(0, 1), 4)

    # perfil descriptivo de cada grupo, con su proceso representativo
    perfil_grupos = []
    for g in grupos_ordenados:
        sel = tabla["grupo"] == g
        if not sel.any():
            continue
        sub = tabla.loc[sel]
        p = {"grupo": int(g), "tamano": int(sel.sum()),
             "pct": round(100.0 * sel.sum() / len(tabla), 1),
             "afinidad_media": round(float(sub["afinidad"].mean()), 4),
             "compat_media": round(float(sub["compatibilidad"].mean()), 4),
             "compat_max": round(float(sub["compatibilidad"].max()), 4),
             "n_oportunidades": int((sub["compatibilidad"] >= UMBRAL_MEDIA).sum()),
             "modalidad_dominante": sub["modalidad"].mode().iloc[0],
             "presupuesto_medio": float(sub["presupuesto"].mean()),
             "ocid_medoide": modelo["medoides_ocid"][g]}
        for c in cm.NUMERICAS:
            p[c] = round(float(sub[c].mean()), 4)
        perfil_grupos.append(p)

    # --- ÍNDICE DE COMPATIBILIDAD ENTRE GRUPOS -----------------------------
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

    # --- VEREDICTO ---------------------------------------------------------
    # Se mide sobre las oportunidades REALES del grupo accionable, no sobre el
    # tamaño del grupo: un grupo de 4.695 procesos sin coincidencia de rubro
    # no es un resultado, es la ausencia de uno.
    oport = tabla[(tabla["grupo"] == grupo_proveedor)
                  & (tabla["compatibilidad"] >= UMBRAL_MEDIA)]
    n_oport = len(oport)
    if n_oport == 0:
        nivel = "NULA"
        media_oport = float(tabla.loc[tabla["grupo"] == grupo_proveedor,
                                      "compatibilidad"].mean())
    else:
        media_oport = float(oport["compatibilidad"].mean())
        if media_oport >= UMBRAL_ALTA:
            nivel = "ALTA"
        elif media_oport >= UMBRAL_MEDIA:
            nivel = "MEDIA"
        else:
            nivel = "BAJA"

    return {
        "ruc": ruc, "provincia": fila["provincia"],
        "adjudicaciones": int(fila["num_adjudicaciones"]),
        "n_cpc": int(fila["n_cpc_historicos"]),
        "monto_promedio": float(fila["monto_promedio_ganado"]),
        "tabla": tabla, "perfil_grupos": perfil_grupos,
        "grupo_proveedor": grupo_proveedor,
        "nivel": nivel, "n_oportunidades": n_oport,
        "compat_media_oportunidades": round(media_oport, 4),
        "segundos": round(time.time() - t0, 2),
        "n_procesos": len(tabla),
    }


@st.cache_data(show_spinner=False)
def referencia_grupos():
    """Perfil y ejemplos de cada grupo, para el arranque sin historial."""
    ej = rucs_de_ejemplo()
    if not ej:
        return [], None
    r = analizar_proveedor(ej[1]["ruc"] if len(ej) > 1 else ej[0]["ruc"])
    if not r or "perfil_grupos" not in r:
        return [], None
    return r["perfil_grupos"], r["tabla"]


# ============================================================ presentación
def mil(x, dec=0):
    """Formatea un número con punto como separador de miles (uso EC)."""
    return f"{x:,.{dec}f}".replace(",", "·").replace(".", ",").replace("·", ".")


def estilos():
    st.markdown("""
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1350px;}
      #MainMenu, footer {visibility: hidden;}

      .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #0ea5e9 100%);
        color: #fff; padding: 1.6rem 1.9rem; border-radius: 16px;
        margin-bottom: 1.4rem;
      }
      .hero h1 {font-size: 1.85rem; font-weight: 700; margin: 0 0 .35rem 0;
                color: #fff; letter-spacing: -.02em;}
      .hero p  {font-size: 1rem; margin: 0; opacity: .92;}

      .veredicto {
        border-radius: 16px; padding: 1.5rem 1.8rem; margin-bottom: 1.3rem;
        border: 2px solid;
      }
      .veredicto .nivel {font-size: 1.55rem; font-weight: 700; margin-bottom: .5rem;
                         letter-spacing: -.01em;}
      .veredicto .cifra {font-size: 2.6rem; font-weight: 800; line-height: 1.05;
                         margin-bottom: .2rem;}
      .veredicto .expl  {font-size: 1.02rem; line-height: 1.5; margin: 0; opacity: .95;}

      .ficha {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: .85rem 1rem; height: 100%;
      }
      .ficha .et  {font-size: .78rem; color: #64748b; text-transform: uppercase;
                   letter-spacing: .04em; margin-bottom: .3rem; font-weight: 600;}
      .ficha .va  {font-size: 1.5rem; font-weight: 700; color: #0f172a; line-height: 1.1;}
      .ficha .su  {font-size: .8rem; color: #64748b; margin-top: .2rem;}

      .tarjeta {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.1rem 1.3rem; margin-bottom: .9rem;
      }
      .tarjeta-igual {min-height: 378px;}
      .tarjeta h4 {margin: 0 0 .4rem 0; font-size: 1.1rem; color: #0f172a;}
      .tarjeta p  {margin: 0; font-size: .93rem; color: #475569; line-height: 1.5;}

      .seccion {font-size: 1.28rem; font-weight: 700; color: #0f172a;
                margin: 1.7rem 0 .3rem 0; letter-spacing: -.01em;}
      .subseccion {font-size: .92rem; color: #64748b; margin-bottom: .9rem;}

      .chip {display:inline-block; padding:.16rem .6rem; border-radius:999px;
             font-size:.78rem; font-weight:600; margin-right:.35rem;}
      .chip-a {background:#dcfce7; color:#15803d;}
      .chip-m {background:#fef9c3; color:#a16207;}
      .chip-b {background:#ffedd5; color:#c2410c;}

      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
    """, unsafe_allow_html=True)


def cabecera(subtitulo):
    st.markdown(
        f'<div class="hero"><h1>🎯 Compatibilidad proveedor–licitación</h1>'
        f'<p>{subtitulo}</p></div>', unsafe_allow_html=True)


def ficha(etiqueta, valor, sub=""):
    return (f'<div class="ficha"><div class="et">{etiqueta}</div>'
            f'<div class="va">{valor}</div>'
            + (f'<div class="su">{sub}</div>' if sub else "")
            + '</div>')


def pie_de_pagina(datos):
    st.divider()
    m = datos["modelo"]
    st.caption(
        f"Corte de datos: {datos['fecha_corte']} · "
        f"Modelo: {m['algoritmo_nombre']}, k={m['k']} · "
        f"Seleccionado por comparación de 6 algoritmos "
        f"(silueta-Gower ponderada {m['silueta_promedio_portafolio']:.4f}) · "
        f"{AVISO_DATOS}")


# ------------------------------------------------------------ pantalla 1
def pantalla_inicio(datos):
    cabecera("Descubra qué licitaciones vigentes se ajustan a lo que su "
             "empresa ya ha vendido al Estado.")

    izq, der = st.columns([1.05, 1], gap="large")
    with izq:
        st.markdown('<div class="seccion">Consulte su RUC</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="subseccion">13 dígitos, sin guiones ni '
                    'espacios.</div>', unsafe_allow_html=True)
        st.text_input("RUC del proveedor", key="entrada_ruc", max_chars=13,
                      placeholder="0000000000001", label_visibility="collapsed")
        st.button("Buscar mis oportunidades", type="primary", width="stretch",
                  key="btn_consultar",
                  on_click=lambda: st.session_state.update(
                      {"consulta": st.session_state.entrada_ruc.strip()}))
        st.markdown(
            f'<div style="margin-top:1.2rem;font-size:.9rem;color:#475569;'
            f'line-height:1.55">Analizamos <b>'
            f'{mil(len(datos["procesos"]))}</b> procesos vigentes y le decimos '
            f'cuáles coinciden con su historial de adjudicaciones, en qué '
            f'medida y por qué.</div>',
            unsafe_allow_html=True)

    with der:
        ejemplos = rucs_de_ejemplo()
        if ejemplos:
            st.markdown('<div class="seccion">O pruebe con un ejemplo</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="subseccion">Proveedores reales del corte '
                        'de datos, usados como referencia del modelo.</div>',
                        unsafe_allow_html=True)
            for e in ejemplos:
                c1, c2 = st.columns([3.1, 1])
                c1.markdown(
                    f'<div style="padding-top:.35rem"><code>{e["ruc"]}</code>'
                    f'<br><span style="font-size:.85rem;color:#64748b">'
                    f'{e["provincia"]} · {e["competitivas"]} adjudicaciones '
                    f'competitivas</span></div>', unsafe_allow_html=True)
                c2.button("Usar", key=f"ej_{e['ruc']}", width="stretch",
                          on_click=lambda r=e["ruc"]: st.session_state.update(
                              {"consulta": r}))
    pie_de_pagina(datos)


# ------------------------------------------------------------ pantalla 2
def texto_veredicto(r):
    """Frase de negocio que explica el veredicto, sin jerga del modelo."""
    n, total = r["n_oportunidades"], r["n_procesos"]
    if r["nivel"] == "ALTA":
        return (f"Encontramos procesos vigentes que coinciden de forma clara "
                f"con los rubros y el tipo de bienes o servicios que su empresa "
                f"ya ha adjudicado. Son los primeros que conviene revisar.")
    if r["nivel"] == "MEDIA":
        return (f"Hay procesos con coincidencia parcial: comparten rubro o "
                f"descripción con su historial, pero no de forma tan directa. "
                f"Vale la pena revisarlos uno por uno antes de decidir.")
    if r["nivel"] == "BAJA":
        return (f"La coincidencia con su historial es débil. Puede haber "
                f"oportunidades, pero el modelo no las distingue con claridad "
                f"del resto del mercado.")
    return (f"Entre los {mil(total)} procesos vigentes no encontramos ninguno "
            f"que coincida de forma significativa con su historial. Esto "
            f"ocurre cuando la empresa vende en rubros muy variados —y "
            f"entonces ningún grupo la representa— o cuando su especialidad "
            f"no tiene procesos abiertos en este corte de datos.")


def bloque_veredicto(r):
    est = NIVELES[r["nivel"]]
    n = mil(r["n_oportunidades"])
    total = mil(r["n_procesos"])
    if r["nivel"] == "NULA":
        cifra = f'<div class="cifra">Sin coincidencias</div>'
        sub = f'<div style="font-size:.95rem;opacity:.85">de {total} procesos vigentes</div>'
    else:
        cifra = f'<div class="cifra">{n}</div>'
        sub = (f'<div style="font-size:.95rem;opacity:.85">'
               f'oportunidades de {total} procesos vigentes</div>')
    st.markdown(
        f'<div class="veredicto" style="background:{est["fondo"]};'
        f'border-color:{est["borde"]};color:{est["color"]}">'
        f'<div class="nivel">{est["icono"]} {est["titulo"]}</div>'
        f'{cifra}{sub}'
        f'<p class="expl" style="margin-top:.7rem">{texto_veredicto(r)}</p>'
        f'</div>', unsafe_allow_html=True)


def etiqueta_nivel(v):
    if v >= UMBRAL_ALTA:
        return "Alta"
    if v >= UMBRAL_MEDIA:
        return "Media"
    if v >= UMBRAL_BAJA:
        return "Baja"
    return "Mínima"


def vista_procesos(sub, n=300):
    """Convierte la tabla interna en la vista que ve el proveedor."""
    sub = sub.sort_values("compatibilidad", ascending=False).head(n)
    return pd.DataFrame({
        "Qué se licita": sub["objeto"].str.slice(0, 58),
        "Provincia": sub["provincia_buyer"].str.title(),
        "Modalidad": sub["modalidad"],
        "Presupuesto": sub["presupuesto"].round(2),
        "Qué tanto le calza": sub["compatibilidad"],
        "Nivel": [etiqueta_nivel(v) for v in sub["compatibilidad"]],
        "Código del proceso": sub["ocid"],
    })


CONFIG_TABLA = {
    "Qué se licita": st.column_config.TextColumn(width=300),
    "Provincia": st.column_config.TextColumn(width=110),
    "Modalidad": st.column_config.TextColumn(width=150),
    "Presupuesto": st.column_config.NumberColumn(format="dollar", width=110),
    "Qué tanto le calza": st.column_config.ProgressColumn(
        format="%.2f", min_value=0.0, max_value=1.0, width=170,
        help="0 = ninguna coincidencia con su historial; 1 = coincidencia total "
             "de rubro y descripción."),
    "Nivel": st.column_config.TextColumn(width=80),
    "Código del proceso": st.column_config.TextColumn(width=250),
}


def panel_tecnico(datos, r=None):
    """Todo el detalle metodológico, plegado. Es la capa de sustentación."""
    with st.expander("🔬 Detalle técnico del modelo", expanded=False):
        m = datos["modelo"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algoritmo", m["algoritmo_nombre"])
        c2.metric("Grupos (k)", m["k"])
        c3.metric("Silueta-Gower", f"{m['silueta_promedio_portafolio']:.4f}")
        c4.metric("Procesos", mil(len(datos["procesos"])))

        st.caption(
            "La silueta se calcula sobre distancia de Gower ponderada "
            "(6 variables numéricas con peso 1 y la modalidad con peso 1/3). "
            "El ganador se eligió comparando seis algoritmos bajo condiciones "
            "idénticas, con dos líneas base de control y cuatro reglas de "
            "descalificación.")

        comp = tabla_comparacion_modelos()
        if comp is not None and len(comp):
            st.markdown("**Comparación de los seis algoritmos**")
            agg = (comp.groupby("algoritmo")
                       .agg(silueta=("silueta_gower", "mean"),
                            k=("mejor_k", "median"),
                            entropia=("entropia", "mean"),
                            cobertura=("cobertura", "mean"),
                            descalificado=("descalificado", "any"))
                       .reset_index()
                       .sort_values("silueta", ascending=False))
            agg["Estado"] = np.where(agg["descalificado"],
                                     "DESCALIFICADO", "admitido")
            ganador = m["algoritmo_nombre"]
            agg["Estado"] = [
                "🏆 GANADOR" if str(a).lower().startswith(str(ganador).lower()[:8])
                and not d else e
                for a, d, e in zip(agg["algoritmo"], agg["descalificado"],
                                   agg["Estado"])]
            st.dataframe(
                agg.rename(columns={"algoritmo": "Algoritmo",
                                    "silueta": "Silueta-Gower", "k": "k",
                                    "entropia": "Entropía",
                                    "cobertura": "Cobertura %"})
                   [["Algoritmo", "Silueta-Gower", "k", "Entropía",
                     "Cobertura %", "Estado"]],
                hide_index=True, width="stretch",
                column_config={
                    "Silueta-Gower": st.column_config.NumberColumn(format="%.4f"),
                    "Entropía": st.column_config.NumberColumn(format="%.3f"),
                    "Cobertura %": st.column_config.NumberColumn(format="%.1f"),
                })
            st.caption(
                "Los dos algoritmos con MAYOR silueta quedaron descalificados: "
                "el jerárquico concentra casi todo en un grupo (entropía muy "
                "por debajo del mínimo exigido de 0.50) y el de densidad "
                "descarta como ruido una parte grande de los procesos "
                "(cobertura por debajo del 85 % exigido). Sin las reglas de "
                "descalificación el ganador nominal habría sido degenerado.")

        if r is not None:
            st.markdown("**Perfil de los grupos en esta consulta**")
            filas = []
            for i, p in enumerate(r["perfil_grupos"]):
                fila = {"Grupo": p["grupo"],
                        "Rol": "accionable" if i == 0 else "fondo",
                        "Procesos": p["tamano"], "%": p["pct"],
                        "Compatib. media": p["compat_media"],
                        "Afinidad media": p["afinidad_media"],
                        "Modalidad dominante": p["modalidad_dominante"],
                        "Medoide (OCID)": p["ocid_medoide"]}
                for c, et in ETIQUETAS_VARIABLES.items():
                    fila[et] = p[c]
                filas.append(fila)
            st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
            st.caption(
                "«Afinidad media» es la cercanía al medoide: mide TIPICIDAD "
                "dentro del grupo, no compatibilidad con el proveedor. Por eso "
                "la lista de oportunidades se ordena por compatibilidad "
                "(rubro CPC + similitud textual) y no por afinidad. El medoide "
                "es un proceso real del grupo, no un promedio.")
            st.markdown(
                f"**Atípicos:** {mil(int(r['tabla']['atipico'].sum()))} de "
                f"{mil(r['n_procesos'])} procesos marcados por el Isolation "
                f"Forest global entrenado en la Fase 2.2.")
            st.caption(
                "Limitación documentada: el Isolation Forest ajustado POR GRUPO "
                "devuelve ~5 % por construcción (contamination es un parámetro, "
                "no una medición); la columna informativa es la del bosque global.")

        st.markdown("**Limitaciones conocidas**")
        st.markdown(
            "- La silueta es casi plana entre k=3 y k=10 (0.242 – 0.277): los "
            "datos son un continuo, no tienen un número natural de grupos.\n"
            "- De los tres grupos sólo uno es accionable; los otros dos se "
            "separan sobre todo por distancia geográfica.\n"
            "- La compatibilidad mide coincidencia con el historial; **no es "
            "una probabilidad de adjudicación**.")


def pantalla_resultado(datos, r):
    perfiles = r["perfil_grupos"]
    principal = perfiles[0]
    cabecera(f"RUC {r['ruc']} · {r['provincia']} · {r['adjudicaciones']} "
             f"adjudicaciones históricas en {r['n_cpc']} rubros")

    bloque_veredicto(r)

    tabla = r["tabla"]
    oport = tabla[(tabla["grupo"] == principal["grupo"])
                  & (tabla["compatibilidad"] >= UMBRAL_MEDIA)]
    resto = tabla.drop(oport.index)

    c1, c2, c3, c4 = st.columns(4)
    presupuesto = oport["presupuesto"].sum() if len(oport) else 0.0
    c1.markdown(ficha("Oportunidades", mil(r["n_oportunidades"]),
                      "coincidencia significativa"), unsafe_allow_html=True)
    c2.markdown(ficha("Mejor coincidencia",
                      f'{tabla["compatibilidad"].max():.2f}'.replace(".", ","),
                      "de un máximo de 1,00"), unsafe_allow_html=True)
    c3.markdown(ficha("Presupuesto en juego", f'US$ {mil(presupuesto)}',
                      "suma de las oportunidades"), unsafe_allow_html=True)
    c4.markdown(ficha("Tiempo de análisis",
                      f'{r["segundos"]:.2f} s'.replace(".", ","),
                      f'{mil(r["n_procesos"])} procesos'),
                unsafe_allow_html=True)

    if len(oport):
        st.markdown('<div class="seccion">Sus oportunidades</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="subseccion">Ordenadas por qué tanto coinciden '
                    'con su historial. La barra combina coincidencia de rubro '
                    'y similitud de la descripción.</div>',
                    unsafe_allow_html=True)
        vista = vista_procesos(oport)
        st.dataframe(vista, hide_index=True, width="stretch", height=430,
                     column_config=CONFIG_TABLA)
        st.download_button(
            "⬇  Descargar mis oportunidades (CSV)",
            data=vista.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"oportunidades_{r['ruc']}.csv", mime="text/csv")
    else:
        st.markdown('<div class="seccion">Qué puede hacer</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="tarjeta"><h4>El modelo no encontró coincidencias</h4>'
            '<p>No significa que no haya nada para su empresa: significa que '
            'su historial es demasiado amplio o demasiado específico para que '
            'este modelo lo distinga. Puede revisar el mercado completo abajo, '
            'u orientarse por rubro en la pantalla de exploración.</p></div>',
            unsafe_allow_html=True)

    with st.expander(f"📋 Ver el resto del mercado "
                     f"({mil(len(resto))} procesos con baja coincidencia)",
                     expanded=False):
        st.caption("Procesos vigentes que el modelo no considera compatibles "
                   "con su historial. Ordenados igualmente por coincidencia.")
        st.dataframe(vista_procesos(resto, 200), hide_index=True,
                     width="stretch", height=380, column_config=CONFIG_TABLA)

    panel_tecnico(datos, r)

    st.button("← Nueva consulta", on_click=lambda: st.session_state.update(
        {"consulta": None, "grupo_activo": None}))
    pie_de_pagina(datos)


# ------------------------------------------------------------ pantalla 3
def describir_grupo(p, es_primero, ejemplos):
    """Describe un grupo en lenguaje de negocio, sin jerga del modelo."""
    if es_primero:
        titulo = "Nicho definido"
        cuerpo = ("Procesos muy parecidos entre sí en rubro y descripción. "
                  "Si su empresa trabaja en una especialidad concreta, es "
                  "probable que sus oportunidades estén aquí.")
    elif p["distancia_km"] < 120:
        titulo = "Cercanía geográfica"
        cuerpo = (f"Entidades próximas entre sí ({p['distancia_km']:.0f} km en "
                  f"promedio) pero con rubros variados. Interesa si su ventaja "
                  f"es logística más que técnica.")
    else:
        titulo = "Mercado general"
        cuerpo = (f"El grueso de los procesos vigentes: entidades distantes "
                  f"({p['distancia_km']:.0f} km en promedio) y rubros muy "
                  f"diversos.")
    muestras = "".join(
        f'<li style="margin-bottom:.25rem">{e}</li>' for e in ejemplos[:3])
    st.markdown(
        f'<div class="tarjeta tarjeta-igual"><h4>{titulo} '
        f'<span style="font-weight:400;color:#94a3b8;font-size:.9rem">'
        f'· {mil(p["tamano"])} procesos ({p["pct"]} %)</span></h4>'
        f'<p>{cuerpo}</p>'
        f'<p style="margin-top:.6rem;font-size:.85rem;color:#64748b">'
        f'<b>Modalidad más frecuente:</b> {p["modalidad_dominante"]}<br>'
        f'<b>Presupuesto medio:</b> US$ {mil(p["presupuesto_medio"])}</p>'
        f'<p style="margin-top:.5rem;font-size:.85rem;color:#475569">'
        f'<b>Ejemplos de lo que se licita aquí:</b></p>'
        f'<ul style="margin:.2rem 0 0 1rem;font-size:.85rem;color:#475569">'
        f'{muestras}</ul></div>',
        unsafe_allow_html=True)


def pantalla_exploracion(datos, ruc, motivo=None):
    cabecera("No encontramos historial de adjudicaciones para este RUC.")

    st.markdown(
        f'<div class="veredicto" style="background:{NIVELES["NULA"]["fondo"]};'
        f'border-color:{NIVELES["NULA"]["borde"]};color:{NIVELES["NULA"]["color"]}">'
        f'<div class="nivel">ℹ️ Sin historial en este corte de datos</div>'
        f'<p class="expl">'
        + (motivo if motivo else
           f'El RUC <code>{ruc}</code> no registra adjudicaciones en el '
           f'histórico 2024–2025. Sin historial no podemos calcular '
           f'coincidencia de rubros ni similitud de descripción, que es lo que '
           f'el modelo necesita para recomendarle procesos.')
        + '</p></div>', unsafe_allow_html=True)

    perfiles, tabla_ref = referencia_grupos()
    if not perfiles:
        st.error("No se pudo construir el perfil de referencia de los grupos.")
        pie_de_pagina(datos)
        return

    st.markdown('<div class="seccion">Ubíquese por tipo de mercado</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subseccion">Así divide el modelo los procesos '
                'vigentes. Revise cuál se parece más a lo que su empresa '
                'ofrece y explore sus procesos.</div>', unsafe_allow_html=True)

    cols = st.columns(len(perfiles), gap="medium")
    for i, (col, p) in enumerate(zip(cols, perfiles)):
        with col:
            sub = tabla_ref[tabla_ref["grupo"] == p["grupo"]]
            sub = sub.sort_values("compatibilidad", ascending=False)
            # objetos DISTINTOS: el corte repite el mismo texto muchas veces
            vistos, ejemplos = set(), []
            for x in sub["objeto"]:
                t = str(x).strip()[:52].rstrip(",")
                if t and t.upper() not in vistos:
                    vistos.add(t.upper())
                    ejemplos.append(t)
                if len(ejemplos) == 3:
                    break
            describir_grupo(p, i == 0, ejemplos)
            st.button(f"Ver estos {mil(p['tamano'])} procesos",
                      key=f"ver_{p['grupo']}_{i}", width="stretch",
                      on_click=lambda g=p["grupo"]: st.session_state.update(
                          {"grupo_activo": g}))

    g = st.session_state.get("grupo_activo")
    if g is not None and tabla_ref is not None:
        sub = tabla_ref[tabla_ref["grupo"] == g]
        st.markdown('<div class="seccion">Procesos de este mercado</div>',
                    unsafe_allow_html=True)
        st.caption("Vista de referencia: la coincidencia exacta depende del "
                   "historial de cada proveedor.")
        st.dataframe(vista_procesos(sub, 200), hide_index=True,
                     width="stretch", height=400, column_config=CONFIG_TABLA)

    panel_tecnico(datos)
    st.button("← Nueva consulta", on_click=lambda: st.session_state.update(
        {"consulta": None, "grupo_activo": None}))
    pie_de_pagina(datos)


# ------------------------------------------------------------------ main
def main():
    estilos()
    datos, err = cargar_todo()
    if err:
        cabecera("No se pudo iniciar el panel.")
        st.error(err, icon="🚫")
        st.stop()

    consulta = st.session_state.get("consulta")
    if not consulta:
        pantalla_inicio(datos)
        return

    ruc = re.sub(r"\D", "", str(consulta))
    if len(ruc) != 13:
        pantalla_inicio(datos)
        st.error(f"El RUC debe tener exactamente 13 dígitos; recibí {len(ruc)}. "
                 f"Verifique el número e intente de nuevo.", icon="🚫")
        return

    with st.spinner("Analizando procesos vigentes…"):
        resultado = analizar_proveedor(ruc)
    if resultado is None:
        pantalla_exploracion(datos, ruc)
    elif resultado.get("sin_datos_suficientes"):
        pantalla_exploracion(datos, ruc, motivo=resultado["motivo"])
    else:
        pantalla_resultado(datos, resultado)


if __name__ == "__main__":
    main()
