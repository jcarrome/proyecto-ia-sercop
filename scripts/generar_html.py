# -*- coding: utf-8 -*-
"""
Genera resultados/comparacion_modelos.html: página autocontenida con gráficos
para comparar los seis algoritmos. Lee sólo los CSV/PNG ya producidos.
"""
import base64
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "resultados")
OUT = os.path.join(RES, "comparacion_modelos.html")

t6 = pd.read_csv(os.path.join(RES, "tabla6_comparacion.csv"))
t7 = pd.read_csv(os.path.join(RES, "tabla7_lineas_base.csv"))
t8 = pd.read_csv(os.path.join(RES, "tabla8_perfiles.csv"))
t9 = pd.read_csv(os.path.join(RES, "tabla9_atipicos.csv"))

# ---- paleta categórica validada (slots 1..8), claro / oscuro
SERIES = [
    ("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"), ("#e87ba4", "#d55181"), ("#008300", "#008300"),
    ("#4a3aa7", "#9085e9"), ("#e34948", "#e66767"),
]
ORDEN = ["K-Medoids (PAM)", "DBSCAN", "GMM", "Jerárquico (promedio)",
         "K-Means", "K-Prototypes"]
COLOR = {a: i for i, a in enumerate(ORDEN)}

UMBRAL_COB, UMBRAL_ARI = 90.0, 0.60

sil_rand = float(t7.loc[t7["Algoritmo"].str.contains("aleatoria"), "Silueta (Gower)"].iloc[0])
sil_mod = float(t7.loc[t7["Algoritmo"].str.contains("modalidad"), "Silueta (Gower)"].iloc[0])


def descartes(r):
    m = []
    if r["Cobertura"] < UMBRAL_COB:
        m.append(f"cobertura {r['Cobertura']:.2f}% &lt; 90%")
    if r["ARI"] < UMBRAL_ARI:
        m.append(f"ARI {r['ARI']:.3f} &lt; 0.60")
    return m


t6 = t6.copy()
t6["_desc"] = t6.apply(descartes, axis=1)
t6["_ok"] = t6["_desc"].map(lambda x: len(x) == 0)
adm = t6[t6["_ok"]].sort_values("Silueta (Gower)", ascending=False)
GANADOR = adm.iloc[0]["Algoritmo"]
SIL_GAN = float(adm.iloc[0]["Silueta (Gower)"])
SEGUNDO = float(adm.iloc[1]["Silueta (Gower)"])


def barras(df, campo, fmt, vmin, vmax, ref=None, invertir=False, sub=None):
    """Barras horizontales con etiqueta directa (obligatoria: contraste <3:1 en claro)."""
    filas = []
    d = df.sort_values(campo, ascending=invertir)
    for _, r in d.iterrows():
        a = r["Algoritmo"]
        v = float(r[campo])
        pct = 0 if vmax == vmin else max(0.5, 100 * (v - vmin) / (vmax - vmin))
        ci = COLOR.get(a, 7)
        muerto = "" if r.get("_ok", True) else " muerto"
        extra = f'<span class="sub">{sub(r)}</span>' if sub else ""
        filas.append(f'''
      <div class="fila{muerto}">
        <div class="etq">{a}{' <span class="tag-gana">gana</span>' if a == GANADOR else ''}
          {'<span class="tag-fuera">descartado</span>' if not r.get("_ok", True) else ''}</div>
        <div class="pista" tabindex="0" data-tip="{a}: {fmt.format(v)}">
          <div class="barra c{ci}" style="width:{pct:.2f}%"></div>
          <span class="val">{fmt.format(v)}</span>{extra}
        </div>
      </div>''')
    linea = ""
    if ref:
        for rv, rl, cls in ref:
            # La pista de barras empieza tras la etiqueta (190px) + gap (12px) = 202px,
            # así que el % hay que aplicarlo al ancho restante, no al total.
            f = (rv - vmin) / (vmax - vmin)
            linea += (f'<div class="ref {cls}" '
                      f'style="left:calc(202px + (100% - 202px) * {f:.4f})">'
                      f'<span>{rl}</span></div>')
    return f'<div class="chart">{linea}{"".join(filas)}</div>'


def png_b64(nombre):
    p = os.path.join(RES, nombre)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def figura(nombre, titulo, nota):
    b = png_b64(nombre)
    if b is None:
        return (f'<figure class="fig falta"><figcaption><b>{titulo}</b><br>'
                f'<span class="nota">No generada: el notebook aún no había llegado a esta '
                f'celda cuando se armó esta página.</span></figcaption></figure>')
    return (f'<figure class="fig"><img alt="{titulo}" src="data:image/png;base64,{b}">'
            f'<figcaption><b>{titulo}</b><br><span class="nota">{nota}</span></figcaption></figure>')


# ---------- tabla 6 en HTML
def tabla_html(df, cols, fmts, clases=None):
    th = "".join(f"<th>{c}</th>" for c in cols)
    trs = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            f = fmts.get(c)
            tds.append(f"<td>{f.format(v) if f and pd.notna(v) else ('' if pd.isna(v) else v)}</td>")
        cls = clases(r) if clases else ""
        trs.append(f'<tr class="{cls}">{"".join(tds)}</tr>')
    return f'<div class="tabla-wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


filas_dec = []
for _, r in t6.sort_values("Silueta (Gower)", ascending=False).iterrows():
    if r["_ok"]:
        filas_dec.append(f'<li class="pasa"><b>{r["Algoritmo"]}</b> — pasa los dos filtros '
                         f'(cobertura {r["Cobertura"]:.2f}%, ARI {r["ARI"]:.3f})</li>')
    else:
        filas_dec.append(f'<li class="falla"><b>{r["Algoritmo"]}</b> — {"; ".join(r["_desc"])}</li>')

css_series = "\n".join(
    f"  .c{i}{{--s:{lo};}}" for i, (lo, dk) in enumerate(SERIES))
# Sin anidamiento CSS: selectores completos para cada scope de tema oscuro.
css_series_dark = "\n".join(
    f'  :root:where(:not([data-theme="light"])) .c{i}{{--s:{dk};}}'
    for i, (lo, dk) in enumerate(SERIES))
css_series_dark_attr = "\n".join(
    f'  :root[data-theme="dark"] .c{i}{{--s:{dk};}}'
    for i, (lo, dk) in enumerate(SERIES))

HTML = f"""<title>Comparación de seis algoritmos de agrupamiento — SERCOP/OCDS</title>
<style>
:root {{
  --surface-1:#fcfcfb; --plane:#f9f9f7; --text-1:#0b0b0b; --text-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --good:#0ca30c; --critical:#d03b3b; --warn:#fab219;
  color-scheme:light;
}}
{css_series}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    --surface-1:#1a1a19; --plane:#0d0d0d; --text-1:#fff; --text-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
    color-scheme:dark;
  }}
{css_series_dark}
}}
:root[data-theme="dark"] {{
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-1:#fff; --text-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  color-scheme:dark;
}}
{css_series_dark_attr}
* {{ box-sizing:border-box; }}
body {{
  margin:0; padding:32px 20px 72px; background:var(--plane); color:var(--text-1);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
}}
.wrap {{ max-width:980px; margin:0 auto; }}
h1 {{ font-size:26px; line-height:1.25; margin:0 0 6px; letter-spacing:-.01em; }}
h2 {{ font-size:19px; margin:38px 0 4px; letter-spacing:-.005em; }}
h3 {{ font-size:15px; margin:22px 0 6px; color:var(--text-2); font-weight:600; }}
.sub-h {{ color:var(--text-2); margin:0 0 4px; }}
.nota {{ color:var(--muted); font-size:12.5px; }}
.card {{
  background:var(--surface-1); border:1px solid var(--ring); border-radius:12px;
  padding:20px 22px; margin:14px 0;
}}
.hero {{ display:flex; flex-wrap:wrap; gap:26px; align-items:flex-start; }}
.hero .cifra {{ font-size:44px; font-weight:650; line-height:1; letter-spacing:-.02em; }}
.hero .cap {{ color:var(--text-2); font-size:13px; margin-top:5px; }}
.aviso {{
  border-left:3px solid var(--critical); background:color-mix(in srgb,var(--critical) 7%,transparent);
  padding:13px 16px; border-radius:0 8px 8px 0; margin:14px 0;
}}
.aviso b {{ color:var(--critical); }}

/* ---- gráfico de barras ---- */
.chart {{ position:relative; margin:14px 0 6px; }}
.fila {{ display:grid; grid-template-columns:190px 1fr; gap:12px; align-items:center;
        padding:5px 0; }}
/* atenuado por color, NO por opacity: opacity volvería translúcido el recorte
   de fondo de .val y la línea de referencia se vería a través del número. */
.fila.muerto .etq, .fila.muerto .val, .fila.muerto .sub {{ color:var(--muted); }}
.etq {{ font-size:13px; color:var(--text-2); text-align:right; }}
.pista {{ position:relative; height:26px; display:flex; align-items:center; outline:none; }}
.pista:focus-visible {{ box-shadow:0 0 0 2px var(--s,#2a78d6); border-radius:5px; }}
.barra {{
  height:15px; background:var(--s); border-radius:0 4px 4px 0;
  box-shadow:0 0 0 2px var(--surface-1);
}}
.fila.muerto .barra {{
  background:repeating-linear-gradient(135deg,var(--s) 0 5px,transparent 5px 9px);
  outline:1.5px solid var(--s); outline-offset:-1px;
}}
.val {{ font-size:12.5px; margin-left:9px; color:var(--text-1);
        font-variant-numeric:tabular-nums; white-space:nowrap;
        /* recorte de superficie: la etiqueta gana a la línea de referencia */
        position:relative; z-index:4; background:var(--plane); padding:1px 4px;
        border-radius:3px; }}
.sub {{ font-size:11.5px; margin-left:7px; color:var(--muted); white-space:nowrap; }}
.ref {{ position:absolute; top:2px; bottom:-2px; width:0; border-left:2px dashed var(--axis);
        pointer-events:none; z-index:3; }}
.ref.critico {{ border-left:2px dashed var(--critical); }}
.ref.linea-base {{ border-left:2.5px dashed var(--text-1); }}
.ref span {{ position:absolute; top:-17px; left:6px; font-size:11px; font-weight:600;
             color:var(--text-2); white-space:nowrap; background:var(--plane);
             padding:1px 5px; border-radius:4px; }}
.ref.critico span {{ color:var(--critical); }}
/* está al 89% del ancho: la etiqueta va a la izquierda para no desbordar */
.ref.linea-base span {{ color:var(--text-1); left:auto; right:6px; }}
.tag-gana {{ background:var(--good); color:#fff; font-size:10.5px; padding:1px 6px;
             border-radius:9px; vertical-align:middle; }}
.tag-fuera {{ background:var(--critical); color:#fff; font-size:10.5px; padding:1px 6px;
              border-radius:9px; vertical-align:middle; }}
.pista[data-tip]:hover::after {{
  content:attr(data-tip); position:absolute; left:0; bottom:100%; z-index:5;
  background:var(--text-1); color:var(--surface-1); font-size:12px;
  padding:5px 9px; border-radius:6px; white-space:nowrap; pointer-events:none;
}}

/* ---- tablas ---- */
.tabla-wrap {{ overflow-x:auto; margin:12px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; min-width:560px; }}
th, td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--grid);
          font-variant-numeric:tabular-nums; white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; font-variant-numeric:normal;
                                  white-space:normal; }}
th {{ color:var(--text-2); font-weight:600; border-bottom:1.5px solid var(--axis); }}
tr.gana td {{ background:color-mix(in srgb,var(--good) 9%,transparent); font-weight:600; }}
tr.gana td:first-child {{ box-shadow:inset 3px 0 0 var(--good); }}
tr.fuera td {{ color:var(--muted); }}
tr.fuera td:first-child {{ box-shadow:inset 3px 0 0 var(--critical); }}

ul.decision {{ list-style:none; padding:0; margin:10px 0; }}
ul.decision li {{ padding:6px 0 6px 24px; position:relative; font-size:13.5px; }}
ul.decision li::before {{ position:absolute; left:0; top:6px; font-weight:700; }}
ul.decision li.pasa::before {{ content:"✓"; color:var(--good); }}
ul.decision li.falla::before {{ content:"✕"; color:var(--critical); }}

.fig {{ margin:16px 0; }}
.fig img {{ max-width:100%; height:auto; border:1px solid var(--ring); border-radius:9px;
            background:#fff; }}
.fig figcaption {{ margin-top:7px; font-size:13px; color:var(--text-2); }}
.fig.falta figcaption {{ border-left:3px solid var(--warn); padding-left:11px; }}
.pie {{ margin-top:40px; padding-top:16px; border-top:1px solid var(--grid);
        font-size:12.5px; color:var(--muted); }}
@media (max-width:640px) {{
  .fila {{ grid-template-columns:1fr; gap:2px; }}
  .etq {{ text-align:left; }}
  .ref {{ display:none; }}
}}
</style>

<div class="wrap">

<h1>Comparación de seis algoritmos de agrupamiento</h1>
<p class="sub-h">Segmentación de compatibilidad proveedor–licitación · datos abiertos SERCOP (OCDS)<br>
Proyecto final de Inteligencia Artificial — ESPOL, CCPG1044, Grupo #3</p>

<div class="card hero">
  <div>
    <div class="cifra">{GANADOR}</div>
    <div class="cap">Ganador según la regla de decisión fijada de antemano</div>
  </div>
  <div>
    <div class="cifra">{SIL_GAN:.4f}</div>
    <div class="cap">Silueta sobre Gower · k = {int(adm.iloc[0]['k usado/obtenido'])} grupos</div>
  </div>
  <div>
    <div class="cifra">13 848</div>
    <div class="cap">procesos activos en la matriz (de 14 660)</div>
  </div>
</div>

<div class="aviso">
  <b>Ningún candidato supera la línea base trivial.</b> Agrupar por
  <code>modalidad_contratacion</code> sin más da una silueta de <b>{sil_mod:.4f}</b>, por encima
  del ganador ({SIL_GAN:.4f}) y de los seis algoritmos. Todos superan la línea base aleatoria
  ({sil_rand:.4f}), pero ese es el listón bajo. Es un resultado válido y se reporta como tal.
</div>

<h2>Silueta sobre la matriz de Gower</h2>
<p class="nota">Árbitro único: una sola matriz de Gower, <code>metric='precomputed'</code>, compartida
por los seis. Cada algoritmo se ajusta en su propio espacio pero se evalúa aquí. Más alto es mejor.</p>
<div style="height:22px"></div>
{barras(t6, "Silueta (Gower)", "{:.4f}", 0, 0.68,
        ref=[(sil_mod, f"← línea base por modalidad: {sil_mod:.4f}", "linea-base")],
        sub=lambda r: f"k={int(r['k usado/obtenido'])}")}
<p class="nota">La línea negra a trazos marca la línea base trivial. <b>Ninguna barra la alcanza</b>:
ése es el resultado que hay que reportar.</p>

<h2>Cobertura — filtro eliminatorio</h2>
<p class="nota">% de procesos asignados a un grupo real. DBSCAN etiqueta ruido (−1). Se descarta
todo candidato por debajo del 90 %.</p>
{barras(t6, "Cobertura", "{:.2f}%", 80, 101,
        ref=[(UMBRAL_COB, "umbral 90%", "critico")])}

<h2>Estabilidad — filtro eliminatorio</h2>
<p class="nota">ARI medio de 20 reejecuciones sobre submuestras del 80 %, contra la partición de
referencia. Se descarta por debajo de 0.60.</p>
{barras(t6, "ARI", "{:.3f}", 0, 1.05,
        ref=[(UMBRAL_ARI, "umbral 0.60", "critico")],
        sub=lambda r: f"± {r['ARI desv.']:.3f}")}

<h2>Cómo se aplicó la regla de decisión</h2>
<div class="card">
  <ul class="decision">
    {"".join(filas_dec)}
  </ul>
  <p class="nota" style="margin-top:12px">
    Entre los admitidos gana la mayor silueta sobre Gower: <b>{GANADOR}</b> con {SIL_GAN:.4f}.
    El segundo admitido queda en {SEGUNDO:.4f}; la diferencia es
    <b>{SIL_GAN - SEGUNDO:.4f}</b>, por encima del umbral de 0.02, así que
    <b>no hay empate técnico</b> y no hizo falta desempatar por interpretabilidad de centroides.
  </p>
  <p class="nota">
    DBSCAN tenía la mejor silueta de los seis ({float(t6.loc[t6['Algoritmo']=='DBSCAN','Silueta (Gower)'].iloc[0]):.4f})
    pero dejó {100 - float(t6.loc[t6['Algoritmo']=='DBSCAN','Cobertura'].iloc[0]):.2f} % de los procesos
    como ruido, y el filtro de cobertura lo elimina antes de comparar siluetas.
  </p>
</div>

<h2>Criterio secundario: Davies-Bouldin y Calinski-Harabasz</h2>
<p class="nota">Calculados sobre la <b>matriz codificada</b> (numéricas z-score + one-hot), no sobre
Gower: sus implementaciones asumen euclidiana y no aceptan matriz precalculada. Por eso son
criterio secundario — y aquí apuntan a un ganador distinto, lo que conviene no esconder.</p>

<h3>Davies-Bouldin — <span class="nota">más bajo es mejor</span></h3>
{barras(t6, "Davies-Bouldin", "{:.3f}", 0, 2.5, invertir=True)}

<h3>Calinski-Harabasz — <span class="nota">más alto es mejor</span></h3>
{barras(t6, "Calinski-Harabasz", "{:,.0f}", 0, 4500)}

<h2>Tabla 6 — comparación completa</h2>
{tabla_html(t6, ["Algoritmo", "k usado/obtenido", "Silueta (Gower)", "Davies-Bouldin",
                 "Calinski-Harabasz", "Cobertura", "ARI", "ARI desv."],
            {"Silueta (Gower)": "{:.4f}", "Davies-Bouldin": "{:.3f}",
             "Calinski-Harabasz": "{:,.0f}", "Cobertura": "{:.2f}%",
             "ARI": "{:.4f}", "ARI desv.": "{:.4f}"},
            clases=lambda r: "gana" if r["Algoritmo"] == GANADOR else ("fuera" if not r["_ok"] else ""))}

<h2>Tabla 7 — líneas base de control</h2>
{tabla_html(t7, ["Algoritmo", "k usado/obtenido", "Silueta (Gower)", "Davies-Bouldin",
                 "Calinski-Harabasz", "Cobertura", "ARI"],
            {"Silueta (Gower)": "{:.4f}", "Davies-Bouldin": "{:.3f}",
             "Calinski-Harabasz": "{:,.1f}", "Cobertura": "{:.2f}%", "ARI": "{:.4f}"})}
<p class="nota">El ARI de la línea base por modalidad es 1.000 por construcción: es una partición
determinista, no depende de la submuestra. No indica calidad.</p>

<h2>Tabla 8 — perfiles de los grupos del ganador</h2>
<p class="nota">La columna «Lectura de negocio» se deja vacía a propósito, para que la llenes tú.</p>
{tabla_html(t8, ["Grupo", "Procesos", "Similitud semántica", "Coincid. CPC",
                 "Desv. presupuesto", "Distancia (km)", "Modalidad dominante"],
            {"Similitud semántica": "{:.4f}", "Coincid. CPC": "{:.4f}",
             "Desv. presupuesto": "{:+.3f}", "Distancia (km)": "{:.1f}",
             "Procesos": "{:,.0f}"})}
<p class="nota">El grupo <b>6</b> es el único con afinidad real al proveedor consultante:
coincidencia CPC 0.996 y similitud semántica 0.0996, ambas un orden de magnitud por encima del
resto, sobre 2 257 procesos de Subasta Inversa Electrónica.</p>

<h2>Tabla 9 — atípicos por grupo (Isolation Forest, contamination = 0.05)</h2>
{tabla_html(t9, ["Grupo", "Procesos", "Atípicos", "%", "Variable que más se aleja"],
            {"Procesos": "{:,.0f}", "%": "{:.2f}%"})}

<h2>Figuras del notebook</h2>
{figura("fig_codo_por_algoritmo.png", "Curva del codo y silueta vs k, por algoritmo",
        "Izquierda: dispersión intra-grupo W(k) sobre Gower, comparable entre algoritmos. "
        "Derecha: silueta sobre Gower frente a k, con las dos líneas base como referencia.")}
{figura("fig_silueta_por_grupo_ganador.png", f"Silueta por grupo — {GANADOR}",
        "Distribución del coeficiente de silueta dentro de cada uno de los 10 grupos.")}
{figura("fig_kdistancias_dbscan.png", "Gráfico de k-distancias (DBSCAN)",
        "Distancia de Gower al 10.º vecino más cercano, ordenada. De aquí sale el eps barrido.")}
{figura("fig_mds_grupos_ganador.png", "Proyección MDS sobre Gower",
        "Submuestra estratificada: SMACOF es O(n²) por iteración y no es viable con n = 13 848. "
        "Es sólo una figura; ninguna métrica de la tabla 6 depende de ella.")}

<div class="pie">
<b>Cómo leer esta página.</b> Todas las cifras salen de la ejecución real del notebook
<code>notebooks/comparacion_portafolio.ipynb</code> sobre los archivos OCDS locales
(<code>data/2024–2026.jsonl.gz</code>, 90 076 procesos). Ninguna API consumida, ningún valor estimado.
Semilla <code>random_state=42</code> en todo.<br><br>
<b>Limitaciones declaradas.</b> (1) <code>scikit-learn-extra</code> no importa con numpy 2.x, así que
PAM está implementado sobre la matriz de Gower con inicialización BUILD + refinamiento alternante;
el SWAP exhaustivo de PAM clásico es O(k(n−k)²) por iteración e intratable con n = 13 848.
(2) La matriz de Gower se guarda en float32 (0.71 GB) en vez de float64 (1.43 GB) por la memoria
disponible. (3) 812 procesos activos (5.5 %) quedaron fuera por datos faltantes — 508 sin ítems,
304 sin provincia del comprador; no se imputó nada. (4) «Catálogo Electrónico» no aparece entre los
procesos activos, así que la regla de colapsar convenios es inocua sobre esta matriz.
</div>

</div>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"HTML escrito: {OUT}")
print(f"ganador={GANADOR}  silueta={SIL_GAN:.4f}  base_modalidad={sil_mod:.4f}")
for n in ["fig_codo_por_algoritmo.png", "fig_silueta_por_grupo_ganador.png",
          "fig_kdistancias_dbscan.png", "fig_mds_grupos_ganador.png"]:
    p = os.path.join(RES, n)
    print(f"   {'OK   ' if os.path.exists(p) else 'FALTA'} {n}")
print(f"tamaño: {os.path.getsize(OUT)/1024:.0f} KB")
