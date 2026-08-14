# Segmentación de compatibilidad proveedor–licitación (SERCOP / OCDS)

Proyecto final de Inteligencia Artificial — ESPOL, CCPG1044, Grupo #3.

Agrupa los procesos de contratación pública **activos** del SERCOP según qué tan
compatibles son con el historial de un proveedor, y entrega un panel local donde
el proveedor consulta su RUC y obtiene los procesos ordenados por afinidad.

Todo corre **100 % local**: sin APIs, sin conexión a internet, sin servicios de
terceros.

---

## Qué hay en el repositorio

| Carpeta | Contenido |
|---|---|
| `scripts/` | Toda la tubería: perfiles, extracción de activos, comparación de modelos, generación de reportes |
| `app/` | `dashboard.py` — el panel de Streamlit (Fase 3) |
| `resultados/` | Salidas y evidencia: modelo entrenado, tablas de comparación, figuras, capturas |
| `resultados/fase2_v1/`, `fase2_v21/` | Snapshots de las rondas anteriores de comparación (evidencia del proceso de revalidación) |
| `notebooks/` | Cuaderno con la comparación del portafolio |
| `data/` | **No versionado** — ver «Datos de entrada» |

---

## Datos de entrada (no vienen en el repositorio)

El repositorio **no incluye** los datos crudos: son 56 MB de descargas públicas
del portal de datos abiertos del SERCOP en formato OCDS (releases compilados,
JSON por líneas comprimido). Hay que colocarlos en `data/`:

```
data/2024.jsonl.gz     33,7 MB
data/2025.jsonl.gz     22,0 MB
data/2026.jsonl.gz      0,6 MB
```

**Sin estos archivos igual se puede levantar el panel**, porque los perfiles y
los procesos activos ya están calculados y versionados en `resultados/*.parquet`.
Los datos crudos sólo hacen falta para **reconstruir** la tubería desde cero
(Fase 1 y 2).

Fecha de corte de los datos activos: **2026-07-04**.

---

## Instalación

Requiere **Python 3.11**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Para las capturas automáticas del panel (opcional):

```bash
python -m playwright install chromium
```

---

## Cómo levantar el panel (Fase 3)

Es lo único que hace falta para ver el resultado funcionando:

```bash
streamlit run app/dashboard.py
```

Abre `http://localhost:8501`. La pantalla inicial ofrece tres RUCs de ejemplo
válidos; también se puede escribir cualquier RUC de 13 dígitos. Cada consulta
tarda entre 0,5 y 0,9 s sobre los 7 637 procesos activos.

Las tres pantallas: consulta, resultado (tabla ordenada por afinidad) y
exploración de grupos para RUCs sin historial.

---

## Cómo reconstruir todo desde cero

Requiere los tres `.gz` en `data/`. En este orden:

```bash
python scripts/construir_perfiles.py     # perfiles de proveedores      (~8 s)
python scripts/extraer_activos.py        # procesos activos             (~5 s)
python scripts/comparar_modelos.py       # portafolio de 6 modelos      (~6,4 min)
```

El tercero entrena y compara los seis algoritmos, aplica las reglas de
descalificación, elige el ganador y guarda `resultados/modelo_ganador.pkl`.
Todo es determinista: `seed = 42` en cada algoritmo que la acepta, así que dos
corridas dan resultados idénticos.

Scripts auxiliares (generan la evidencia del informe, no son parte de la tubería):

```bash
python scripts/generar_capturas.py       # capturas del panel con Playwright
python scripts/generar_html.py           # reporte HTML de la comparación
python scripts/generar_notebook.py       # cuaderno de la comparación
```

---

## Resultados

Se compararon **seis** algoritmos bajo condiciones idénticas (mismas variables,
mismo preprocesamiento, mismos tres proveedores de prueba, k de 3 a 10), con la
**silueta sobre distancia de Gower ponderada** como único árbitro.

**Ganador: K-Medoids (PAM), k = 3**, silueta promedio de portafolio **0,2769**.

Ninguna métrica se tomó sola: cada candidato debía superar dos líneas base de
control (etiquetas aleatorias y la partición trivial por modalidad) y pasar
cuatro reglas de descalificación —ARI contra `modalidad_norm`, entropía de
tamaños, cobertura y colinealidad de variables—. Dos de los seis algoritmos
quedaron descalificados **pese a tener la silueta más alta**: el jerárquico
(0,5900) porque metía casi todo en un solo grupo, y el de densidad (0,3261)
porque descartaba el 23 % de los procesos como ruido.

El detalle está en `resultados/comparacion_modelos.csv`,
`resultados/candidatos_evaluados.csv` y `resultados/log_fase2_2.txt`.

---

## Variables del modelo

Seis numéricas y una categórica:

| Variable | Qué mide |
|---|---|
| `distancia_km` | Haversine entre la capital provincial del proveedor y la del comprador |
| `cpc_jaccard4` | Jaccard entre los CPC históricos del proveedor y los del proceso (4 dígitos) |
| `sim_tfidf` | Coseno TF-IDF entre el corpus del proveedor y el objeto del proceso |
| `desviacion_presupuesto` | Distancia del presupuesto del proceso al monto típico del proveedor |
| `actividad_cpc_comprador` | Cuánto compra ese comprador en ese rubro |
| `afinidad_comprador` | Historial de adjudicaciones entre ese proveedor y ese comprador |
| `modalidad_norm` *(categórica)* | Modalidad de contratación, normalizada a 9 categorías |

Preprocesamiento fijo: winsorización p1/p99 → `StandardScaler` → Gower ponderada
(la categórica pesa 1/3 frente a 1 de cada numérica). Los límites de
winsorización y el escalador quedan guardados dentro de `modelo_ganador.pkl`
para que la inferencia del panel use exactamente la misma transformación.

---

## Limitaciones conocidas

Están documentadas porque afectan cómo se leen los resultados:

- **La estructura de grupos es débil.** La silueta es casi plana entre k=3 y
  k=10 (0,242 – 0,277): los datos son un continuo, no tienen un número natural
  de grupos. k=3 ganó por 0,017 sobre k=7, un margen delgado.
- **De los tres grupos, sólo uno es accionable.** Los otros dos se separan
  básicamente por distancia geográfica, con rubros igual de irrelevantes en
  ambos. La granularidad útil la da el ranking continuo dentro del grupo, no la
  etiqueta.
- **`IsolationForest(contamination=0.05)` por grupo devuelve ~5 % por
  construcción**: `contamination` es un parámetro, no una medición. La columna
  informativa es la del bosque global.
- **`cpc_match` quedó fuera** por resultar constante en 0; se verificó que no era
  un error de mapeo (adjudicaciones y licitaciones comparten esquema CPC,
  1 437 códigos en común).

---

## Nota sobre los datos

`resultados/perfiles_proveedores.parquet` contiene RUC y razón social de los
proveedores. Es información pública del portal de contratación abierta del
SERCOP, pero **conviene mantener el repositorio privado** si no hay una razón
explícita para publicarla.
