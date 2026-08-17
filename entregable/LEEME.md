# Segmentación de compatibilidad proveedor–licitación (SERCOP / OCDS)

**Proyecto final de Inteligencia Artificial — ESPOL, CCPG1044, Grupo #3**

Sistema que, dado el RUC de un proveedor, agrupa los procesos de contratación
pública **vigentes** según qué tan compatibles son con su historial de
adjudicaciones, y le presenta las oportunidades ordenadas.

Corre **100 % local**: sin APIs, sin conexión a internet, sin servicios de
terceros.

---

## Dónde está cada punto del entregable

| Punto solicitado | Ubicación en este paquete |
|---|---|
| **a) El código fuente — el notebook** | `notebooks/modelo_final.ipynb` (ya ejecutado, con todas las salidas) y la tubería completa en `scripts/` |
| **b) Un conjunto de ejemplos para correr su modelo** | `ejemplos/` — 10 casos con su resultado esperado y un script que los corre y valida |
| **c) La interface desarrollada** | `app/dashboard.py` + `.streamlit/config.toml`. Capturas de las 6 pantallas en `capturas/` |
| **d) Librerías no públicas utilizadas** | `librerias/LEEME.txt` — **ninguna**; todas son de PyPI, versiones fijadas en `requirements.txt` |
| **e) El borrador del póster** | `poster/Poster_Grupo3_P2.pptx` y su PDF |

Además: `resultados/` trae el **modelo ya entrenado** (`modelo_ganador.pkl`) y
las tablas de evidencia, para que todo funcione sin reentrenar ni descargar los
datos crudos.

---

## Instalación

Requiere **Python 3.11**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Las dos cosas que conviene correr

**1. Los ejemplos** (unos 10 segundos):

```bash
python ejemplos/correr_ejemplos.py
```

Pasa los 10 RUCs por el modelo y compara con el resultado esperado. Devuelve
código 0 si todos coinciden. Detalle de cada caso en `ejemplos/LEEME.txt`.

**2. La interfaz** (abre en `http://localhost:8501`):

```bash
streamlit run app/dashboard.py
```

La pantalla inicial ofrece tres RUCs de ejemplo con un botón. Cada consulta
tarda 0,4–0,6 s sobre los 7.637 procesos vigentes.

**3. El notebook** ya viene ejecutado; para volver a correrlo:

```bash
jupyter notebook notebooks/modelo_final.ipynb
```

---

## Qué se hizo

Se compararon **seis algoritmos** de agrupamiento en igualdad de condiciones
—mismas variables, mismo preprocesamiento, mismos tres proveedores de prueba,
k de 3 a 10, semilla 42— con la **silueta sobre distancia de Gower ponderada**
como árbitro único declarado de antemano.

**Ganador: K-Medoids (PAM), k = 3**, silueta 0,2769 (promedio de tres
proveedores).

| Algoritmo | Silueta | Estado |
|---|---|---|
| Jerárquico (average) | 0,5900 | **DESCALIFICADO** — entropía 0,23 < 0,50 |
| DBSCAN / HDBSCAN | 0,3261 | **DESCALIFICADO** — cobertura 77 % < 85 % |
| **K-Medoids (PAM)** | **0,2769** | **GANADOR** |
| K-Means | 0,2581 | admitido |
| K-Prototypes | 0,2540 | admitido |
| Gaussian Mixture | 0,2341 | admitido |

**El hallazgo central:** los dos algoritmos con la silueta más alta son los dos
que quedaron descalificados. El jerárquico mete casi todo en un grupo; el de
densidad descarta el 23 % de los procesos como ruido y calcula su silueta sólo
sobre lo que sí clasifica. Sin las reglas fijadas de antemano, el ganador
nominal habría sido degenerado.

Cada candidato tenía que superar además **dos líneas base de control**
(etiquetas aleatorias y la partición trivial por modalidad) y pasar **cuatro
reglas de descalificación**: ARI contra `modalidad_norm`, entropía de tamaños,
cobertura y colinealidad de variables.

**Trayectoria de tres rondas.** El ARI del ganador contra la modalidad fue
**1,000 → 0,380 → −0,039**. En la primera ronda el «modelo» era la columna de
modalidad reetiquetada; en la última es independiente de ella.

---

## Limitaciones (están en el notebook, sección 12)

Lo que conviene saber antes de leer los resultados:

- **La estructura de grupos es débil.** La silueta es casi plana entre k=3 y
  k=10 (0,242–0,277): los datos son un continuo, no tienen un número natural de
  grupos. k=3 ganó por 0,017 sobre k=7, y los ocho valores de k pasan las cuatro
  reglas.
- **De los tres grupos, sólo uno es accionable.** Los otros dos se separan por
  distancia geográfica, con coincidencia de rubro casi nula en ambos.
- **Se eliminaron tres variables defectuosas** detectadas por diagnóstico:
  `cpc_match` (constante en 0), `modalidad_afinidad` (η²=1,000 contra la
  categórica: era una recodificación) y `log_presupuesto` (r=1,0000 con
  `desviacion_presupuesto`).
- **El Isolation Forest por grupo no mide nada:** `contamination=0.05` fuerza el
  5 % que devuelve. La columna válida es la del bosque global.
- **La compatibilidad no es una probabilidad de adjudicación.** Mide parecido
  con el historial; no modela competencia, precio ni capacidad de cumplimiento.
- **Un proveedor muy diversificado no obtiene resultado.** El caso extremo del
  corte (30.721 adjudicaciones en 313 rubros) da coincidencia 0,008 y la
  interfaz lo declara «sin coincidencias» en vez de inventar una recomendación.
  Es el ejemplo n.º 4 del conjunto de pruebas.

---

## Datos

Los **56 MB de datos crudos no van en este paquete**: son descargas públicas del
portal de datos abiertos del SERCOP en formato OCDS. Sólo hacen falta para
reconstruir la tubería desde cero, y el paquete ya trae los perfiles y los
procesos calculados.

Para reconstruir todo, con `data/{2024,2025,2026}.jsonl.gz` en su sitio:

```bash
python scripts/construir_perfiles.py    # 12.220 proveedores        (~8 s)
python scripts/extraer_activos.py       # 7.637 procesos vigentes   (~5 s)
python scripts/comparar_modelos.py      # 6 algoritmos              (~7,7 min)
```

Todo es determinista con semilla 42: dos corridas dan resultados idénticos.

**Corte de datos: 2026-07-04.**

---

## Repositorio

Código completo con historial en
[github.com/jcarrome/proyecto-ia-sercop](https://github.com/jcarrome/proyecto-ia-sercop)

## Nota

`resultados/perfiles_proveedores.parquet` contiene RUC y razón social de los
proveedores. Es información pública del portal de contratación abierta del
SERCOP.
