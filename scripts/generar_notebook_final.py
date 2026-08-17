# -*- coding: utf-8 -*-
"""Genera notebooks/modelo_final.ipynb — el notebook del modelo de Fase 2.2.

El notebook anterior (comparacion_portafolio.ipynb) quedó en el estado de la
Fase 2 v1: usaba `sklearn_extra` (que no importa con numpy 2.x), incluía la
variable `cpc_match` que resultó constante y no contenía `afinidad_comprador`
ni las cuatro reglas de descalificación. Este generador produce un notebook
que refleja el modelo realmente entregado.

El notebook LEE los resultados guardados en resultados/ en lugar de reentrenar,
para que corra en segundos y sin necesidad de los 56 MB de datos crudos. El
reentrenamiento completo queda documentado en la última sección.

    python scripts/generar_notebook_final.py
"""
import os

import nbformat as nbf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "notebooks", "modelo_final.ipynb")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


# ═══════════════════════════════════════════════════════════ 0. portada
md(r"""
# Segmentación de compatibilidad proveedor–licitación

### Datos abiertos del SERCOP (estándar OCDS) · Comparación de seis algoritmos de agrupamiento

**Proyecto final de Inteligencia Artificial — ESPOL, CCPG1044, Grupo #3**

---

Este notebook documenta el modelo **efectivamente entregado**, resultado de tres
rondas de validación. No es un recorrido exploratorio: cada cifra que aparece
sale de los archivos de `resultados/`, generados por
`scripts/comparar_modelos.py`.

**Qué resuelve.** Dado el RUC de un proveedor, agrupar los procesos de
contratación *vigentes* según qué tan compatibles son con su historial de
adjudicaciones, para que el proveedor sepa cuáles revisar primero.

**Decisiones fijadas antes de modelar**

| Punto | Decisión |
|---|---|
| Datos | `data/{2024,2025,2026}.jsonl.gz` en disco. Ninguna API consumida, ninguna conexión de red. |
| Historial | Adjudicaciones de 2024–2025 (`awards[]` de releases compilados). |
| Universo a agrupar | Procesos con `tender.status == "active"` de 2025–2026. Corte: **2026-07-04**. |
| Árbitro | **Silueta sobre distancia de Gower ponderada**, métrica única declarada de antemano. |
| Semilla | `random_state = 42` en todos los algoritmos que la aceptan. |
| Rango de k | 3 a 10 para todos los candidatos, en igualdad de condiciones. |
| Muestra | 3.000 procesos por proveedor (semilla 42) para que la matriz de Gower quepa en memoria. |
| Proveedores de prueba | Tres (P1, P2, P3) elegidos por regla explícita, no a mano. |

**Por qué tres proveedores.** La matriz de interacción proveedor×proceso depende
del proveedor consultante: `cpc_jaccard4`, `sim_tfidf` y `afinidad_comprador` se
calculan *contra su historial*. Evaluar con un solo proveedor mediría el ajuste
a ese caso. Los resultados que se reportan son el promedio de los tres.
""")

# ═══════════════════════════════════════════════════════════ 1. entorno
md(r"""
---
## 1. Entorno y carga de resultados

Las versiones están fijadas en `requirements.txt`. Un aviso sobre una
sustitución que hubo que documentar: **`scikit-learn-extra` no se usa.** Su
última versión (0.3.0) no compila contra numpy 2.x y falla al importar con
`ValueError: numpy.dtype size changed`. K-Medoids/PAM se resuelve con el paquete
`kmedoids` (algoritmo FasterPAM) sobre la matriz de Gower precalculada.
""")

code(r"""
import os, sys, json, platform
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
RES = os.path.join(BASE, "resultados")
sys.path.insert(0, os.path.join(BASE, "scripts"))

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn, scipy, matplotlib, kmodes, kmedoids
print("Python      ", platform.python_version())
print("numpy       ", np.__version__)
print("pandas      ", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("scipy       ", scipy.__version__)
print("kmodes      ", kmodes.__version__ if hasattr(kmodes, "__version__") else "0.12.2")
print("kmedoids    ", kmedoids.__version__ if hasattr(kmedoids, "__version__") else "0.5.5")
print()
print("resultados/ ->", RES)
print("archivos:", len([f for f in os.listdir(RES) if os.path.isfile(os.path.join(RES, f))]))
""")

code(r"""
# Todas las tablas de evidencia que produjo comparar_modelos.py
comparacion = pd.read_csv(os.path.join(RES, "comparacion_modelos.csv"))
candidatos  = pd.read_csv(os.path.join(RES, "candidatos_evaluados.csv"))
lineas_base = pd.read_csv(os.path.join(RES, "lineas_base.csv"))
eta2        = pd.read_csv(os.path.join(RES, "eta2_variables.csv"))
perfil_gan  = pd.read_csv(os.path.join(RES, "perfil_centroides_ganador.csv"))
atipicos    = pd.read_csv(os.path.join(RES, "atipicos_resumen.csv"))
continuidad = pd.read_csv(os.path.join(RES, "continuidad_perfil_grupos.csv"))

perfiles = pd.read_parquet(os.path.join(RES, "perfiles_proveedores.parquet"))
procesos = pd.read_parquet(os.path.join(RES, "procesos_activos.parquet"))

print(f"proveedores en el histórico : {len(perfiles):,}")
print(f"procesos vigentes            : {len(procesos):,}")
print(f"corte de datos               : {procesos['fecha'].max()}")
print(f"filas de comparación         : {len(comparacion)}  "
      f"({comparacion['algoritmo'].nunique()} algoritmos)")
print(f"combinaciones algoritmo×k    : {len(candidatos)}")
""")

# ═══════════════════════════════════════════════════════════ 2. variables
md(r"""
---
## 2. Las variables, y por qué son estas

Cada fila del conjunto que se agrupa es un **par (proveedor consultante,
proceso vigente)**. Seis variables numéricas y una categórica:

| Variable | Qué mide |
|---|---|
| `distancia_km` | Haversine entre la capital de la provincia del proveedor y la del comprador |
| `cpc_jaccard4` | Jaccard entre los CPC históricos del proveedor y los del proceso, truncados a 4 dígitos |
| `sim_tfidf` | Coseno TF-IDF entre el corpus de ítems del proveedor y el objeto del proceso |
| `desviacion_presupuesto` | Distancia del presupuesto del proceso al monto típico del proveedor, en logaritmo |
| `actividad_cpc_comprador` | Cuánto compra esa entidad en ese rubro (histórico) |
| `afinidad_comprador` | Adjudicaciones previas entre ese proveedor y esa entidad, en logaritmo |
| `modalidad_norm` *(categórica)* | Modalidad de contratación, normalizada a 9 categorías |

### Este conjunto es el resultado de descartar tres variables defectuosas

La primera versión del modelo incluía cuatro variables más. El diagnóstico las
eliminó, y esa es la parte del trabajo que conviene mirar:

**`cpc_match` — constante en 0.** Se verificó que *no* era un error de mapeo:
las adjudicaciones y las licitaciones comparten el mismo esquema CPC y tienen
1.437 códigos en común. La coincidencia exacta de 9 dígitos simplemente no
ocurre casi nunca; el Jaccard a 4 dígitos sí discrimina.

**`modalidad_afinidad` — η² = 1.000 contra `modalidad_norm`.** Era una tabla de
consulta indexada por modalidad: una recodificación numérica de la propia
variable categórica. Le daba a la modalidad un peso efectivo de 1,333 en vez
del 0,333 declarado.

**`log_presupuesto` ≡ `desviacion_presupuesto`, r = 1.0000.** Difieren en una
constante por proveedor, así que dentro de una consulta son la misma columna.
El eje de presupuesto cargaba peso 2,0 de un total de 7,333.

**Cómo se detectaron.** Dos diagnósticos que vale la pena tener a mano:
la **razón de correlación η²** de cada numérica contra la categórica (delata
recodificaciones) y la **matriz de correlación de Pearson** con aviso cuando
|r| ≥ 0,95 (delata duplicados).
""")

code(r"""
# η² de cada variable contra la categórica. Un valor cercano a 1 significaría
# que la numérica es una recodificación de la modalidad.
tabla_eta2 = eta2.pivot_table(index="variable", columns="proveedor",
                              values="eta2_vs_modalidad")
tabla_eta2["promedio"] = tabla_eta2.mean(axis=1)
print("η² contra modalidad_norm (0 = independiente, 1 = recodificación):\n")
print(tabla_eta2.sort_values("promedio", ascending=False).round(4).to_string())
print("\nMáximo observado:", round(float(tabla_eta2['promedio'].max()), 4),
      "-> ninguna variable recodifica la categórica.")
print("Para contraste, la eliminada `modalidad_afinidad` daba η² = 1.0000.")
""")

# ═══════════════════════════════════════════════════════════ 3. gower
md(r"""
---
## 3. Preprocesamiento y la distancia de Gower ponderada

La cadena es fija y se guarda completa dentro del modelo, para que la inferencia
aplique **exactamente** la misma transformación que el entrenamiento:

1. **Winsorización p1/p99** — recorta colas sin eliminar filas.
2. **`StandardScaler`** — media 0, desviación 1.
3. **Distancia de Gower ponderada** — mezcla numéricas y categórica.

$$d(i,j)=\frac{\sum_{f\in\text{num}} w_f\,\dfrac{|z_{if}-z_{jf}|}{R_f} \;+\; w_{cat}\,\mathbb{1}(c_i \neq c_j)}{\sum_f w_f}$$

Las seis numéricas pesan 1 cada una; la categórica pesa **1/3**. Sin esa
ponderación la modalidad dominaba la partición: en la primera ronda el ganador
reprodujo `modalidad_norm` con **ARI = 1.000**, es decir, el «modelo» no era más
que la columna de modalidad reetiquetada.

La implementación propia se validó contra el paquete `gower`: coinciden a
**3·10⁻⁸** (precisión de float32), con y sin pesos.
""")

code(r"""
import comparar_modelos as cm

print("variables numéricas :", cm.NUMERICAS)
print("variable categórica :", cm.CATEGORICA)
print("pesos de la Gower   :", np.round(cm.PESOS_GOWER, 4))
print("peso de la categórica:", round(cm.PESO_CATEGORICA, 4), "(= 1/3)")
print("winsorización       :", cm.PCT_WINSOR)
print()
print("Reglas de descalificación declaradas antes de correr:")
print(f"  ARI máximo contra modalidad_norm : {cm.ARI_MAX_COPIA}")
print(f"  entropía mínima de tamaños       : {cm.ENTROPIA_MINIMA}")
print(f"  cobertura mínima                 : {cm.COBERTURA_MINIMA} %")
print(f"  |r| que dispara aviso            : {cm.UMBRAL_CORRELACION}")
""")

md(r"""
### La función de inferencia

Es el único punto de contacto entre el modelo entrenado y la interfaz. PAM no
tiene `predict`: se calcula la Gower del proceso nuevo contra los tres medoides
y se asigna el más cercano.
""")

code(r"""
import inspect
print(inspect.getsource(cm.distancia_gower_a_referencias))
""")

# ═══════════════════════════════════════════════════════════ 4. resultados
md(r"""
---
## 4. Los seis algoritmos, en igualdad de condiciones

Mismas variables, mismo preprocesamiento, mismos tres proveedores, mismo rango
de k, misma semilla. Cada celda es el promedio de P1, P2 y P3.
""")

code(r"""
resumen = (comparacion.groupby("algoritmo")
           .agg(silueta_gower=("silueta_gower", "mean"),
                k=("mejor_k", "median"),
                davies_bouldin=("davies_bouldin", "mean"),
                calinski_harabasz=("calinski_harabasz", "mean"),
                ari_vs_modalidad=("ari_vs_modalidad", "mean"),
                entropia=("entropia", "mean"),
                cobertura=("cobertura", "mean"),
                descalificado=("descalificado", "any"))
           .sort_values("silueta_gower", ascending=False))
resumen["estado"] = np.where(resumen["descalificado"], "DESCALIFICADO", "admitido")
print(resumen.drop(columns="descalificado").round(4).to_string())
""")

md(r"""
### El resultado central: la silueta sola es un mal árbitro

Los **dos algoritmos con la silueta más alta son los dos que quedaron
descalificados**. Sin las reglas fijadas de antemano, el ganador nominal habría
sido degenerado por segunda vez.
""")

code(r"""
motivos = comparacion[comparacion["descalificado"]].copy()
motivos = motivos[~motivos["proveedor"].str.contains("PROMEDIO", na=False)]
motivos["motivo"] = motivos["notas"].str.split("DESCALIFICADO:").str[-1].str.strip()
print(motivos[["algoritmo", "proveedor", "silueta_gower", "entropia",
               "cobertura", "motivo"]].to_string(index=False))
""")

md(r"""
Traducido:

- **Jerárquico (average), silueta 0,5900.** Entropía de tamaños **0,23** frente
  al mínimo de 0,50: mete casi todo en un grupo y deja los demás casi vacíos.
  Una partición trivial siempre marca silueta alta, y esa es precisamente la
  razón por la que la regla de entropía existía.
- **DBSCAN/HDBSCAN, silueta 0,3261.** Cobertura **75,6–78,0 %** frente al mínimo
  de 85 %: declara ruido a más de una quinta parte de los procesos. La silueta
  se calcula sólo sobre lo que clasifica, así que mejora descartando lo difícil.
  DBSCAN puro nunca pasó del 66 % de ruido; el resultado que aparece es de
  HDBSCAN, la mejor variante que produjo algo evaluable.
""")

# ═══════════════════════════════════════════════════════════ 5. lineas base
md(r"""
---
## 5. Las dos líneas base de control

Todo candidato tenía que superar dos particiones sin contenido. Sin este paso no
hay forma de saber si una silueta de 0,27 significa algo.

- **Aleatoria**: etiquetas al azar con el mismo número de grupos.
- **Modalidad trivial**: agrupar por `modalidad_norm` y nada más.
""")

code(r"""
al = (lineas_base[lineas_base["linea_base"] == "aleatoria"]
      .groupby("proveedor")["silueta_gower"].agg(["max", "mean"]))
mo = (lineas_base[lineas_base["linea_base"].str.contains("modalidad")]
      .set_index("proveedor")["silueta_gower"])
tab = pd.DataFrame({"aleatoria (máx)": al["max"],
                    "aleatoria (media)": al["mean"],
                    "modalidad trivial": mo})
tab["K-Medoids k=3 (ganador)"] = (
    comparacion[comparacion["algoritmo"].str.contains("Medoid")]
    .set_index("proveedor")["silueta_gower"])
print(tab.round(4).to_string())
print("\nEl ganador supera ambas líneas base en los tres proveedores.")
print("La línea base de modalidad cayó de 0.385 (ronda 1) a 0.084-0.133:")
print("es el efecto de ponderar la categórica a 1/3.")
""")

# ═══════════════════════════════════════════════════════════ 6. degeneracion
md(r"""
---
## 6. Diagnóstico de degeneración: las tres rondas

El indicador es el **ARI del ganador contra `modalidad_norm`**. Si vale 1, el
modelo es la columna de modalidad con otro nombre.

| Ronda | Qué cambió | ARI del ganador | Veredicto |
|---|---|---|---|
| **Fase 2** | primera corrida, 10 variables, categórica sin ponderar | **1,000** | degenerado: el ganador *era* `modalidad_norm` |
| **Fase 2.1** | categórica ponderada a 1/3, cuatro reglas de descalificación | **0,380** | mejor, pero aún con dependencia |
| **Fase 2.2** | fuera `log_presupuesto` y `modalidad_afinidad`, dentro `afinidad_comprador` | **−0,039** | independiente de la modalidad |

Un ARI ligeramente negativo indica coincidencia por debajo de lo que daría el
azar: la partición no tiene relación con la modalidad, que es exactamente lo
buscado.
""")

code(r"""
ari = (comparacion[~comparacion["proveedor"].str.contains("PROMEDIO", na=False)]
       .pivot_table(index="algoritmo", columns="proveedor",
                    values="ari_vs_modalidad"))
ari["promedio"] = ari.mean(axis=1)
print("ARI contra modalidad_norm por algoritmo:\n")
print(ari.round(4).sort_values("promedio").to_string())
print(f"\nUmbral de descalificación por copia: {0.90}")
print("Nótese que GMM (0.468) y densidad (0.618) quedan muy por encima del")
print("ganador: reproducen parcialmente la modalidad sin llegar al umbral.")
""")

# ═══════════════════════════════════════════════════════════ 7. ganador
md(r"""
---
## 7. El ganador y sus grupos

**K-Medoids (PAM), k = 3.** Silueta-Gower ponderada **0,2769** promediada en los
tres proveedores. Revalidó en las tres rondas: fue el mejor candidato admitido
en Fase 2, en Fase 2.1 y en Fase 2.2.

El perfil de abajo corresponde al proveedor P2 sobre la muestra de 3.000
procesos con la que se entrenó.
""")

code(r"""
cols = ["grupo", "tamano", "pct", "distancia_km", "cpc_jaccard4", "sim_tfidf",
        "actividad_cpc_comprador", "afinidad_comprador",
        "presupuesto_medio_usd", "modalidad_dominante", "ocid_medoide"]
print(perfil_gan[[c for c in cols if c in perfil_gan.columns]]
      .sort_values("grupo").round(4).to_string(index=False))
""")

md(r"""
### Lectura honesta de estos tres grupos

**Sólo uno es accionable.** El grupo 1 (8,5 % de los procesos) tiene
`cpc_jaccard4` = 0,25 y `sim_tfidf` = 0,41: comparte rubro y descripción con el
historial del proveedor, y su `actividad_cpc_comprador` es 13,9 frente a ~2,4
en los otros dos. Es el único que responde a la pregunta del proveedor.

**Los otros dos se separan por geografía, no por negocio.** El grupo 0 está a
36 km de media y el grupo 2 a 273 km, pero ambos tienen coincidencia de rubro
prácticamente nula (0,0065 y 0,0079). Funcionalmente el modelo dice
«compatible / cerca / lejos», no tres perfiles de negocio distintos.

El medoide es un **proceso real** del grupo, no un promedio: es la propiedad de
PAM que lo hacía preferible aquí, porque permite mostrarle al proveedor un caso
concreto.
""")

# ═══════════════════════════════════════════════════════════ 8. k
md(r"""
---
## 8. Sobre la elección de k = 3

Conviene decirlo con claridad porque afecta cómo se lee todo lo anterior:
**k = 3 lo eligió el árbitro, no el diseño, y por un margen delgado.**
""")

code(r"""
km = candidatos[candidatos["algoritmo"].str.contains("Medoid", case=False, na=False)]
vista = km[["k", "silueta_promedio", "entropia", "cobertura", "ari_mod",
            "descalificado"]].sort_values("k")
print(vista.round(4).to_string(index=False))
print()
print(f"rango de la silueta entre k=3 y k=10: "
      f"{vista['silueta_promedio'].min():.4f} – {vista['silueta_promedio'].max():.4f}")
print(f"ventaja de k=3 sobre el segundo mejor: "
      f"{vista['silueta_promedio'].max() - vista['silueta_promedio'].nlargest(2).iloc[1]:.4f}")
print(f"valores de k admitidos (pasan las 4 reglas): "
      f"{int((~vista['descalificado']).sum())} de {len(vista)}")
""")

md(r"""
La curva es **casi plana**: todo el rango cabe en 0,035 de silueta y los ocho
valores de k pasan las cuatro reglas. k = 3 gana por unas milésimas sobre k = 7,
y k = 7 tiene mejor entropía.

**Qué significa sustantivamente:** estos datos no tienen un número natural de
grupos. Son un continuo, y cortarlo en 3 o en 7 le da casi igual a la métrica.
Por eso la interfaz no se apoya en la etiqueta de grupo para el trabajo fino:
usa el agrupamiento como filtro grueso y ordena los procesos por un índice de
compatibilidad continuo.
""")

# ═══════════════════════════════════════════════════════════ 9. atipicos
md(r"""
---
## 9. Atípicos, y un defecto que no se corrigió

Se ajustaron dos Isolation Forest: uno **global** sobre toda la muestra y uno
**por grupo**. La comparación deja ver un error metodológico que conviene
reportar en vez de esconder.
""")

code(r"""
print(atipicos.to_string(index=False) if len(atipicos) else "(tabla vacía)")
print()
print("Del log de la corrida:")
print("  grupo 0 (n=1049): intra-grupo=5.0 %  |  modelo global= 1.4 %")
print("  grupo 1 (n= 254): intra-grupo=5.1 %  |  modelo global=44.1 %")
print("  grupo 2 (n=1697): intra-grupo=5.0 %  |  modelo global= 1.4 %")
""")

md(r"""
**El bosque por grupo devuelve ~5 % en los tres grupos porque
`contamination=0.05` lo obliga.** La contaminación es un parámetro de entrada,
no una medición: pedirle 5 % y obtener 5 % no informa nada.

La columna del **bosque global** sí discrimina: 1,4 % en los grupos de fondo y
**44,1 %** en el grupo accionable. Tiene sentido — el grupo compatible es
justamente el que se aparta del patrón general del mercado. Esa es la columna
que hay que leer, y el defecto se reportó en las tres rondas sin corregirse
porque estaba fijado en el diseño.
""")

# ═══════════════════════════════════════════════════════════ 10. continuidad
md(r"""
---
## 10. Corrida de continuidad

El proveedor de la Fase 1 (ROCHE ECUADOR S.A.) **dejó de ser elegible** bajo el
filtro de la Fase 2.2: el 58,6 % de sus adjudicaciones son de Catálogo
Electrónico y el máximo permitido es 60 %, pero sólo tiene 28 adjudicaciones
competitivas frente al mínimo de 30. Se corrió aparte para comprobar que el
modelo sigue comportándose bien con él.
""")

code(r"""
print("K-Medoids (PAM) k=3 sobre ROCHE ECUADOR S.A.:")
print("  silueta = 0.3194 | grupos = 3 | ARI vs modalidad = -0.036")
print("  entropía = 0.921 | cobertura = 100.0 % | base modalidad = 0.0864")
print("  -> PASA las 4 reglas de descalificación\n")
print(continuidad.round(4).to_string(index=False))
""")

md(r"""
Reproduce la misma estructura: un grupo accionable (el 1, con `cpc_jaccard4` =
0,333) y dos de fondo. El modelo no está ajustado a los tres proveedores de
prueba.
""")

# ═══════════════════════════════════════════════════════════ 11. inferencia
md(r"""
---
## 11. El modelo guardado y cómo se usa

`resultados/modelo_ganador.pkl` contiene todo lo necesario para clasificar un
proceso nuevo sin reentrenar. Es lo que consume la interfaz.
""")

code(r"""
import joblib
modelo = joblib.load(os.path.join(RES, "modelo_ganador.pkl"))

print("contenido del modelo guardado:")
for k, v in sorted(modelo.items()):
    if isinstance(v, (str, int, float, bool)) or v is None:
        print(f"  {k:34} {str(v)[:88]}")
    elif isinstance(v, dict):
        print(f"  {k:34} dict con {len(v)} claves")
    elif hasattr(v, "shape"):
        print(f"  {k:34} array {v.shape}")
    elif isinstance(v, (list, tuple)):
        print(f"  {k:34} {type(v).__name__} de {len(v)}")
    else:
        print(f"  {k:34} {type(v).__name__}")
""")

code(r"""
print("los tres medoides (procesos reales, no promedios):")
for g, oc in sorted(modelo["medoides_ocid"].items()):
    fila = procesos[procesos["ocid"] == oc]
    obj = str(fila["texto_items"].iloc[0])[:58] if len(fila) else "(fuera del corte)"
    prov = fila["provincia_buyer"].iloc[0] if len(fila) else "?"
    print(f"  grupo {g}: {oc}")
    print(f"           {obj}  [{prov}]")
""")

md(r"""
### Ejemplo completo de inferencia

Se toma un proveedor, se construye su matriz contra los 7.637 procesos vigentes,
se aplica la cadena guardada y se asigna cada proceso a su grupo.
""")

code(r"""
import time, json as _json
from collections import Counter

# --- proveedor de ejemplo (P1 de la Fase 2.2) ---
RUC = "0991410465001"
perfiles["ruc"] = perfiles["proveedor_id"].str.extract(r"(\d{13})")[0]
fila = perfiles[perfiles["ruc"] == RUC].iloc[0].copy()

with open(os.path.join(RES, "categorias_modalidad.json"), encoding="utf-8") as f:
    conservadas = set(_json.load(f)["conservadas"])
cruda = _json.loads(fila["modalidades_hist_json"])
norm = Counter()
for k_, v_ in cruda.items():
    norm[cm.normalizar_modalidad(k_, conservadas)] += v_
fila["modalidades_norm"] = dict(norm)

idx_act = cm.indice_actividad(pd.read_parquet(os.path.join(RES, "actividad_buyer_cpc.parquet")))
idx_pb  = cm.indice_proveedor_buyer(pd.read_parquet(os.path.join(RES, "proveedor_buyer.parquet")))

t0 = time.time()
X, _ = cm.construir_matriz(fila, procesos, idx_act, idx_pb)

# 1) winsorizar con los LÍMITES GUARDADOS
crudas = np.array(X[cm.NUMERICAS].to_numpy(dtype=float), copy=True)
for j, c in enumerate(cm.NUMERICAS):
    lo, hi = modelo["winsor_limites"][c]
    crudas[:, j] = np.clip(crudas[:, j], lo, hi)
# 2) estandarizar con el ESCALADOR GUARDADO
Z = (crudas - np.asarray(modelo["escalador_media"])) / np.asarray(modelo["escalador_escala"])
# 3) Gower ponderada contra los medoides -> grupo más cercano
D = cm.distancia_gower_a_referencias(
    Z, X[cm.CATEGORICA].astype(str).to_numpy(),
    np.asarray(modelo["medoides_num_std"], dtype=float),
    np.asarray(modelo["medoides_modalidad"], dtype=object),
    np.asarray(modelo["gower_rangos"], dtype=float),
    pesos=np.asarray(modelo["pesos_gower"], dtype=float))
grupos_ord = sorted(modelo["medoides_ocid"].keys())
etiquetas = np.array([grupos_ord[i] for i in np.argmin(D, axis=1)])
dt = time.time() - t0

print(f"proveedor {RUC} ({fila['provincia']}), "
      f"{int(fila['num_adjudicaciones'])} adjudicaciones, "
      f"{int(fila['n_cpc_historicos'])} rubros")
print(f"{len(procesos):,} procesos clasificados en {dt:.2f} s\n")

X["grupo"] = etiquetas
X["compatibilidad"] = 0.5 * X["cpc_jaccard4"].clip(0, 1) + 0.5 * X["sim_tfidf"].clip(0, 1)
res = X.groupby("grupo").agg(
    procesos=("grupo", "size"),
    cpc_jaccard4=("cpc_jaccard4", "mean"),
    sim_tfidf=("sim_tfidf", "mean"),
    distancia_km=("distancia_km", "mean"),
    compatibilidad=("compatibilidad", "mean"))
res["pct"] = (100 * res["procesos"] / len(X)).round(1)
print(res.round(4).to_string())
""")

code(r"""
# El grupo accionable es el de mayor compatibilidad, NO el de mayor afinidad.
accionable = res["compatibilidad"].idxmax()
oport = X[(X["grupo"] == accionable) & (X["compatibilidad"] >= 0.15)]
print(f"grupo accionable: {accionable}")
print(f"oportunidades con compatibilidad >= 0.15: {len(oport):,} "
      f"de {len(X):,} procesos ({100*len(oport)/len(X):.1f} %)\n")

top = oport.nlargest(8, "compatibilidad")
muestra = pd.DataFrame({
    "objeto": procesos.loc[top.index, "texto_items"].str.slice(0, 52).to_numpy(),
    "provincia": procesos.loc[top.index, "provincia_buyer"].to_numpy(),
    "compat": top["compatibilidad"].round(3).to_numpy(),
    "cpc": top["cpc_jaccard4"].round(3).to_numpy(),
    "tfidf": top["sim_tfidf"].round(3).to_numpy(),
})
print(muestra.to_string(index=False))
""")

# ═══════════════════════════════════════════════════════════ 12. limitaciones
md(r"""
---
## 12. Defectos encontrados y limitaciones

Lo que un lector crítico debería saber antes de usar esto:

**Tres variables defectuosas, detectadas y eliminadas.** `cpc_match` constante
en 0; `modalidad_afinidad` con η² = 1,000 contra la categórica;
`desviacion_presupuesto` idéntica a `log_presupuesto` con r = 1,0000. Las tres
inflaban artificialmente el peso de la modalidad o del presupuesto.

**La estructura de grupos es débil.** La silueta es casi plana entre k = 3 y
k = 10 (0,242–0,277). No hay un número natural de grupos; k = 3 ganó por 0,017
sobre k = 7. Los datos son un continuo.

**De los tres grupos, sólo uno es accionable.** Los otros dos se separan por
distancia geográfica con coincidencia de rubro casi nula en ambos. La
granularidad útil la aporta el orden continuo dentro del grupo, no la etiqueta.

**El Isolation Forest por grupo no mide nada.** `contamination=0.05` fuerza el
5 % que devuelve. Se reportó tres veces; la columna válida es la del bosque
global.

**La afinidad al medoide no es compatibilidad.** Mide tipicidad dentro del
grupo. En estos datos el grupo con mayor afinidad media es el difuso, que no
comparte rubros: para P2, el grupo 0 tiene afinidad 0,864 y coincidencia de
rubro 0,007, mientras el grupo accionable tiene afinidad 0,711 y coincidencia
0,250. La interfaz ordena por compatibilidad explícita, no por afinidad.

**La compatibilidad no es una probabilidad de adjudicación.** Mide parecido con
el historial. No modela competencia, precio ni capacidad de cumplimiento.

**Un proveedor muy diversificado no obtiene resultado.** El caso extremo del
corte (30.721 adjudicaciones en 313 rubros) da coincidencia media de 0,001: tan
amplio que ningún grupo lo representa. La interfaz lo declara «sin
coincidencias» en vez de inventar una recomendación.

**Sustitución de biblioteca.** `scikit-learn-extra` no importa con numpy 2.x;
PAM se resolvió con `kmedoids` (FasterPAM). Ambas implementan PAM sobre una
matriz de distancias precalculada.
""")

# ═══════════════════════════════════════════════════════════ 13. reproducir
md(r"""
---
## 13. Reproducción completa

Este notebook lee resultados guardados para correr en segundos. Para
reconstruirlo todo desde los datos crudos hacen falta los tres `.gz` en `data/`
(56 MB del portal de datos abiertos del SERCOP, no versionados) y este orden:

```bash
python scripts/construir_perfiles.py    # perfiles de 12.220 proveedores   (~8 s)
python scripts/extraer_activos.py       # 7.637 procesos vigentes          (~5 s)
python scripts/comparar_modelos.py      # portafolio de 6 algoritmos       (~7,7 min)
```

Todo es determinista con `seed = 42`: dos corridas dan resultados idénticos.
La tercera etapa entrena y compara los seis algoritmos, aplica las líneas base y
las cuatro reglas, elige el ganador y escribe `modelo_ganador.pkl`.

Para la interfaz:

```bash
streamlit run app/dashboard.py
```

**Tiempos medidos** en la máquina de desarrollo (Windows 11, Python 3.11):
perfiles 8 s · activos 5 s · comparación 461 s (7,7 min) · consulta del panel
0,4–0,6 s sobre los 7.637 procesos vigentes.
""")

code(r"""
print("Comprobación final: el modelo carga y clasifica.")
print(f"  algoritmo   : {modelo['algoritmo_nombre']}")
print(f"  k           : {modelo['k']}")
print(f"  fase        : {modelo['fase']}")
print(f"  semilla     : {modelo['seed']}")
print(f"  silueta     : {modelo['silueta_promedio_portafolio']:.4f} "
      f"(promedio de P1, P2, P3)")
print(f"  medoides    : {len(modelo['medoides_ocid'])}")
print(f"  grupos asignados en el ejemplo: "
      f"{sorted(pd.unique(etiquetas).tolist())}")
print("\nOK")
""")

# ═══════════════════════════════════════════════════════════ escritura
nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[notebook] escrito: {OUT}")
print(f"[notebook] celdas: {len(C)} "
      f"({sum(1 for c in C if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in C if c.cell_type == 'code')} código)")
