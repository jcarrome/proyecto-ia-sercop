// Genera resultados/Reporte_Comparacion_Algoritmos_Grupo3.docx
// Ejecutar con:  NODE_PATH=<npm root -g> node scripts/generar_reporte_docx.js
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, PageOrientation, TableOfContents, Footer, PageNumber,
  LevelFormat, convertInchesToTwip,
} = require('docx');

const BASE = path.dirname(__dirname);
const RES = path.join(BASE, 'resultados');
const D = JSON.parse(fs.readFileSync(path.join(RES, 'reporte_datos.json'), 'utf8'));

const CONTENT_W = 9360;          // 12240 - 2*1440 (Carta, márgenes 1")
const AZUL = '1F4E79', GRIS = 'F2F2F2', VERDE = 'E2F0D9', ROJO = 'FBE4E4';

// ---------- utilidades ----------
// Separador decimal español (coma), coherente con la prosa del reporte.
const nf = (v, d = 4) =>
  (v === null || v === undefined || Number.isNaN(v)) ? ''
    : Number(v).toFixed(d).replace('.', ',');
// Miles separados por espacio y decimales por coma, igual que en la prosa.
const miles = (v, d = 0) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  const [ent, dec] = Number(v).toFixed(d).split('.');
  const e = ent.replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  return dec ? `${e},${dec}` : e;
};

function P(text, opts = {}) {
  const { bold, italics, size = 21, align, spacing, color, font } = opts;
  return new Paragraph({
    alignment: align,
    spacing: spacing || { after: 110 },
    children: [new TextRun({ text, bold, italics, size, color, font })],
  });
}

// Párrafo con tramos de formato mixto: [['texto',{bold:true}], ...]
function Pmix(tramos, opts = {}) {
  return new Paragraph({
    alignment: opts.align,
    spacing: opts.spacing || { after: 110 },
    children: tramos.map(([t, o]) => new TextRun(Object.assign({ text: t, size: 21 }, o || {}))),
  });
}

function H(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 260, after: 120 },
    children: [new TextRun({ text, color: AZUL, bold: true })],
  });
}

function celda(texto, { w, bold, fill, align, size = 18 }) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: fill ? { type: ShadingType.CLEAR, fill, color: 'auto' } : undefined,
    margins: { top: 55, bottom: 55, left: 85, right: 85 },
    children: [new Paragraph({
      alignment: align || AlignmentType.RIGHT,
      spacing: { after: 0 },
      children: [new TextRun({ text: String(texto), bold, size })],
    })],
  });
}

/** tabla: cols=[{k,label,w,fmt,align}], filas=array de objetos */
function tabla(cols, filas, resaltar, sinCabecera) {
  const widths = cols.map(c => c.w);
  const head = new TableRow({
    tableHeader: true,
    children: cols.map(c => celda(c.label, {
      w: c.w, bold: true, fill: 'DCE6F1',
      align: c.align || AlignmentType.RIGHT,
    })),
  });
  const rows = filas.map(f => {
    const fill = resaltar ? resaltar(f) : undefined;
    return new TableRow({
      children: cols.map(c => {
        const v = f[c.k];
        const txt = c.fmt ? c.fmt(v, f) : (v === null || v === undefined ? '' : String(v));
        return celda(txt, { w: c.w, fill, align: c.align || AlignmentType.RIGHT });
      }),
    });
  });
  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA' },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA' },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'DDDDDD' },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: sinCabecera ? rows : [head, ...rows],
  });
}

// dimensiones de un PNG (cabecera IHDR)
function pngSize(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function figura(archivo, pie, maxW = 620) {
  const p = path.join(RES, archivo);
  if (!fs.existsSync(p)) {
    return [P(`[FALTA la figura ${archivo}]`, { italics: true, color: 'C00000' })];
  }
  const { w, h } = pngSize(p);
  const width = Math.min(maxW, w);
  const height = Math.round(h * width / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [new ImageRun({
        type: 'png', data: fs.readFileSync(p),
        transformation: { width, height },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: pie, size: 17, italics: true, color: '555555' })],
    }),
  ];
}

function vinieta(text) {
  return new Paragraph({
    numbering: { reference: 'vinietas', level: 0 },
    spacing: { after: 70 },
    children: [new TextRun({ text, size: 21 })],
  });
}

function nota(text) {
  return new Paragraph({
    spacing: { before: 60, after: 160 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: AZUL, space: 10 } },
    indent: { left: 180 },
    children: [new TextRun({ text, size: 19, italics: true, color: '444444' })],
  });
}

// ---------- datos ----------
const inv = D.inventario;
const g = D.ganador;
const lb = D.lineas_base;
const t6 = D.tabla6;
const SLUG = {
  'K-Medoids (PAM)': 'pam', 'DBSCAN': 'dbscan', 'GMM': 'gmm',
  'Jerárquico (promedio)': 'jerarquico', 'K-Means': 'kmeans', 'K-Prototypes': 'kprototypes',
};
const ORDEN = ['K-Medoids (PAM)', 'DBSCAN', 'GMM', 'Jerárquico (promedio)', 'K-Means', 'K-Prototypes'];
const porNombre = Object.fromEntries(t6.map(r => [r.Algoritmo, r]));
const balPorNombre = Object.fromEntries(D.balance.map(r => [r.Modelo, r]));

const COMENTARIO = {
  'K-Medoids (PAM)':
    'Es el ganador. Produce la mayor silueta entre los admitidos y, a diferencia del jerárquico, ' +
    'reparte los procesos de forma equilibrada: ningún grupo baja del 2,5 % ni supera el 21,6 %. ' +
    'Sus medoides son procesos reales del conjunto, lo que hace directa la asignación de un ' +
    'proveedor nuevo y facilita explicar cada grupo señalando un expediente concreto. Nota de ' +
    'cautela: su silueta seguía subiendo en k = 10, el borde del barrido fijado en 3–10, así que ' +
    'el óptimo podría estar más allá y no se exploró.',
  'DBSCAN':
    'Obtuvo la mejor silueta de los seis (0,5807) pero queda DESCARTADO por la regla de decisión: ' +
    'dejó el 15,50 % de los procesos como ruido, por debajo del mínimo de 90 % de cobertura. ' +
    'Su ventaja en silueta es en parte consecuencia de eso mismo — al descartar los puntos ' +
    'fronterizos, los grupos que quedan se ven más limpios de lo que son. Además fragmenta: ' +
    'de sus 17 grupos, 5 reúnen menos del 1 % de los procesos cada uno y el menor tiene 20.',
  'GMM':
    'Segundo admitido, con silueta 0,4439. El k = 10 lo fija el BIC, según manda el protocolo, no ' +
    'la silueta; de haberse elegido por silueta habría sido k = 7. Reparte peor que PAM: su grupo ' +
    'mayor concentra el 36,5 % y tres grupos quedan por debajo del 1 %. Su estabilidad (ARI 0,787) ' +
    'es la segunda más baja de los admitidos.',
  'Jerárquico (promedio)':
    'ATENCIÓN: su resultado es engañoso. Con k = 3 el enlace promedio produce una partición ' +
    'degenerada — 13 732 procesos (99,2 %) en un solo grupo y dos grupos residuales de 48 y 68. ' +
    'Su ARI de 0,9997 no indica robustez sino que una partición casi constante es trivialmente ' +
    'estable, y su silueta de 0,3987 mide sobre todo lo compacto que es ese bloque único. ' +
    'La regla de decisión no incluye un filtro de balance, así que formalmente pasa los dos ' +
    'umbrales; en la práctica no segmenta nada. Es el efecto de encadenamiento clásico del ' +
    'enlace promedio sobre datos con densidad desigual.',
  'K-Means':
    'La silueta más baja junto con K-Prototypes (0,2156) y, con diferencia, el menos estable de ' +
    'todos: ARI 0,738 con una desviación de 0,222, un orden de magnitud peor que PAM o el ' +
    'jerárquico. Trabaja sobre el espacio codificado con one-hot, donde la modalidad aporta 19 ' +
    'dimensiones binarias frente a 5 numéricas, y la distancia euclidiana no es la métrica que ' +
    'después lo juzga. Dos de sus cinco grupos son residuales (105 y 49 procesos).',
  'K-Prototypes':
    'Comparte con K-Means la peor silueta (0,2022), pese a ser el algoritmo teóricamente más ' +
    'apropiado: trata la modalidad como categórica nativa en vez de expandirla en one-hot. ' +
    'Su virtud es el equilibrio — es la partición mejor repartida de las seis (entropía 0,909, ' +
    'ningún grupo por debajo de 2 268 procesos) — pero con k = 3 la segmentación es demasiado ' +
    'gruesa para ser útil. Fue además el más caro: 732 s en el barrido de k, frente a 53 s de PAM.',
};

// ---------- documento ----------
const hijos = [];

// Portada
hijos.push(
  new Paragraph({ spacing: { before: 1600, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'ESCUELA SUPERIOR POLITÉCNICA DEL LITORAL', size: 22, color: '555555' })] }),
  new Paragraph({ spacing: { after: 700 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'CCPG1044 — Inteligencia Artificial · Proyecto final · Grupo #3', size: 22, color: '555555' })] }),
  new Paragraph({ spacing: { after: 160 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Comparación de seis algoritmos de agrupamiento', bold: true, size: 44, color: AZUL })] }),
  new Paragraph({ spacing: { after: 900 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Segmentación de compatibilidad proveedor–licitación\ncon datos abiertos del SERCOP (estándar OCDS)'.replace('\n', ' · '), size: 26, color: '333333' })] }),
);
hijos.push(tabla(
  [{ k: 'a', label: '', w: 3200, align: AlignmentType.LEFT }, { k: 'b', label: '', w: 6160, align: AlignmentType.LEFT }],
  [
    { a: 'Ganador', b: `${g.algoritmo} · k = ${g.k}` },
    { a: 'Silueta sobre Gower', b: nf(g.silueta) },
    { a: 'Procesos analizados', b: `${miles(inv.matriz.filas)} licitaciones activas` },
    { a: 'Proveedor consultante', b: `${inv.proveedor_consultante.nombre}` },
    { a: 'Fuente', b: 'Archivos OCDS locales 2024–2026 (90 076 procesos). Ninguna API consumida.' },
    { a: 'Semilla', b: 'random_state = 42 en todos los ajustes' },
  ], null, true));
hijos.push(new Paragraph({ children: [new PageBreak()] }));

// Índice
hijos.push(H('Contenido', HeadingLevel.HEADING_1));
hijos.push(new TableOfContents('Contenido', { hyperlink: true, headingStyleRange: '1-2' }));
hijos.push(new Paragraph({ children: [new PageBreak()] }));

// 1. Resumen ejecutivo
hijos.push(H('1. Resumen ejecutivo', HeadingLevel.HEADING_1));
hijos.push(Pmix([
  ['Gana ', {}], [`${g.algoritmo} con k = ${g.k}`, { bold: true }],
  [`, con una silueta sobre Gower de `, {}], [nf(g.silueta), { bold: true }],
  [`, cobertura del 100 % y un ARI de estabilidad de ${nf(g.ari, 3)} ± ${nf(g.ari_desv, 3)}. `, {}],
  [`El segundo admitido, ${g.segundo}, queda en ${nf(g.silueta_segundo)}; la diferencia de ${nf(g.diferencia)} supera el umbral de 0,02, así que no hubo empate técnico y no fue necesario desempatar por interpretabilidad de centroides.`, {}],
]));
hijos.push(Pmix([
  ['Se descartó un único candidato: ', {}], ['DBSCAN', { bold: true }],
  [`, por cobertura del ${nf(porNombre['DBSCAN'].Cobertura, 2)} %, por debajo del mínimo del 90 %. Es un descarte costoso, porque era el algoritmo con mejor silueta de los seis (${nf(porNombre['DBSCAN']['Silueta (Gower)'])}). Ningún candidato incumplió el umbral de estabilidad.`, {}],
]));
hijos.push(P('Tres resultados merecen atención por encima del veredicto:', { bold: true, spacing: { before: 160, after: 100 } }));
hijos.push(vinieta(
  `Ningún candidato supera la línea base trivial. Agrupar los procesos por modalidad_contratacion y nada más da una silueta de ${nf(lb.modalidad)}, por encima del ganador (${nf(g.silueta)}) y de los seis algoritmos. Todos superan holgadamente la línea base aleatoria (${nf(lb.aleatoria)}), pero ése es el listón bajo. Con estas seis variables y este árbitro, la modalidad sola explica la estructura mejor que cualquier algoritmo del portafolio.`));
hijos.push(vinieta(
  'El agrupamiento jerárquico produce una partición degenerada. Con k = 3 concentra el 99,2 % de los procesos en un solo grupo. Su ARI de 0,9997 no mide robustez: una partición casi constante es trivialmente estable. Formalmente pasa los dos filtros porque la regla de decisión no incluye un criterio de balance, pero no segmenta nada.'));
hijos.push(vinieta(
  'Davies-Bouldin y Calinski-Harabasz nombran ganadores distintos — K-Means y K-Prototypes respectivamente. Es esperable, porque se calculan sobre la matriz codificada con distancia euclidiana y no sobre Gower, y por eso el protocolo los fija como criterio secundario. Pero la discrepancia es grande y conviene no ocultarla.'));
hijos.push(nota(
  'Todas las cifras de este reporte provienen de la ejecución real del notebook sobre los archivos en disco. No hay valores estimados, interpolados ni rellenados a mano. Las etiquetas de los seis modelos se recalcularon de forma independiente para las figuras por modelo y reprodujeron exactamente las siluetas de la Tabla 6.'));

// 2. Datos
hijos.push(H('2. Datos y verificación previa', HeadingLevel.HEADING_1));
hijos.push(P(`El análisis parte de tres archivos OCDS comprimidos en disco (2024, 2025 y el año en curso), con ${miles(inv.lineas_leidas)} procesos y ${miles(inv.ocid_unicos)} identificadores únicos: no hubo un solo duplicado que colapsar entre archivos. No se consumió ninguna API.`));
hijos.push(tabla(
  [{ k: 'a', label: 'Verificación', w: 4600, align: AlignmentType.LEFT },
   { k: 'b', label: 'Resultado', w: 4760, align: AlignmentType.LEFT }],
  [
    { a: 'Procesos con tender.status = "active"', b: `${miles(inv.activos_tender_status)} (umbral del protocolo: 1 000 → se continúa)` },
    { a: 'Procesos "complete" (historial del proveedor)', b: miles(inv.completos) },
    { a: 'Dimensiones de la matriz', b: `${miles(inv.matriz.filas)} filas × ${inv.matriz.columnas_semanticas} columnas` },
    { a: 'Matriz codificada (one-hot)', b: `${miles(inv.matriz.filas)} × ${inv.matriz.columnas_codificadas_onehot} (${inv.matriz.modalidades_distintas} modalidades)` },
    { a: 'Matriz de Gower (n²)', b: `${miles(inv.gower_n2.celdas)} celdas — ${inv.gower_n2.float64_GB} GB en float64, ${inv.gower_n2.float32_GB} GB en float32` },
  ]));
hijos.push(P(`De los ${miles(inv.activos_tender_status)} procesos activos quedaron fuera 812 (5,5 %) por datos faltantes: 508 sin ítems en ninguna de las dos rutas posibles y 304 sin provincia del comprador, imprescindible para la distancia geográfica. No se imputó ningún valor.`, { spacing: { before: 140, after: 110 } }));
hijos.push(nota(
  `El historial del proveedor consultante se construyó únicamente con procesos en estado "complete", para no filtrar información desde las mismas filas que después se agrupan. El proveedor es ${inv.proveedor_consultante.nombre}, con ${inv.proveedor_consultante.procesos_historicos} procesos adjudicados, ${inv.proveedor_consultante.cpc_distintos} códigos CPC distintos y sede en ${inv.proveedor_consultante.provincia}.`));
hijos.push(P('Un hallazgo del corte que conviene registrar: la categoría «Catálogo Electrónico» no aparece entre los procesos activos — los convenios de catálogo están todos en estado "complete". La regla de colapsar los convenios en una sola categoría quedó aplicada, pero resulta inocua sobre esta matriz.'));

// 3. Protocolo
hijos.push(H('3. Protocolo de comparación', HeadingLevel.HEADING_1));
hijos.push(P('Los seis algoritmos reciben exactamente la misma matriz y el mismo pre-procesamiento: log(1+x) sobre los montos —incorporado en la desviación de presupuesto, que es log1p del monto del proceso menos log1p de la mediana histórica del proveedor— y estandarización z-score de las cinco numéricas. La modalidad es la única categórica.'));
hijos.push(Pmix([
  ['El árbitro es único: ', {}],
  ['se calcula una sola matriz de distancias de Gower y se comparte. La silueta se evalúa siempre sobre ella con metric=\'precomputed\', aunque cada algoritmo se ajuste en el espacio que le corresponde.', { bold: true }],
  [' Sin ese árbitro común las siluetas no serían comparables entre un método que trabaja sobre one-hot y otro que trabaja sobre la matriz de distancias.', {}],
]));
hijos.push(P('Davies-Bouldin y Calinski-Harabasz se calculan sobre la matriz codificada porque sus implementaciones asumen distancia euclidiana y no aceptan una matriz precalculada. Quedan explícitamente etiquetados como criterio secundario.'));
hijos.push(tabla(
  [{ k: 'a', label: 'Candidato', w: 2500, align: AlignmentType.LEFT },
   { k: 'b', label: 'Configuración', w: 6860, align: AlignmentType.LEFT }],
  [
    { a: 'K-Prototypes', b: 'kmodes; gamma automático; init=\'Cao\'; n_init=10; barrido k = 3–10' },
    { a: 'K-Means', b: 'one-hot de la modalidad; k-means++; n_init=10; barrido k = 3–10' },
    { a: 'Jerárquico', b: 'sobre Gower precalculada; enlace promedio; barrido k = 3–10' },
    { a: 'K-Medoids / PAM', b: 'sobre Gower precalculada; medoides = procesos reales; barrido k = 3–10' },
    { a: 'DBSCAN', b: 'sobre Gower; eps del gráfico de k-distancias; el nº de grupos es SALIDA' },
    { a: 'GMM', b: 'covarianza completa; selección de k por BIC' },
  ]));

// 4. Resultados
hijos.push(new Paragraph({ children: [new PageBreak()] }));
hijos.push(H('4. Resultados', HeadingLevel.HEADING_1));
hijos.push(H('4.1 Tabla 6 — Comparación de los seis candidatos', HeadingLevel.HEADING_2));
hijos.push(tabla(
  [
    { k: 'Algoritmo', label: 'Algoritmo', w: 2000, align: AlignmentType.LEFT },
    { k: 'k usado/obtenido', label: 'k', w: 520 },
    { k: 'Silueta (Gower)', label: 'Silueta (Gower)', w: 1240, fmt: v => nf(v) },
    { k: 'Davies-Bouldin', label: 'Davies-Bouldin', w: 1240, fmt: v => nf(v, 3) },
    { k: 'Calinski-Harabasz', label: 'Calinski-H.', w: 1180, fmt: v => miles(v) },
    { k: 'Cobertura', label: 'Cobertura', w: 1100, fmt: v => nf(v, 2) + ' %' },
    { k: 'ARI', label: 'ARI', w: 1080, fmt: v => nf(v, 3) },
    { k: 'ARI desv.', label: '± desv.', w: 1000, fmt: v => nf(v, 3) },
  ],
  ORDEN.map(a => porNombre[a]),
  f => f.Algoritmo === g.algoritmo ? VERDE : (f.Cobertura < 90 ? ROJO : undefined)));
hijos.push(nota('Fila verde: ganador. Fila roja: descartado por la regla de decisión. Davies-Bouldin y Calinski-Harabasz están calculados sobre la matriz codificada (euclidiana), no sobre Gower — criterio secundario.'));

hijos.push(H('4.2 Tabla 7 — Líneas base de control', HeadingLevel.HEADING_2));
hijos.push(tabla(
  [
    { k: 'Algoritmo', label: 'Referencia', w: 2600, align: AlignmentType.LEFT },
    { k: 'k usado/obtenido', label: 'k', w: 560 },
    { k: 'Silueta (Gower)', label: 'Silueta (Gower)', w: 1400, fmt: v => nf(v) },
    { k: 'Davies-Bouldin', label: 'Davies-Bouldin', w: 1400, fmt: v => nf(v, 3) },
    { k: 'Calinski-Harabasz', label: 'Calinski-H.', w: 1300, fmt: v => miles(v, 1) },
    { k: 'Cobertura', label: 'Cobertura', w: 1100, fmt: v => nf(v, 2) + ' %' },
    { k: 'ARI', label: 'ARI', w: 1000, fmt: v => nf(v, 3) },
  ],
  D.tabla7));
hijos.push(Pmix([
  ['La línea base por modalidad alcanza una silueta de ', {}], [nf(lb.modalidad), { bold: true }],
  [`, superior a la del ganador (${nf(g.silueta)}) y a la de los seis algoritmos. `, {}],
  ['Ningún candidato supera esta referencia trivial.', { bold: true }],
  [' Hay dos razones que lo explican sin excusarlo: la modalidad pesa un sexto en la distancia de Gower, y agrupar por ella anula ese componente dentro de cada grupo; y las modalidades ya vienen correlacionadas con el monto y el tipo de ítem, de modo que arrastran parte de la señal de las otras variables.', {}],
]));
hijos.push(P('Su ARI de 1,000 es trivial: la partición por modalidad es determinista y no depende de la submuestra, así que no indica calidad alguna.', { spacing: { after: 140 } }));

// 5. Regla de decisión
hijos.push(H('5. Aplicación de la regla de decisión', HeadingLevel.HEADING_1));
hijos.push(P('La regla se fijó antes de ver los resultados: se descarta todo candidato con cobertura inferior al 90 % o ARI de estabilidad inferior a 0,60; entre los que pasan gana la mayor silueta sobre Gower; una diferencia menor a 0,02 entre los dos mejores constituye empate técnico.'));
D.descartados.forEach(d => hijos.push(Pmix([
  ['DESCARTADO — ', { bold: true, color: 'C00000' }],
  [`${d.algoritmo}: ${d.motivos.join('; ')}.`, {}],
])));
hijos.push(P('Admitidos, ordenados por silueta sobre Gower:', { spacing: { before: 140, after: 90 } }));
D.admitidos.forEach((a, i) => hijos.push(vinieta(
  `${a.algoritmo} — silueta ${nf(a.silueta)}, cobertura ${nf(a.cobertura, 2)} %, ARI ${nf(a.ari, 3)}${i === 0 ? '  ← GANADOR' : ''}`)));
hijos.push(Pmix([
  [`La diferencia entre los dos mejores admitidos es ${nf(g.diferencia)}, por encima del umbral de 0,02: `, {}],
  ['no hay empate técnico', { bold: true }],
  [' y el veredicto se resuelve directamente por silueta, sin recurrir a la interpretabilidad de los centroides.', {}],
], { spacing: { before: 140, after: 140 } }));

// 6. Diagnóstico por modelo (sin salto forzado: dejaba una página casi vacía)
hijos.push(H('6. Diagnóstico por modelo', HeadingLevel.HEADING_1));
hijos.push(P('Cada modelo se presenta con tres paneles: su proyección sobre la matriz de Gower, la distribución de la silueta dentro de cada grupo, y el tamaño de los grupos. La proyección MDS es la misma incrustación en los seis, calculada una sola vez sobre una submuestra estratificada de 2 000 procesos, de modo que las seis imágenes son directamente comparables entre sí: lo único que cambia de una a otra es el color, es decir, la partición.'));

ORDEN.forEach(a => {
  const r = porNombre[a], b = balPorNombre[a];
  hijos.push(H(`6.${ORDEN.indexOf(a) + 1} ${a}`, HeadingLevel.HEADING_2));
  hijos.push(tabla(
    [
      { k: 'a', label: 'k', w: 700 },
      { k: 'b', label: 'Silueta (Gower)', w: 1600 },
      { k: 'c', label: 'Cobertura', w: 1400 },
      { k: 'd', label: 'ARI ± desv.', w: 1800 },
      { k: 'e', label: 'Grupo mayor', w: 1500 },
      { k: 'f', label: 'Grupo menor', w: 1300 },
      { k: 'h', label: 'Entropía', w: 1060 },
    ],
    [{
      a: r['k usado/obtenido'], b: nf(r['Silueta (Gower)']),
      c: nf(r.Cobertura, 2) + ' %', d: `${nf(r.ARI, 3)} ± ${nf(r['ARI desv.'], 3)}`,
      e: b['Mayor (%)'] + ' %', f: miles(b['Menor (n)']), h: nf(b['Entropía norm.'], 3),
    }]));
  hijos.push(P(COMENTARIO[a], { spacing: { before: 140, after: 120 } }));
  figura(`fig_modelo_${SLUG[a]}.png`,
    `${a} — proyección MDS sobre Gower, silueta por grupo y tamaño de los grupos.`)
    .forEach(x => hijos.push(x));
  hijos.push(new Paragraph({ children: [new PageBreak()] }));
});

// 7. Panel comparativo + balance
hijos.push(H('7. Los seis modelos en el mismo espacio', HeadingLevel.HEADING_1));
figura('fig_panel_comparativo.png',
  'Los seis modelos sobre la misma proyección MDS de la matriz de Gower. Sólo cambia el color: la geometría de los puntos es idéntica en los seis paneles.')
  .forEach(x => hijos.push(x));
hijos.push(P('La estructura en bandas que se aprecia en los seis paneles corresponde en buena medida a la modalidad de contratación, que es la variable categórica y aporta un sexto de la distancia de Gower. Esto es coherente con el hecho de que la línea base por modalidad obtenga la mejor silueta de todo el estudio.'));

hijos.push(H('7.1 Balance de los grupos', HeadingLevel.HEADING_2));
hijos.push(P('El protocolo no incluye un criterio de balance, pero sin él la Tabla 6 se puede leer mal. La entropía normalizada vale 1 cuando todos los grupos tienen el mismo tamaño y tiende a 0 cuando uno concentra casi todo.'));
hijos.push(tabla(
  [
    { k: 'Modelo', label: 'Modelo', w: 2400, align: AlignmentType.LEFT },
    { k: 'Grupos', label: 'Grupos', w: 800 },
    { k: 'Mayor (%)', label: 'Mayor (%)', w: 1100, fmt: v => nf(v, 1) },
    { k: 'Dos mayores (%)', label: '2 mayores (%)', w: 1500, fmt: v => nf(v, 1) },
    { k: 'Menor (n)', label: 'Menor (n)', w: 1100, fmt: v => miles(v) },
    { k: 'Grupos <1%', label: 'Grupos <1 %', w: 1300 },
    { k: 'Entropía norm.', label: 'Entropía', w: 1160, fmt: v => nf(v, 3) },
  ],
  D.balance,
  f => f.Modelo === g.algoritmo ? VERDE : (f['Entropía norm.'] < 0.3 ? ROJO : undefined)));
hijos.push(Pmix([
  ['El jerárquico queda retratado aquí: entropía 0,049, con ', {}],
  [`${miles(D.tamanos_jerarquico['2'])} de ${miles(inv.matriz.filas)} procesos en un único grupo`, { bold: true }],
  [` y dos grupos residuales de ${D.tamanos_jerarquico['0']} y ${D.tamanos_jerarquico['1']}. `, {}],
  ['El ganador, en cambio, es el más equilibrado de los que producen diez grupos: ningún grupo por debajo de 351 procesos y ninguno por encima del 21,6 %.', {}],
], { spacing: { before: 140, after: 140 } }));

// 8. Perfiles del ganador
hijos.push(new Paragraph({ children: [new PageBreak()] }));
hijos.push(H('8. Perfiles de los grupos del ganador', HeadingLevel.HEADING_1));
hijos.push(H('8.1 Tabla 8 — Perfiles', HeadingLevel.HEADING_2));
hijos.push(tabla(
  [
    { k: 'Grupo', label: 'Grupo', w: 620 },
    { k: 'Procesos', label: 'Procesos', w: 900, fmt: v => miles(v) },
    { k: 'Similitud semántica', label: 'Simil. semánt.', w: 1180, fmt: v => nf(v) },
    { k: 'Coincid. CPC', label: 'Coincid. CPC', w: 1120, fmt: v => nf(v) },
    { k: 'Desv. presupuesto', label: 'Desv. presup.', w: 1180, fmt: v => (v >= 0 ? '+' : '') + nf(v, 3) },
    { k: 'Distancia (km)', label: 'Dist. (km)', w: 980, fmt: v => nf(v, 1) },
    { k: 'Modalidad dominante', label: 'Modalidad dominante', w: 3380, align: AlignmentType.LEFT },
  ],
  D.tabla8,
  f => Number(f['Coincid. CPC']) > 0.5 ? VERDE : undefined));
hijos.push(nota('La columna «Lectura de negocio» se deja deliberadamente vacía en el CSV entregado, para que la complete el equipo.'));
hijos.push(Pmix([
  ['Sólo el grupo 6 muestra afinidad real con el proveedor consultante', { bold: true }],
  [': coincidencia CPC de 0,996 y similitud semántica de 0,0996, ambas un orden de magnitud por encima del resto, sobre 2 257 procesos de Subasta Inversa Electrónica. Los otros nueve grupos separan bien geometría y presupuesto, pero describen licitaciones con las que el proveedor no tiene relación temática. Para uso operativo, la segmentación reduce el universo de 13 848 licitaciones activas a un grupo de 2 257 que merece revisión.', {}],
]));

hijos.push(H('8.2 Tabla 9 — Atípicos por grupo (Isolation Forest)', HeadingLevel.HEADING_2));
hijos.push(P('Un Isolation Forest por grupo, con contamination = 0,05 sobre las cinco variables numéricas. La última columna indica qué variable se aleja más, medida en desviaciones típicas del propio grupo.'));
hijos.push(tabla(
  [
    { k: 'Grupo', label: 'Grupo', w: 800 },
    { k: 'Procesos', label: 'Procesos', w: 1300, fmt: v => miles(v) },
    { k: 'Atípicos', label: 'Atípicos', w: 1200 },
    { k: '%', label: '%', w: 900, fmt: v => nf(v, 2) },
    { k: 'Variable que más se aleja', label: 'Variable que más se aleja', w: 5160, align: AlignmentType.LEFT },
  ],
  D.tabla9));
hijos.push(P('El porcentaje se mantiene en torno al 5 % en los diez grupos, como impone el parámetro. Lo informativo es la última columna: en seis de los diez grupos la variable que más separa a los atípicos es la actividad del comprador en los CPC del proveedor, y en los otros cuatro la coincidencia de CPC.'));

// 9. Asignabilidad
hijos.push(H('9. Asignabilidad de un proveedor nuevo', HeadingLevel.HEADING_1));
hijos.push(tabla(
  [
    { k: 'Algoritmo', label: 'Algoritmo', w: 2200, align: AlignmentType.LEFT },
    { k: 'Asignación de un proveedor nuevo', label: 'Procedimiento', w: 7160, align: AlignmentType.LEFT },
  ],
  D.asignabilidad));

// 10. Figuras globales
hijos.push(new Paragraph({ children: [new PageBreak()] }));
hijos.push(H('10. Figuras del barrido y del ganador', HeadingLevel.HEADING_1));
figura('fig_codo_por_algoritmo.png',
  'Izquierda: dispersión intra-grupo W(k) sobre Gower, medida comparable entre algoritmos. Derecha: silueta sobre Gower frente a k, con las dos líneas base como referencia horizontal.')
  .forEach(x => hijos.push(x));
hijos.push(P('En el panel derecho se aprecia que ninguna curva alcanza la línea base por modalidad, y que la curva de PAM sigue ascendiendo en k = 10, el borde del barrido fijado por el protocolo.'));
figura('fig_kdistancias_dbscan.png',
  'Gráfico de k-distancias con min_samples = 10: distancia de Gower al décimo vecino más cercano, ordenada. De aquí se obtuvo el rango de eps barrido para DBSCAN.')
  .forEach(x => hijos.push(x));
figura('fig_silueta_por_grupo_ganador.png',
  `Silueta por grupo del ganador (${g.algoritmo}, k = ${g.k}) sobre el conjunto completo.`)
  .forEach(x => hijos.push(x));
figura('fig_mds_grupos_ganador.png',
  'Proyección MDS sobre Gower de los grupos del ganador, submuestra estratificada.')
  .forEach(x => hijos.push(x));

// 11. Limitaciones
hijos.push(new Paragraph({ children: [new PageBreak()] }));
hijos.push(H('11. Limitaciones y desviaciones declaradas', HeadingLevel.HEADING_1));
[
  'scikit-learn-extra no se puede importar en este entorno: falla con «ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject», por estar compilado contra numpy 1.x mientras el entorno tiene numpy 2.4.6. En consecuencia, PAM está implementado sobre la matriz de Gower con inicialización BUILD seguida de refinamiento alternante. No es el SWAP exhaustivo de PAM clásico, que es O(k(n−k)²) por iteración e intratable con n = 13 848. Los medoides siguen siendo procesos reales del conjunto, como exige el protocolo.',
  'La matriz de Gower se almacena en float32 (0,71 GB) en lugar de float64 (1,43 GB) por la memoria física disponible en la máquina. Se convierte a float64 sólo donde SciPy lo exige, en el cálculo del enlace jerárquico.',
  'La proyección MDS se calcula sobre una submuestra estratificada de 2 000 procesos. SMACOF es O(n²) por iteración y no resulta viable con n = 13 848. Es únicamente una figura: ninguna métrica de la Tabla 6 depende de ella.',
  'Davies-Bouldin y Calinski-Harabasz se calculan sobre la matriz codificada con distancia euclidiana, no sobre Gower, porque sus implementaciones no aceptan una matriz precalculada. Se reportan como criterio secundario.',
  '812 procesos activos (5,5 % del total) quedaron fuera de la matriz por datos faltantes: 508 sin ítems y 304 sin provincia del comprador. No se imputó ningún valor.',
  'La categoría «Catálogo Electrónico» no aparece entre los procesos activos, de modo que la regla de colapsar los convenios de catálogo se aplicó pero resulta inocua sobre esta matriz.',
  'La elección del proveedor consultante no venía fijada por el enunciado. Se seleccionó tras un diagnóstico comparativo de dieciséis candidatos, documentado en scripts/diag_proveedor.py, priorizando aquel cuyas variables de interacción resultaran más discriminantes sobre el universo activo.',
  'El barrido de k está fijado en 3–10 por el protocolo. La silueta de PAM aún crecía en k = 10, por lo que su óptimo podría situarse fuera del rango explorado; no se amplió para no alterar el protocolo.',
].forEach(t => hijos.push(vinieta(t)));

// 12. Reproducibilidad
hijos.push(H('12. Reproducibilidad', HeadingLevel.HEADING_1));
hijos.push(P('Semilla random_state = 42 en todos los ajustes. Entorno virtual con Python 3.11.15.'));
hijos.push(tabla(
  [{ k: 'a', label: 'Componente', w: 3200, align: AlignmentType.LEFT },
   { k: 'b', label: 'Versión', w: 6160, align: AlignmentType.LEFT }],
  [
    { a: 'Python', b: '3.11.15' }, { a: 'numpy', b: '2.4.6' }, { a: 'pandas', b: '3.0.5' },
    { a: 'scipy', b: '1.17.1' }, { a: 'scikit-learn', b: '1.9.0' }, { a: 'kmodes', b: '0.12.2' },
    { a: 'matplotlib', b: '3.11.1' },
    { a: 'scikit-learn-extra', b: 'instalado pero NO importable (ver sección 11)' },
  ]));
hijos.push(H('12.1 Tiempos de cómputo', HeadingLevel.HEADING_2));
const tks = Object.keys(D.tiempos).filter(k => k !== 'TOTAL medido');
hijos.push(tabla(
  [{ k: 'a', label: 'Etapa', w: 6160, align: AlignmentType.LEFT },
   { k: 'b', label: 'Segundos', w: 3200 }],
  tks.map(k => ({ a: k, b: nf(D.tiempos[k], 2) }))
    .concat([{ a: 'TOTAL medido', b: nf(D.tiempos['TOTAL medido'], 2) }])));
hijos.push(P('El barrido de K-Prototypes (732 s) y la fase de estabilidad (1 137 s) concentran el 87 % del tiempo total. La matriz de Gower, que es la estructura más pesada en memoria, se construye en menos de 4 segundos.', { spacing: { before: 120 } }));

hijos.push(H('12.2 Archivos generados', HeadingLevel.HEADING_2));
[
  'notebooks/comparacion_portafolio.ipynb — ejecución completa con sus salidas',
  'resultados/tabla6_comparacion.csv, tabla7_lineas_base.csv, tabla8_perfiles.csv, tabla9_atipicos.csv',
  'resultados/asignabilidad.csv, log_ejecucion.txt, matriz_procesos.csv',
  'resultados/fig_modelo_*.png (uno por algoritmo) y fig_panel_comparativo.png',
  'resultados/fig_codo_por_algoritmo.png, fig_silueta_por_grupo_ganador.png, fig_mds_grupos_ganador.png, fig_kdistancias_dbscan.png',
  'resultados/comparacion_modelos.html — versión interactiva de esta comparación',
].forEach(t => hijos.push(vinieta(t)));

// ---------- ensamblar ----------
const doc = new Document({
  creator: 'Grupo #3 — CCPG1044 ESPOL',
  title: 'Comparación de seis algoritmos de agrupamiento — SERCOP/OCDS',
  description: 'Proyecto final de Inteligencia Artificial',
  features: { updateFields: true },
  numbering: {
    config: [{
      reference: 'vinietas',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '•',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 460, hanging: 260 } } },
      }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: 'Calibri', size: 21 }, paragraph: { spacing: { line: 276 } } },
      heading1: { run: { font: 'Calibri', size: 30, bold: true, color: AZUL } },
      heading2: { run: { font: 'Calibri', size: 25, bold: true, color: AZUL } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({
            children: ['CCPG1044 · Grupo #3 · ', PageNumber.CURRENT, ' / ', PageNumber.TOTAL_PAGES],
            size: 17, color: '888888',
          })],
        })],
      }),
    },
    children: hijos,
  }],
});

const OUT = path.join(RES, 'Reporte_Comparacion_Algoritmos_Grupo3.docx');
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log('escrito: ' + OUT);
  console.log('tamaño: ' + (buf.length / 1024).toFixed(0) + ' KB');
});
