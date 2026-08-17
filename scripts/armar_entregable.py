# -*- coding: utf-8 -*-
"""
Arma el ZIP del entregable con los cinco puntos pedidos.

    python scripts/armar_entregable.py [Apellido_Nombre]

Produce entregable/<nombre>.zip con:
    a) el código fuente — notebooks/modelo_final.ipynb + scripts/
    b) un conjunto de ejemplos para correr el modelo — ejemplos/
    c) la interfaz desarrollada — app/ + .streamlit/ + capturas/
    d) las librerías no públicas utilizadas — librerias/ (ninguna; se explica)
    e) el borrador del póster — poster/

Copia además resultados/ con el modelo entrenado, para que el paquete corra sin
reentrenar y sin los 56 MB de datos crudos.

La estructura conserva las rutas del proyecto, así que el ZIP extraído es
ejecutable tal cual: `python ejemplos/correr_ejemplos.py` y
`streamlit run app/dashboard.py` funcionan sin tocar nada.

No modifica el póster ni ningún archivo del proyecto: sólo copia.
"""
import os
import shutil
import sys
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESCARGAS = os.path.join(os.path.expanduser("~"), "Downloads")
NOMBRE = sys.argv[1] if len(sys.argv) > 1 else "Romero_Uchubanda_Juan_Carlos"

TRABAJO = os.path.join(BASE, "entregable", "_armado")
DESTINO = os.path.join(TRABAJO, NOMBRE)
ZIP = os.path.join(BASE, "entregable", NOMBRE + ".zip")

# archivos de resultados/ que el paquete necesita para funcionar
NECESARIOS = ("modelo_ganador.pkl", "perfiles_proveedores.parquet",
              "procesos_activos.parquet", "actividad_buyer_cpc.parquet",
              "proveedor_buyer.parquet", "categorias_modalidad.json")
# tablas y figuras de evidencia que el notebook lee y el informe cita
EVIDENCIA = ("comparacion_modelos.csv", "candidatos_evaluados.csv",
             "lineas_base.csv", "eta2_variables.csv",
             "perfil_centroides_ganador.csv", "atipicos_resumen.csv",
             "continuidad_perfil_grupos.csv", "comparacion_detalle_k.csv",
             "log_fase2_2.txt", "etiquetas_ganador.npz",
             "comparacion_barras.png", "curvas_P1.png", "curvas_P2.png",
             "curvas_P3.png")
POSTER = ("Poster_Grupo3_P2.pptx", "Poster_Grupo3_P2.pdf")
IGNORAR_DIRS = ("__pycache__", ".ipynb_checkpoints")

copiados = []
avisos = []


def copiar_archivo(origen, destino_rel):
    if not os.path.exists(origen):
        avisos.append(f"no se encontró {origen}")
        return False
    dst = os.path.join(DESTINO, destino_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(origen, dst)
    copiados.append((destino_rel.replace("\\", "/"), os.path.getsize(origen)))
    return True


def copiar_carpeta(origen, destino_rel, patrones=None):
    if not os.path.isdir(origen):
        avisos.append(f"no existe la carpeta {origen}")
        return
    for raiz, dirs, files in os.walk(origen):
        dirs[:] = [d for d in dirs if d not in IGNORAR_DIRS]
        for f in sorted(files):
            if patrones and not f.endswith(patrones):
                continue
            src = os.path.join(raiz, f)
            copiar_archivo(src, os.path.join(
                destino_rel, os.path.relpath(src, origen)))


def main():
    if os.path.exists(TRABAJO):
        shutil.rmtree(TRABAJO)
    os.makedirs(DESTINO)

    print(f"[entregable] armando «{NOMBRE}»")

    # (a) código fuente
    copiar_archivo(os.path.join(BASE, "notebooks", "modelo_final.ipynb"),
                   os.path.join("notebooks", "modelo_final.ipynb"))
    copiar_carpeta(os.path.join(BASE, "scripts"), "scripts", patrones=(".py",))

    # (b) ejemplos
    copiar_carpeta(os.path.join(BASE, "ejemplos"), "ejemplos")

    # (c) interfaz
    copiar_archivo(os.path.join(BASE, "app", "dashboard.py"),
                   os.path.join("app", "dashboard.py"))
    copiar_archivo(os.path.join(BASE, ".streamlit", "config.toml"),
                   os.path.join(".streamlit", "config.toml"))
    copiar_carpeta(os.path.join(BASE, "resultados", "capturas"), "capturas",
                   patrones=(".png",))

    # (d) librerías
    copiar_carpeta(os.path.join(BASE, "librerias"), "librerias")

    # (e) póster — se busca en Descargas
    for f in POSTER:
        copiar_archivo(os.path.join(DESCARGAS, f), os.path.join("poster", f))

    # modelo entrenado y evidencia
    for f in NECESARIOS + EVIDENCIA:
        copiar_archivo(os.path.join(BASE, "resultados", f),
                       os.path.join("resultados", f))

    # índice y dependencias
    copiar_archivo(os.path.join(BASE, "requirements.txt"), "requirements.txt")
    copiar_archivo(os.path.join(BASE, "entregable", "LEEME.md"), "LEEME.md")

    # comprobación de los cinco puntos antes de comprimir
    print("\n[entregable] comprobando los cinco puntos:")
    exigidos = {
        "a) notebook": "notebooks/modelo_final.ipynb",
        "a) tubería": "scripts/comparar_modelos.py",
        "b) ejemplos": "ejemplos/correr_ejemplos.py",
        "b) casos": "ejemplos/ejemplos_rucs.csv",
        "c) interfaz": "app/dashboard.py",
        "d) librerías": "librerias/LEEME.txt",
        "e) póster": "poster/Poster_Grupo3_P2.pptx",
        "modelo": "resultados/modelo_ganador.pkl",
        "índice": "LEEME.md",
    }
    presentes = {r for r, _ in copiados}
    faltan = []
    for etiqueta, ruta in exigidos.items():
        ok = ruta in presentes
        print(f"  {'[ok]' if ok else '[NO]'} {etiqueta:16} {ruta}")
        if not ok:
            faltan.append(ruta)
    if faltan:
        print(f"\n[ERROR] faltan {len(faltan)} elementos exigidos; no se "
              f"genera el ZIP.", file=sys.stderr)
        for r in faltan:
            print(f"  - {r}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(ZIP), exist_ok=True)
    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for raiz, dirs, files in os.walk(DESTINO):
            dirs[:] = [d for d in dirs if d not in IGNORAR_DIRS]
            for f in sorted(files):
                src = os.path.join(raiz, f)
                z.write(src, os.path.join(
                    NOMBRE, os.path.relpath(src, DESTINO)))

    # testzip() debe correr sobre el archivo YA CERRADO: llamarlo dentro del
    # bloque de escritura da un falso positivo en la primera entrada.
    with zipfile.ZipFile(ZIP) as z:
        malo = z.testzip()
        n_en_zip = len(z.namelist())

    shutil.rmtree(TRABAJO, ignore_errors=True)

    sin_comprimir = sum(s for _, s in copiados)
    print(f"\n[entregable] {ZIP}")
    print(f"[entregable] {n_en_zip} archivos | "
          f"{sin_comprimir/1024/1024:.1f} MB sin comprimir -> "
          f"{os.path.getsize(ZIP)/1024/1024:.2f} MB comprimido")
    print(f"[entregable] integridad: "
          f"{'OK' if malo is None else 'CORRUPTO en ' + malo}")
    if n_en_zip != len(copiados):
        print(f"[entregable] AVISO: se copiaron {len(copiados)} archivos pero "
              f"el ZIP tiene {n_en_zip}")
    if avisos:
        print("\n[entregable] avisos:")
        for a in avisos:
            print(f"  - {a}")
    print("\nPara comprobar el paquete: extráigalo y desde su raíz corra")
    print("    python ejemplos/correr_ejemplos.py")
    print("    streamlit run app/dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
