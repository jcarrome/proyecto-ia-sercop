# -*- coding: utf-8 -*-
"""
FASE 3 — Capturas del dashboard para el reporte final.

Levanta la app de Streamlit en un puerto libre, la recorre con Playwright y
guarda las tres pantallas en resultados/capturas/ a 1400 px de ancho.

    python scripts/generar_capturas.py

No muestra razones sociales: usa los proveedores de ejemplo (P1/P2/P3), que se
identifican sólo por RUC y provincia. Si la automatización falla, imprime las
instrucciones para tomar las capturas a mano.
"""
import os
import socket
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(BASE, "app", "dashboard.py")
SALIDA = os.path.join(BASE, "resultados", "capturas")
ANCHO = 1400
ALTO = 1000
RUC_SIN_HISTORIAL = "9999999999999"
# proveedor real muy diversificado: el modelo NO encuentra coincidencias y lo
# dice; es la captura que documenta el caso de fallo honesto
RUC_DIVERSIFICADO = "1790732657001"


def puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def esperar_servidor(url, timeout=90):
    import urllib.error
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def instrucciones_manuales(motivo):
    print("\n[capturas] NO se pudieron generar automáticamente.")
    print(f"[capturas] Motivo: {motivo}")
    print("\nPASOS PARA TOMARLAS A MANO:")
    print("  1. En una terminal:  streamlit run app/dashboard.py")
    print("  2. Abra http://localhost:8501 y ponga la ventana en 1400 px de ancho")
    print("     (F12 -> icono de dispositivo -> tamaño personalizado 1400x1000).")
    print(f"  3. pantalla_inicio.png              -> la pantalla inicial, sin consultar nada.")
    print(f"  4. pantalla_resultado.png           -> pulse «Usar» en el primer")
    print(f"                                         proveedor de ejemplo.")
    print(f"  5. pantalla_sin_coincidencias.png   -> «Nueva consulta», escriba")
    print(f"                                         {RUC_DIVERSIFICADO}.")
    print(f"  6. pantalla_exploracion.png         -> «Nueva consulta», escriba")
    print(f"                                         {RUC_SIN_HISTORIAL}.")
    print(f"  7. pantalla_detalle_tecnico.png     -> despliegue «Detalle técnico»")
    print(f"                                         y capture ese bloque.")
    print(f"  8. Guarde los archivos en: {SALIDA}")


def main():
    os.makedirs(SALIDA, exist_ok=True)
    if not os.path.exists(APP):
        instrucciones_manuales(f"no existe {APP}")
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        instrucciones_manuales(
            "playwright no está instalado (pip install playwright && "
            "python -m playwright install chromium)")
        return 1

    puerto = puerto_libre()
    url = f"http://localhost:{puerto}"
    print(f"[capturas] levantando Streamlit en {url} ...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", APP,
         "--server.port", str(puerto), "--server.headless", "true",
         "--browser.gatherUsageStats", "false",
         "--client.toolbarMode", "minimal",
         "--server.fileWatcherType", "none"],
        cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    generadas = []
    try:
        if not esperar_servidor(url):
            proc.terminate()
            instrucciones_manuales("Streamlit no respondió en 90 s")
            return 1
        print("[capturas] servidor arriba", flush=True)

        with sync_playwright() as p:
            navegador = p.chromium.launch()
            pagina = navegador.new_page(viewport={"width": ANCHO, "height": ALTO})

            def captura(nombre, espera_texto, timeout=60000):
                pagina.wait_for_selector(f"text={espera_texto}", timeout=timeout)
                pagina.wait_for_timeout(2500)     # deja asentar el render
                ruta = os.path.join(SALIDA, nombre)
                pagina.screenshot(path=ruta, full_page=True)
                generadas.append(ruta)
                print(f"[capturas] {ruta}", flush=True)

            # --- 1) pantalla de inicio ---------------------------------
            pagina.goto(url, wait_until="networkidle", timeout=90000)
            captura("pantalla_inicio.png", "Consulte su RUC")

            # --- 2) resultado (proveedor de ejemplo) -------------------
            pagina.get_by_role("button", name="Usar").first.click()
            captura("pantalla_resultado.png", "Sus oportunidades")

            # --- 3) ficha de detalle de un proceso ---------------------
            # La tabla se pinta en <canvas>: no hay celdas en el DOM. La
            # selección se hace en la casilla de la izquierda, por coordenadas.
            grid = pagina.locator("div[data-testid='stDataFrame']").first
            grid.scroll_into_view_if_needed()
            pagina.wait_for_timeout(800)
            caja = grid.bounding_box()
            pagina.mouse.click(caja["x"] + 22, caja["y"] + 35 + 35 * 2 + 17)
            pagina.wait_for_selector("text=Por qué apareció en su lista",
                                     timeout=60000)
            pagina.wait_for_timeout(2200)
            bloque = pagina.locator("div.detalle").first
            bloque.scroll_into_view_if_needed()
            pagina.wait_for_timeout(700)
            c = bloque.bounding_box()
            ruta = os.path.join(SALIDA, "pantalla_detalle_proceso.png")
            pagina.screenshot(path=ruta,
                              clip={"x": c["x"] - 10, "y": c["y"] - 10,
                                    "width": c["width"] + 20, "height": 760})
            generadas.append(ruta)
            print(f"[capturas] {ruta}", flush=True)

            # --- 4) sin coincidencias (proveedor muy diversificado) ----
            def nueva_consulta(ruc):
                pagina.get_by_role("button", name="Nueva consulta").first.click()
                pagina.wait_for_selector("text=Consulte su RUC", timeout=60000)
                pagina.wait_for_timeout(1400)
                caja = pagina.locator("input[type='text']").first
                caja.click()
                caja.fill(ruc)
                caja.press("Enter")
                pagina.wait_for_timeout(1400)
                pagina.get_by_role(
                    "button", name="Buscar mis oportunidades").first.click()

            nueva_consulta(RUC_DIVERSIFICADO)
            captura("pantalla_sin_coincidencias.png", "Sin resultado concluyente")

            # --- 5) exploración (RUC sin historial) --------------------
            nueva_consulta(RUC_SIN_HISTORIAL)
            captura("pantalla_exploracion.png", "Ubíquese por tipo de mercado")

            # --- 6) panel técnico desplegado ---------------------------
            pagina.get_by_text("Detalle técnico del modelo").first.click()
            pagina.wait_for_selector("text=Comparación de los seis algoritmos",
                                     timeout=60000)
            pagina.wait_for_timeout(2200)
            ruta = os.path.join(SALIDA, "pantalla_detalle_tecnico.png")
            pagina.locator("div[data-testid='stExpander']").last.screenshot(
                path=ruta)
            generadas.append(ruta)
            print(f"[capturas] {ruta}", flush=True)

            navegador.close()
    except Exception as e:
        instrucciones_manuales(f"{type(e).__name__}: {e}")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n[capturas] {len(generadas)} capturas generadas en {SALIDA}")
    for g in generadas:
        print(f"  {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
