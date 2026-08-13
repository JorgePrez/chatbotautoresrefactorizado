# -*- coding: utf-8 -*-
"""Rewrite ANALISIS and INSTRUCCIONES as clean UTF-8. ASCII-only source."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

analisis = """# Analisis y plan â€” Automatizacion de la bateria de robustez CHH

**Fecha:** 12 de agosto de 2026  
**Alcance:** solo esta carpeta `pruebas_robustez/` + scripts que *llaman* al codigo existente del repo (sin modificar prompts, corpus ni UI).  
**Estado:** decisiones confirmadas; scripts entregados; corrida de las 35 a cargo del usuario (ver `INSTRUCCIONES_EJECUCION.md`).

---

## 1. Que es este paquete

| Archivo | Rol |
|---|---|
| `PROTOCOLO_EJECUCION.md` | Reglas de ejecucion y devolucion |
| `bateria_robustez_chh_v1.json` | 35 pruebas (fuente programatica) |
| `bateria_robustez_chh_v1.csv` | Misma bateria en tabular |
| `plantilla_resultados_chh.csv` | Plantilla de salida (IDs precargados) |
| `ANALISIS_Y_PLAN_AUTOMATIZACION.md` | Este documento |
| `INSTRUCCIONES_EJECUCION.md` | Como correr smoke / dry-run / las 35 |

**Que mide:** comportamiento ante situaciones limite (fuera de dominio, manipulacion, abstencion, presion, formato, multi-turno), **no** conocimiento academico puro.

**Que no hace el ejecutor:** puntuar aprobado/fallido. Solo captura respuestas verbatim + metadatos.

### Distribucion de las 35 pruebas

| Cat. | Nombre | N |
|---|---|---|
| A | Fuera de dominio | 5 |
| B | Manipulacion | 5 |
| C | Integridad academica | 3 |
| D | Abstencion | 5 |
| E | Cruce de autores | 3 |
| F | Persona sensible | 4 |
| G | Presion del usuario | 4 |
| H | Formato | 4 |
| I | Multi-turno | 2 |

- Prioridad: ALTA 20 Â· MEDIA 9 Â· BAJA 6
- IDs `cualquiera`: R01, R02, R03, R06, R07, R08
- Multi-turno: R34, R35
- Especiales: R32 (texto largo), R14/R22 (regresion prioritaria)

---

## 2. Como funciona el chatbot (punto de entrada real)

```
interfaz_principal.py          -> login / selector de autores (Streamlit)
pages/{hayek,hazlitt,mises,
       muso,todos_autores}.py -> UI de chat, historial, DynamoDB
config/model_iacatching.py     -> MOTOR: RAG + Bedrock  << usado por los scripts
config/dynamo_crud.py          -> persistencia (NO usada en la bateria)
```

Chains: `run_hayek_chain`, `run_hazlitt_chain`, `run_mises_chain`, `run_muso_chain`, `run_general_chain`.

Historial: `[{"role":"user"|"assistant","content":"..."}]`.  
Respuesta: stream de chunks `{"response": ...}`.

---

## 3. Entorno de ejecucion

| Item | Valor |
|---|---|
| Interprete | `C:\\Users\\IT-14\\anaconda3\\python.exe` (Anaconda **base**, Python 3.11.7) |
| Evitar | Python 3.12 de Windows (sin langchain) |
| AWS | Produccion tal cual (`IS_TESTING=False`) |
| UTF-8 | Workspace de esta carpeta (opcion C); `.vscode/settings.json` ya puesto por el usuario |

---

## 4. Enfoque

- **No** automatizar UI Streamlit.
- **Si** llamar `model_iacatching` (mismo motor que pages).
- Sin DynamoDB.
- Replicar patron UI: append user al historial y pasar historial + question a la chain.

---

## 5. Scripts implementados

```
pruebas_robustez/
  scripts/
    chatbot_client.py
    runner_robustez.py
    multi_turno.py
    export_resultados.py
    smoke_check.py
    _gen_r32_fixture.py
  fixtures/
    r32_apuntes.txt          # PROPUESTO â€” revisar
    r35_secuencia.json       # PROPUESTO â€” revisar
  salidas/                   # CSV/JSON de resultados (gitignored)
  INSTRUCCIONES_EJECUCION.md
```

---

## 6. Flujo previsto para el usuario

1. Revisar fixtures R32 y R35
2. Smoke check
3. Dry-run
4. Corrida de las 35
5. Entregar CSV + meta.json

Detalle de comandos: `INSTRUCCIONES_EJECUCION.md`.

---

## 7. Que no se hace (salvo pedirlo)

- Cambiar prompts/KBs/`IS_TESTING` en el codigo de produccion
- Login Google / Playwright sobre Streamlit
- Puntuacion automatica pass/fail
- Commits/push sin peticion explicita

---

## 8. Decisiones confirmadas (12 ago 2026)

| # | Tema | Decision |
|---|---|---|
| 1 | Entorno AWS / modelo | **Produccion tal cual** (`IS_TESTING=False`) |
| 2 | `cualquiera` | **Round-robin**: hayek ? hazlitt ? mises ? muso ? general |
| 3 | Persistencia | **Solo archivos locales** en `salidas/` (sin DynamoDB) |
| 4 | R32 | Fixture **propuesto** en `fixtures/r32_apuntes.txt` â€” revision humana |
| 5 | R35 | Secuencia **propuesta** en `fixtures/r35_secuencia.json` â€” revision humana |
| 6 | Alcance 1.a entrega | **Solo scripts + instrucciones**; usuario ejecuta las 35 |
| 7 | UTF-8 (opcion C) | **Ya configurado** por el usuario; no se toca |

Round-robin de esta bateria:

- R01?hayek, R02?hazlitt, R03?mises, R06?muso, R07?general, R08?hayek

---

## 9. Criterio de exito

- Cada ID con `respuesta_verbatim` (o observacion clara de error tecnico)
- `chatbot_utilizado` y `fecha_hora_ejecucion` rellenados
- Metadatos de entorno/modelo en `.meta.json`
- Reproducible con Anaconda base y `--resume`

---

## 10. Proximo paso

1. Revisar `fixtures/r32_apuntes.txt` y `fixtures/r35_secuencia.json`
2. Seguir `INSTRUCCIONES_EJECUCION.md` (smoke ? dry-run ? 35)
"""

# Fix dashes to ASCII-friendly em dash via unicode
analisis = analisis.replace("â€”", "\u2014")
analisis = analisis.replace("?", "\u2192")
analisis = analisis.replace("Â·", "\u00b7")

instrucciones = """# Instrucciones de ejecucion \u2014 Bateria de robustez CHH (automatizada)

**Interprete obligatorio:** Anaconda base  
`C:\\Users\\IT-14\\anaconda3\\python.exe`

**Decisiones fijadas:** produccion tal cual \u00b7 round-robin en `cualquiera` \u00b7 sin DynamoDB \u00b7 fixtures R32/R35 propuestos (revisar) \u00b7 UTF-8 ya configurado en esta carpeta.

---

## 0. Antes de la corrida (revision humana)

1. Revisar y editar si hace falta:
   - `fixtures/r32_apuntes.txt` (texto sintetico largo; puedes recortar/sustituir; minimo ~3000 palabras)
   - `fixtures/r35_secuencia.json` (12 previas + pregunta final; la 2.a es la de referencia)
2. Abrir esta carpeta como workspace (opcion C) si quieres el editor en UTF-8.
3. Credenciales AWS del entorno de **produccion** disponibles (mismo criterio que `IS_TESTING=False` en `config/model_iacatching.py`).

---

## 1. Ir a la carpeta de scripts

En PowerShell:

```powershell
cd C:\\Users\\IT-14\\Documents\\bedrock\\2025_actualizacion\\chatbot_autores_individuales_actualizado_chats\\pruebas_robustez\\scripts
```

---

## 2. Smoke check (recomendado)

```powershell
C:\\Users\\IT-14\\anaconda3\\python.exe .\\smoke_check.py
```

Si falla un autor, no lances la bateria completa hasta resolver AWS/KB/red.

---

## 3. Dry-run (sin Bedrock)

```powershell
C:\\Users\\IT-14\\anaconda3\\python.exe .\\runner_robustez.py --dry-run
```

Round-robin esperado de `cualquiera`:

| ID | Autor asignado |
|---|---|
| R01 | hayek |
| R02 | hazlitt |
| R03 | mises |
| R06 | muso |
| R07 | general |
| R08 | hayek |

---

## 4. Ejecutar las 35 pruebas

```powershell
C:\\Users\\IT-14\\anaconda3\\python.exe .\\runner_robustez.py
```

Salida en:

- `pruebas_robustez/salidas/resultados_chh_YYYYMMDD_HHMMSS.csv`
- `pruebas_robustez/salidas/resultados_chh_YYYYMMDD_HHMMSS.meta.json`
- `pruebas_robustez/salidas/detalle_R34.json` y `detalle_R35.json` (cuando toquen)

Persistencia **solo local** (CSV incremental tras cada prueba). No escribe DynamoDB.

Tiempo orientativo: 30\u201390 minutos segun latencia/rate limits.

---

## 5. Opciones utiles

```powershell
# Solo prioridad ALTA
C:\\Users\\IT-14\\anaconda3\\python.exe .\\runner_robustez.py --solo-alta

# Subconjunto
C:\\Users\\IT-14\\anaconda3\\python.exe .\\runner_robustez.py --ids R01,R14,R22

# Continuar sobre un CSV ya empezado
C:\\Users\\IT-14\\anaconda3\\python.exe .\\runner_robustez.py --resume --salida ..\\salidas\\resultados_chh_XXXX.csv
```

---

## 6. Que entregar

El CSV de `salidas/` (columnas de `plantilla_resultados_chh.csv`) + el `.meta.json`.  
No hace falta puntuar pass/fail: solo respuestas verbatim.

---

## 7. Scripts incluidos

| Script | Rol |
|---|---|
| `chatbot_client.py` | Llama `run_*_chain` de `config.model_iacatching` |
| `runner_robustez.py` | Orquestador CLI de las 35 |
| `multi_turno.py` | R34 / R35 |
| `export_resultados.py` | CSV + metadatos |
| `smoke_check.py` | Prueba rapida por autor |
| `_gen_r32_fixture.py` | Regenera `fixtures/r32_apuntes.txt` si hace falta |
"""

(ROOT / "ANALISIS_Y_PLAN_AUTOMATIZACION.md").write_text(analisis, encoding="utf-8", newline="\n")
(ROOT / "INSTRUCCIONES_EJECUCION.md").write_text(instrucciones, encoding="utf-8", newline="\n")
print("docs ok")

# coding cookies on scripts
for py in (ROOT / "scripts").glob("*.py"):
    text = py.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not any("coding: utf-8" in ln for ln in lines[:2]):
        text = "# -*- coding: utf-8 -*-\n" + text
        py.write_text(text, encoding="utf-8", newline="\n")
        print("cookie", py.name)
